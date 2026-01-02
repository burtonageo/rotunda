// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, clippy::missing_safety_doc)]

//! Single-threaded reference-counting pointer types backed by an `Arena`.

use crate::{Arena, buffer::Buffer, handle::Handle, layout_repeat};
use alloc::alloc::{Allocator, Global, Layout};
use core::{
    any::Any,
    borrow::Borrow,
    cell::Cell,
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    hint::assert_unchecked,
    iter::IntoIterator,
    marker::{PhantomData, PhantomPinned},
    mem::{self, ManuallyDrop, MaybeUninit, offset_of},
    num::NonZero,
    ops::{Deref, Index},
    ptr::{self, NonNull},
    slice::{self, SliceIndex},
    str::Utf8Error,
};
#[cfg(feature = "nightly")]
use core::{
    marker::CoercePointee,
    ptr::{Pointee, Thin},
};
#[cfg(feature = "serde")]
use serde_core::ser::{Serialize, Serializer};

/// A reference-counted pointer to some memory backed by an [`Arena`],
/// analogous to [`Rc<T>`].
///
/// See the [module documentation] for more information.
///
/// [`Arena`]: ../struct.Arena.html
/// [`Rc<T>`]: https://doc.rust-lang.org/stable/std/rc/struct.Rc.html
/// [module documentation]: ./index.html
#[cfg_attr(feature = "nightly", derive(CoercePointee))]
#[repr(transparent)]
pub struct RcHandle<'a, #[cfg_attr(feature = "nightly", pointee)] T: ?Sized, A: Allocator = Global>
{
    ptr: NonNull<RcHandleInner<'a, T, A>>,
    _boo: PhantomData<RcHandleInner<'a, T, A>>,
}

const _: () = assert!(mem::size_of::<RcHandle<()>>() == mem::size_of::<NonNull<()>>());
const _: () = assert!(mem::align_of::<RcHandle<()>>() == mem::align_of::<NonNull<()>>());

impl<'a, T, A: Allocator> RcHandle<'a, T, A> {
    /// Create a new `RcHandle` containing the given `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let rc_handle = RcHandle::new_in(&arena, 25);
    /// assert_eq!(&rc_handle, &25);
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_in(arena: &'a Arena<A>, value: T) -> Self {
        let handle = RcHandle::new_uninit_in(arena);
        RcHandle::init(handle, value)
    }

    /// Create a new `RcHandle` containing the return value of the given function.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let rc_handle = RcHandle::new_with(&arena, || "Hello!");
    /// assert_eq!(&rc_handle, &"Hello!");
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_with<F: FnOnce() -> T>(arena: &'a Arena<A>, f: F) -> Self {
        let handle = RcHandle::new_uninit_in(arena);
        RcHandle::init(handle, f())
    }

    #[track_caller]
    #[must_use]
    #[inline]
    pub unsafe fn init_with<F: FnOnce(&mut MaybeUninit<T>)>(arena: &'a Arena<A>, f: F) -> Self {
        let mut handle = RcHandle::new_uninit_in(arena);
        unsafe {
            f(RcHandle::get_mut_unchecked(&mut handle));
            RcHandle::assume_init(handle)
        }
    }
}

impl<'a, T, A: Allocator> RcHandle<'a, [T], A> {
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_slice_from_fn_in<F: FnMut(usize) -> T>(
        arena: &'a Arena<A>,
        slice_len: usize,
        mut f: F,
    ) -> Self {
        struct Guard<'a, T> {
            slice: &'a mut [MaybeUninit<T>],
            len: usize,
        }

        impl<'a, T> Drop for Guard<'a, T> {
            #[inline]
            fn drop(&mut self) {
                let slice = unsafe {
                    let ptr = self.slice.as_mut_ptr().cast::<T>();
                    slice::from_raw_parts_mut(ptr, self.len)
                };
                unsafe { ptr::drop_in_place(slice) };
            }
        }

        let mut handle = RcHandle::new_slice_uninit_in(arena, slice_len);

        let slice = unsafe { RcHandle::get_mut_unchecked(&mut handle) };

        let mut guard = Guard { slice, len: 0 };

        for i in 0..slice_len {
            let slot = unsafe { guard.slice.get_unchecked_mut(i) };
            slot.write(f(i));
            guard.len += 1;
        }

        mem::forget(guard);

        unsafe { RcHandle::assume_init_slice(handle) }
    }

    #[inline]
    pub fn try_into_buffer(this: Self) -> Result<Buffer<'a, T, A>, Self> {
        Self::try_into_handle(this).map(Buffer::from_slice_handle)
    }

    #[inline]
    pub fn into_buffer(this: Self) -> Option<Buffer<'a, T, A>> {
        Self::try_into_buffer(this).ok()
    }
}

impl<'a, T, const N: usize, A: Allocator> RcHandle<'a, [T; N], A> {
    /// Create a new `RcHandle` to a fixed size array, where each element is initialized
    /// by calling `f` on its index into the array.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let rc_handle = RcHandle::<'_, [usize; 10]>::new_array_from_fn_in(&arena, |i| i);
    /// assert_eq!(rc_handle.as_ref(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_array_from_fn_in<F: FnMut(usize) -> T>(arena: &'a Arena<A>, f: F) -> Self {
        let rc = RcHandle::new_slice_from_fn_in(arena, N, f);
        let ptr = rc.ptr.cast::<RcHandleInner<'_, [T; N], A>>();
        mem::forget(rc);
        RcHandle {
            ptr,
            _boo: PhantomData,
        }
    }
}

impl<'a, T: Copy, A: Allocator, const N: usize> RcHandle<'a, [T; N], A> {
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_array_splat_in(arena: &'a Arena<A>, value: T) -> Self {
        let handle = RcHandle::<'_, MaybeUninit<[T; N]>, A>::new_uninit_in(arena);
        let mut handle = RcHandle::<'_, MaybeUninit<[T; N]>, A>::transpose_inner_uninit(handle);

        {
            let handle_ref = unsafe { RcHandle::get_mut_unchecked(&mut handle) };

            for i in 0..N {
                let slot = unsafe { handle_ref.get_unchecked_mut(i) };
                slot.write(value);
            }
        }

        unsafe { RcHandle::assume_init_array(handle) }
    }
}

impl<'a, T: Copy, A: Allocator> RcHandle<'a, [T], A> {
    /// Create a new `RcHandle` containing a copy of the given slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let rc_handle = RcHandle::new_slice_in(&arena, &[1, 2, 3, 4, 5]);
    /// assert_eq!(&rc_handle, &[1, 2, 3, 4, 5]);
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_slice_in<S: ?Sized + AsRef<[T]>>(arena: &'a Arena<A>, slice: &'_ S) -> Self {
        let slice = slice.as_ref();
        let mut rc_handle = RcHandle::new_slice_uninit_in(arena, slice.len());

        unsafe {
            let slice =
                { slice::from_raw_parts(slice.as_ptr().cast::<MaybeUninit<T>>(), slice.len()) };

            RcHandle::get_mut_unchecked(&mut rc_handle).copy_from_slice(slice);
            RcHandle::assume_init_slice(rc_handle)
        }
    }

    /// Create a new `RcHandle` to a shared value of a slice with size `slice_len`, and
    /// each element initialized to `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let rc_handle = RcHandle::new_splat_in(&arena, 5, 128u32);
    /// assert_eq!(&rc_handle, &[128u32; 5]);
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_splat_in(arena: &'a Arena<A>, slice_len: usize, value: T) -> RcHandle<'a, [T], A> {
        let mut hndl = RcHandle::new_slice_uninit_in(arena, slice_len);
        let slots = unsafe { RcHandle::get_mut_unchecked(&mut hndl) };

        slots.fill(MaybeUninit::new(value));

        unsafe { RcHandle::assume_init_slice(hndl) }
    }
}

impl<'a, T: Default, A: Allocator> RcHandle<'a, T, A> {
    /// Create a new `RcHandle` containing the default value.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let handle = RcHandle::<bool>::new_default_in(&arena);
    /// assert_eq!(*handle, false);
    /// ```
    #[must_use]
    #[inline]
    pub fn new_default_in(arena: &'a Arena<A>) -> Self {
        RcHandle::new_with(arena, Default::default)
    }
}

impl<'a, T: ?Sized, A: Allocator> RcHandle<'a, T, A> {
    #[inline]
    pub const fn arena(this: &Self) -> &'a Arena<A> {
        Self::inner(this).arena
    }

    /// Returns a raw pointer to the contents of this `RcHandle`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let handle = RcHandle::new_in(&arena, 1902usize);
    ///
    /// let ptr: *const usize = RcHandle::as_ptr(&handle);
    ///
    /// unsafe {
    ///     assert_eq!(ptr.as_ref(), Some(&1902));
    /// }
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_ptr(this: &Self) -> *const T {
        unsafe {
            this.ptr
                .byte_add(offset_of!(RcHandleInner<'_, (), A>, data))
                .as_ptr()
                .cast_const() as *const _
        }
    }

    /// Consumes the `RcHandle`, returning the underlying wrapped pointer.
    ///
    /// To avoid a memory leak, the pointer should be converted back using
    /// [`RcHandle::from_raw()`].
    ///
    /// # Examples
    ///
    /// ```
    /// # #![cfg_attr(feature = "nightly", feature(allocator_api))]
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let rc = RcHandle::new_in(&arena, 85);
    /// let raw = RcHandle::into_raw(rc);
    /// unsafe { assert_eq!(*raw, 85); }
    /// # #[cfg(all(feature = "allocator-api2", not(feature = "nightly")))]
    /// # use allocator_api2::alloc::Global;
    /// # #[cfg(feature = "nightly")]
    /// # use std::alloc::Global;
    /// # let _ = unsafe { RcHandle::<'_, _, Global>::from_raw(raw) };
    /// ```
    #[must_use]
    #[inline]
    pub const fn into_raw(self) -> *const T {
        let ptr = Self::as_ptr(&self);
        let _this = ManuallyDrop::new(self);
        ptr
    }

    /// Create a new `WeakHandle` from this `RcHandle`.
    ///
    /// Upgrading the returned `WeakHandle` will increment this `RcHandle`'s refcount.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let rc = RcHandle::new_in(&arena, 25);
    ///
    /// let weak = RcHandle::downgrade(&rc);
    /// # assert!(RcHandle::ptr_eq(&rc, &weak));
    /// ```
    #[must_use]
    #[inline]
    pub const fn downgrade(this: &Self) -> WeakHandle<'a, T, A> {
        WeakHandle { ptr: this.ptr }
    }

    /// Returns a mutable reference to the contents of this `RcHandle` if
    /// it uniquely owns its contents.
    ///
    /// If there are other live `RcHandle`s to the shared inner value,
    /// this method returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut rc = RcHandle::new_in(&arena, 25);
    ///
    /// if let Some(inner) = RcHandle::get_mut(&mut rc) {
    ///     *inner = 42;
    /// }
    ///
    /// assert_eq!(*rc, 42);
    ///
    /// let rc_2 = RcHandle::clone(&rc);
    ///
    /// assert!(RcHandle::get_mut(&mut rc).is_none());
    /// ```
    #[must_use]
    #[inline]
    pub const fn get_mut(this: &mut Self) -> Option<&mut T> {
        if Self::is_unique(this) {
            unsafe { Some(Self::get_mut_unchecked(this)) }
        } else {
            None
        }
    }

    /// Returns a mutable reference to the contents of this `RcHandle`.
    ///
    /// # Safety
    ///
    /// You should only call this method if you are sure that the `RcHandle`
    /// is the only live handle to the shared value, otherwise it is possible
    /// to mutably alias the shared value.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// const DATA: &'_ str = "This message";
    ///
    /// const BUF_SIZE: usize = 20;
    /// let mut rc_handle = RcHandle::new_splat_in(&arena, BUF_SIZE, 0u8);
    /// # assert!(rc_handle.len() >= DATA.len());
    ///
    /// unsafe {
    ///     let contents = RcHandle::get_mut_unchecked(&mut rc_handle);
    ///     core::ptr::copy_nonoverlapping(DATA.as_bytes().as_ptr(), contents.as_mut_ptr(), DATA.len());
    /// }
    ///
    /// assert_eq!(&rc_handle[..DATA.len()], DATA.as_bytes());
    /// assert_eq!(&rc_handle[DATA.len()..], &[0u8; const { BUF_SIZE - DATA.len() }]);
    /// assert_eq!(rc_handle.len(), BUF_SIZE);
    /// ```
    #[must_use]
    #[inline]
    pub const unsafe fn get_mut_unchecked(this: &mut Self) -> &mut T {
        unsafe { &mut this.ptr.as_mut().data }
    }

    /// Attempt to convert the `RcHandle` into a uniquely owned `Handle`.
    ///
    /// If there are other live `RcHandle`s to the shared value, then
    /// the original `RcHandle` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::{RcHandle, WeakHandle}};
    ///
    /// let arena = Arena::new();
    ///
    /// let handle = RcHandle::new_in(&arena, 25);
    /// let weak = RcHandle::downgrade(&handle);
    ///
    /// {
    ///     let handle_2 = RcHandle::clone(&handle);
    ///     let result = RcHandle::try_into_handle(handle_2);
    ///     assert!(result.is_err());
    /// }
    ///
    /// let handle = RcHandle::try_into_handle(handle).unwrap();
    /// assert_eq!(*handle, 25);
    /// assert!(WeakHandle::upgrade(&weak).is_none());
    /// ```
    #[inline]
    pub fn try_into_handle(this: Self) -> Result<Handle<'a, T, A>, Self> {
        if Self::is_unique(&this) {
            let arena = RcHandle::arena(&this);

            // Decrement the ref count to `0` to invalidate all
            // `WeakHandle`s which point here.
            Self::inner(&this).mark_inaccessible();

            let raw = this
                .ptr
                .as_ptr()
                .map_addr(|addr| addr + offset_of!(RcHandleInner<'_, (), A>, data))
                as *mut T;

            let _this = ManuallyDrop::new(this);
            unsafe { Ok(Handle::from_raw_in(raw, arena)) }
        } else {
            Err(this)
        }
    }

    /// Converts this `RcHandle` into a uniquely owned `Handle`.
    ///
    /// If there are other live `RcHandle`s to the shared value, this method
    /// returns `None`.
    ///
    /// Note that any `WeakHandle`s into this handle will not be able to
    /// be resurrected with [`WeakHandle::try_resurrect()`], as it cannot
    /// ensure that there is not a live `Handle` to the data part of the shared
    /// value.
    ///
    /// If you will need to resurrect any `WeakHandle`s to this data, use
    /// [`Handle::into_inner()`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let rc = RcHandle::new_str_in(&arena, "Hello!");
    ///
    /// {
    ///     let rc2 = RcHandle::clone(&rc);
    ///     assert_eq!(RcHandle::into_handle(rc2), None);
    /// }
    ///
    /// let handle = RcHandle::into_handle(rc).unwrap();
    /// assert_eq!(handle, "Hello!");
    /// ```
    #[must_use]
    #[inline]
    pub fn into_handle(this: Self) -> Option<Handle<'a, T, A>> {
        Self::try_into_handle(this).ok()
    }

    /// Returns the number of strong references to the shared value.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let handle = RcHandle::new_in(&arena, ());
    ///
    /// assert_eq!(RcHandle::ref_count(&handle), 1);
    ///
    /// let handle_2 = RcHandle::clone(&handle);
    ///
    /// assert_eq!(RcHandle::ref_count(&handle), 2);
    /// # let _ = handle_2;
    /// ```
    #[must_use]
    #[inline]
    pub const fn ref_count(this: &Self) -> usize {
        let count = Self::inner(this).count.get();
        debug_assert!(count > 0);
        count
    }

    /// Returns the number of strong references to the shared value.
    ///
    /// As there must be at least one `RcHandle` owner of the shared value,
    /// it is always valid to represent the count as a `NonZero<usize>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    /// use core::cell::Cell;
    ///
    /// let arena = Arena::new();
    ///
    /// let handle = RcHandle::new_in(&arena, Cell::new(1u32));
    ///
    /// assert_eq!(RcHandle::non_zero_ref_count(&handle).get(), 1);
    /// ```
    #[must_use]
    #[inline]
    pub const fn non_zero_ref_count(this: &Self) -> NonZero<usize> {
        unsafe { NonZero::new_unchecked(Self::ref_count(this)) }
    }

    /// Check if this is the only `RcHandle` which has ownership of the shared value.
    ///
    /// If this is `true`, then [`RcHandle::get_mut()`] will succeed.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::marker::PhantomData;
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let rc = RcHandle::new_in(&arena, PhantomData::<usize>);
    ///
    /// assert!(RcHandle::is_unique(&rc));
    ///
    /// let rc_2 = RcHandle::clone(&rc);
    ///
    /// assert!(!RcHandle::is_unique(&rc));
    /// ```
    #[must_use]
    #[inline]
    pub const fn is_unique(this: &Self) -> bool {
        RcHandle::ref_count(this) == 1
    }

    /// Returns `true` if the pointer value of `self` is equal to `other`.
    ///
    /// If this returns `true`, then both handles must be pointing to the same
    /// shared value.
    ///
    /// This method takes `other` by `Into<WeakHandle<'a, U>>`, which means that
    /// this equality can be compared to both `RcHandle`s and `WeakHandle`s.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let (rc, rc_2) = (RcHandle::new_in(&arena, 0usize), RcHandle::new_in(&arena, 0usize));
    /// let cloned = RcHandle::clone(&rc);
    ///
    /// assert!(RcHandle::ptr_eq(&rc, &cloned));
    /// assert!(!RcHandle::ptr_eq(&rc, &rc_2));
    /// ```
    #[must_use]
    #[inline]
    pub fn ptr_eq<'b, A2: Allocator + 'b, Rhs: Into<WeakHandle<'b, U, A2>>, U: ?Sized>(
        this: &Self,
        other: Rhs,
    ) -> bool {
        ptr::eq(
            Self::as_ptr(this).cast::<()>(),
            WeakHandle::as_ptr(&other.into()).cast::<()>(),
        )
    }

    /// Hash the pointer into the given `hasher`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::hash::{Hasher, DefaultHasher};
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let mut hasher = DefaultHasher::new();
    ///
    /// let arena = Arena::new();
    ///
    /// let rc = RcHandle::new_in(&arena, [0u8; 255]);
    ///
    /// RcHandle::ptr_hash(&rc, &mut hasher);
    /// let code = hasher.finish();
    /// ```
    #[inline]
    pub fn ptr_hash<H: Hasher>(this: &Self, hasher: &mut H) {
        ptr::hash(RcHandle::as_ptr(this), hasher);
    }

    #[inline]
    pub unsafe fn from_raw(raw: *const T) -> Self {
        let raw = unsafe { raw.byte_sub(offset_of!(RcHandleInner<'_, (), A>, data)) as *mut _ };
        unsafe { Self::from_raw_inner(raw) }
    }

    #[must_use]
    #[inline]
    const unsafe fn from_raw_inner(raw: *const RcHandleInner<'a, T, A>) -> Self {
        let ptr = unsafe { NonNull::new_unchecked(raw.cast_mut()) };
        let handle = Self {
            ptr,
            _boo: PhantomData,
        };

        debug_assert!(Self::ref_count(&handle) > 0);
        handle
    }

    #[must_use]
    #[inline]
    const fn inner(this: &Self) -> &RcHandleInner<'a, T, A> {
        unsafe { this.ptr.as_ref() }
    }

    #[must_use]
    #[inline]
    const fn clone_const(&self) -> RcHandle<'a, T, A> {
        let inner = Self::inner(self);
        inner.increment_refcount();

        Self {
            ptr: self.ptr,
            _boo: PhantomData,
        }
    }

    #[must_use]
    #[inline]
    const fn as_ref_const(&self) -> &T {
        &Self::inner(self).data
    }
}

impl<'a, T: Unpin, A: Allocator> RcHandle<'a, T, A> {
    /// Consumes the `RcHandle`, returning the inner value.
    ///
    /// If this `RcHandle` does not have sole ownership of the wrapped value, then
    /// it is returned in the `Err` variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let rc = RcHandle::new_in(&arena, 25i32);
    ///
    /// let data: i32 = RcHandle::try_unwrap(rc).unwrap();
    /// assert_eq!(data, 25);
    /// ```
    #[inline]
    pub fn try_unwrap(this: Self) -> Result<T, RcHandle<'a, T, A>> {
        let inner = this.ptr;
        let value = Self::try_into_handle(this).map(Handle::into_inner)?;

        unsafe {
            // The handle has been moved out from, so it is valid to reset the count
            // to `0` so that other `WeakHandle`s can resurrect this `RcHandle`.
            inner.as_ref().count.set(0);
        }

        Ok(value)
    }

    /// Consumes the `RcHandle`, returning the inner value.
    ///
    /// If this `RcHandle` does not have sole ownership of the wrapped value, then
    /// `None` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let rc = RcHandle::new_in(&arena, 4224i32);
    ///
    /// let data: Option<i32> = RcHandle::into_inner(rc);
    /// assert_eq!(data, Some(4224));
    /// ```
    #[must_use]
    #[inline]
    pub fn into_inner(this: Self) -> Option<T> {
        Self::try_unwrap(this).ok()
    }
}

impl<'a, A: Allocator> RcHandle<'a, dyn Any, A> {
    #[inline]
    pub fn downcast<T: Any>(self) -> Result<RcHandle<'a, T, A>, RcHandle<'a, dyn Any, A>> {
        if (*self).is::<T>() {
            unsafe { Ok(Self::downcast_unchecked::<T>(self)) }
        } else {
            Err(self)
        }
    }

    #[must_use]
    #[inline]
    pub unsafe fn downcast_unchecked<T: Any>(self) -> RcHandle<'a, T, A> {
        unsafe {
            let ptr = Self::into_raw(self);
            RcHandle::from_raw(ptr as *const T)
        }
    }
}

impl<'a, T, A: Allocator> RcHandle<'a, MaybeUninit<T>, A> {
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_uninit_in(arena: &'a Arena<A>) -> RcHandle<'a, MaybeUninit<T>, A> {
        let layout = const { rc_inner_layout_for_value_layout(Layout::new::<T>()) };

        let ptr = arena
            .alloc_raw(layout)
            .cast::<RcHandleInner<'a, MaybeUninit<T>, A>>();

        unsafe {
            RcHandleInner::init_raw(ptr, arena);
            RcHandle::from_raw_inner(ptr.as_ptr().cast_const())
        }
    }

    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_uninit_zeroed_in(arena: &'a Arena<A>) -> RcHandle<'a, MaybeUninit<T>, A> {
        let layout = const { rc_inner_layout_for_value_layout(Layout::new::<T>()) };

        let ptr = arena
            .alloc_raw_zeroed(layout)
            .cast::<RcHandleInner<'_, MaybeUninit<T>, A>>();

        unsafe {
            RcHandleInner::init_raw(ptr, arena);
            RcHandle::from_raw_inner(ptr.as_ptr().cast_const())
        }
    }

    /// Returns a new `RcHandle` which assumes that the contents have been initialized.
    ///
    /// It is the programmer's responsibility to ensure that the `RcHandle` is only initialized
    /// once. If the uninitialized `RcHandle` is cloned and initialized multiple times, or if
    /// the initialized `RcHandle` is dropped before any uninitialized clones, then the destructor
    /// may be skipped.
    ///
    /// # Safety
    ///
    /// It is the programmer's responsibility to ensure that all invariants which relate to
    /// `MaybeUninit::assume_init()` are upheld - namely that the contained value has
    /// been fully initialized.
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut rc_uninit = RcHandle::new_uninit_in(&arena);
    ///
    /// let rc = unsafe {
    ///     RcHandle::get_mut_unchecked(&mut rc_uninit).write(6000usize);
    ///     RcHandle::assume_init(rc_uninit)
    /// };
    ///
    /// assert_eq!(*rc, 6000);
    /// ```
    #[must_use]
    #[inline]
    pub const unsafe fn assume_init(this: Self) -> RcHandle<'a, T, A> {
        unsafe { mem::transmute(this) }
    }

    /// Initialize the `RcHandle` with the given value and return the initialized handle.
    ///
    /// It is the programmer's responsibility to ensure that the `RcHandle` is only initialized
    /// once. If the uninitialized `RcHandle` is cloned and initialized multiple times, or if
    /// the initialized `RcHandle` is dropped before any uninitialized clones, then the destructor
    /// may be skipped.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let rc_uninit = RcHandle::new_uninit_in(&arena);
    /// let rc = RcHandle::init(rc_uninit, 420usize);
    ///
    /// assert_eq!(*rc, 420);
    /// ```
    #[must_use]
    #[inline]
    pub const fn init(mut this: Self, value: T) -> RcHandle<'a, T, A> {
        unsafe {
            let inner = RcHandle::get_mut_unchecked(&mut this);
            let _ = inner.write(value);

            RcHandle::assume_init(this)
        }
    }
}

impl<'a, T, A: Allocator, const N: usize> RcHandle<'a, MaybeUninit<[T; N]>, A> {
    /// Transpose a `RcHandle<MaybeUninit<[T; N]>>` to an `RcHandle<[MaybeUninit<T>; N]>`.
    #[must_use]
    #[inline]
    pub const fn transpose_inner_uninit(this: Self) -> RcHandle<'a, [MaybeUninit<T>; N], A> {
        unsafe { mem::transmute(this) }
    }
}

impl<'a, T, A: Allocator, const N: usize> RcHandle<'a, [MaybeUninit<T>; N], A> {
    #[must_use]
    #[inline]
    pub const unsafe fn assume_init_array(this: Self) -> RcHandle<'a, [T; N], A> {
        unsafe { mem::transmute(this) }
    }

    #[must_use]
    #[inline]
    pub const fn transpose_outer_uninit(this: Self) -> RcHandle<'a, MaybeUninit<[T; N]>, A> {
        unsafe { mem::transmute(this) }
    }
}

impl<'a, T, A: Allocator> RcHandle<'a, [MaybeUninit<T>], A> {
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_slice_uninit_in(arena: &'a Arena<A>, slice_len: usize) -> Self {
        let (array_layout, ..) =
            layout_repeat(&Layout::new::<T>(), slice_len).expect("size overflow");
        let inner_layout = rc_inner_layout_for_value_layout(array_layout);

        unsafe {
            let ptr = arena
                .alloc_raw(inner_layout)
                .cast::<RcHandleInner<'_, MaybeUninit<T>, A>>();

            RcHandleInner::init_raw(ptr, arena);

            let ptr = RcHandleInner::cast_to_slice(ptr.as_ptr(), slice_len);
            RcHandle::from_raw_inner(ptr)
        }
    }

    #[must_use]
    #[inline]
    pub const unsafe fn assume_init_slice(this: Self) -> RcHandle<'a, [T], A> {
        let ptr = this.ptr.as_ptr() as *mut RcHandleInner<'_, [T], A>;
        let ptr = unsafe { NonNull::new_unchecked(ptr) };
        let _this = ManuallyDrop::new(this);

        RcHandle {
            ptr,
            _boo: PhantomData,
        }
    }
}

impl<'a, A: Allocator> RcHandle<'a, str, A> {
    /// Create a new `RcHandle` containing the given `string`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let handle = RcHandle::new_str_in(&arena, "Some Data");
    ///
    /// assert_eq!(&handle, "Some Data");
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_str_in<S: ?Sized + AsRef<str>>(arena: &'a Arena<A>, string: &'_ S) -> Self {
        unsafe { Self::new_from_utf8_unchecked_in_inner(arena, string.as_ref().as_bytes()) }
    }

    /// Create a new `RcHandle` containing the given bytes as a string.
    ///
    /// # Errors
    ///
    /// If `bytes` is not a valid utf8 encoded string, a `Utf8Error` will be returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let valid = RcHandle::new_from_utf8_in(&arena, b"Hello!").unwrap();
    /// assert_eq!(&valid, "Hello!");
    /// ```
    #[track_caller]
    #[inline]
    pub fn new_from_utf8_in<B: ?Sized + AsRef<[u8]>>(
        arena: &'a Arena<A>,
        bytes: &B,
    ) -> Result<Self, Utf8Error> {
        str::from_utf8(bytes.as_ref()).map(|s| RcHandle::new_str_in(arena, s))
    }

    /// Create a new `RcHandle` containing the given bytes as a string.
    /// This factory method will not perform any validation.
    ///
    /// # Safety
    ///
    /// If `bytes` is not a valid utf8 encoded string, then the returned `RcHandle` will
    /// not be able to uphold safety invariants around the wrapped `str`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let valid = unsafe { RcHandle::new_from_utf8_unchecked_in(&arena, b"This is a message") };
    /// assert_eq!(&valid, "This is a message");
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub unsafe fn new_from_utf8_unchecked_in<B: ?Sized + AsRef<[u8]>>(
        arena: &'a Arena<A>,
        bytes: &B,
    ) -> Self {
        unsafe { Self::new_from_utf8_unchecked_in_inner(arena, bytes.as_ref()) }
    }

    #[track_caller]
    #[must_use]
    #[inline]
    unsafe fn new_from_utf8_unchecked_in_inner(arena: &'a Arena<A>, bytes: &[u8]) -> Self {
        let string_len = bytes.len();
        let mut rc_handle = RcHandle::new_slice_uninit_in(arena, string_len);

        unsafe {
            let slice = RcHandle::get_mut_unchecked(&mut rc_handle);
            let string_bytes = {
                let data = bytes.as_ptr().cast::<MaybeUninit<u8>>();
                slice::from_raw_parts(data, string_len)
            };

            slice.copy_from_slice(string_bytes);
        }

        let ptr = unsafe {
            NonNull::new_unchecked(rc_handle.ptr.as_ptr() as *mut RcHandleInner<'_, str, A>)
        };
        mem::forget(rc_handle);

        RcHandle {
            ptr,
            _boo: PhantomData,
        }
    }

    /// Consumes this `RcHandle`, returning a new `RcHandle` over the bytes of the string.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let rc = RcHandle::new_str_in(&arena, "Hello");
    /// let bytes = RcHandle::into_bytes(rc);
    /// assert_eq!(&bytes, &[b'H', b'e', b'l', b'l', b'o']);
    /// ```
    #[inline]
    pub const fn into_bytes(this: Self) -> RcHandle<'a, [u8], A> {
        unsafe { mem::transmute(this) }
    }
}

impl<'a, A: Allocator> RcHandle<'a, [u8], A> {
    /// Converts the `RcHandle<'_, [u8]>` into a `RcHandle<'_, str>`, performing validation
    /// on the contents. A `Clone::clone` of the original data is returned on success.
    ///
    /// # Errors
    ///
    /// If the `RcHandle` does not contain a byte array which is a valid utf8 string, then
    /// an error will be returned. The original bytes are not consumed by this function.
    ///
    /// # Panics
    ///
    /// This method will panic if calling the `clone()` method would panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let bytes = RcHandle::new_slice_in(&arena, "今日は".as_bytes());
    ///
    /// let string = RcHandle::to_utf8(&bytes).unwrap();
    /// assert_eq!(&string, "今日は");
    /// ```
    #[inline]
    pub const fn to_utf8(this: &Self) -> Result<RcHandle<'a, str, A>, Utf8Error> {
        match str::from_utf8(this.as_ref_const()) {
            Ok(_) => unsafe { Ok(Self::into_utf8_unchecked(this.clone_const())) },
            Err(e) => Err(e),
        }
    }

    /// Converts the `RcHandle<'_, [u8]>` into a `RcHandle<'_, str>`, without validating
    /// the contents of the slice.
    ///
    /// # Safety
    ///
    /// If `this` is not a valid utf8 encoded string, then the returned `RcHandle` will
    /// not be able to uphold safety invariants around the wrapped `str`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::new();
    ///
    /// let bytes = RcHandle::new_slice_in(&arena, &[72u8, 101, 108, 108, 111, 0]);
    /// let hello = unsafe { RcHandle::into_utf8_unchecked(bytes) };
    ///
    /// assert_eq!(&hello, "Hello\0");
    /// ```
    #[inline]
    pub const unsafe fn into_utf8_unchecked(this: Self) -> RcHandle<'a, str, A> {
        unsafe { mem::transmute(this) }
    }
}

#[cfg(feature = "nightly")]
impl<'a, T: ?Sized + Pointee, A: Allocator> RcHandle<'a, T, A> {
    #[must_use]
    #[inline]
    pub unsafe fn from_raw_parts(
        data: *const impl Thin,
        metadata: <T as Pointee>::Metadata,
    ) -> Self {
        let ptr = ptr::from_raw_parts(data, metadata);
        unsafe { Self::from_raw(ptr) }
    }

    #[must_use]
    #[inline]
    pub const fn to_raw_parts(self) -> (*const (), <T as Pointee>::Metadata) {
        let raw = Self::into_raw(self);
        <*const T>::to_raw_parts(raw)
    }
}

impl<'a, T: ?Sized + fmt::Debug, A: Allocator> fmt::Debug for RcHandle<'a, T, A> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_ref(), fmtr)
    }
}

impl<'a, T: ?Sized + fmt::Display, A: Allocator> fmt::Display for RcHandle<'a, T, A> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_ref(), fmtr)
    }
}

impl<'a, T: ?Sized, A: Allocator> fmt::Pointer for RcHandle<'a, T, A> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, fmtr)
    }
}

impl<'a, T: ?Sized, A: Allocator> AsRef<T> for RcHandle<'a, T, A> {
    #[inline]
    fn as_ref(&self) -> &T {
        self.as_ref_const()
    }
}

impl<'a, A: Allocator> AsRef<[u8]> for RcHandle<'a, str, A> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl<'a, T: ?Sized, A: Allocator> Borrow<T> for RcHandle<'a, T, A> {
    #[inline]
    fn borrow(&self) -> &T {
        &Self::inner(self).data
    }
}

impl<'a, T: ?Sized, A: Allocator> Deref for RcHandle<'a, T, A> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &Self::inner(self).data
    }
}

impl<'a, T, A: Allocator, I: SliceIndex<[T]>> Index<I> for RcHandle<'a, [T], A> {
    type Output = <[T] as Index<I>>::Output;
    #[track_caller]
    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        self.as_ref().index(index)
    }
}

impl<'a, T: ?Sized, A: Allocator> Clone for RcHandle<'a, T, A> {
    /// Increment the reference count of a `RcHandle` pointer, and return a new `RcHandle` to the
    /// shared data. The underlying shared value is not cloned, so this operation is computationally
    /// cheap.
    ///
    /// # Panics
    ///
    /// This method will panic if incrementing the reference count would overflow [`usize::MAX - 1`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::RcHandle};
    ///
    /// let arena = Arena::with_block_size(u32::MAX as usize);
    ///
    /// let rc = RcHandle::new_splat_in(&arena, 1024 * 2, 0u8);
    ///
    /// // Clone the `rc`. This increments the reference count, but does not copy any of the
    /// // underlying data.
    /// let rc_2 = RcHandle::clone(&rc);
    /// ```
    ///
    /// [`usize::MAX - 1`]: https://doc.rust-lang.org/stable/std/primitive.usize.html#associatedconstant.MAX
    #[track_caller]
    #[inline]
    fn clone(&self) -> Self {
        self.clone_const()
    }
}

impl<'a, T: ?Sized + PartialEq, A: Allocator> PartialEq for RcHandle<'a, T, A> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_ref().eq(other.as_ref())
    }
}

impl<'a, T: ?Sized + PartialEq, A: Allocator> PartialEq<T> for RcHandle<'a, T, A> {
    #[inline]
    fn eq(&self, other: &T) -> bool {
        self.as_ref().eq(other)
    }
}

impl<'a, T: PartialEq<U>, U, A: Allocator, const N: usize> PartialEq<[U; N]>
    for RcHandle<'a, [T], A>
{
    #[inline]
    fn eq(&self, other: &[U; N]) -> bool {
        PartialEq::eq(self.as_ref(), &other[..])
    }
}

impl<'a, T: ?Sized + Eq, A: Allocator> Eq for RcHandle<'a, T, A> {}

impl<'a, T: ?Sized + PartialOrd, A: Allocator> PartialOrd for RcHandle<'a, T, A> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_ref().partial_cmp(other.as_ref())
    }
}

impl<'a, T: ?Sized + PartialOrd, A: Allocator> PartialOrd<T> for RcHandle<'a, T, A> {
    #[inline]
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        self.as_ref().partial_cmp(other)
    }
}

impl<'a, T: ?Sized + Ord, A: Allocator> Ord for RcHandle<'a, T, A> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_ref().cmp(other.as_ref())
    }
}

impl<'a, T: ?Sized + Hash, A: Allocator> Hash for RcHandle<'a, T, A> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state)
    }
}

impl<'a, T, const N: usize, A: Allocator> TryFrom<RcHandle<'a, [T], A>>
    for RcHandle<'a, [T; N], A>
{
    type Error = RcHandle<'a, [T], A>;
    #[inline]
    fn try_from(value: RcHandle<'a, [T], A>) -> Result<Self, Self::Error> {
        if value.len() == N {
            unsafe {
                let ptr = RcHandle::into_raw(value);
                Ok(RcHandle::from_raw(ptr.cast::<[T; N]>()))
            }
        } else {
            Err(value)
        }
    }
}

impl<'a, 'h, T: ?Sized, A: Allocator> TryFrom<&'h WeakHandle<'a, T, A>> for RcHandle<'a, T, A> {
    type Error = &'h WeakHandle<'a, T, A>;
    #[inline]
    fn try_from(value: &'h WeakHandle<'a, T, A>) -> Result<Self, Self::Error> {
        WeakHandle::upgrade(value).ok_or(value)
    }
}

impl<'rc: 'a, 'a, T: 'a, A: Allocator> IntoIterator for &'rc RcHandle<'a, [T], A> {
    type Item = &'a T;
    type IntoIter = <&'a [T] as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        AsRef::<[T]>::as_ref(self).iter()
    }
}

#[cfg(feature = "serde")]
impl<'a, T: ?Sized + Serialize, A: Allocator> Serialize for RcHandle<'a, T, A> {
    #[inline]
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.as_ref().serialize(serializer)
    }
}

impl<'a, T: ?Sized, A: Allocator> Drop for RcHandle<'a, T, A> {
    #[track_caller]
    #[inline]
    fn drop(&mut self) {
        let should_drop = {
            let inner = Self::inner(self);
            inner.decrement_refcount();
            !inner.is_live()
        };

        if should_drop {
            unsafe {
                ptr::drop_in_place(&raw mut self.ptr.as_mut().data);
            }

            // cleanup of backing allocation happens in `Arena::reset()` or `Arena::drop()`.
        }
    }
}

/// A weak reference-counted pointer to some memory backed by an [`Arena`],
/// analogous to [`Weak<T>`].
///
/// [`Arena`]: ./struct.Arena.html
/// [`Weak<T>`]: https://doc.rust-lang.org/stable/std/rc/struct.Weak.html
#[cfg_attr(feature = "nightly", derive(CoercePointee))]
#[repr(transparent)]
pub struct WeakHandle<
    'a,
    #[cfg_attr(feature = "nightly", pointee)] T: ?Sized,
    A: Allocator = Global,
> {
    ptr: NonNull<RcHandleInner<'a, T, A>>,
}

impl<'a, T, A: Allocator> WeakHandle<'a, T, A> {
    /// Constructs a new `WeakHandle<T>` without allocating any memory. Calling [`WeakHandle::upgrade()`]
    /// on the return value will always return `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::rc_handle::WeakHandle;
    ///
    /// let weak = WeakHandle::<'_, i32>::new();
    /// # let _ = weak;
    /// ```
    ///
    /// [`WeakHandle::upgrade()`]: ./struct.WeakHandle.html#method.upgrade
    #[inline]
    pub const fn new() -> Self {
        let ptr = unsafe {
            let sentinel = ptr::without_provenance_mut(DANGLING_SENTINEL);
            NonNull::new_unchecked(sentinel)
        };

        Self { ptr }
    }

    /// Reinitializes the shared slot with the given `value` if it has
    /// been destroyed.
    ///
    /// # Errors
    ///
    /// This method will return the `value` parameter as an `Err` if there
    /// are still live `RcHandle`s to this shared value, or if the
    /// `WeakHandle` was created through [`WeakHandle::new()`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::{RcHandle, WeakHandle}};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut weak_handle = WeakHandle::new();
    ///
    /// assert_eq!(WeakHandle::try_resurrect(&weak_handle, 124).unwrap_err(), 124);
    ///
    /// {
    ///     let rc_handle = RcHandle::new_in(&arena, 99);
    ///     weak_handle = RcHandle::downgrade(&rc_handle);
    /// }
    ///
    /// let strong = WeakHandle::try_resurrect(&weak_handle, 42).unwrap();
    /// assert_eq!(*strong, 42);
    ///
    /// let thirteen = WeakHandle::try_resurrect(&weak_handle, 13).unwrap_err();
    /// assert_eq!(thirteen, 13);
    /// ```
    #[inline]
    pub fn try_resurrect(&self, value: T) -> Result<RcHandle<'a, T, A>, T> {
        Self::try_resurrect_with(self, || value).map_err(|f| f())
    }

    /// Reinitializes the shared slot with the result of `f` if it has
    /// been destroyed.
    ///
    /// # Errors
    ///
    /// This method will return the `f` parameter as an `Err` if there
    /// are still live `RcHandle`s to this shared value, or if the
    /// `WeakHandle` was created through [`WeakHandle::new()`].
    ///
    /// If this operation fails, `f` will not be evaluated.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::{RcHandle, WeakHandle}};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut weak_handle = WeakHandle::new();
    ///
    /// {
    ///     let handle = RcHandle::new_in(&arena, 198);
    ///     weak_handle = RcHandle::downgrade(&handle);
    /// }
    ///
    /// let handle = WeakHandle::upgrade(&weak_handle)
    ///     .or_else(|| {
    ///         WeakHandle::try_resurrect_with(&weak_handle, || 17).ok()
    ///     })
    ///     .unwrap_or_else(|| unreachable!());
    ///
    /// assert_eq!(handle, 17);
    /// ```
    #[inline]
    pub fn try_resurrect_with<F: FnOnce() -> T>(&self, f: F) -> Result<RcHandle<'a, T, A>, F> {
        if !self.is_accessible() || self.is_valid() {
            return Err(f);
        }

        unsafe { Ok(Self::resurrect_unchecked_with(self, f)) }
    }

    /// Attempt to upgrade the given `WeakHandle` to an `RcHandle`, otherwise reinitialize with `f`.
    ///
    /// If this `WeakHandle` was constructed with [`WeakHandle::new()`], this method will return `None`.
    ///
    /// # Example
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer, rc_handle::{RcHandle, WeakHandle}};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut weak = WeakHandle::new();
    ///
    /// assert!(weak.try_upgrade_or_resurrect_with(|| Buffer::empty_in(&arena)).is_none());
    ///
    /// {
    ///     let buffer = Buffer::new_in(&arena, [1, 2, 3, 4]);
    ///     let handle = RcHandle::new_in(&arena, buffer);
    ///     weak = RcHandle::downgrade(&handle);
    /// }
    ///
    /// let rc = weak.try_upgrade_or_resurrect_with(|| Buffer::new_in(&arena, [5, 6, 7, 8])).unwrap();
    /// assert_eq!(*rc, &[5, 6, 7, 8]);
    ///
    /// let rc_2 = weak.try_upgrade_or_resurrect_with(|| Buffer::empty_in(&arena)).unwrap();
    /// assert_eq!(*rc_2, &[5, 6, 7, 8]);
    ///
    /// let mut weak_2 = WeakHandle::new();
    /// let rc_3 = weak_2
    ///     .try_upgrade_or_resurrect_with(|| Buffer::new_in(&arena, [1, 3, 5, 7]))
    ///     .unwrap_or_else(|| RcHandle::new_in(&arena, Buffer::new_in(&arena, [2, 4, 6, 8])));
    ///
    /// assert_eq!(*rc_3, &[2, 4, 6, 8]);
    /// weak_2 = RcHandle::downgrade(&rc_3);
    /// ```
    #[track_caller]
    #[inline]
    pub fn try_upgrade_or_resurrect_with<F: FnOnce() -> T>(
        &self,
        f: F,
    ) -> Option<RcHandle<'a, T, A>> {
        if self.is_dangling() {
            None
        } else {
            let rc = Self::upgrade(self)
                .unwrap_or_else(|| unsafe { Self::resurrect_unchecked_with(self, f) });
            Some(rc)
        }
    }

    /// Attempt to upgrade the given `WeakHandle` to an `RcHandle`, otherwise reinitialize with `f`.
    ///
    /// Once this function has been called, the given `WeakHandle` will point to the value owned by
    /// the returned `RcHandle`.
    ///
    /// If this `WeakHandle` was constructed with [`WeakHandle::new()`], a new `RcHandle` will be allocated
    /// from the given `arena`.
    ///
    /// # Panics
    ///
    /// This method will panic if there is an allocation error in `Arena` while allocating a new `RcHandle`.
    ///
    /// # Example
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer, rc_handle::{RcHandle, WeakHandle}};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut weak = WeakHandle::new();
    ///
    /// let rc = weak.upgrade_or_resurrect_with(&arena, || Buffer::new_in(&arena, [0, 1, 2, 3, 5]));
    /// assert_eq!(*rc, &[0, 1, 2, 3, 5]);
    /// assert!(WeakHandle::ptr_eq(&weak, &rc));
    /// ```
    #[track_caller]
    #[inline]
    pub fn upgrade_or_resurrect_with<F: FnOnce() -> T>(
        &mut self,
        arena: &'a Arena<A>,
        f: F,
    ) -> RcHandle<'a, T, A> {
        if self.is_dangling() {
            let rc = RcHandle::new_with(arena, f);
            *self = RcHandle::downgrade(&rc);
            rc
        } else {
            Self::upgrade(self)
                .unwrap_or_else(|| unsafe { Self::resurrect_unchecked_with(self, f) })
        }
    }

    #[track_caller]
    #[must_use]
    #[inline]
    unsafe fn resurrect_unchecked_with<F: FnOnce() -> T>(&self, f: F) -> RcHandle<'a, T, A> {
        unsafe {
            assert_unchecked(!self.is_dangling());
            assert_unchecked(WeakHandle::ref_count(self) == 0);

            {
                let inner = self.ptr.as_ptr();
                let dst = inner
                    .map_addr(|addr| addr + offset_of!(RcHandleInner<'a, T, A>, data))
                    .cast::<T>();

                ptr::write(dst, f());
            }

            self.ptr.as_ref().increment_refcount();

            RcHandle {
                ptr: self.ptr,
                _boo: PhantomData,
            }
        }
    }
}

impl<'a, T: ?Sized, A: Allocator> WeakHandle<'a, T, A> {
    /// Upgrade the `WeakHandle` to return a `RcHandle` to access the data.
    ///
    /// If the reference count is `0`, this method returns `None`.
    ///
    /// # Panics
    ///
    /// This method will panic if incrementing the refcount would overflow [`usize::MAX - 1`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::{RcHandle, WeakHandle}};
    ///
    /// let arena = Arena::new();
    ///
    /// let weak;
    ///
    /// {
    ///     let rc_handle = RcHandle::new_in(&arena, 25);
    ///     weak = RcHandle::downgrade(&rc_handle);
    ///
    ///     let rc_handle_2 = WeakHandle::upgrade(&weak).unwrap();
    ///     assert_eq!(*rc_handle_2, 25);
    /// }
    ///
    /// assert!(WeakHandle::upgrade(&weak).is_none());
    /// ```
    ///
    /// [`usize::MAX - 1`]: https://doc.rust-lang.org/stable/std/primitive.usize.html#associatedconstant.MAX
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn upgrade(&self) -> Option<RcHandle<'a, T, A>> {
        self.inner().map(|inner| {
            inner.increment_refcount();
            RcHandle {
                ptr: self.ptr,
                _boo: PhantomData,
            }
        })
    }

    /// Returns the number of active references to the shared value.
    ///
    /// If this method returns `0`, then the contained value has been dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::{RcHandle, WeakHandle}};
    ///
    /// let arena = Arena::new();
    ///
    /// let handle = RcHandle::new_str_in(&arena, "Hello!");
    ///
    /// let weak = RcHandle::downgrade(&handle);
    /// assert_eq!(weak.ref_count(), 1);
    ///
    /// let handle_2 = RcHandle::clone(&handle);
    ///
    /// let weak = RcHandle::downgrade(&handle);
    /// assert_eq!(weak.ref_count(), 2);
    /// ```
    #[must_use]
    #[inline]
    pub fn ref_count(&self) -> usize {
        match self.inner() {
            Some(inner) if inner.is_accessible() => inner.count.get(),
            _ => 0,
        }
    }

    #[must_use]
    #[inline]
    pub fn arena(&self) -> Option<&'a Arena<A>> {
        match self.inner() {
            Some(inner) if inner.is_live() => Some(&inner.arena),
            _ => None,
        }
    }

    /// Access a pointer to the shared value.
    ///
    /// # Safety
    ///
    /// It is not valid to dereference this pointer unless you are certain that
    /// the shared value is still valid. If this `WeakHandle` is pointing to
    /// a value whose `ref_count()` is `0`, this method will return a dangling pointer.
    ///
    /// To check that the pointer is still valid, it is recommended to upgrade the
    /// `WeakHandle` to an `RcHandle`, and call `RcHandle::get_ptr()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::{RcHandle, WeakHandle}};
    ///
    /// let arena = Arena::new();
    ///
    /// let rc = RcHandle::new_in(&arena, 17);
    ///
    /// let weak = RcHandle::downgrade(&rc);
    ///
    /// let ptr: *const i32 = WeakHandle::as_ptr(&weak);
    /// ```
    #[must_use]
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        if self.is_dangling() {
            self.ptr.as_ptr() as *const _
        } else {
            self.ptr
                .as_ptr()
                .map_addr(|addr| addr + offset_of!(RcHandleInner<'_, (), A>, data))
                as *const _
        }
    }

    /// Returns `true` if two `WeakHandle`-convertible types point to the same value.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::{RcHandle, WeakHandle}};
    ///
    /// let arena = Arena::new();
    ///
    /// let handle_1 = RcHandle::new_in(&arena, 25);
    /// let handle_2 = RcHandle::clone(&handle_1);
    ///
    /// let handle_3 = RcHandle::new_in(&arena, 25);
    ///
    /// {
    ///     let weak_1 = RcHandle::downgrade(&handle_1);
    ///     let weak_2 = RcHandle::downgrade(&handle_2);
    ///
    ///     assert!(WeakHandle::ptr_eq(&weak_1, &weak_2));
    ///     assert!(!WeakHandle::ptr_eq(&weak_1, &handle_3));
    /// }
    /// ```
    #[must_use]
    #[inline]
    pub fn ptr_eq<'b, Rhs: Into<WeakHandle<'b, U, A2>>, U: ?Sized, A2: Allocator + 'b>(
        &self,
        rhs: Rhs,
    ) -> bool {
        let rhs = rhs.into();
        let (lhs, rhs) = (self.ptr.as_ptr(), rhs.ptr.as_ptr());
        ptr::eq(lhs.cast::<()>(), rhs.cast::<()>())
    }

    /// Hashes the pointer value of this `WeakHandle` into the given `hasher`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::hash::{Hasher, DefaultHasher};
    /// use rotunda::rc_handle::WeakHandle;
    ///
    /// let weak = WeakHandle::<i32>::new();
    /// let mut hasher = DefaultHasher::new();
    ///
    /// WeakHandle::ptr_hash(&weak, &mut hasher);
    /// let code = hasher.finish();
    /// ```
    #[inline]
    pub fn ptr_hash<H: Hasher>(this: &Self, hasher: &mut H) {
        ptr::hash(WeakHandle::as_ptr(this), hasher);
    }

    #[must_use]
    #[inline]
    fn inner(&self) -> Option<&RcHandleInner<'a, T, A>> {
        if self.is_valid() {
            unsafe { Some(self.ptr.as_ref()) }
        } else {
            None
        }
    }

    #[must_use]
    #[inline]
    fn is_valid(&self) -> bool {
        !self.is_dangling() && unsafe { self.ptr.as_ref().is_live() }
    }

    #[must_use]
    #[inline]
    fn is_accessible(&self) -> bool {
        !self.is_dangling() && unsafe { self.ptr.as_ref().is_accessible() }
    }

    #[must_use]
    #[inline]
    fn is_dangling(&self) -> bool {
        is_dangling(self.ptr.as_ptr())
    }

    #[must_use]
    #[inline]
    pub fn into_raw_in(self) -> (*const T, Option<&'a Arena<A>>) {
        let (data_ptr, arena) = (Self::as_ptr(&self), Self::arena(&self));
        let _this = ManuallyDrop::new(self);
        (data_ptr, arena)
    }

    #[inline]
    pub unsafe fn from_raw_in(raw: *const T, arena: &'a Arena<A>) -> Self {
        let _ = arena;
        unsafe { Self::from_raw_with_alloc(raw) }
    }

    #[inline]
    unsafe fn from_raw_with_alloc(raw: *const T) -> Self {
        let ptr = if is_dangling(raw) {
            raw.with_addr(DANGLING_SENTINEL)
        } else {
            raw.map_addr(|addr| addr - offset_of!(RcHandleInner<'a, (), Global>, data))
        };

        let ptr = unsafe { NonNull::new_unchecked(ptr.cast_mut() as *mut _) };

        Self { ptr }
    }
}

impl<'a, T: ?Sized> WeakHandle<'a, T, Global> {
    /// Consumes the `WeakHandle`, returning the underlying wrapped pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, rc_handle::{RcHandle, WeakHandle}};
    ///
    /// let arena = Arena::new();
    ///
    /// let rc = RcHandle::new_in(&arena, 127i8);
    ///
    /// let weak = RcHandle::downgrade(&rc);
    ///
    /// let weak_raw = WeakHandle::into_raw(weak);
    /// unsafe { assert_eq!(*weak_raw, 127); }
    ///
    /// # let _ = unsafe { WeakHandle::<'_, _>::from_raw(weak_raw) };
    /// ```
    #[must_use]
    #[inline]
    pub fn into_raw(self) -> *const T {
        let (ptr, _) = Self::into_raw_in(self);
        ptr
    }

    #[inline]
    pub unsafe fn from_raw(raw: *const T) -> Self {
        unsafe { Self::from_raw_with_alloc(raw) }
    }
}

#[cfg(feature = "nightly")]
impl<'a, T: ?Sized + Pointee, A: Allocator> WeakHandle<'a, T, A> {
    #[must_use]
    #[inline]
    pub unsafe fn from_raw_parts_in(
        ptr: *const impl Thin,
        metadata: <T as Pointee>::Metadata,
        arena: &'a Arena<A>,
    ) -> Self {
        let ptr = ptr::from_raw_parts(ptr, metadata);
        unsafe { Self::from_raw_in(ptr, arena) }
    }

    #[must_use]
    #[inline]
    pub fn to_raw_parts_in(self) -> (*const (), <T as Pointee>::Metadata, Option<&'a Arena<A>>) {
        let (raw, arena) = Self::into_raw_in(self);
        let (data, meta) =  <*const T>::to_raw_parts(raw);
        (data, meta, arena)
    }
}
#[cfg(feature = "nightly")]
impl<'a, T: ?Sized + Pointee> WeakHandle<'a, T, Global> {
    #[must_use]
    #[inline]
    pub unsafe fn from_raw_parts(
        ptr: *const impl Thin,
        metadata: <T as Pointee>::Metadata,
    ) -> Self {
        let ptr = ptr::from_raw_parts(ptr, metadata);
        unsafe { WeakHandle::from_raw(ptr) }
    }
}

impl<'a, T: ?Sized + fmt::Debug, A: Allocator> fmt::Debug for WeakHandle<'a, T, A> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(inner) = WeakHandle::inner(self) {
            fmt::Debug::fmt(&inner.data, fmtr)
        } else {
            let payload: &'_ dyn fmt::Debug = if self.is_dangling() {
                struct Dangling;
                impl fmt::Debug for Dangling {
                    #[inline]
                    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
                        fmtr.write_str("<Dangling>")
                    }
                }

                &Dangling
            } else {
                struct Destroyed;
                impl fmt::Debug for Destroyed {
                    #[inline]
                    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
                        fmtr.write_str("<Destroyed>")
                    }
                }

                &Destroyed
            };

            fmtr.debug_tuple("WeakHandle").field(payload).finish()
        }
    }
}

impl<'a, T: ?Sized, A: Allocator> fmt::Pointer for WeakHandle<'a, T, A> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, fmtr)
    }
}

impl<'a, T, A: Allocator> Default for WeakHandle<'a, T, A> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T: ?Sized, A: Allocator> Clone for WeakHandle<'a, T, A> {
    #[inline]
    fn clone(&self) -> Self {
        WeakHandle { ptr: self.ptr }
    }
}

impl<'h, 'a, T: ?Sized, A: Allocator> From<&'h RcHandle<'a, T, A>> for WeakHandle<'a, T, A> {
    #[inline]
    fn from(value: &'h RcHandle<'a, T, A>) -> Self {
        RcHandle::downgrade(value)
    }
}

impl<'h, 'a, T: ?Sized, A: Allocator> From<&'h WeakHandle<'a, T, A>> for WeakHandle<'a, T, A> {
    #[inline]
    fn from(value: &'h WeakHandle<'a, T, A>) -> Self {
        value.clone()
    }
}

#[repr(C)]
struct RcHandleInner<'a, T: ?Sized, A: Allocator> {
    _boo: PhantomData<PhantomPinned>,
    count: Cell<usize>,
    arena: &'a Arena<A>,
    data: T,
}

impl<'a, T: ?Sized, A: Allocator> RcHandleInner<'a, T, A> {
    const COUNT_INACCESSIBLE: usize = usize::MAX;

    #[inline]
    unsafe fn init_raw(this: NonNull<Self>, arena: &'a Arena<A>) {
        let count_ptr = this
            .map_addr(|addr| addr.saturating_add(offset_of!(RcHandleInner<'_, T, A>, count)))
            .cast::<Cell<usize>>();

        unsafe {
            ptr::write(count_ptr.as_ptr(), Cell::new(1));
        }

        let arena_ptr = this
            .map_addr(|addr| addr.saturating_add(offset_of!(RcHandleInner<'_, T, A>, arena)))
            .cast::<&'a Arena<A>>();

        unsafe {
            ptr::write(arena_ptr.as_ptr(), arena);
        }
    }

    #[inline]
    const fn mark_inaccessible(&self) {
        self.count.replace(Self::COUNT_INACCESSIBLE);
    }

    #[inline]
    const fn is_accessible(&self) -> bool {
        let count = self.count.get();
        count < Self::COUNT_INACCESSIBLE
    }

    #[inline]
    const fn is_live(&self) -> bool {
        let count = self.count.get();
        count > 0 && count < Self::COUNT_INACCESSIBLE
    }

    #[track_caller]
    #[inline]
    const fn increment_refcount(&self) {
        let new_count = match self.count.get().checked_add(1) {
            Some(value) if value != Self::COUNT_INACCESSIBLE => value,
            _ => modify_refcount_failed("refcount overflow"),
        };

        self.count.replace(new_count);
    }

    #[track_caller]
    #[inline]
    const fn decrement_refcount(&self) {
        let count = self.count.get();
        if count == Self::COUNT_INACCESSIBLE {
            return;
        }

        let new_count = match count.checked_sub(1) {
            Some(value) => value,
            None => modify_refcount_failed("refcount underflow"),
        };

        self.count.replace(new_count);
    }
}

#[track_caller]
#[cold]
#[inline(never)]
const fn modify_refcount_failed(message: &str) -> ! {
    panic!("{}", message);
}

impl<'a, T, A: Allocator> RcHandleInner<'a, T, A> {
    #[must_use]
    #[inline]
    unsafe fn cast_to_slice(
        this: *mut RcHandleInner<'a, T, A>,
        slice_len: usize,
    ) -> *mut RcHandleInner<'a, [T], A> {
        let data_ptr = this
            .map_addr(|addr| addr + offset_of!(RcHandleInner<'_, (), Global>, data))
            .cast::<T>();
        let ptr = ptr::slice_from_raw_parts_mut(data_ptr, slice_len);
        ptr.map_addr(|addr| addr - offset_of!(RcHandleInner<'_, (), Global>, data))
            as *mut RcHandleInner<'a, [T], A>
    }
}

const DANGLING_SENTINEL: usize = usize::MAX;

#[track_caller]
#[must_use]
#[inline]
const fn rc_inner_layout_for_value_layout(layout: Layout) -> Layout {
    match Layout::new::<RcHandleInner<'_, (), Global>>().extend(layout) {
        Ok((layout, _)) => layout.pad_to_align(),
        Err(_) => panic!("bad layout"),
    }
}

#[must_use]
#[inline]
fn is_dangling<T: ?Sized>(ptr: *const T) -> bool {
    ptr.cast::<()>().addr() == DANGLING_SENTINEL
}
