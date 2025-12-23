#![allow(missing_docs, clippy::missing_safety_doc)]

//! Single-threaded reference-counting pointer types backed by an `Arena`.

use crate::{Arena, handle::Handle};
use alloc::alloc::{Allocator, Layout};
use core::{
    any::Any,
    borrow::Borrow,
    cell::Cell,
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    iter::IntoIterator,
    marker::{CoercePointee, PhantomData, PhantomPinned},
    mem::{self, ManuallyDrop, MaybeUninit, offset_of},
    ops::{Deref, Index},
    ptr::{self, NonNull, Pointee, Thin},
    slice::{self, SliceIndex},
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
#[derive(CoercePointee)]
#[repr(transparent)]
pub struct RcHandle<'a, T: ?Sized> {
    ptr: NonNull<RcHandleInner<T>>,
    _boo: PhantomData<(&'a Arena, RcHandleInner<T>)>,
}

const _: () = assert!(mem::size_of::<RcHandle<()>>() == mem::size_of::<NonNull<()>>());
const _: () = assert!(mem::size_of::<Option<RcHandle<()>>>() == mem::size_of::<Handle<()>>());
const _: () = assert!(mem::align_of::<RcHandle<()>>() == mem::align_of::<NonNull<()>>());

impl<'a, T> RcHandle<'a, T> {
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
    pub fn new_in<A: Allocator>(arena: &'a Arena<A>, value: T) -> Self {
        let handle = RcHandle::new_uninit_in(arena);
        unsafe { RcHandle::init(handle, value) }
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
    pub fn new_with<A: Allocator, F: FnOnce() -> T>(arena: &'a Arena<A>, f: F) -> Self {
        let handle = RcHandle::new_uninit_in(arena);
        unsafe { RcHandle::init(handle, f()) }
    }

    #[track_caller]
    #[must_use]
    #[inline]
    pub unsafe fn init_with<A: Allocator, F: FnOnce(&mut MaybeUninit<T>)>(
        arena: &'a Arena<A>,
        f: F,
    ) -> Self {
        let mut handle = RcHandle::new_uninit_in(arena);
        unsafe {
            f(RcHandle::get_mut_unchecked(&mut handle));
            RcHandle::assume_init(handle)
        }
    }
}

impl<'a, T> RcHandle<'a, [T]> {
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_slice_from_fn_in<A: Allocator, F: FnMut(usize) -> T>(
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
}

impl<'a, T, const N: usize> RcHandle<'a, [T; N]> {
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_array_from_fn_in<A: Allocator, F: FnMut(usize) -> T>(
        arena: &'a Arena<A>,
        mut f: F,
    ) -> Self {
        struct Guard<'a, T> {
            data: &'a mut [MaybeUninit<T>],
            initted: usize,
        }

        impl<'a, T> Drop for Guard<'a, T> {
            #[inline]
            fn drop(&mut self) {
                unsafe {
                    let slice =
                        slice::from_raw_parts_mut(self.data.as_mut_ptr().cast::<T>(), self.initted);
                    ptr::drop_in_place(slice);
                }
            }
        }

        let handle = RcHandle::<'_, MaybeUninit<[T; N]>>::new_uninit_in(arena);
        let mut handle = RcHandle::<'_, MaybeUninit<[T; N]>>::transpose_inner_uninit(handle);

        {
            let handle_ref = unsafe { RcHandle::get_mut_unchecked(&mut handle) };
            let mut guard = Guard {
                data: &mut handle_ref[..],
                initted: 0,
            };

            for i in 0..N {
                let slot = unsafe { guard.data.get_unchecked_mut(i) };
                slot.write(f(i));
                guard.initted += 1;
            }

            mem::forget(guard);
        }

        unsafe { RcHandle::assume_init_array(handle) }
    }
}

impl<'a, T: Copy, const N: usize> RcHandle<'a, [T; N]> {
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_array_splat_in<A: Allocator>(arena: &'a Arena<A>, value: T) -> Self {
        let handle = RcHandle::<'_, MaybeUninit<[T; N]>>::new_uninit_in(arena);
        let mut handle = RcHandle::<'_, MaybeUninit<[T; N]>>::transpose_inner_uninit(handle);

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

impl<'a, T: Copy> RcHandle<'a, [T]> {
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_splat<A: Allocator>(
        arena: &'a Arena<A>,
        slice_len: usize,
        value: T,
    ) -> RcHandle<'a, [T]> {
        let mut hndl = RcHandle::new_slice_uninit_in(arena, slice_len);
        let slots = unsafe { RcHandle::get_mut_unchecked(&mut hndl) };

        slots.fill(MaybeUninit::new(value));

        unsafe { RcHandle::assume_init_slice(hndl) }
    }
}

impl<'a, T: Default> RcHandle<'a, T> {
    #[must_use]
    #[inline]
    pub fn new_default_in<A: Allocator>(arena: &'a Arena<A>) -> Self {
        RcHandle::new_with(arena, Default::default)
    }
}

impl<'a, T: ?Sized> RcHandle<'a, T> {
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
        &Self::inner(this).data
    }

    #[must_use]
    #[inline]
    pub const fn into_raw(self) -> *const T {
        let ptr = Self::as_ptr(&self);
        let _this = ManuallyDrop::new(self);
        ptr
    }

    #[must_use]
    #[inline]
    pub unsafe fn from_raw(raw: *const T) -> Self {
        unsafe {
            let ptr = raw.map_addr(|addr| addr - offset_of!(RcHandleInner<()>, data))
                as *const RcHandleInner<T>;
            Self::from_raw_inner(ptr)
        }
    }

    #[inline]
    pub unsafe fn increment_count(raw: *const T) {
        unsafe {
            let handle = Self::from_raw(raw);
            let copy = handle.clone();
            let _ = (ManuallyDrop::new(handle), ManuallyDrop::new(copy));
        }
    }

    #[inline]
    pub unsafe fn decrement_count(raw: *const T) {
        unsafe {
            drop(Self::from_raw(raw));
        }
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
    pub const fn downgrade(this: &Self) -> WeakHandle<'a, T> {
        WeakHandle {
            ptr: this.ptr,
            _boo: PhantomData,
        }
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
    /// let string = "This message";
    ///
    /// let mut rc_handle = RcHandle::new_splat(&arena, 20, 0u8);
    ///
    /// unsafe {
    ///     let contents = RcHandle::get_mut_unchecked(&mut rc_handle);
    ///     core::ptr::copy_nonoverlapping(string.as_bytes().as_ptr(), contents.as_mut_ptr(), string.len());
    /// }
    ///
    /// assert_eq!(*&rc_handle[..string.len()], *"This message".as_bytes());
    /// assert_eq!(rc_handle.len(), 20);
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
    pub fn try_into_handle(this: Self) -> Result<Handle<'a, T>, Self> {
        if Self::is_unique(&this) {
            // Decrement the ref count to `0` to invalidate all
            // `WeakHandle`s which point here.
            Self::inner(&this).decrement_refcount();

            let raw = this
                .ptr
                .as_ptr()
                .map_addr(|addr| addr + offset_of!(RcHandleInner<()>, data))
                as *mut T;

            let _this = ManuallyDrop::new(this);
            unsafe { Ok(Handle::from_raw(raw)) }
        } else {
            Err(this)
        }
    }

    /// Converts this `RcHandle` into a uniquely owned `Handle`.
    ///
    /// If there are other live `RcHandle`s to the shared value, this method
    /// returns `None`.
    #[must_use]
    #[inline]
    pub fn into_handle(this: Self) -> Option<Handle<'a, T>> {
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

    #[must_use]
    #[inline]
    pub const fn is_unique(this: &Self) -> bool {
        RcHandle::ref_count(this) == 1
    }

    #[must_use]
    #[inline]
    pub fn ptr_eq<Rhs: Into<WeakHandle<'a, T>>>(this: &Self, other: Rhs) -> bool {
        ptr::eq(Self::as_ptr(this), WeakHandle::as_ptr(&other.into()))
    }

    #[inline]
    pub fn ptr_hash<H: Hasher>(this: &Self, hasher: &mut H) {
        ptr::hash(RcHandle::as_ptr(this), hasher);
    }

    #[must_use]
    #[inline]
    const unsafe fn from_raw_inner(raw: *const RcHandleInner<T>) -> Self {
        let ptr = unsafe { NonNull::new_unchecked(raw as *mut _) };
        let handle = Self {
            ptr,
            _boo: PhantomData,
        };

        debug_assert!(Self::ref_count(&handle) > 0);
        handle
    }

    #[must_use]
    #[inline]
    const fn inner(this: &Self) -> &RcHandleInner<T> {
        unsafe { this.ptr.as_ref() }
    }
}

impl<'a, T: Unpin> RcHandle<'a, T> {
    #[inline]
    pub fn try_unwrap(this: Self) -> Result<T, RcHandle<'a, T>> {
        Self::try_into_handle(this).map(Handle::into_inner)
    }

    #[must_use]
    #[inline]
    pub fn into_inner(this: Self) -> Option<T> {
        Self::try_unwrap(this).ok()
    }
}

impl<'a> RcHandle<'a, dyn Any> {
    #[inline]
    pub fn downcast<T: Any>(self) -> Result<RcHandle<'a, T>, RcHandle<'a, dyn Any>> {
        if (*self).is::<T>() {
            unsafe { Ok(Self::downcast_unchecked::<T>(self)) }
        } else {
            Err(self)
        }
    }

    #[must_use]
    #[inline]
    pub unsafe fn downcast_unchecked<T: Any>(self) -> RcHandle<'a, T> {
        unsafe {
            let ptr = Self::into_raw(self);
            RcHandle::from_raw(ptr as *const T)
        }
    }
}

impl<'a, T> RcHandle<'a, MaybeUninit<T>> {
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_uninit_in<A: Allocator>(arena: &'a Arena<A>) -> RcHandle<'a, MaybeUninit<T>> {
        let layout = const { rc_inner_layout_for_value_layout(Layout::new::<T>()) };

        let ptr = arena
            .alloc_raw(layout)
            .cast::<RcHandleInner<MaybeUninit<T>>>();

        unsafe {
            let count_ptr = ptr
                .map_addr(|addr| addr.saturating_add(offset_of!(RcHandleInner<T>, count)))
                .cast::<Cell<usize>>()
                .as_ptr();

            ptr::write(count_ptr, Cell::new(1));
            RcHandle::from_raw_inner(ptr.as_ptr().cast_const())
        }
    }

    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_uninit_zeroed_in<A: Allocator>(arena: &'a Arena<A>) -> RcHandle<'a, MaybeUninit<T>> {
        let layout = const { rc_inner_layout_for_value_layout(Layout::new::<T>()) };

        let ptr = arena
            .alloc_raw_zeroed(layout)
            .cast::<RcHandleInner<MaybeUninit<T>>>();

        unsafe {
            let count_ptr = ptr
                .map_addr(|addr| addr.saturating_add(offset_of!(RcHandleInner<T>, count)))
                .cast::<Cell<usize>>()
                .as_ptr();

            ptr::write(count_ptr, Cell::new(1));
            RcHandle::from_raw_inner(ptr.as_ptr().cast_const())
        }
    }

    #[must_use]
    #[inline]
    pub const unsafe fn assume_init(this: Self) -> RcHandle<'a, T> {
        unsafe { mem::transmute(this) }
    }

    #[must_use]
    #[inline]
    pub const unsafe fn init(mut this: Self, value: T) -> RcHandle<'a, T> {
        unsafe {
            let inner = RcHandle::get_mut_unchecked(&mut this);
            let _ = inner.write(value);

            RcHandle::assume_init(this)
        }
    }
}

impl<'a, T, const N: usize> RcHandle<'a, MaybeUninit<[T; N]>> {
    #[must_use]
    #[inline]
    pub const fn transpose_inner_uninit(this: Self) -> RcHandle<'a, [MaybeUninit<T>; N]> {
        unsafe { mem::transmute(this) }
    }
}

impl<'a, T, const N: usize> RcHandle<'a, [MaybeUninit<T>; N]> {
    #[must_use]
    #[inline]
    pub const unsafe fn assume_init_array(this: Self) -> RcHandle<'a, [T; N]> {
        unsafe { mem::transmute(this) }
    }

    #[must_use]
    #[inline]
    pub const fn transpose_outer_uninit(this: Self) -> RcHandle<'a, MaybeUninit<[T; N]>> {
        unsafe { mem::transmute(this) }
    }
}

impl<'a, T> RcHandle<'a, [MaybeUninit<T>]> {
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_slice_uninit_in<A: Allocator>(arena: &'a Arena<A>, slice_len: usize) -> Self {
        let (array_layout, ..) = Layout::new::<T>().repeat(slice_len).expect("size overflow");
        let inner_layout = rc_inner_layout_for_value_layout(array_layout);

        unsafe {
            let ptr = arena
                .alloc_raw(inner_layout)
                .cast::<RcHandleInner<MaybeUninit<T>>>()
                .as_ptr();

            let count_ptr = ptr
                .map_addr(|addr| addr.saturating_add(offset_of!(RcHandleInner<T>, count)))
                .cast::<Cell<usize>>();

            ptr::write(count_ptr, Cell::new(1));
            let ptr = RcHandleInner::cast_to_slice(ptr, slice_len);
            RcHandle::from_raw_inner(ptr)
        }
    }

    #[must_use]
    #[inline]
    pub const unsafe fn assume_init_slice(this: Self) -> RcHandle<'a, [T]> {
        let ptr = this.ptr.as_ptr() as *mut RcHandleInner<[T]>;
        let ptr = unsafe { NonNull::new_unchecked(ptr) };
        let _this = ManuallyDrop::new(this);

        RcHandle {
            ptr,
            _boo: PhantomData,
        }
    }
}

impl<'a> RcHandle<'a, str> {
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
    pub fn new_str_in<A: Allocator>(arena: &'a Arena<A>, string: &'_ str) -> Self {
        let string_len = string.len();
        let mut rc_handle =
            RcHandle::<'a, [MaybeUninit<u8>]>::new_slice_uninit_in(&arena, string_len);

        unsafe {
            let slice = RcHandle::get_mut_unchecked(&mut rc_handle);
            let string_bytes = {
                let data = string.as_bytes().as_ptr().cast::<MaybeUninit<u8>>();
                slice::from_raw_parts(data, string_len)
            };

            slice.copy_from_slice(string_bytes);
        }

        let ptr = rc_handle.ptr.as_ptr() as *mut RcHandleInner<str>;
        let ptr = unsafe { NonNull::new_unchecked(ptr) };
        mem::forget(rc_handle);
        RcHandle {
            ptr,
            _boo: PhantomData,
        }
    }
}

impl<'a, T: ?Sized + Pointee> RcHandle<'a, T> {
    #[must_use]
    #[inline]
    pub unsafe fn from_raw_parts(
        data: *const impl Thin,
        metadata: <T as Pointee>::Metadata,
    ) -> Self {
        let ptr = ptr::from_raw_parts(data, metadata);
        unsafe { Self::from_raw(ptr) }
    }
}

impl<'a, T: ?Sized + fmt::Debug> fmt::Debug for RcHandle<'a, T> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.as_ref(), fmtr)
    }
}

impl<'a, T: ?Sized + fmt::Display> fmt::Display for RcHandle<'a, T> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.as_ref(), fmtr)
    }
}

impl<'a, T: ?Sized> fmt::Pointer for RcHandle<'a, T> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, fmtr)
    }
}

impl<'a, T: ?Sized> AsRef<T> for RcHandle<'a, T> {
    #[inline]
    fn as_ref(&self) -> &T {
        &Self::inner(self).data
    }
}

impl<'a, T: ?Sized> Borrow<T> for RcHandle<'a, T> {
    #[inline]
    fn borrow(&self) -> &T {
        &Self::inner(self).data
    }
}

impl<'a, T: ?Sized> Deref for RcHandle<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &Self::inner(self).data
    }
}

impl<'a, T, I: SliceIndex<[T]>> Index<I> for RcHandle<'a, [T]> {
    type Output = <[T] as Index<I>>::Output;
    #[track_caller]
    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        self.as_ref().index(index)
    }
}

impl<'a, T: ?Sized> Clone for RcHandle<'a, T> {
    #[track_caller]
    #[inline]
    fn clone(&self) -> Self {
        let inner = Self::inner(self);
        inner.increment_refcount();

        Self {
            ptr: self.ptr,
            _boo: PhantomData,
        }
    }
}

impl<'a, T: ?Sized + PartialEq> PartialEq for RcHandle<'a, T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_ref().eq(other.as_ref())
    }
}

impl<'a, T: ?Sized + PartialEq> PartialEq<T> for RcHandle<'a, T> {
    #[inline]
    fn eq(&self, other: &T) -> bool {
        self.as_ref().eq(other)
    }
}

impl<'a, T: PartialEq<U>, U, const N: usize> PartialEq<[U; N]> for RcHandle<'a, [T]> {
    #[inline]
    fn eq(&self, other: &[U; N]) -> bool {
        PartialEq::eq(self.as_ref(), &other[..])
    }
}

impl<'a, T: ?Sized + Eq> Eq for RcHandle<'a, T> {}

impl<'a, T: ?Sized + PartialOrd> PartialOrd for RcHandle<'a, T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_ref().partial_cmp(other.as_ref())
    }
}

impl<'a, T: ?Sized + PartialOrd> PartialOrd<T> for RcHandle<'a, T> {
    #[inline]
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        self.as_ref().partial_cmp(other)
    }
}

impl<'a, T: ?Sized + Ord> Ord for RcHandle<'a, T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_ref().cmp(other.as_ref())
    }
}

impl<'a, T: ?Sized + Hash> Hash for RcHandle<'a, T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state)
    }
}

impl<'a, T, const N: usize> TryFrom<RcHandle<'a, [T]>> for RcHandle<'a, [T; N]> {
    type Error = RcHandle<'a, [T]>;
    #[inline]
    fn try_from(value: RcHandle<'a, [T]>) -> Result<Self, Self::Error> {
        if value.len() == N {
            unsafe {
                let ptr = RcHandle::into_raw(value).cast::<[T; N]>();
                Ok(RcHandle::from_raw(ptr))
            }
        } else {
            Err(value)
        }
    }
}

impl<'a, 'h, T: ?Sized> TryFrom<&'h WeakHandle<'a, T>> for RcHandle<'a, T> {
    type Error = &'h WeakHandle<'a, T>;
    #[inline]
    fn try_from(value: &'h WeakHandle<'a, T>) -> Result<Self, Self::Error> {
        WeakHandle::upgrade(value).ok_or(value)
    }
}

impl<'rc: 'a, 'a, T: 'a> IntoIterator for &'rc RcHandle<'a, [T]> {
    type Item = &'a T;
    type IntoIter = <&'a [T] as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        AsRef::<[T]>::as_ref(self).iter()
    }
}

#[cfg(feature = "serde")]
impl<'a, T: ?Sized + Serialize> Serialize for RcHandle<'a, T> {
    #[inline]
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.as_ref().serialize(serializer)
    }
}

impl<'a, T: ?Sized> Drop for RcHandle<'a, T> {
    #[track_caller]
    #[inline]
    fn drop(&mut self) {
        let inner = Self::inner(self);
        inner.decrement_refcount();

        if inner.count.get() == 0 {
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
#[derive(CoercePointee)]
#[repr(transparent)]
pub struct WeakHandle<'a, T: ?Sized> {
    ptr: NonNull<RcHandleInner<T>>,
    _boo: PhantomData<&'a Arena>,
}

impl<'a, T> WeakHandle<'a, T> {
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

        Self {
            ptr,
            _boo: PhantomData,
        }
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
    pub fn try_resurrect(&self, value: T) -> Result<RcHandle<'a, T>, T> {
        if WeakHandle::ref_count(&self) > 0 || is_dangling(self.ptr.as_ptr()) {
            return Err(value);
        }

        unsafe {
            let inner = self.ptr.as_ptr();
            let dst = inner
                .map_addr(|addr| addr + offset_of!(RcHandleInner<T>, data))
                .cast::<T>();
            ptr::write(dst, value);
            (&*inner).increment_refcount();
        }

        Ok(RcHandle {
            ptr: self.ptr,
            _boo: PhantomData,
        })
    }
}

impl<'a, T: ?Sized> WeakHandle<'a, T> {
    /// Upgrade the `WeakHandle` to return a `RcHandle` to access the data.
    ///
    /// If the reference count is `0`, this method returns `None`.
    ///
    /// # Panics
    ///
    /// If the reference count overflows, this method will panic.
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
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn upgrade(&self) -> Option<RcHandle<'a, T>> {
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
        self.inner()
            .map(|inner| inner.count.get())
            .unwrap_or_default()
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
                .map_addr(|addr| addr + offset_of!(RcHandleInner<()>, data)) as *const _
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
    pub fn ptr_eq<Rhs: Into<WeakHandle<'a, T>>>(&self, rhs: Rhs) -> bool {
        let rhs = rhs.into();
        let (lhs, rhs) = (self.ptr.as_ptr(), rhs.ptr.as_ptr());
        ptr::eq(lhs, rhs)
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
    pub fn into_raw(self) -> *const T {
        let data_ptr = Self::as_ptr(&self);
        let _this = ManuallyDrop::new(self);
        data_ptr
    }

    #[inline]
    pub unsafe fn from_raw(raw: *const T) -> Self {
        let ptr = if is_dangling(raw) {
            raw.with_addr(DANGLING_SENTINEL)
        } else {
            raw.map_addr(|addr| addr - offset_of!(RcHandleInner<()>, data))
        };

        let ptr = unsafe { NonNull::new_unchecked(ptr.cast_mut() as *mut RcHandleInner<T>) };

        Self {
            ptr,
            _boo: PhantomData,
        }
    }

    #[must_use]
    #[inline]
    fn inner(&self) -> Option<&RcHandleInner<T>> {
        if self.is_valid() {
            unsafe { Some(self.ptr.as_ref()) }
        } else {
            None
        }
    }

    #[must_use]
    #[inline]
    fn is_valid(&self) -> bool {
        !self.is_dangling() && unsafe { self.ptr.as_ref().count.get() != 0 }
    }

    #[must_use]
    #[inline]
    fn is_dangling(&self) -> bool {
        is_dangling(self.ptr.as_ptr())
    }
}

impl<'a, T: ?Sized + Pointee> WeakHandle<'a, T> {
    #[must_use]
    #[inline]
    pub unsafe fn from_raw_parts(
        ptr: *const impl Thin,
        metadata: <T as Pointee>::Metadata,
    ) -> Self {
        let ptr = ptr::from_raw_parts(ptr, metadata);
        unsafe { Self::from_raw(ptr) }
    }
}

impl<'a, T: ?Sized + fmt::Debug> fmt::Debug for WeakHandle<'a, T> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(strong) = WeakHandle::upgrade(self) {
            fmt::Debug::fmt(strong.as_ref(), fmtr)
        } else {
            struct Destroyed;
            impl fmt::Debug for Destroyed {
                fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
                    fmtr.write_str("<Destroyed>")
                }
            }

            let mut debug_tuple = fmtr.debug_tuple("WeakHandle");
            debug_tuple.field(&Destroyed).finish()
        }
    }
}

impl<'a, T: ?Sized> fmt::Pointer for WeakHandle<'a, T> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, fmtr)
    }
}

impl<'a, T> Default for WeakHandle<'a, T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T: ?Sized> Clone for WeakHandle<'a, T> {
    #[inline]
    fn clone(&self) -> Self {
        WeakHandle {
            ptr: self.ptr,
            _boo: PhantomData,
        }
    }
}

impl<'h, 'a: 'h, T: ?Sized> From<&'h RcHandle<'a, T>> for WeakHandle<'a, T> {
    #[inline]
    fn from(value: &'h RcHandle<'a, T>) -> Self {
        RcHandle::downgrade(value)
    }
}

impl<'h, 'a: 'h, T: ?Sized> From<&'h WeakHandle<'a, T>> for WeakHandle<'a, T> {
    #[inline]
    fn from(value: &'h WeakHandle<'a, T>) -> Self {
        value.clone()
    }
}

#[repr(C)]
struct RcHandleInner<T: ?Sized> {
    _boo: PhantomData<PhantomPinned>,
    count: Cell<usize>,
    data: T,
}

impl<T: ?Sized> RcHandleInner<T> {
    #[track_caller]
    #[inline]
    const fn increment_refcount(&self) {
        let new_count = match self.count.get().checked_add(1) {
            Some(value) => value,
            None => modify_refcount_failed("refcount overflow"),
        };

        self.count.replace(new_count);
    }

    #[track_caller]
    #[inline]
    const fn decrement_refcount(&self) {
        let new_count = match self.count.get().checked_sub(1) {
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

impl<T> RcHandleInner<T> {
    #[must_use]
    #[inline]
    unsafe fn cast_to_slice(
        this: *mut RcHandleInner<T>,
        slice_len: usize,
    ) -> *mut RcHandleInner<[T]> {
        let data_ptr: *mut T = this
            .map_addr(|addr| addr + offset_of!(RcHandleInner<()>, data))
            .cast::<T>();
        let ptr = ptr::from_raw_parts_mut::<[T]>(data_ptr, slice_len);
        ptr.map_addr(|addr| addr - offset_of!(RcHandleInner<()>, data)) as *mut RcHandleInner<[T]>
    }
}

const DANGLING_SENTINEL: usize = usize::MAX;

#[track_caller]
#[must_use]
#[inline]
const fn rc_inner_layout_for_value_layout(layout: Layout) -> Layout {
    match Layout::new::<RcHandleInner<()>>().extend(layout) {
        Ok((layout, _)) => layout.pad_to_align(),
        Err(_) => panic!("bad layout"),
    }
}

#[must_use]
#[inline]
fn is_dangling<T: ?Sized>(ptr: *const T) -> bool {
    ptr.cast::<()>().addr() == DANGLING_SENTINEL
}
