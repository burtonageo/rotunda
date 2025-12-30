// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, clippy::missing_safety_doc)]

//! A singly-owned mutable pointer backed by an `Arena`.

use crate::{
    Arena,
    buffer::{Buffer, IntoIterHandles},
    layout_repeat,
    rc_handle::RcHandle,
    string_buffer::{FromUtf8Error, StringBuffer},
};
use alloc::alloc::{Allocator, Global, Layout};
use core::{
    any::Any,
    borrow::{Borrow, BorrowMut},
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    hint::assert_unchecked,
    iter::IntoIterator,
    marker::{PhantomData, Unpin},
    mem::{self, ManuallyDrop, MaybeUninit},
    ops::{Deref, DerefMut, Index, IndexMut},
    panic::{RefUnwindSafe, UnwindSafe},
    pin::Pin,
    ptr::{self, NonNull},
    slice::SliceIndex,
    str,
};
#[cfg(feature = "nightly")]
use core::{
    marker::Unsize,
    ops::CoerceUnsized,
    ptr::{Pointee, Thin},
};
#[cfg(feature = "serde")]
use serde_core::{Serialize, Serializer};
#[cfg(feature = "std")]
use std::io::{self, BufRead, IoSlice, IoSliceMut, Read, Write};

/// An owned, mutable pointer to some memory backed by an [`Arena`], analogous to
/// [`Box<T>`].
///
/// See the [module documentation] for more information.
///
/// [`Arena`]: ../struct.Arena.html
/// [`Box<T>`]: https://doc.rust-lang.org/stable/std/boxed/struct.Box.html
/// [module documentation]: ./index.html
pub struct Handle<'a, T: ?Sized, A: Allocator = Global> {
    ptr: NonNull<T>,
    arena: &'a Arena<A>,
    _boo: PhantomData<T>,
}

// A handle can be sent to other threads if the type within is
// thread-safe - it is guaranteed to drop before the arena is
// deallocated thanks to borrowing rules.
unsafe impl<'a, T: ?Sized + Send, A: Allocator> Send for Handle<'a, T, A> {}
unsafe impl<'a, T: ?Sized + Sync, A: Allocator> Sync for Handle<'a, T, A> {}

impl<'a, T, A: Allocator> Handle<'a, T, A> {
    /// Create a new `Handle` containing the given `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    /// let handle = Handle::new_in(&arena, 156);
    /// # let _ = handle;
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_in(arena: &'a Arena<A>, value: T) -> Self {
        let handle = Handle::new_uninit_in(arena);
        Handle::init(handle, value)
    }

    /// Create a new `Handle` containing the return value of the given function.
    ///
    /// # Example
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    /// let handle = Handle::new_with(&arena, || 23 + 45);
    /// ```
    #[must_use]
    #[inline]
    pub fn new_with<F: FnOnce() -> T>(arena: &'a Arena<A>, f: F) -> Self {
        let handle = Handle::new_uninit_in(arena);
        Handle::init(handle, f())
    }

    /// Create a new `Handle`, using the given function to initialize its contents.
    ///
    /// # Safety
    ///
    /// The function must fully initialize the allocated value, as this function will
    /// assume that the value has been fully initialized when the given `f` is run.
    ///
    /// If this contract is not held, then this `Handle` may permit access to uninitialized
    /// data.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::ptr::{self, addr_of_mut};
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// struct Data {
    ///     integer: u32,
    ///     string: &'static str,
    /// }
    ///
    /// let arena = Arena::new();
    /// let handle: Handle<'_, Data> = unsafe {
    ///     Handle::init_with(&arena, |data| unsafe {
    ///         let data_ptr: *mut Data = data.as_mut_ptr();
    ///         ptr::write(addr_of_mut!((*data_ptr).integer), 25);
    ///         ptr::write(addr_of_mut!((*data_ptr).string), "Default Data");
    ///     })
    /// };
    ///
    /// assert_eq!(handle.integer, 25);
    /// assert_eq!(handle.string, "Default Data");
    /// # let _ = handle;
    /// ```
    #[must_use]
    #[inline]
    pub unsafe fn init_with<F: FnOnce(&mut MaybeUninit<T>)>(arena: &'a Arena<A>, f: F) -> Self {
        let mut handle = Handle::new_uninit_in(arena);
        f(handle.as_mut());
        unsafe { Handle::assume_init(handle) }
    }

    /// Converts the handle into a `Handle<[T]>` with a slice length of `1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    /// let handle = Handle::new_in(&arena, 'c');
    /// let slice_handle = Handle::into_slice(handle);
    /// assert_eq!(&*slice_handle, &['c']);
    /// ```
    #[must_use]
    #[inline]
    pub const fn into_slice(this: Self) -> Handle<'a, [T], A> {
        let (ptr, arena) = Handle::into_raw(this);
        unsafe {
            let slice = ptr::slice_from_raw_parts_mut(ptr, 1);
            Handle::from_raw_in(slice, arena)
        }
    }

    /// Converts the handle into a `Handle<[T; 1]>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    /// let handle = Handle::new_in(&arena, 266);
    ///
    /// let array_handle = Handle::into_array(handle);
    /// assert_eq!(array_handle, [266; 1]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn into_array(this: Self) -> Handle<'a, [T; 1], A> {
        let (ptr, arena) = Handle::into_raw(this);
        unsafe { Handle::from_raw_in(ptr.cast::<[T; 1]>(), arena) }
    }
}

impl<'a, T, A: Allocator> Handle<'a, [T; 1], A> {
    /// Create a `Handle<T>` from a handle to an array containing a single element.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    /// let array_handle = Handle::new_in(&arena, [25; 1]);
    ///
    /// let handle = Handle::from_array(array_handle);
    /// assert_eq!(handle, 25);
    /// ```
    #[must_use]
    #[inline]
    pub const fn from_array(this: Self) -> Handle<'a, T, A> {
        let (ptr, arena) = Handle::into_raw(this);
        unsafe { Handle::from_raw_in(ptr.cast::<T>(), arena) }
    }
}

impl<'a, T: Default, A: Allocator> Handle<'a, T, A> {
    /// Create a new `Handle` in `arena`, containing the default value of `T`.
    ///
    /// # Example
    ///
    /// ```
    /// # use rotunda::{Arena, handle::Handle};
    /// let arena = Arena::new();
    /// let handle = Handle::<Vec<i32>>::new_default_in(&arena);
    /// assert_eq!(&*handle, &<Vec<i32>>::default());
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_default_in(arena: &'a Arena<A>) -> Self {
        Handle::new_in(arena, Default::default())
    }
}

impl<'a, T, A: Allocator> Handle<'a, MaybeUninit<T>, A> {
    /// Create a new `Handle` in `arena`, containing an uninitialized `MaybeUninit<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rotunda::{Arena, handle::Handle};
    /// # use core::mem::MaybeUninit;
    /// let arena = Arena::new();
    ///
    /// let handle: Handle<MaybeUninit<i32>> = Handle::new_uninit_in(&arena);
    /// # let _handle = handle;
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_uninit_in(arena: &'a Arena<A>) -> Self {
        let layout = Layout::new::<T>().pad_to_align();
        let ptr = arena.alloc_raw(layout).cast::<MaybeUninit<T>>();
        unsafe { Handle::from_raw_in(ptr.as_ptr(), arena) }
    }

    /// Create a new `Handle` in `arena`, containing a zeroed `MaybeUninit<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rotunda::{Arena, handle::Handle};
    /// # use core::mem::MaybeUninit;
    /// let arena = Arena::new();
    ///
    /// let handle: Handle<MaybeUninit<i32>> = Handle::new_uninit_zeroed_in(&arena);
    /// # let _handle = handle;
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_uninit_zeroed_in(arena: &'a Arena<A>) -> Self {
        let layout = Layout::new::<T>().pad_to_align();
        let ptr = arena.alloc_raw_zeroed(layout).cast::<MaybeUninit<T>>();
        unsafe { Handle::from_raw_in(ptr.as_ptr(), arena) }
    }
}

impl<'a, T, const N: usize, A: Allocator> Handle<'a, [MaybeUninit<T>; N], A> {
    #[track_caller]
    #[inline]
    #[must_use]
    pub fn new_array_uninit_in(arena: &'a Arena<A>) -> Self {
        let handle = Handle::new_uninit_in(arena);
        Handle::<'_, MaybeUninit<[T; N]>, A>::transpose_inner_uninit(handle)
    }

    #[track_caller]
    #[inline]
    #[must_use]
    pub fn new_array_uninit_zeroed_in(arena: &'a Arena<A>) -> Self {
        let handle = Handle::new_uninit_zeroed_in(arena);
        Handle::transpose_inner_uninit(handle)
    }
}

impl<'a, T, A: Allocator> Handle<'a, [MaybeUninit<T>], A> {
    #[track_caller]
    #[inline]
    #[must_use]
    pub fn new_slice_uninit_in(arena: &'a Arena<A>, slice_len: usize) -> Self {
        let type_layout = Layout::new::<T>();
        let (array_layout, ..) = { layout_repeat(&type_layout, slice_len).expect("size overflow") };

        let ptr = {
            let ptr = arena.alloc_raw(array_layout).cast::<MaybeUninit<T>>();
            NonNull::slice_from_raw_parts(ptr, slice_len)
        };

        unsafe { Handle::from_raw_in(ptr.as_ptr(), arena) }
    }
}

impl<'a, T, const N: usize, A: Allocator> Handle<'a, [T; N], A> {
    /// Create a new `Handle` array.
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_array_from_fn_in<F: FnMut(usize) -> T>(arena: &'a Arena<A>, f: F) -> Self {
        let buffer = Buffer::from_fn_in(arena, N, f);
        unsafe { Handle::into_array_unchecked::<N>(buffer.into_slice_handle()) }
    }
}

impl<'a, T, const N: usize, A: Allocator> Handle<'a, MaybeUninit<[T; N]>, A> {
    #[must_use]
    #[inline]
    pub fn transpose_inner_uninit(self) -> Handle<'a, [MaybeUninit<T>; N], A> {
        unsafe { mem::transmute(self) }
    }
}

impl<'a, T, const N: usize, A: Allocator> Handle<'a, [MaybeUninit<T>; N], A> {
    #[must_use]
    #[inline]
    pub fn transpose_outer_uninit(self) -> Handle<'a, MaybeUninit<[T; N]>, A> {
        unsafe { mem::transmute(self) }
    }

    #[must_use]
    #[inline]
    pub const unsafe fn assume_init_array(self) -> Handle<'a, [T; N], A> {
        unsafe { mem::transmute(self) }
    }
}

impl<'a, T: ?Sized, A: Allocator> Handle<'a, T, A> {
    #[inline]
    pub const fn arena(&'_ self) -> &'a Arena<A> {
        self.arena
    }

    /// Converts the `Handle` into a raw pointer, also returning the `Arena` from
    /// which it was allocated.
    ///
    /// # Notes
    ///
    /// Ownership of the resource managed by the `Handle` is transferred to
    /// the caller. It is their responsibility to ensure that the contents
    /// of the pointer are dropped when necessary.
    ///
    /// `Handle` has no drop logic - it is valid to use either [`Handle::from_raw_with_alloc`]
    /// or [`core::ptr::drop_in_place`] to clean up the resource.
    ///
    /// The pointer returned by `Handle` should not be accessed if the memory backing
    /// it in the `Arena` is dellocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    /// let handle = Handle::new_in(&arena, 42);
    ///
    /// let (ptr, arena) = Handle::into_raw(handle);
    /// # unsafe { core::ptr::drop_in_place(ptr); }
    /// ```
    ///
    /// [`Handle::from_raw_with_alloc`]: ./struct.Handle.html#method.from_raw_with_alloc
    /// [`core::ptr::drop_in_place`]: ./https://doc.rust-lang.org/stable/core/ptr/fn.drop_in_place.html
    #[must_use]
    #[inline]
    pub const fn into_raw(mut this: Self) -> (*mut T, &'a Arena<A>) {
        let (raw, arena) = (Handle::as_mut_ptr(&mut this), this.arena);
        let _this = ManuallyDrop::new(this);
        (raw, arena)
    }

    /// Access the contained value through a const pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use core::ptr;
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    /// let handle = Handle::new_in(&arena, 42);
    ///
    /// let ptr = Handle::as_ptr(&handle);
    /// # let _ = ptr;
    /// ```
    #[inline]
    #[must_use]
    pub const fn as_ptr(this: &Self) -> *const T {
        this.ptr.as_ptr().cast_const()
    }

    /// Access the contained value through a mutable pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use core::ptr;
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    /// let mut handle = Handle::new_in(&arena, 42);
    ///
    /// let ptr = Handle::as_mut_ptr(&mut handle);
    /// # let _ = ptr;
    /// ```
    #[inline]
    #[must_use]
    pub const fn as_mut_ptr(this: &mut Self) -> *mut T {
        this.ptr.as_ptr()
    }

    /// Wrap the `Handle` in a `Pin`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use core::ptr;
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    /// let handle = Handle::new_in(&arena, 42);
    ///
    /// let pinned = Handle::into_pin(handle);
    /// # let _ = pinned;
    /// ```
    #[inline]
    #[must_use]
    pub const fn into_pin(this: Self) -> Pin<Self> {
        unsafe { Pin::new_unchecked(this) }
    }

    /// Take ownership of the given `raw` pointer in the given `Arena`.
    ///
    /// # Safety
    ///
    /// It is the caller's responsibiity to ensure that `raw` has been created from
    /// a previous call to [`Handle::into_raw`], and that the `Arena` backing it's
    /// allocation has not cleared the block in use by the allocation.
    ///
    /// If this precondition is not held, then the `Handle` may point to dangling
    /// memory.
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    /// let handle = Handle::new_in(&arena, 42u8);
    ///
    /// let (raw, arena) = Handle::into_raw(handle);
    ///
    /// unsafe {
    ///     *&mut (*raw) = 255;
    /// }
    ///
    /// let handle = unsafe { Handle::from_raw_in(raw, arena) };
    /// assert_eq!(handle, 255);
    /// ```
    ///
    /// [`Handle::into_raw`]: ./struct.Handle.html#method.into_raw
    #[inline]
    pub const unsafe fn from_raw_in(raw: *mut T, arena: &'a Arena<A>) -> Self {
        let ptr = unsafe { NonNull::new_unchecked(raw) };

        Self {
            ptr,
            arena,
            _boo: PhantomData,
        }
    }

    #[must_use]
    #[inline]
    pub(crate) const fn as_nonnull(this: &Self) -> NonNull<T> {
        this.ptr
    }

    #[must_use]
    #[inline]
    const fn as_ref_const(&self) -> &T {
        unsafe { self.ptr.as_ref() }
    }

    #[must_use]
    #[inline]
    const fn as_mut_const(&mut self) -> &mut T {
        unsafe { self.ptr.as_mut() }
    }
}

impl<'a, T: Unpin, A: Allocator> Handle<'a, T, A> {
    /// Returns the inner value of the `Handle`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    ///
    /// let handle = Handle::new_in(&arena, vec![1, 2, 3, 4]);
    ///
    /// let data = Handle::into_inner(handle);
    /// assert_eq!(&data, &[1, 2, 3, 4]);
    /// ```
    #[inline]
    pub const fn into_inner(this: Self) -> T {
        let inner = unsafe { this.ptr.read() };
        let _this = ManuallyDrop::new(this);
        inner
    }

    /// Extracts the inner value of the `Handle`, returning the value and the empty `Handle`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut handle = Handle::new_in(&arena, 1024);
    /// let (data, empty_handle) = Handle::extract_inner(handle);
    ///
    /// assert_eq!(data, 1024);
    ///
    /// let new_handle = Handle::init(empty_handle, 2048);
    /// # let _ = new_handle;
    /// ```
    #[inline]
    pub const fn extract_inner(this: Self) -> (T, Handle<'a, MaybeUninit<T>, A>) {
        let arena = this.arena;
        let inner = unsafe { this.ptr.read() };
        let handle =
            unsafe { Handle::from_raw_in(this.ptr.cast::<MaybeUninit<T>>().as_ptr(), arena) };

        let _this = ManuallyDrop::new(this);
        (inner, handle)
    }

    /// Replace the contents of this `Handle` with the given `value`.
    ///
    /// The previous contents of the `Handle` are returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut handle = Handle::new_in(&arena, 25);
    /// let prev = Handle::replace(&mut handle, 42);
    ///
    /// assert_eq!(*handle, 42);
    /// assert_eq!(prev, 25);
    /// ```
    #[inline]
    pub const fn replace(this: &mut Self, mut value: T) -> T {
        unsafe {
            mem::swap(this.ptr.as_mut(), &mut value);
        }
        value
    }
}

impl<'a, A: Allocator> Handle<'a, dyn Any, A> {
    #[inline]
    pub fn downcast<T: Any>(self) -> Result<Handle<'a, T, A>, Handle<'a, dyn Any, A>> {
        if self.is::<T>() {
            unsafe { Ok(Self::downcast_unchecked::<T>(self)) }
        } else {
            Err(self)
        }
    }

    #[must_use]
    #[inline]
    pub unsafe fn downcast_unchecked<T: Any>(self) -> Handle<'a, T, A> {
        unsafe {
            let (ptr, arena) = Self::into_raw(self);
            Handle::from_raw_in(ptr as *mut T, arena)
        }
    }
}

impl<'a, A: Allocator> Handle<'a, dyn Any + Send, A> {
    #[inline]
    pub fn downcast<T: Any>(self) -> Result<Handle<'a, T, A>, Handle<'a, dyn Any + Send, A>> {
        if self.is::<T>() {
            unsafe { Ok(Self::downcast_unchecked::<T>(self)) }
        } else {
            Err(self)
        }
    }

    #[must_use]
    #[inline]
    pub unsafe fn downcast_unchecked<T: Any>(self) -> Handle<'a, T, A> {
        unsafe {
            let (ptr, arena) = Self::into_raw(self);
            Handle::from_raw_in(ptr as *mut T, arena)
        }
    }
}

impl<'a, A: Allocator> Handle<'a, dyn Any + Send + Sync, A> {
    #[inline]
    pub fn downcast<T: Any>(
        self,
    ) -> Result<Handle<'a, T, A>, Handle<'a, dyn Any + Send + Sync, A>> {
        if self.is::<T>() {
            unsafe { Ok(Self::downcast_unchecked::<T>(self)) }
        } else {
            Err(self)
        }
    }

    #[must_use]
    #[inline]
    pub unsafe fn downcast_unchecked<T: Any>(self) -> Handle<'a, T, A> {
        unsafe {
            let (ptr, arena) = Self::into_raw(self);
            Handle::from_raw_in(ptr as *mut T, arena)
        }
    }
}

#[cfg(feature = "nightly")]
impl<'a, T: ?Sized + Pointee, A: Allocator> Handle<'a, T, A> {
    #[must_use]
    #[inline]
    pub const unsafe fn from_raw_parts_in(
        ptr: *mut impl Thin,
        metadata: <T as Pointee>::Metadata,
        arena: &'a Arena<A>,
    ) -> Self {
        let ptr = ptr::from_raw_parts_mut(ptr, metadata);
        unsafe { Handle::from_raw_in(ptr, arena) }
    }

    #[must_use]
    #[inline]
    pub const fn to_raw_parts(self) -> (*const (), <T as Pointee>::Metadata, &'a Arena<A>) {
        let (raw, arena) = Self::into_raw(self);
        let (data, metadata) = <*const T>::to_raw_parts(raw);
        (data, metadata, arena)
    }
}

impl<'a, T, A: Allocator> Handle<'a, [T], A> {
    #[inline]
    pub const fn empty_in(arena: &'a Arena<A>) -> Self {
        unsafe { Self::slice_from_raw_parts_in(ptr::dangling_mut(), 0, arena) }
    }

    #[must_use]
    #[inline]
    pub const unsafe fn slice_from_raw_parts_in(
        data: *mut T,
        len: usize,
        arena: &'a Arena<A>,
    ) -> Self {
        let slice = ptr::slice_from_raw_parts_mut(data, len);
        unsafe { Self::from_raw_in(slice, arena) }
    }

    /// Create a `Handle` slice of length `slice_len`, where each element is initialized by `f`.
    ///
    /// The function `f` is called with the index of each item in the `Handle`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    ///
    /// let handle = Handle::new_slice_with_fn_in(&arena, 5, |i| i * 2);
    ///
    /// assert_eq!(handle.as_ref(), &[0, 2, 4, 6, 8]);
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_slice_with_fn_in<F: FnMut(usize) -> T>(
        arena: &'a Arena<A>,
        slice_len: usize,
        f: F,
    ) -> Self {
        let buf = Buffer::from_fn_in(arena, slice_len, f);
        buf.into_slice_handle()
    }

    /// Create a `Handle` slice containing the contents of the given iterator.
    ///
    /// # Panics
    ///
    /// If there is not enough room in the current `Block` to move all items from
    /// the `iter` into the `Handle`, this method will panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    ///
    /// let handle = Handle::new_slice_from_iter_in(
    ///     &arena,
    ///     [0, 1, 2, 3, 4]);
    ///
    /// assert_eq!(handle.as_ref(), &[0, 1, 2, 3, 4]);
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_slice_from_iter_in<I: IntoIterator<Item = T>>(arena: &'a Arena<A>, iter: I) -> Self
    where
        I::IntoIter: ExactSizeIterator,
    {
        Buffer::new_in(arena, iter).into_slice_handle()
    }

    /// Create a `Buffer` from this `Handle`.
    ///
    /// See [`Buffer::from_slice_handle`] for more details.
    ///
    /// [`Buffer::from_slice_handle`]: ../buffer/struct.Buffer.html#method.from_slice_handle
    #[must_use]
    #[inline]
    pub const fn into_buffer(this: Self) -> Buffer<'a, T, A> {
        Buffer::from_slice_handle(this)
    }

    #[must_use]
    #[inline]
    pub fn split_off(this: &mut Self, at: usize) -> Handle<'a, [T], A> {
        let (lhs, rhs) = {
            let empty =
                unsafe { Handle::slice_from_raw_parts_in(ptr::dangling_mut(), 0, this.arena) };
            let this = mem::replace(this, empty);
            Handle::split_at(this, at)
        };

        *this = lhs;
        rhs
    }

    /// Split the `Handle` at the given `mid`, returning the left and right
    /// sides of the array.
    ///
    /// # Panics
    ///
    /// This method will panic if `mid` is greater than or equal to `self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    ///
    /// let handle = Handle::new_slice_from_iter_in(&arena, [1, 2, 3, 4, 5, 6]);
    ///
    /// let (lhs, rhs) = Handle::split_at(handle, 3);
    ///
    /// assert_eq!(lhs.as_ref(), &[1, 2, 3]);
    /// assert_eq!(rhs.as_ref(), &[4, 5, 6]);
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub const fn split_at(this: Self, mid: usize) -> (Handle<'a, [T], A>, Handle<'a, [T], A>) {
        if Self::check_split(&this, mid) {
            unsafe { Self::split_at_unchecked(this, mid) }
        } else {
            panic!("mid > len")
        }
    }

    /// Split the `Handle` at the given `mid`, returning the left and right
    /// sides of the array.
    ///
    /// If `mid` is greater than or equal to `self.len()`, then the original
    /// `Handle` is returned in the `Err` variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    ///
    /// let handle = Handle::new_slice_from_iter_in(&arena, [1, 2, 3, 4, 5, 6]);
    ///
    /// let Err(handle) = Handle::split_at_checked(handle, 7) else {
    ///     unreachable!();
    /// };
    ///
    /// let (lhs, rhs) = Handle::split_at_checked(handle, 3).unwrap();
    ///
    /// assert_eq!(lhs.as_ref(), &[1, 2, 3]);
    /// assert_eq!(rhs.as_ref(), &[4, 5, 6]);
    /// ```
    #[inline]
    pub const fn split_at_checked(this: Self, mid: usize) -> Result<(Self, Self), Self> {
        if Self::check_split(&this, mid) {
            unsafe { Ok(Self::split_at_unchecked(this, mid)) }
        } else {
            Err(this)
        }
    }

    /// Split the `Handle` at the given `mid`, returning the left and right
    /// sides of the array.
    ///
    /// # Safety
    ///
    /// The given `mid` must be less than the length of the `Buffer`, otherwise this
    /// method will trigger undefined behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    ///
    /// let handle = Handle::new_slice_from_iter_in(&arena, [1, 2, 3, 4, 5, 6, 7]);
    ///
    /// let (lhs, rhs) = unsafe { Handle::split_at_unchecked(handle, 1) };
    ///
    /// assert_eq!(lhs.as_ref(), &[1]);
    /// assert_eq!(rhs.as_ref(), &[2, 3, 4, 5, 6, 7]);
    /// ```
    #[must_use]
    #[inline]
    pub const unsafe fn split_at_unchecked(
        this: Self,
        mid: usize,
    ) -> (Handle<'a, [T], A>, Handle<'a, [T], A>) {
        let len = unsafe { this.ptr.as_ref().len() };
        unsafe {
            assert_unchecked(mid <= len);
        }

        let (ptr, arena) = Self::into_raw(this);
        let ptr = ptr as *mut T;

        let lhs = ptr::slice_from_raw_parts_mut(ptr, mid);
        let rhs = unsafe { ptr::slice_from_raw_parts_mut(ptr.add(mid), len.unchecked_sub(mid)) };

        unsafe {
            (
                Handle::from_raw_in(lhs, arena),
                Handle::from_raw_in(rhs, arena),
            )
        }
    }

    #[must_use]
    #[inline]
    pub fn iter(&self) -> <&'_ [T] as IntoIterator>::IntoIter {
        self.as_ref().iter()
    }

    #[must_use]
    #[inline]
    pub fn iter_mut(&mut self) -> <&'_ mut [T] as IntoIterator>::IntoIter {
        self.as_mut().iter_mut()
    }

    #[must_use]
    #[inline]
    pub fn iter_handles(this: Self) -> IntoIterHandles<'a, T, A> {
        Handle::into_buffer(this).iter_handles()
    }

    /// Transpose a `Handle` of slice `T` into a slice of `Uninit<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    ///
    /// let handle = Handle::new_slice_from_iter_in(&arena, [1, 2, 3, 4, 5, 6, 7]);
    ///
    /// let handle = Handle::transpose_into_uninit(handle);
    /// # let _ = handle;
    /// ```
    #[must_use]
    #[inline]
    pub const fn transpose_into_uninit(this: Self) -> Handle<'a, [MaybeUninit<T>], A> {
        let arena = this.arena;
        let ptr = unsafe { NonNull::new_unchecked(this.ptr.as_ptr() as *mut [MaybeUninit<T>]) };
        let _this = ManuallyDrop::new(this);

        Handle {
            ptr,
            arena,
            _boo: PhantomData,
        }
    }

    #[inline]
    #[must_use]
    const fn check_split(this: &Self, mid: usize) -> bool {
        mid <= this.ptr.len()
    }

    #[inline]
    pub(crate) const unsafe fn set_len(this: &mut Self, new_len: usize) {
        let ptr = this.ptr.as_ptr() as *mut T;
        let ptr = unsafe { NonNull::new_unchecked(ptr::slice_from_raw_parts_mut(ptr, new_len)) };

        this.ptr = ptr;
    }

    #[must_use]
    #[inline]
    pub(crate) unsafe fn into_array_unchecked<const N: usize>(this: Self) -> Handle<'a, [T; N], A> {
        let (ptr, arena) = Self::into_raw(this);
        unsafe { Handle::from_raw_in(ptr as *mut [T; N], arena) }
    }
}

impl<'a, T: Copy, A: Allocator> Handle<'a, [T], A> {
    /// Create a new slice handle `slice_len` long, filled with `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    /// let handle = Handle::new_slice_splat_in(&arena, 255, 123);
    ///
    /// assert_eq!(handle.as_ref(), &[123; 255]);
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_slice_splat_in(arena: &'a Arena<A>, slice_len: usize, value: T) -> Self {
        let mut buf = Buffer::with_capacity_in(arena, slice_len);
        buf.extend((0..slice_len).map(|_| value));
        buf.into_slice_handle()
    }
}

impl<'a, T, A: Allocator> Handle<'a, MaybeUninit<T>, A> {
    /// Consumes the `Handle`, returning a new `Handle` which treats its contents
    /// as fully initialized.
    ///
    /// # Safety
    ///
    /// This method must be called on a `Handle` which has had its contents fully
    /// initialized, otherwise it may permit access to uninitialized memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    ///
    /// let handle = unsafe {
    ///     let mut handle = Handle::new_uninit_in(&arena);
    ///     handle.as_mut().write(28);
    ///     Handle::assume_init(handle)
    /// };
    ///
    /// assert_eq!(*handle, 28);
    /// ```
    #[inline]
    pub const unsafe fn assume_init(this: Self) -> Handle<'a, T, A> {
        let ptr = this.ptr.cast();
        let arena = this.arena;
        let _this = ManuallyDrop::new(this);
        Handle {
            ptr,
            arena,
            _boo: PhantomData,
        }
    }

    /// Writes the `value` into the `Handle` and initilizes it.
    ///
    /// Owership over `value` is transferred into the `Handle`. Any
    /// data previously written to the `Handle` is overwritten.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    ///
    /// let handle = Handle::new_uninit_in(&arena);
    /// let handle = Handle::init(handle, 25);
    ///
    /// assert_eq!(*handle, 25);
    /// ```
    #[inline]
    pub const fn init(this: Self, value: T) -> Handle<'a, T, A> {
        unsafe {
            ptr::write(this.ptr.as_ptr(), MaybeUninit::new(value));
            Self::assume_init(this)
        }
    }
}

impl<'a, T, A: Allocator> Handle<'a, [MaybeUninit<T>], A> {
    #[inline]
    pub const unsafe fn assume_init_slice(this: Self) -> Handle<'a, [T], A> {
        let (ptr, arena) = Handle::into_raw(this);
        let len = ptr.len();
        let ptr = ptr::slice_from_raw_parts_mut(ptr as *mut T, len);

        unsafe { Handle::from_raw_in(ptr, arena) }
    }
}

impl<'a, A: Allocator> Handle<'a, str, A> {
    /// Create a new `Handle` containing the given `string`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let arena = Arena::new();
    ///
    /// let handle = Handle::new_str_in(&arena, "Some Data");
    ///
    /// assert_eq!(&handle, "Some Data");
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_str_in<S: ?Sized + AsRef<str>>(arena: &'a Arena<A>, string: &S) -> Self {
        fn inner<'a, A: Allocator>(arena: &'a Arena<A>, string: &str) -> Handle<'a, str, A> {
            let len = string.len();
            let hndl = Handle::<'_, [MaybeUninit<u8>], A>::new_slice_uninit_in(arena, len);
            let (data, arena) = Handle::into_raw(hndl);
            let data = data as *mut MaybeUninit<u8> as *mut _;

            unsafe {
                ptr::copy_nonoverlapping(string.as_ptr(), data, string.len());

                let ptr = ptr::slice_from_raw_parts_mut(data, len);
                Handle::from_raw_in(ptr as *mut str, arena)
            }
        }

        inner(arena, string.as_ref())
    }

    #[inline]
    pub const fn from_utf8(bytes: Handle<'a, [u8], A>) -> Result<Self, FromUtf8Error<'a, A>> {
        match str::from_utf8(bytes.as_ref_const()) {
            Ok(_) => unsafe { Ok(Self::from_utf8_unchecked(bytes)) },
            Err(error) => Err(FromUtf8Error::new(Buffer::from_slice_handle(bytes), error)),
        }
    }

    #[inline]
    pub const unsafe fn from_utf8_unchecked(bytes: Handle<'a, [u8], A>) -> Self {
        unsafe { mem::transmute(bytes) }
    }

    #[inline]
    pub const fn into_bytes(this: Self) -> Handle<'a, [u8], A> {
        unsafe { mem::transmute(this) }
    }
}

impl<'a, T: ?Sized + fmt::Debug, A: Allocator> fmt::Debug for Handle<'a, T, A> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_ref(), fmtr)
    }
}

impl<'a, T: ?Sized + fmt::Display, A: Allocator> fmt::Display for Handle<'a, T, A> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_ref(), fmtr)
    }
}

impl<'a, T: ?Sized, A: Allocator> fmt::Pointer for Handle<'a, T, A> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, fmtr)
    }
}

impl<'a, T: ?Sized, A: Allocator> AsRef<T> for Handle<'a, T, A> {
    #[inline]
    fn as_ref(&self) -> &T {
        self.as_ref_const()
    }
}

impl<'a, T: ?Sized, A: Allocator> AsMut<T> for Handle<'a, T, A> {
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        self.as_mut_const()
    }
}

impl<'a, A: Allocator> AsRef<[u8]> for Handle<'a, str, A> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl<'a, T: ?Sized, A: Allocator> Borrow<T> for Handle<'a, T, A> {
    #[inline]
    fn borrow(&self) -> &T {
        self.as_ref_const()
    }
}

impl<'a, T: ?Sized, A: Allocator> BorrowMut<T> for Handle<'a, T, A> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut T {
        self.as_mut_const()
    }
}

impl<'a, T, A: Allocator, I: SliceIndex<[T]>> Index<I> for Handle<'a, [T], A> {
    type Output = <[T] as Index<I>>::Output;
    #[track_caller]
    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        self.as_ref().index(index)
    }
}

impl<'a, T, A: Allocator, I: SliceIndex<[T]>> IndexMut<I> for Handle<'a, [T], A> {
    #[track_caller]
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.as_mut().index_mut(index)
    }
}

impl<'sl: 'a, 'a, T: 'a, A: Allocator> IntoIterator for &'sl Handle<'a, [T], A> {
    type Item = &'a T;
    type IntoIter = <&'a [T] as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        AsRef::<[T]>::as_ref(self).iter()
    }
}

impl<'sl: 'a, 'a, T: 'a, A: Allocator> IntoIterator for &'sl mut Handle<'a, [T], A> {
    type Item = &'a mut T;
    type IntoIter = <&'a mut [T] as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        AsMut::<[T]>::as_mut(self).iter_mut()
    }
}

impl<'a, T: Unpin, A: Allocator> IntoIterator for Handle<'a, [T], A> {
    type Item = T;
    type IntoIter = <Buffer<'a, T, A> as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Buffer::from_slice_handle(self).into_iter()
    }
}

impl<'a, T, A: Allocator> From<Buffer<'a, T, A>> for Handle<'a, [T], A> {
    #[inline]
    fn from(value: Buffer<'a, T, A>) -> Self {
        value.into_slice_handle()
    }
}

impl<'a, A: Allocator> From<StringBuffer<'a, A>> for Handle<'a, str, A> {
    #[inline]
    fn from(value: StringBuffer<'a, A>) -> Self {
        value.into_str_handle()
    }
}

impl<'a, T, A: Allocator> From<Handle<'a, T, A>> for Handle<'a, [T], A> {
    #[inline]
    fn from(value: Handle<'a, T, A>) -> Self {
        Handle::into_slice(value)
    }
}

impl<'a, T, A: Allocator> From<Handle<'a, T, A>> for Handle<'a, [T; 1], A> {
    #[inline]
    fn from(value: Handle<'a, T, A>) -> Self {
        Handle::into_array(value)
    }
}

impl<'a, T, A: Allocator> From<Handle<'a, [T; 1], A>> for Handle<'a, T, A> {
    #[inline]
    fn from(value: Handle<'a, [T; 1], A>) -> Self {
        Handle::from_array(value)
    }
}

impl<'a, A: Allocator> TryFrom<Handle<'a, [u8], A>> for Handle<'a, str, A> {
    type Error = FromUtf8Error<'a, A>;
    #[inline]
    fn try_from(value: Handle<'a, [u8], A>) -> Result<Self, Self::Error> {
        Handle::from_utf8(value)
    }
}

impl<'a> From<Handle<'a, str>> for Handle<'a, [u8]> {
    #[inline]
    fn from(value: Handle<'a, str>) -> Self {
        Handle::into_bytes(value)
    }
}

impl<'a, T: ?Sized, A: Allocator> Drop for Handle<'a, T, A> {
    #[inline]
    fn drop(&mut self) {
        // Size of value must be calculated before it is dropped.
        let size_of_val = mem::size_of_val(self.as_ref_const());

        let p = self.ptr.as_ptr();
        unsafe {
            ptr::drop_in_place(p);
        }

        if self.arena.blocks.is_last_allocation(self.ptr.cast::<()>()) {
            unsafe {
                self.arena.blocks.unbump(size_of_val);
            }
        }
    }
}

impl<'a, T: ?Sized, A: Allocator> TryFrom<RcHandle<'a, T, A>> for Handle<'a, T, A> {
    type Error = RcHandle<'a, T, A>;
    #[inline]
    fn try_from(value: RcHandle<'a, T, A>) -> Result<Self, Self::Error> {
        RcHandle::try_into_handle(value)
    }
}

impl<'a, T, const N: usize, A: Allocator> TryFrom<Handle<'a, [T], A>> for Handle<'a, [T; N], A> {
    type Error = Handle<'a, [T], A>;
    #[inline]
    fn try_from(value: Handle<'a, [T], A>) -> Result<Self, Self::Error> {
        if value.len() == N {
            unsafe {
                let (ptr, arena) = Handle::into_raw(value);
                Ok(Handle::from_raw_in(ptr.cast::<[T; N]>(), arena))
            }
        } else {
            Err(value)
        }
    }
}

impl<'a, T: RefUnwindSafe + ?Sized, A: Allocator + UnwindSafe> UnwindSafe for Handle<'a, T, A> {}

impl<'a, T: RefUnwindSafe + ?Sized, A: Allocator + UnwindSafe> RefUnwindSafe for Handle<'a, T, A> {}

#[cfg(feature = "nightly")]
impl<'a, T: Unsize<U> + ?Sized, U: ?Sized, A: Allocator> CoerceUnsized<Handle<'a, U, A>>
    for Handle<'a, T, A>
{
}

impl<'a, T: ?Sized, A: Allocator> Deref for Handle<'a, T, A> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<'a, T: ?Sized, A: Allocator> DerefMut for Handle<'a, T, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}

impl<'a, T: ?Sized + PartialEq, A: Allocator> PartialEq for Handle<'a, T, A> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_ref().eq(other.as_ref())
    }
}

impl<'a, T: ?Sized + PartialEq, A: Allocator> PartialEq<T> for Handle<'a, T, A> {
    #[inline]
    fn eq(&self, other: &T) -> bool {
        self.as_ref().eq(other)
    }
}

impl<'a, 'h, T: ?Sized + PartialEq, A: Allocator> PartialEq<&'h Handle<'h, T>>
    for Handle<'a, T, A>
{
    #[inline]
    fn eq(&self, other: &&'h Handle<'h, T>) -> bool {
        self.as_ref().eq(other.as_ref())
    }
}

impl<'a, 'v, T: ?Sized + PartialEq, A: Allocator> PartialEq<&'v T> for Handle<'a, T, A> {
    #[inline]
    fn eq(&self, other: &&'v T) -> bool {
        self.as_ref().eq(*other)
    }
}

impl<'a, T: ?Sized + Eq, A: Allocator> Eq for Handle<'a, T, A> {}

impl<'a, T: ?Sized + PartialOrd, A: Allocator> PartialOrd for Handle<'a, T, A> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_ref().partial_cmp(other.as_ref())
    }
}

impl<'a, T: ?Sized + PartialOrd, A: Allocator> PartialOrd<T> for Handle<'a, T, A> {
    #[inline]
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        self.as_ref().partial_cmp(other)
    }
}

impl<'a, T: ?Sized + Ord, A: Allocator> Ord for Handle<'a, T, A> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_ref().cmp(other.as_ref())
    }
}

impl<'a, T: ?Sized + Hash, A: Allocator> Hash for Handle<'a, T, A> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state)
    }
}

#[cfg(feature = "std")]
impl<'a, R: ?Sized + Read, A: Allocator> Read for Handle<'a, R, A> {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.as_mut().read(buf)
    }

    #[inline]
    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        self.as_mut().read_vectored(bufs)
    }

    #[inline]
    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.as_mut().read_exact(buf)
    }

    #[inline]
    fn read_to_end(&mut self, buf: &mut std::vec::Vec<u8>) -> io::Result<usize> {
        self.as_mut().read_to_end(buf)
    }
}

#[cfg(feature = "std")]
impl<'a, R: ?Sized + BufRead, A: Allocator> BufRead for Handle<'a, R, A> {
    #[inline]
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.as_mut().fill_buf()
    }

    #[inline]
    fn consume(&mut self, amount: usize) {
        self.as_mut().consume(amount)
    }
}

#[cfg(feature = "std")]
impl<'a, W: ?Sized + Write, A: Allocator> Write for Handle<'a, W, A> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.as_mut().write(buf)
    }

    #[inline]
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        self.as_mut().write_vectored(bufs)
    }

    #[cfg(feature = "nightly")]
    fn is_write_vectored(&self) -> bool {
        self.as_ref().is_write_vectored()
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.as_mut().write_all(buf)
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        self.as_mut().flush()
    }
}

#[cfg(feature = "serde")]
impl<'a, T: Serialize, A: Allocator> Serialize for Handle<'a, T, A> {
    #[inline]
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.as_ref().serialize(serializer)
    }
}
