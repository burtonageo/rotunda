#![allow(missing_docs, clippy::missing_safety_doc)]

//! A contiguous, growable array of values allocated in an `Arena`.

use crate::{Arena, InvariantLifetime, blocks::lock::BlockLock, handle::Handle};
use alloc::alloc::{Allocator, Layout};
use core::{
    borrow::{Borrow, BorrowMut},
    error::{self, Error},
    fmt,
    hash::{Hash, Hasher},
    hint::assert_unchecked,
    iter::{Fuse, FusedIterator},
    marker::PhantomData,
    mem::{self, ManuallyDrop, MaybeUninit},
    ops::{Deref, DerefMut, Index, IndexMut},
    ptr::{self, NonNull},
    slice::{self, SliceIndex},
};
#[cfg(feature = "std")]
use std::io::{self, Read, Write};

#[macro_export]
macro_rules! buf {
    ([ $($elem:expr),* $(,)? ] in $arena:expr) => {
        $crate::handle::Handle::<[_]>::into_buffer($crate::handle::Handle::new_slice_from_iter_in(&$arena, [$($elem),*].into_iter()))
    };

    ([$elem:expr ; $num:expr] in $arena:expr) => {
        $crate::handle::Handle::into_buffer($crate::handle::Handle::<[_]>::new_slice_splat_in(&$arena, $num, $elem))
    };

    (in $arena:expr; [ $($elem:expr),* $(,)? ]) => {
        buf!([$($elem),*] in $arena)
    };

    (in $arena:expr; [$elem:expr ; $num:expr]) => {
        buf!([$elem ; $num] in $arena)
    };
}

/// A `Buffer` is used to represent a contiguous array of `T`s allocated in an `Arena` block.
///
/// The buffer can be dynamically resized, but can only grow up to its [`capacity()`]. Beyond
/// that,  new `Buffer` will have to be allocated with a larger given capacity.
///
/// See the [module documentation] for more informtion.
///
/// [`capacity()`]: ./struct.Buffer.html#method.capacity
/// [module documentation]: ./index.html
pub struct Buffer<'a, T> {
    handle: Handle<'a, [MaybeUninit<T>]>,
    len: usize,
    _boo: PhantomData<T>,
}

// A buffer can be sent to other threads if the type within is
// thread-safe - it is guaranteed to drop before the arena is
// deallocated thanks to borrowing rules.
unsafe impl<'a, T: Send> Send for Buffer<'a, T> {}
unsafe impl<'a, T: Sync> Sync for Buffer<'a, T> {}

impl<'a, T> Buffer<'a, T> {
    /// Creates a new `Buffer` containing the contents of `iter`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    /// let buffer = Buffer::new_in(&arena, [1, 2, 3, 4, 5]);
    ///
    /// assert_eq!(buffer.as_slice(), &[1, 2, 3, 4, 5]);
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_in<I: IntoIterator<Item = T>, A: Allocator>(arena: &'a Arena<A>, iter: I) -> Self
    where
        I::IntoIter: ExactSizeIterator,
    {
        let iter = iter.into_iter();
        let mut buffer = Buffer::with_capacity_in(arena, iter.len());
        buffer.extend(iter);
        buffer
    }

    /// Create a new empty `Buffer` with the capacity to store up to `capacity` elements,
    /// backed by the given `Arena`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    ///
    /// let buffer = Buffer::<i32>::with_capacity_in(&arena, 10);
    /// assert!(buffer.is_empty());
    /// assert_eq!(buffer.capacity(), 10);
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn with_capacity_in<A: Allocator>(arena: &'a Arena<A>, capacity: usize) -> Self {
        let handle = Handle::new_slice_uninit_in(arena, capacity);
        unsafe { Self::from_raw_parts(handle, 0) }
    }

    /// Create a new `Buffer` from the given function.
    ///
    /// The function `f()` will be called `len` times with the index of each
    /// element as the parameter, and the results will be collected into
    /// the returned `Buffer`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle, buffer::Buffer};
    /// let arena = Arena::new();
    ///
    /// let buffer = Buffer::from_fn_in(&arena, 5, |elem| {
    ///     Handle::new_str_in(&arena, &format!("{}", elem))
    /// });
    ///
    /// assert_eq!(&buffer[0], "0");
    /// assert_eq!(&buffer[1], "1");
    /// assert_eq!(&buffer[2], "2");
    /// assert_eq!(&buffer[3], "3");
    /// assert_eq!(&buffer[4], "4");
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn from_fn_in<A: Allocator, F: FnMut(usize) -> T>(
        arena: &'a Arena<A>,
        len: usize,
        mut f: F,
    ) -> Self {
        let mut buf = Buffer::with_capacity_in(arena, len);
        for i in 0..len {
            unsafe {
                buf.push_unchecked(f(i));
            }
        }

        buf
    }

    #[track_caller]
    #[inline]
    pub fn try_with_growable_in<A, E, F>(arena: &'a Arena<A>, f: F) -> Result<Self, E>
    where
        A: Allocator,
        F: for<'buf> FnOnce(&'buf mut GrowableBuffer<'a, T, A>) -> Result<(), E>,
    {
        let curr_block_cap = arena.curr_block_capacity();
        if curr_block_cap.is_none() {
            arena.force_push_new_block();
        }

        let offset = arena.blocks.offset_to_align_for(&Layout::new::<T>());
        unsafe {
            arena.blocks.bump(offset);
        }

        let head = arena.curr_block_head();
        debug_assert!(head.is_some());

        let backing_storage = unsafe {
            let data = head.unwrap_unchecked().as_ptr();
            let len = data.len() / mem::size_of::<T>();
            let data = ptr::from_raw_parts_mut(data.cast::<T>(), len);
            NonNull::new_unchecked(data)
        };

        let lock = BlockLock::lock(arena);

        let mut growable_buffer = GrowableBuffer {
            backing_storage,
            len: 0,
            cap: 0,
            _boo: PhantomData,
        };

        let result = f(&mut growable_buffer);

        drop(lock);

        match result {
            Ok(_) => {
                unsafe {
                    arena
                        .blocks
                        .bump(offset + growable_buffer.cap * mem::size_of::<T>());
                }
                Ok(growable_buffer.into_buffer())
            }
            Err(e) => Err(e),
        }
    }

    /// Create a new `Buffer` with a growable initial capacity.
    ///
    /// See [`Buffer::try_with_growable_in`] for more info.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    ///
    /// let buffer = Buffer::with_growable_in(&arena, |buf| {
    ///     buf.push("Hello");
    ///     buf.push("World");
    ///     buf.reserve(25);
    /// });
    ///
    /// assert_eq!(buffer[0], "Hello");
    /// assert_eq!(buffer[1], "World");
    ///
    /// assert_eq!(buffer.capacity(), 27);
    /// ```
    /// 
    /// [`Buffer::try_with_growable_in`]: ./struct.Buffer.html#method.try_with_growable_in
    #[track_caller]
    #[inline]
    pub fn with_growable_in<A, F>(arena: &'a Arena<A>, f: F) -> Self
    where
        A: Allocator,
        F: for<'buf> FnOnce(&'buf mut GrowableBuffer<'a, T, A>),
    {
        enum Never {}
        let result = Self::try_with_growable_in::<A, Never, _>(arena, |buffer| {
            f(buffer);
            Ok(())
        });

        match result {
            Ok(buffer) => buffer,
        }
    }

    /// Create a new empty `Buffer` not backed by any `Arena`
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::buffer::Buffer;
    ///
    /// let buffer = Buffer::<u8>::new();
    ///
    /// assert_eq!(buffer.capacity(), 0);
    /// assert_eq!(&buffer, &[]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        unsafe { Self::from_raw_parts(Handle::empty(), 0) }
    }

    /// Compose a `Buffer` from its constituent parts.
    ///
    /// # Safety
    ///
    /// `Handle` must be a valid `Handle` pointing to a live memory allocation
    /// in an `Arena`. Additionally, `len` must be less than `handle.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::{ptr, mem::MaybeUninit};
    /// use rotunda::{Arena, buffer::Buffer, handle::Handle};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut storage = Handle::new_slice_uninit_in(&arena, 15);
    ///
    /// unsafe {
    ///     let storage_ptr = storage.as_mut_ptr();
    ///     for i in 0..3 {
    ///         ptr::write(storage_ptr.add(i), MaybeUninit::new(i));
    ///     }
    /// }
    ///
    /// let buffer = unsafe { Buffer::from_raw_parts(storage, 3) };
    ///
    /// assert_eq!(buffer.as_ref(), &[0, 1, 2]);
    /// ```
    #[must_use]
    #[inline]
    pub const unsafe fn from_raw_parts(handle: Handle<'a, [MaybeUninit<T>]>, len: usize) -> Self {
        Self {
            handle,
            len,
            _boo: PhantomData,
        }
    }

    /// Decompose a `Buffer` into its constituent parts.
    ///
    /// Ownership of the data is relinquished to the developer. It is the developer's responsibility
    /// to ensure that the initialized elements are dropped, otherwise this method will lead to a memory leak.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut buffer = Buffer::with_capacity_in(&arena, 300);
    /// buffer.extend([1, 2, 3, 4, 5]);
    ///
    /// let (storage, len) = Buffer::into_raw_parts(buffer);
    ///
    /// assert_eq!(storage.len(), 300);
    /// assert_eq!(len, 5);
    /// # let _ = unsafe { Buffer::from_raw_parts(storage, len) };
    /// ```
    #[must_use]
    #[inline]
    pub const fn into_raw_parts(self) -> (Handle<'a, [MaybeUninit<T>]>, usize) {
        let handle = unsafe { mem::transmute_copy(&self.handle) };
        let len = self.len;
        let _this = ManuallyDrop::new(self);

        (handle, len)
    }

    /// Access the contents of the `Buffer` as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    ///
    /// let buffer = Buffer::new_in(&arena, [1, 2, 3, 4, 5]);
    ///
    /// assert_eq!(buffer.as_slice(), &[1, 2, 3, 4, 5]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        let ptr = Handle::as_ptr(&self.handle);
        unsafe { slice::from_raw_parts(ptr as *const T, self.len) }
    }

    /// Access the contents of the `Buffer` as a mutable slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut buffer = Buffer::new_in(&arena, [1, 2, 3, 4, 5]);
    ///
    /// {
    ///     let slice = buffer.as_mut_slice();
    ///     slice[0] = 6;
    ///     slice[4] = 6;
    /// }
    ///
    /// assert_eq!(buffer.as_slice(), &[6, 2, 3, 4, 6]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        let ptr = Handle::as_mut_ptr(&mut self.handle);
        unsafe { slice::from_raw_parts_mut(ptr as *mut T, self.len) }
    }

    #[must_use]
    #[inline]
    pub fn iter(&'_ self) -> <&'_ [T] as IntoIterator>::IntoIter {
        self.as_slice().iter()
    }

    #[must_use]
    #[inline]
    pub fn iter_mut(&'_ mut self) -> <&'_ mut [T] as IntoIterator>::IntoIter {
        self.as_mut_slice().iter_mut()
    }

    /// Append the given `value` to the end of the `Buffer`.
    ///
    /// # Panics
    ///
    /// If the `Buffer` contains `capacity()` items, then this method will panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut buffer = Buffer::with_capacity_in(&arena, 10);
    /// buffer.push(21);
    /// buffer.push(512);
    /// buffer.push(999);
    ///
    /// assert_eq!(&buffer, &[21, 512, 999]);
    /// ```
    #[track_caller]
    #[inline]
    pub const fn push(&mut self, value: T) {
        let result = self.try_push(value);
        match result {
            Ok(()) => {
                mem::forget(result);
            }
            Err(_) => panic!("buffer oveflow"),
        }
    }

    /// Attempt to push the given `value` to the end of the `Buffer`.
    ///
    /// If the `Buffer` already contains `capacity()` items, then the original
    /// `value` will be returned as an error.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut buffer = Buffer::with_capacity_in(&arena, 2);
    ///
    /// buffer.try_push(42).unwrap();
    /// buffer.try_push(58).unwrap();
    ///
    /// assert_eq!(buffer.try_push(100), Err(100));
    ///
    /// assert_eq!(&buffer, &[42, 58]);
    /// ```
    #[inline]
    pub const fn try_push(&mut self, value: T) -> Result<(), T> {
        if self.is_full() {
            return Err(value);
        }
        // @SAFETY: the capacity is checked above
        unsafe {
            self.push_unchecked(value);
        }
        Ok(())
    }

    /// Attempt to extend the `Buffer` with the contents of the given `iter`.
    ///
    /// If the `Buffer`'s capacity is exhausted before the iterator is finished,
    /// this method will return an error.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut buffer = Buffer::with_capacity_in(&arena, 8);
    ///
    /// let _ = buffer.try_extend(1..5).unwrap();
    ///
    /// let err = buffer.try_extend(0..130).unwrap_err();
    ///
    /// let (elem, iter) = err.into_inner();
    /// assert_eq!(elem, 4);
    /// assert_eq!(iter.count(), 125);
    /// ```
    #[inline]
    pub fn try_extend<I: IntoIterator<Item = T>>(
        &mut self,
        iter: I,
    ) -> Result<(), TryExtendError<I>> {
        let mut iter = iter.into_iter().fuse();
        while let Some(item) = iter.next() {
            match self.try_push(item) {
                Ok(()) => (),
                Err(item) => {
                    return Err(TryExtendError {
                        curr: item,
                        rest: iter,
                    });
                }
            }
        }
        Ok(())
    }

    /// Remove the last element from the `Buffer`.
    ///
    /// If the `Buffer` is empty, this method returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut buffer = Buffer::new_in(&arena, ["Data 1", "Data 2"]);
    ///
    /// assert_eq!(buffer.pop(), Some("Data 2"));
    /// assert_eq!(buffer.pop(), Some("Data 1"));
    /// assert_eq!(buffer.pop(), None);
    /// ```
    #[inline]
    pub const fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        unsafe {
            self.set_len(self.len - 1);
            let value = Handle::as_nonnull(&self.handle)
                .as_ref()
                .get_unchecked(self.len)
                .assume_init_read();

            Some(value)
        }
    }

    /// Removes the item at `idx` from the `Buffer`.
    ///
    /// The order of the remaining elements are preserved.
    ///
    /// # Panics
    ///
    /// If the given `idx` is out of bounds, this method will panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut buffer = Buffer::new_in(&arena, [2, 3, 4, 5, 6]);
    ///
    /// assert_eq!(buffer.remove(1), 3);
    /// assert_eq!(&buffer, &[2, 4, 5, 6]);
    /// ```
    #[track_caller]
    #[inline]
    pub fn remove(&mut self, idx: usize) -> T {
        match self.try_remove(idx) {
            Some(value) => value,
            None => panic!(
                "removal index (is {}) should be < len (is {})",
                idx,
                self.len(),
            ),
        }
    }

    /// Attempt to remove the item at `idx` from the `Buffer`.
    ///
    /// The order of the remaining elements are preserved.
    ///
    /// If `idx` is out of bounds, this method returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut buffer = Buffer::new_in(&arena, [1, 3, 5, 7, 9]);
    ///
    /// assert_eq!(buffer.try_remove(5), None);
    /// assert_eq!(buffer.try_remove(0), Some(1));
    ///
    /// assert_eq!(&buffer, &[3, 5, 7, 9]);
    /// ```
    #[inline]
    pub fn try_remove(&mut self, idx: usize) -> Option<T> {
        let len = self.len();
        if len > 0 && idx < len {
            unsafe { Some(self.remove_unchecked(idx)) }
        } else {
            None
        }
    }

    /// Swap the element at `idx` with the last element in the list and removes it.
    ///
    /// The removed element is returned. If `idx` is greater than `self.len()`, this
    /// method returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut buffer = Buffer::new_in(&arena, [1, 3, 5, 7, 9]);
    ///
    /// let element = buffer.swap_remove(1);
    ///
    /// assert_eq!(element, Some(3));
    /// assert_eq!(&buffer, &[1, 9, 5, 7]);
    /// ```
    #[inline]
    pub fn swap_remove(&mut self, idx: usize) -> Option<T> {
        match self.len.checked_sub(1) {
            Some(len) if idx < self.len() => {
                self.as_mut_slice().swap(idx, len);
                self.pop()
            }
            _ => None,
        }
    }

    /// Convert the `Buffer` into a `Handle<[T]>`.
    ///
    /// The excess of the `Buffer` is truncated.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut buffer = Buffer::with_capacity_in(&arena, 10);
    /// buffer.extend(1..=5);
    ///
    /// let handle = Buffer::into_slice_handle(buffer);
    ///
    /// assert_eq!(handle.as_ref(), &[1, 2, 3, 4, 5]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn into_slice_handle(mut self) -> Handle<'a, [T]> {
        let ptr = Handle::as_mut_ptr(&mut self.handle);
        let ptr = ptr::slice_from_raw_parts_mut(ptr as *const T as *mut T, self.len);
        let _this = ManuallyDrop::new(self);

        unsafe { Handle::from_raw(ptr) }
    }

    /// Create a `Buffer` from the given `Handle`.
    ///
    /// The `Buffer` will have a capacity equal to the `handle`'s [`len()`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer, handle::Handle};
    ///
    /// let arena = Arena::new();
    ///
    /// let handle = Handle::new_in(&arena, [1, 2, 3, 4, 5]);
    ///
    /// let buffer = Buffer::from_slice_handle(handle);
    ///
    /// assert_eq!(&buffer, &[1, 2, 3, 4, 5]);
    /// ```
    ///
    /// [`len()`]: ./struct.Handle.html#method.len
    #[must_use]
    #[inline]
    pub const fn from_slice_handle(mut handle: Handle<'a, [T]>) -> Self {
        let ptr = Handle::as_mut_ptr(&mut handle);
        let len = ptr.len();
        let new_handle = unsafe {
            let ptr = ptr::slice_from_raw_parts_mut(ptr as *mut MaybeUninit<T>, len);
            Handle::from_raw(ptr)
        };

        let _hndl = ManuallyDrop::new(handle);

        unsafe { Buffer::from_raw_parts(new_handle, len) }
    }

    #[inline]
    pub fn split_at_spare_capacity(self) -> (Handle<'a, [T]>, Handle<'a, [MaybeUninit<T>]>) {
        let (handle, len) = Self::into_raw_parts(self);
        if len == 0 {
            return (Handle::empty(), handle);
        }

        unsafe {
            let (init, uninit) = Handle::split_at_unchecked(handle, len - 1);
            (Handle::assume_init_slice(init), uninit)
        }
    }

    /// Access the spare capacity of the `Buffer` as a mutable slice.
    ///
    /// The returned slice will have a length of `self.capacity() - self.len()`.
    ///
    /// Values written to this slice may be overwritten if the `Buffer` is modified
    /// without calling [`Buffer::set_len`] to ensure that the values are logically
    /// initialized in the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut buffer = Buffer::with_capacity_in(&arena, 25);
    ///
    /// let capacity = buffer.spare_capacity_mut();
    /// capacity[0].write("Hello");
    /// capacity[1].write("World");
    ///
    /// unsafe {
    ///     buffer.set_len(2);
    /// }
    ///
    /// assert_eq!(&buffer, &["Hello", "World"]);
    /// ```
    ///
    /// [`Buffer::set_len`]: ./struct.Buffer.html#method.set_len
    #[must_use]
    #[inline]
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        unsafe { self.handle.as_mut().get_unchecked_mut(self.len..) }
    }

    /// Returns `true` if the current `Buffer` is at capacity.
    ///
    /// If this is `true`, then pushing any more elements into the `Buffer`
    /// will fail.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut buffer = Buffer::with_capacity_in(&arena, 5); 
    /// 
    /// buffer.extend_from_slice_copy(&[1, 2, 3]);
    /// assert!(!buffer.is_full());
    /// 
    /// buffer.extend_from_slice_copy(&[4, 5]);
    /// assert!(buffer.is_full());
    /// ```
    #[must_use]
    #[inline]
    pub const fn is_full(&self) -> bool {
        self.len >= self.capacity()
    }

    #[must_use]
    #[inline]
    pub const fn has_space_for_elems(&self, num_elems: usize) -> bool {
        (self.capacity() - self.len) >= num_elems
    }

    /// Returns the total capacity of the `Buffer`.
    ///
    /// This represents the total number of items which can be stored in the `Buffer`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut buffer = Buffer::new_in(&arena, [1, 2, 3, 4, 5, 6]);
    ///
    /// assert_eq!(buffer.capacity(), 6);
    ///
    /// for _ in 0..2 {
    ///     let _ = buffer.pop();
    /// }
    ///
    /// assert_eq!(buffer.capacity(), 6);
    /// ```
    #[must_use]
    #[inline]
    pub const fn capacity(&self) -> usize {
        Handle::as_ptr(&self.handle).len()
    }

    /// Sets the length of the `Buffer` to `new_len` without modifying
    /// the contents of the `Buffer`.
    ///
    /// If `set_len()` is called with a smaller length than the current `Buffer`'s
    /// length, then the items between the previous length and the new length
    /// will be leaked when the `Buffer` is dropped.
    ///
    /// # Safety
    ///
    /// You must ensure that all of the items contained in the `Buffer` up to `new_len`
    /// are fully initialized. If this does not happen, then it may be possible to access
    /// uninitialized memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::ptr;
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut buffer = Buffer::with_capacity_in(&arena, 5);
    ///
    /// let data = &[1, 2, 3, 4, 5];
    ///
    /// unsafe {
    ///     // Copy some data into the uninitialized part of the `Buffer`
    ///     ptr::copy_nonoverlapping(data.as_ptr(), buffer.as_mut_ptr(), 5);
    ///
    ///     // Then call `set_len()` to initialize the `Buffer` with the newly copied data.
    ///     buffer.set_len(5);
    /// }
    ///
    /// assert_eq!(&buffer, &[1, 2, 3, 4, 5]);
    /// ```
    #[inline]
    pub const unsafe fn set_len(&mut self, new_len: usize) {
        unsafe {
            assert_unchecked(new_len <= self.capacity());
        }
        self.len = new_len;
    }

    #[inline]
    pub const unsafe fn set_capacity(&mut self, new_capacity: usize) {
        unsafe {
            Handle::set_len(&mut self.handle, new_capacity);
        }
    }

    /// Removes all elements from the `Buffer`.
    ///
    /// The elements are dropped in order. If a panic occurs while
    /// dropping the elements, then the undropped elements will be leaked.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut buffer = Buffer::new_in(&arena, [1, 2, 3, 4 ,5]);
    ///
    /// assert_eq!(&buffer, &[1, 2, 3, 4, 5]);
    ///
    /// buffer.clear();
    ///
    /// assert_eq!(&buffer, &[]);
    /// ```
    #[track_caller]
    #[inline]
    pub fn clear(&mut self) {
        unsafe {
            // Pre-set the length to `0` so that contents are inaccessible
            // if there is a `panic!()` while dropping.
            let old_len = self.len;
            self.set_len(0);
            self.drop_initialized_contents(..old_len);
        }
    }

    /// Truncates the `Buffer` so that it has a length of `new_len`.
    ///
    /// Removed elements are dropped.
    ///
    /// If the given `new_len` is larger than the current length, this method does
    /// nothing.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut buffer = Buffer::new_in(&arena, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    ///
    /// buffer.truncate(11);
    /// assert_eq!(&buffer, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    ///
    /// buffer.truncate(3);
    /// assert_eq!(&buffer, &[1, 2, 3]);
    /// ```
    #[track_caller]
    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        let old_len = self.len();
        if new_len < old_len {
            unsafe {
                self.set_len(new_len);
                self.drop_initialized_contents(new_len..old_len);
            }
        }
    }

    #[inline(always)]
    unsafe fn drop_initialized_contents<S>(&mut self, range: S)
    where
        S: SliceIndex<[MaybeUninit<T>], Output = [MaybeUninit<T>]>,
    {
        let contents = unsafe { self.handle.as_mut().get_unchecked_mut(range) };
        for item in contents {
            unsafe {
                MaybeUninit::assume_init_drop(item);
            }
        }
    }

    /// Shift the elements from `idx` to the end of the `Buffer` down to the start
    /// of the `Buffer`, overwriting the elements from `0` up to `idx`.
    ///
    /// No contained elements are dropped in this function.
    ///
    /// # Safety
    ///
    /// `idx` must be in range of the `Buffer`.
    #[inline]
    unsafe fn shift_down(&mut self, idx: usize) {
        unsafe {
            let ptr = self.as_mut_ptr().add(idx);
            let count = self.len().unchecked_sub(idx + 1);
            ptr::copy(ptr.add(1), ptr, count);
        }
    }

    #[must_use]
    #[inline]
    unsafe fn remove_unchecked(&mut self, idx: usize) -> T {
        let value = unsafe { ptr::read(self.as_ptr().add(idx)) };
        let len = self.len();
        unsafe {
            self.shift_down(idx);
            self.set_len(len - 1);
        }

        value
    }

    #[inline]
    pub(super) const unsafe fn push_unchecked(&mut self, value: T) {
        // @SAFETY: Invariant upheld by caller
        unsafe {
            assert_unchecked(!self.is_full());
        }

        unsafe {
            Handle::as_nonnull(&self.handle)
                .as_mut()
                .get_unchecked_mut(self.len)
                .write(value);
        }

        self.len += 1;
    }
}

impl<'a, T: Clone> Buffer<'a, T> {
    /// Resize the `Buffer` to the given `new_len`.
    ///
    /// If `new_len` is smaller than the current length, elements are dropped.
    /// If `new_len` is larger than the current length, then the buffer is
    /// extended by cloning `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut buffer = Buffer::new_in(&arena, [1, 2, 3, 4]);
    /// buffer.resize(2, 0);
    ///
    /// assert_eq!(&buffer, &[1, 2]);
    ///
    /// buffer.resize(4, 8);
    /// assert_eq!(&buffer, &[1, 2, 8, 8]);
    /// ```
    #[track_caller]
    #[inline]
    pub fn resize(&mut self, new_len: usize, value: T) {
        if new_len < self.len {
            self.truncate(new_len);
        } else {
            let spare_cap = self.capacity() - self.len;
            let n = usize_min(new_len - self.len(), spare_cap);

            if n > 0 {
                for _ in 0..n - 1 {
                    self.push(value.clone());
                }

                self.push(value);
            }
        }
    }

    #[track_caller]
    #[inline]
    pub fn extend_from_slice(&mut self, slice: &[T]) {
        let num_elems = self.capacity() - slice.len().max(self.len);
        for item in slice.iter().take(num_elems).cloned() {
            unsafe {
                self.push_unchecked(item);
            }
        }
    }
}

impl<'a, T: Copy> Buffer<'a, T> {
    /// Create a new `Buffer` containing the contents of `slice`.
    ///
    /// # Panics
    ///
    /// This method will panic if there is not enough space in the `Arena`
    /// to store the contents of `slice`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    ///
    /// let buffer = Buffer::new_slice_copied_in(&arena, &[1, 2, 3, 4]);
    /// assert_eq!(&buffer, &[1, 2, 3, 4]);
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_slice_copied_in<A: Allocator>(arena: &'a Arena<A>, slice: &'_ [T]) -> Self {
        let mut buf = Buffer::with_capacity_in(arena, slice.len());
        buf.extend_from_slice_copy(slice);
        buf
    }

    /// Extend the `Buffer` with the contents of `slice`.
    ///
    /// If there isn't enough space in the `Buffer` to store the `slice`, then
    /// it will only copy enough values to fill the spare capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut buffer = Buffer::with_capacity_in(&arena, 5);
    ///
    /// buffer.extend_from_slice_copy(&[1, 2, 3]);
    /// assert_eq!(&buffer, &[1, 2, 3]);
    ///
    /// buffer.extend_from_slice_copy(&[4, 5, 6]);
    /// assert_eq!(&buffer, &[1, 2, 3, 4, 5]);
    /// ```
    #[inline]
    pub const fn extend_from_slice_copy(&mut self, slice: &[T]) {
        let spare_cap = self.capacity() - self.len;
        let count = usize_min(spare_cap, slice.len());

        unsafe {
            ptr::copy_nonoverlapping(
                slice.as_ptr(),
                Handle::as_mut_ptr(&mut self.handle)
                    .cast::<T>()
                    .add(self.len),
                count,
            );
            self.set_len(self.len + count);
        }
    }
}

impl<'a, T: fmt::Debug> fmt::Debug for Buffer<'a, T> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.as_slice(), fmtr)
    }
}

impl<'a, T> Default for Buffer<'a, T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T: PartialEq> PartialEq for Buffer<'a, T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl<'a, T: PartialEq> PartialEq<[T]> for Buffer<'a, T> {
    #[inline]
    fn eq(&self, other: &[T]) -> bool {
        self.as_slice().eq(other)
    }
}

impl<'a, T: PartialEq, const N: usize> PartialEq<[T; N]> for Buffer<'a, T> {
    #[inline]
    fn eq(&self, other: &[T; N]) -> bool {
        PartialEq::eq(self, &other[..])
    }
}

impl<'a, 's, T: PartialEq> PartialEq<&'s [T]> for Buffer<'a, T> {
    #[inline]
    fn eq(&self, other: &&'s [T]) -> bool {
        self.as_slice().eq(*other)
    }
}

impl<'a, 's, T: PartialEq, const N: usize> PartialEq<&'s [T; N]> for Buffer<'a, T> {
    #[inline]
    fn eq(&self, other: &&'s [T; N]) -> bool {
        PartialEq::eq(self, &other[..])
    }
}

impl<'a, 's, T: PartialEq> PartialEq<&'s mut [T]> for Buffer<'a, T> {
    #[inline]
    fn eq(&self, other: &&'s mut [T]) -> bool {
        self.as_slice().eq(*other)
    }
}

impl<'a, 's, T: PartialEq, const N: usize> PartialEq<&'s mut [T; N]> for Buffer<'a, T> {
    #[inline]
    fn eq(&self, other: &&'s mut [T; N]) -> bool {
        PartialEq::eq(self, &other[..])
    }
}

impl<'a, T: PartialEq> PartialEq<Handle<'_, [T]>> for Buffer<'a, T> {
    #[inline]
    fn eq(&self, other: &Handle<'_, [T]>) -> bool {
        self.as_slice().eq(other.as_ref())
    }
}

impl<'a, T: Eq> Eq for Buffer<'a, T> {}

impl<'a, T: Hash> Hash for Buffer<'a, T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state)
    }
}

impl<'a, T> AsRef<[T]> for Buffer<'a, T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T> AsMut<[T]> for Buffer<'a, T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<'a, T> Borrow<[T]> for Buffer<'a, T> {
    #[inline]
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T> BorrowMut<[T]> for Buffer<'a, T> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<'a, T> Deref for Buffer<'a, T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a, T> DerefMut for Buffer<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<'a, T, I: SliceIndex<[T]>> Index<I> for Buffer<'a, T> {
    type Output = <[T] as Index<I>>::Output;
    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<'a, T, I: SliceIndex<[T]>> IndexMut<I> for Buffer<'a, T> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}

impl<'b: 'a, 'a, T> IntoIterator for &'b Buffer<'a, T> {
    type IntoIter = <&'b [T] as IntoIterator>::IntoIter;
    type Item = &'a T;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'b: 'a, 'a, T> IntoIterator for &'b mut Buffer<'a, T> {
    type IntoIter = <&'b mut [T] as IntoIterator>::IntoIter;
    type Item = &'a mut T;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

impl<'a, T> IntoIterator for Buffer<'a, T> {
    type IntoIter = IntoIter<'a, T>;
    type Item = T;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

impl<'a, T> Extend<T> for Buffer<'a, T> {
    #[track_caller]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.try_extend(iter).unwrap_or_else(|e| panic!("{}", e));
    }
}

impl<'a, T> From<Handle<'a, [T]>> for Buffer<'a, T> {
    #[inline]
    fn from(value: Handle<'a, [T]>) -> Self {
        Handle::into_buffer(value)
    }
}

#[cfg(feature = "std")]
impl<'a> Write for Buffer<'a, u8> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let space = self.capacity() - self.len();
        self.extend(buf.into_iter().take(space).copied());
        Ok(space)
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(feature = "std")]
impl<'a> Read for Buffer<'a, u8> {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.as_slice().read(buf)
    }
}

impl<'a, T> Drop for Buffer<'a, T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.drop_initialized_contents(..self.len);
        }
    }
}

/// Used to access a variable-sized `Buffer` allocation in an `Arena`.
///
/// This type is used in the [`Buffer::with_growable`] and [`Buffer::try_with_growable`]
/// methods.
///
/// [`Buffer::with_growable`]: ./struct.Buffer.html#method.with_growable
/// [`Buffer::try_with_growable`]: ./struct.Buffer.html#method.try_with_growable
pub struct GrowableBuffer<'a, T, A: Allocator> {
    backing_storage: NonNull<[MaybeUninit<T>]>,
    len: usize,
    cap: usize,
    _boo: InvariantLifetime<'a, Arena<A>>,
}

impl<'a, T, A: Allocator> GrowableBuffer<'a, T, A> {
    /// Returns the maximum capacity available in the `GrowableBuffer`.
    ///
    /// This is the total spare capacity available in the current block.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::with_block_size(400);
    ///
    /// let data = arena.alloc_ref::<[u8; 4]>([0; 4]);
    ///
    /// let buffer: Buffer<'_, u8> = Buffer::with_growable_in(&arena, |buf| {
    ///     assert_eq!(buf.max_capacity(), 396);
    ///     # buf.push(0u8);
    /// });
    /// # let _ = (data, buffer);
    /// ```
    #[must_use]
    #[inline]
    pub const fn max_capacity(&self) -> usize {
        self.backing_storage.len()
    }

    /// Returns the current capacity availble in the `GrowableBuffer`.
    ///
    /// When the outer `Buffer` is initialized, it will have `capacity()` space
    /// reserved for it in the `Arena`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::with_block_size(400);
    ///
    /// let buffer: Buffer<'_, u8> = Buffer::with_growable_in(&arena, |buf| {
    ///     assert_eq!(buf.capacity(), 0);
    ///     buf.reserve(21);
    ///     assert_eq!(buf.capacity(), 21);
    /// });
    ///
    /// assert_eq!(buffer.capacity(), 21);
    /// ```
    #[must_use]
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.cap
    }

    /// Returns `true` if the current [`len()`] is equal to [`max_capacity()`].
    ///
    /// If this is `true`, it means that no additional space can be reserved in the
    /// `GrowableBuffer`, and no additional values can be pushed into it.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::with_block_size(400);
    ///
    /// let buffer: Buffer<'_, u8> = Buffer::with_growable_in(&arena, |buf| {
    ///     assert!(!buf.is_full());
    ///     buf.extend_from_slice(&[0u8; 400]);
    ///     assert_eq!(buf, &[0; 400]);
    ///     assert!(buf.is_full());
    /// });
    /// # let _ = buffer;
    /// ```
    ///
    /// [`len()`]: ./struct.GrowableBuffer.html#method.len
    /// [`max_capacity()`]: ./struct.GrowableBuffer.html#method.max_capacity
    #[inline]
    pub const fn is_full(&self) -> bool {
        self.len == self.max_capacity()
    }

    /// Checks whether the `GrowableBuffer` has the capacity for an additional `required_capacity`
    /// items.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::with_block_size(18);
    ///
    /// let buffer: Buffer<'_, u64> = Buffer::with_growable_in(&arena, |buf| {
    ///     buf.reserve(2);
    ///     assert!(buf.has_capacity(2));
    ///     buf.push(21);
    ///     assert!(!buf.has_capacity(2));
    /// });
    /// # assert_eq!(&buffer, &[21]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn has_capacity(&self, required_capacity: usize) -> bool {
        let cap = self.capacity();
        cap - self.len >= required_capacity
    }

    /// Attempt to reserve the additionl bytes in the buffer, returning an error
    /// when the arena capacity is exhausted.
    ///
    /// This method will try to eagerly reserve as much as possible. It is possible
    /// to query the `TryReserveError` to see how much space is not available
    /// in the `Buffer`.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::mem;
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::with_block_size(8 * mem::size_of::<i32>());
    /// ```
    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let cap = self.capacity();

        let (new_cap, max_cap) = (cap + additional, self.max_capacity());
        if new_cap > max_cap {
            self.cap = max_cap;
            Err(TryReserveError {
                requested: additional,
                available: new_cap - max_cap,
            })
        } else {
            unsafe {
                self.set_capacity(new_cap);
            }

            Ok(())
        }
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        match self.try_reserve(additional) {
            Ok(_) => (),
            Err(_) => panic!("Could not reserve additional capacity in buffer",),
        }
    }

    #[inline]
    pub fn try_push(&mut self, value: T) -> Result<(), T> {
        if self.is_full() {
            return Err(value);
        }

        // Because this method doesn't need to hit the system memory allocator,
        // it don't need to grow by a growth factor.
        match self.ensure_capacity(1) {
            Ok(_) => (),
            Err(_) => return Err(value),
        }

        unsafe {
            let dst = self.backing_storage.cast::<T>().add(self.len).as_ptr();
            ptr::write(dst, value);
        }

        self.len += 1;

        Ok(())
    }

    /// Append the given `value` to the end of the `GrowableBuffer`.
    ///
    /// # Panics
    ///
    /// This method will panic if there is no space in the `GrowableBuffer`
    /// for the `value`, and no additional capacity can be reserved.
    ///
    /// # Examples
    /// 
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::with_block_size(18);
    ///
    /// let buffer = Buffer::with_growable_in(&arena, |buf| {
    ///     buf.push(99);
    ///     assert_eq!(buf, &[99]);
    /// });
    /// # assert_eq!(&buffer, &[99]);
    /// ```
    #[track_caller]
    #[inline]
    pub fn push(&mut self, value: T) {
        match self.try_push(value) {
            Ok(_) => (),
            Err(_) => panic!("No space for value in this buffer"),
        }
    }

    /// Remove the last element of the `GrowableBuffer` and return it.
    ///
    /// If the `GrowableBuffer` is empty, this method returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::with_block_size(18);
    ///
    /// let buffer = Buffer::with_growable_in(&arena, |buf| {
    ///     buf.extend([1, 2]);
    ///     assert_eq!(buf.pop().unwrap(), 2);
    ///     assert_eq!(buf.pop().unwrap(), 1);
    ///     assert_eq!(buf.pop(), None);
    /// });
    /// # assert_eq!(&buffer, &[]);
    /// ```
    #[must_use]
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        self.len -= 1;
        let value = unsafe { ptr::read(self.backing_storage.cast::<T>().add(self.len).as_ptr()) };

        Some(value)
    }

    #[inline]
    pub const unsafe fn set_len(&mut self, new_len: usize) {
        self.len = new_len;
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        unsafe {
            self.set_capacity(self.len());
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        unsafe {
            ptr::drop_in_place(self.as_mut_slice());
            self.set_len(0);
        }
    }

    #[must_use]
    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        let data = self.backing_storage.cast::<T>().as_ptr();
        unsafe { slice::from_raw_parts(data, self.len) }
    }

    #[must_use]
    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        let data = self.backing_storage.cast::<T>().as_ptr();
        unsafe { slice::from_raw_parts_mut(data, self.len) }
    }

    #[must_use]
    #[inline]
    pub const fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        unsafe {
            let data = self
                .backing_storage
                .cast::<MaybeUninit<T>>()
                .add(self.len)
                .as_ptr();
            slice::from_raw_parts_mut(data, self.cap - self.len)
        }
    }

    #[must_use]
    #[inline]
    pub fn iter(&'_ self) -> <&'_ [T] as IntoIterator>::IntoIter {
        self.as_slice().iter()
    }

    #[must_use]
    #[inline]
    pub fn iter_mut(&'_ mut self) -> <&'_ mut [T] as IntoIterator>::IntoIter {
        self.as_mut_slice().iter_mut()
    }

    #[inline]
    fn ensure_capacity(&mut self, required_capacity: usize) -> Result<(), TryReserveError> {
        if self.has_capacity(required_capacity) {
            return Ok(());
        }
        self.try_reserve(required_capacity)
    }

    #[inline]
    unsafe fn set_capacity(&mut self, new_cap: usize) {
        self.cap = new_cap;
    }

    #[inline]
    fn into_buffer(self) -> Buffer<'a, T> {
        let data = self.backing_storage.as_ptr().cast::<MaybeUninit<T>>();
        let handle = unsafe { Handle::from_raw_parts(data, self.cap) };

        let len = self.len;

        mem::forget(self);

        unsafe { Buffer::from_raw_parts(handle, len) }
    }
}

impl<'a, T: Copy, A: Allocator> GrowableBuffer<'a, T, A> {
    #[track_caller]
    pub fn extend_from_slice_copy(&mut self, slice: &[T]) {
        let len = slice.len();

        self.ensure_capacity(len).expect("could not reserve");

        let slice = unsafe {
            let data = slice.as_ptr().cast::<MaybeUninit<T>>();
            slice::from_raw_parts(data, len)
        };

        self.spare_capacity_mut()[..len].copy_from_slice(slice);

        unsafe {
            self.set_len(self.len() + len);
        }
    }
}

impl<'a, T: Clone, A: Allocator> GrowableBuffer<'a, T, A> {
    #[track_caller]
    pub fn extend_from_slice(&mut self, slice: &[T]) {
        let len = slice.len();

        self.ensure_capacity(len).expect("could not reserve");

        for item in slice {
            self.push(item.clone());
        }
    }
}

impl<'a, T, A: Allocator> Deref for GrowableBuffer<'a, T, A> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a, T, A: Allocator> DerefMut for GrowableBuffer<'a, T, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<'a, T, A: Allocator, Idx: SliceIndex<[T]>> Index<Idx> for GrowableBuffer<'a, T, A> {
    type Output = <[T] as Index<Idx>>::Output;
    #[inline]
    fn index(&self, index: Idx) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<'a, T, A: Allocator, Idx: SliceIndex<[T]>> IndexMut<Idx> for GrowableBuffer<'a, T, A> {
    #[inline]
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}

impl<'a, T, A: Allocator> Borrow<[T]> for GrowableBuffer<'a, T, A> {
    #[inline]
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T, A: Allocator> BorrowMut<[T]> for GrowableBuffer<'a, T, A> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<'a, T, A: Allocator> AsRef<[T]> for GrowableBuffer<'a, T, A> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T, A: Allocator> AsMut<[T]> for GrowableBuffer<'a, T, A> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<'s, 'a, T, A: Allocator> IntoIterator for &'s GrowableBuffer<'a, T, A> {
    type Item = &'s T;
    type IntoIter = <&'s [T] as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'s, 'a, T, A: Allocator> IntoIterator for &'s mut GrowableBuffer<'a, T, A> {
    type Item = &'s mut T;
    type IntoIter = <&'s mut [T] as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'a, T: fmt::Debug, A: Allocator> fmt::Debug for GrowableBuffer<'a, T, A> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmtr.debug_list().entries(self).finish()
    }
}

impl<'a, T, A: Allocator> Extend<T> for GrowableBuffer<'a, T, A> {
    #[track_caller]
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();

        let additional = {
            let hint = iter.size_hint();
            hint.1.unwrap_or(hint.0)
        };

        if self.ensure_capacity(additional).is_err() {
            panic!(
                "Could not reserve space for {} additional items when extending buffer",
                additional
            );
        }

        for item in iter {
            self.push(item);
        }
    }
}

impl<'a, T: PartialEq, A: Allocator> PartialEq for GrowableBuffer<'a, T, A> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(self.as_slice(), other.as_slice())
    }
}

impl<'a, T: PartialEq, A: Allocator> PartialEq<[T]> for GrowableBuffer<'a, T, A> {
    #[inline]
    fn eq(&self, other: &[T]) -> bool {
        PartialEq::eq(self.as_slice(), other)
    }
}

impl<'a, 's, T: PartialEq, A: Allocator> PartialEq<&'s [T]> for GrowableBuffer<'a, T, A> {
    #[inline]
    fn eq(&self, other: &&'s [T]) -> bool {
        PartialEq::eq(self.as_slice(), *other)
    }
}

impl<'a, 's, T: PartialEq, A: Allocator> PartialEq<&'s mut [T]> for GrowableBuffer<'a, T, A> {
    #[inline]
    fn eq(&self, other: &&'s mut [T]) -> bool {
        PartialEq::eq(self.as_slice(), *other)
    }
}

impl<'a, T: PartialEq, A: Allocator, const N: usize> PartialEq<[T; N]>
    for GrowableBuffer<'a, T, A>
{
    #[inline]
    fn eq(&self, other: &[T; N]) -> bool {
        PartialEq::eq(self.as_slice(), &other[..])
    }
}

impl<'a, 's, T: PartialEq, A: Allocator, const N: usize> PartialEq<&'s [T; N]>
    for GrowableBuffer<'a, T, A>
{
    #[inline]
    fn eq(&self, other: &&'s [T; N]) -> bool {
        PartialEq::eq(self.as_slice(), *other)
    }
}

impl<'a, 's, T: PartialEq, A: Allocator, const N: usize> PartialEq<&'s mut [T; N]>
    for GrowableBuffer<'a, T, A>
{
    #[inline]
    fn eq(&self, other: &&'s mut [T; N]) -> bool {
        PartialEq::eq(self.as_slice(), *other)
    }
}

impl<'a, T, A: Allocator> Drop for GrowableBuffer<'a, T, A> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            ptr::drop_in_place(self.as_mut_slice());
        }
    }
}

#[cfg(feature = "std")]
impl<'a, A: Allocator> Write for GrowableBuffer<'a, u8, A> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let len = buf.len();
        let space = match self.try_reserve(len).map_err(|e| e.available) {
            Ok(_) => len,
            Err(space) => space,
        };

        self.extend(buf.into_iter().take(space).copied());
        Ok(space)
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(feature = "std")]
impl<'a, A: Allocator> Read for GrowableBuffer<'a, u8, A> {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.as_slice().read(buf)
    }
}

#[derive(Debug)]
pub struct TryReserveError {
    requested: usize,
    available: usize,
}

impl fmt::Display for TryReserveError {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            fmtr,
            "could not write {} bytes to buffer, only {} available",
            self.requested, self.available
        )
    }
}

impl error::Error for TryReserveError {}

pub struct TryExtendError<I: IntoIterator> {
    curr: I::Item,
    rest: Fuse<I::IntoIter>,
}

impl<I: IntoIterator> TryExtendError<I> {
    #[must_use]
    #[inline]
    pub fn into_inner(self) -> (I::Item, Fuse<I::IntoIter>) {
        let TryExtendError { curr, rest } = self;
        (curr, rest)
    }
}

impl<I: IntoIterator> fmt::Debug for TryExtendError<I> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmtr.debug_struct("TryExtendError").finish_non_exhaustive()
    }
}

impl<I: IntoIterator> fmt::Display for TryExtendError<I> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmtr.write_str("overflow while extending buffer")
    }
}

impl<I: IntoIterator> Error for TryExtendError<I> {}

/// A by-value iterator over a [`Buffer`].
///
/// This type is returned by [`Buffer::into_iter`].
///
/// [`Buffer`]: ./struct.Buffer.html
/// [`Buffer::into_iter`]: ./struct.Buffer.html#method.into_iter
pub struct IntoIter<'a, T> {
    data: Handle<'a, [MaybeUninit<T>]>,
    front_idx: usize,
    back_idx: usize,
}

impl<'a, T> IntoIter<'a, T> {
    /// Access the elements yet to be iterated through an immutable slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    /// let buffer = Buffer::new_in(&arena, [1, 2, 3, 4, 5]);
    ///
    /// let mut iter = buffer.into_iter();
    ///
    /// let _ = iter.next().unwrap();
    /// let _ = iter.next_back().unwrap();
    ///
    /// assert_eq!(iter.as_slice(), &[2, 3, 4]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.data_start(), self.len_const()) }
    }

    /// Access the elements yet to be iterated through a mutable slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    /// let buffer = Buffer::new_in(&arena, [1, 2, 3, 4, 5]);
    ///
    /// let mut iter = buffer.into_iter();
    ///
    /// iter.as_mut_slice().fill(255);
    ///
    /// for item in iter {
    ///     assert_eq!(item, 255);
    /// }
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.data_start_mut(), self.len_const()) }
    }

    /// Consumes the iterator, returning a `Buffer` containing the unyielded values.
    ///
    /// The returned `Buffer` will have the same capacity as the original `Buffer`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    /// let buffer = Buffer::new_in(&arena, [0, 1, 2, 3, 4]);
    ///
    /// let mut iter = buffer.into_iter();
    ///
    /// let _ = iter.next().unwrap();
    /// let _ = iter.next().unwrap();
    ///
    /// let buffer = iter.into_buffer();
    /// assert_eq!(&buffer, &[2, 3, 4]);
    /// assert_eq!(buffer.capacity(), 5);
    /// ```
    #[must_use]
    #[inline]
    pub const fn into_buffer(self) -> Buffer<'a, T> {
        let cap = Handle::<'_, [MaybeUninit<T>]>::as_ptr(&self.data).len();
        let new_len = self.len_const();
        let start = self.front_idx;

        let mut buf = unsafe {
            let handle = ptr::read(&self.data);
            mem::forget(self);
            Buffer::from_raw_parts(handle, cap)
        };

        let ptr = Handle::as_mut_ptr(&mut buf.handle) as *mut T;

        unsafe {
            ptr::copy(ptr.add(start), ptr, new_len);
            buf.set_len(new_len);
        }

        buf
    }

    /// Consumes the iterator, returning a `Handle<[T]>` containing the unyielded values.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, buffer::Buffer};
    ///
    /// let arena = Arena::new();
    /// let buffer = Buffer::new_in(&arena, [1024usize; 300]);
    ///
    /// let mut iter = buffer.into_iter();
    ///
    /// for item in (&mut iter).take(150) {
    ///     let _ = item;
    /// }
    ///
    /// let slice_handle = iter.into_slice_handle();
    /// assert_eq!(slice_handle.as_ref(), &[1024; 150]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn into_slice_handle(self) -> Handle<'a, [T]> {
        self.into_buffer().into_slice_handle()
    }

    #[must_use]
    #[inline]
    const fn new(buffer: Buffer<'a, T>) -> Self {
        let len = buffer.as_slice().len();
        let data = Handle::transpose_into_uninit(buffer.into_slice_handle());

        Self {
            data,
            front_idx: 0,
            back_idx: len,
        }
    }

    #[must_use]
    #[inline]
    const fn data_start(&self) -> *const T {
        unsafe { Handle::as_ptr(&self.data).cast::<T>().add(self.front_idx) }
    }

    #[must_use]
    #[inline]
    const fn data_start_mut(&mut self) -> *mut T {
        unsafe {
            Handle::as_mut_ptr(&mut self.data)
                .cast::<T>()
                .add(self.front_idx)
        }
    }

    #[must_use]
    #[inline]
    const fn len_const(&self) -> usize {
        self.back_idx - self.front_idx
    }

    #[must_use]
    #[inline]
    unsafe fn read_at_unchecked(&self, idx: usize) -> &MaybeUninit<T> {
        unsafe { self.data.as_ref().get_unchecked(idx) }
    }
}

impl<'a, T: fmt::Debug> fmt::Debug for IntoIter<'a, T> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct Ellipsis;

        impl fmt::Debug for Ellipsis {
            #[inline]
            fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmtr.write_str("…")
            }
        }

        let mut debug_list = fmtr.debug_list();

        if self.front_idx != 0 {
            debug_list.entry(&Ellipsis);
        }

        debug_list.entries(self.as_slice());

        if self.back_idx != self.data.len() {
            debug_list.entry(&Ellipsis);
        }

        debug_list.finish()
    }
}

impl<'a, T> Iterator for IntoIter<'a, T> {
    type Item = T;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front_idx >= self.back_idx {
            return None;
        }

        let item = unsafe { self.read_at_unchecked(self.front_idx).assume_init_read() };

        self.front_idx += 1;
        Some(item)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = ExactSizeIterator::len(self);
        (len, Some(len))
    }
}

impl<'a, T> DoubleEndedIterator for IntoIter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front_idx >= self.back_idx {
            return None;
        }

        let item = unsafe {
            self.read_at_unchecked(self.back_idx.saturating_sub(1))
                .assume_init_read()
        };

        self.back_idx -= 1;
        Some(item)
    }
}

impl<'a, T> ExactSizeIterator for IntoIter<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        self.len_const()
    }
}

impl<'a, T> FusedIterator for IntoIter<'a, T> {}

impl<'a, T> Drop for IntoIter<'a, T> {
    #[inline]
    fn drop(&mut self) {
        let (front, back) = (self.front_idx, self.back_idx.saturating_sub(1));
        if front >= back {
            return;
        }

        let slice = &mut self.as_mut_slice()[front..back];
        unsafe {
            ptr::drop_in_place(slice);
        }
    }
}

#[inline(always)]
const fn usize_min(n1: usize, n2: usize) -> usize {
    if n1 < n2 { n1 } else { n2 }
}
