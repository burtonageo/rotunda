#![allow(missing_docs, clippy::missing_safety_doc)]

//! A contiguous, growable array of values allocated in an `Arena`.

use crate::{Arena, handle::Handle};
use alloc::alloc::{Allocator, Layout};
use core::{
    borrow::{Borrow, BorrowMut},
    error::{self, Error},
    fmt,
    hash::{Hash, Hasher},
    hint::assert_unchecked,
    iter::FusedIterator,
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
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_in<I: IntoIterator<Item = T>, A: Allocator>(arena: &'a Arena<A>, iter: I) -> Self
    where
        I::IntoIter: ExactSizeIterator,
    {
        #[derive(Debug)]
        enum Never {}

        let result = unsafe {
            Buffer::growable_impl::<A, Never, _>(arena, None, |mut buffer| {
                buffer.extend(iter);
                Ok(buffer.into())
            })
        };

        result.expect("could not allocate buffer")
    }

    #[track_caller]
    #[must_use]
    #[inline]
    pub fn with_capacity_in<A: Allocator>(arena: &'a Arena<A>, capacity: usize) -> Self {
        let handle = Handle::new_slice_uninit_in(arena, capacity);
        unsafe { Self::from_raw_parts(handle, 0) }
    }

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
    pub fn try_with_growable<A, E, F>(
        arena: &'a Arena<A>,
        f: F,
    ) -> Result<Self, WithGrowableError<E>>
    where
        A: Allocator,
        F: 'static + FnOnce(GrowableBuffer<'a, T, A>) -> Result<Buffer<'a, T>, E>,
    {
        Self::try_with_growable_guaranteeing_capacity(arena, 0, f)
    }

    #[track_caller]
    #[inline]
    pub fn with_growable<A, F>(arena: &'a Arena<A>, f: F) -> Self
    where
        A: Allocator,
        F: 'static + FnOnce(GrowableBuffer<'a, T, A>) -> Buffer<'a, T>,
    {
        Self::with_growable_guaranteeing_capacity(arena, 0, f)
    }

    #[inline]
    pub fn try_with_growable_guaranteeing_capacity<A, E, F>(
        arena: &'a Arena<A>,
        required_capacity: usize,
        f: F,
    ) -> Result<Self, WithGrowableError<E>>
    where
        A: Allocator,
        F: 'static + FnOnce(GrowableBuffer<'a, T, A>) -> Result<Buffer<'a, T>, E>,
    {
        unsafe { Self::growable_impl::<A, E, _>(arena, Some(required_capacity), f) }
    }

    #[track_caller]
    #[must_use]
    #[inline]
    pub fn with_growable_guaranteeing_capacity<A, F>(
        arena: &'a Arena<A>,
        required_capacity: usize,
        f: F,
    ) -> Self
    where
        A: Allocator,
        F: 'static + FnOnce(GrowableBuffer<'a, T, A>) -> Buffer<'a, T>,
    {
        enum Never {}
        let result = unsafe {
            Self::growable_impl::<A, Never, _>(arena, Some(required_capacity), |buffer| {
                Ok(f(buffer))
            })
        };

        match result {
            Ok(buffer) => buffer,
            Err(WithGrowableError::CapacityFail) => {
                panic!("Required capacity not available in this arena")
            }
        }
    }

    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        unsafe { Self::from_raw_parts(Handle::empty(), 0) }
    }

    #[must_use]
    #[inline]
    pub const unsafe fn from_raw_parts(handle: Handle<'a, [MaybeUninit<T>]>, len: usize) -> Self {
        Self {
            handle,
            len,
            _boo: PhantomData,
        }
    }

    #[must_use]
    #[inline]
    pub const fn into_raw_parts(self) -> (Handle<'a, [MaybeUninit<T>]>, usize) {
        let handle = unsafe { mem::transmute_copy(&self.handle) };
        let len = self.len;
        let _this = ManuallyDrop::new(self);

        (handle, len)
    }

    #[must_use]
    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        let ptr = Handle::as_ptr(&self.handle);
        unsafe { slice::from_raw_parts(ptr as *const T, self.len) }
    }

    #[must_use]
    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        let ptr = Handle::as_mut_ptr(&mut self.handle);
        unsafe { slice::from_raw_parts_mut(ptr as *mut T, self.len) }
    }

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

    #[inline]
    pub fn try_extend<I: IntoIterator<Item = T>>(
        &mut self,
        iter: I,
    ) -> Result<(), TryExtendError<I>> {
        let mut iter = iter.into_iter();
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

    #[inline]
    pub fn try_remove(&mut self, idx: usize) -> Option<T> {
        let len = self.len();
        if len > 0 && idx < len {
            unsafe { Some(self.remove_unchecked(idx)) }
        } else {
            None
        }
    }

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
    pub fn swap_remove(&mut self, idx: usize) -> Option<T> {
        match self.len.checked_sub(1) {
            Some(len) if idx < self.len() => {
                self.as_mut_slice().swap(idx, len);
                self.pop()
            }
            _ => None,
        }
    }

    #[must_use]
    #[inline]
    pub const fn into_slice_handle(mut self) -> Handle<'a, [T]> {
        let ptr = Handle::as_mut_ptr(&mut self.handle);
        let ptr = ptr::slice_from_raw_parts_mut(ptr as *const T as *mut T, self.len);
        let _this = ManuallyDrop::new(self);

        unsafe { Handle::from_raw(ptr) }
    }

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

    #[inline]
    #[must_use]
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        unsafe { self.handle.as_mut().get_unchecked_mut(self.len..) }
    }

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

    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        Handle::as_ptr(&self.handle).len()
    }

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

    #[track_caller]
    #[inline]
    pub fn clear(&mut self) {
        unsafe {
            // Pre-set the length to `0` so that contents are inaccessible
            // if there is a `panic!()` while dropping.
            self.set_len(0);
            self.drop_initialized_contents(..);
        }
    }

    #[track_caller]
    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        if new_len < self.len() {
            unsafe {
                self.set_len(new_len);
                self.drop_initialized_contents(new_len..);
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

    #[inline]
    unsafe fn shift_down(&mut self, idx: usize) {
        unsafe {
            let ptr = self.as_mut_ptr().add(idx);
            let count = self.len().unchecked_sub(idx + 1);
            ptr::copy(ptr.add(1), ptr, count);
        }
    }

    #[track_caller]
    #[inline]
    unsafe fn growable_impl<A, E, F>(
        arena: &'a Arena<A>,
        required_min_capacity_hint: Option<usize>,
        f: F,
    ) -> Result<Self, WithGrowableError<E>>
    where
        A: Allocator,
        F: FnOnce(GrowableBuffer<'a, T, A>) -> Result<Buffer<'a, T>, E>,
    {
        if arena.block_size() < required_min_capacity_hint.unwrap_or(0) {
            return Err(WithGrowableError::CapacityFail);
        }

        let curr_block_cap = arena.curr_block_capacity();
        match curr_block_cap {
            None => arena.force_push_new_block(),
            Some(cap) if cap < required_min_capacity_hint.unwrap_or(usize::MAX) => {
                arena.force_push_new_block()
            }
            _ => (),
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

        let growable_buffer = GrowableBuffer {
            backing_storage,
            len: 0,
            cap: 0,
            _boo: PhantomData,
        };

        struct Unbump<'a, A: Allocator>(&'a Arena<A>, usize);
        impl<'a, A: Allocator> Drop for Unbump<'a, A> {
            fn drop(&mut self) {
                unsafe {
                    self.0.blocks.unbump(self.1);
                }
            }
        }

        let unbump = Unbump(arena, offset);
        let result = f(growable_buffer);

        if let Ok(ref buf) = result {
            mem::forget(unbump);
            unsafe {
                arena.blocks.bump(buf.handle.len() * mem::size_of::<T>());
            }
        }

        result.map_err(WithGrowableError::Inner)
    }
}

impl<'a, T: Clone> Buffer<'a, T> {
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
    #[must_use]
    #[track_caller]
    #[inline]
    pub fn new_slice_copied_in<A: Allocator>(arena: &'a Arena<A>, slice: &'_ [T]) -> Self {
        let mut buf = Buffer::with_capacity_in(arena, slice.len());
        buf.extend_from_slice_copy(slice);
        buf
    }

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

pub struct GrowableBuffer<'a, T, A: Allocator> {
    backing_storage: NonNull<[MaybeUninit<T>]>,
    cap: usize,
    len: usize,
    _boo: PhantomData<&'a Arena<A>>,
}

#[allow(dead_code, unused)]
impl<'a, T, A: Allocator> GrowableBuffer<'a, T, A> {
    #[must_use]
    #[inline]
    pub const fn max_capacity(&self) -> usize {
        self.backing_storage.len()
    }

    #[must_use]
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.cap
    }

    #[inline]
    pub const fn is_full(&self) -> bool {
        self.len == self.max_capacity()
    }

    #[must_use]
    #[inline]
    pub const fn has_capacity(&self, required_capacity: usize) -> bool {
        self.cap - self.len > required_capacity
    }

    #[inline]
    pub const fn reserve(&mut self, additional: usize) -> Result<(), usize> {
        let (new_cap, max_cap) = (self.cap + additional, self.max_capacity());
        if new_cap > max_cap {
            self.cap = max_cap;
            Err(new_cap - max_cap)
        } else {
            self.cap = new_cap;
            Ok(())
        }
    }

    #[inline]
    pub const fn try_push(&mut self, value: T) -> Result<(), T> {
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

    #[inline]
    pub fn push(&mut self, value: T) {
        match self.try_push(value) {
            Ok(_) => (),
            Err(_) => panic!("No space for value in this buffer"),
        }
    }

    #[must_use]
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
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
    pub fn clear(&mut self) {
        unsafe {
            ptr::drop_in_place(self.as_mut_slice());
            self.set_len(0);
        }
    }

    #[must_use]
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        let data = self.backing_storage.cast::<T>().as_ptr();
        unsafe { slice::from_raw_parts(data, self.len) }
    }

    #[must_use]
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let data = self.backing_storage.cast::<T>().as_ptr();
        unsafe { slice::from_raw_parts_mut(data, self.len) }
    }

    #[inline]
    pub fn into_buffer(self) -> Buffer<'a, T> {
        let data = self.backing_storage.as_ptr().cast::<MaybeUninit<T>>();
        let handle = unsafe { Handle::from_raw_parts(data, self.cap) };

        let len = self.len;

        mem::forget(self);

        unsafe { Buffer::from_raw_parts(handle, len) }
    }

    #[inline]
    pub fn iter(&'_ self) -> <&'_ [T] as IntoIterator>::IntoIter {
        self.as_slice().iter()
    }

    #[inline]
    pub fn iter_mut(&'_ mut self) -> <&'_ mut [T] as IntoIterator>::IntoIter {
        self.as_mut_slice().iter_mut()
    }

    #[inline]
    const fn ensure_capacity(&mut self, required_capacity: usize) -> Result<(), usize> {
        if self.has_capacity(required_capacity) {
            return Ok(());
        }
        self.reserve(required_capacity)
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

impl<'a, T, A: Allocator> From<GrowableBuffer<'a, T, A>> for Buffer<'a, T> {
    #[inline]
    fn from(value: GrowableBuffer<'a, T, A>) -> Self {
        value.into_buffer()
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

#[derive(Debug)]
pub enum WithGrowableError<E> {
    CapacityFail,
    Inner(E),
}

impl<E: fmt::Display> fmt::Display for WithGrowableError<E> {
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::CapacityFail => fmtr.write_str("could not create buffer with required capacity"),
            Self::Inner(ref e) => fmt::Display::fmt(e, fmtr),
        }
    }
}

impl<E: error::Error + 'static> error::Error for WithGrowableError<E> {
    #[inline]
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        if let Self::Inner(ref e) = *self {
            Some(e)
        } else {
            None
        }
    }
}

pub struct TryExtendError<I: IntoIterator> {
    curr: I::Item,
    rest: I::IntoIter,
}

impl<I: IntoIterator> TryExtendError<I> {
    #[must_use]
    #[inline]
    pub fn into_inner(self) -> (I::Item, I::IntoIter) {
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

pub struct IntoIter<'a, T> {
    data: Handle<'a, [MaybeUninit<T>]>,
    front_idx: usize,
    back_idx: usize,
}

impl<'a, T> IntoIter<'a, T> {
    #[must_use]
    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.data_start(), self.len_const()) }
    }

    #[must_use]
    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.data_start_mut(), self.len_const()) }
    }

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
