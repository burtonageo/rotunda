#![no_std]
#![cfg_attr(feature = "nightly_ptr_metadata", feature(ptr_metadata))]
#![cfg_attr(feature = "nightly_coerce_pointee", feature(derive_coerce_pointee))]
#![cfg_attr(feature = "nightly_can_vector", feature(can_vector))]
#![feature(alloc_layout_extra, allocator_api)]
#![warn(
    missing_docs,
    clippy::empty_line_after_doc_comments,
    clippy::missing_safety_doc
)]
#![deny(unsafe_attr_outside_unsafe, unsafe_op_in_unsafe_fn)]

//! This module contains types for using the arena allocation strategy. See the [`Arena`] type
//! for more information on how to use arena allocation.
//!
//! # Arena allocation
//!
//! In many workloads, the lifetime of a heap allocation is scoped to a particular function call.
//! For example:
//!
//! ```
//! # #[derive(Default)]
//! # struct SomeType {};
//! # #[derive(Default)]
//! # struct OtherType {};
//! # fn use_values(_: &SomeType, _: &OtherType) {}
//! fn some_function() {
//!     let value = Box::new(SomeType::default());
//!     let value_2 = Box::new(OtherType::default());
//!
//!     // Use `value` and `value_2` in the body of `some_function()`
//!     // ...
//!     use_values(&*value, &*value_2);
//!
//!     // `value` and `value_2` are dropped and deallocated here as the function returns.
//! }
//! ```
//!
//! If many values need to be allocated and deallocated for a function body, this can be
//! inefficient as the heap allocator needs to do work to allocate and deallocate each
//! individual value.
//!
//! To compensate for this, many strategies can be used so that values are allocated from
//! a single block of memory, reducing the work that the memory allocator needs to do.
//! One of these strategies is called arena allocation.
//!
//! The idea behind arena allocation is to preallocate a large block of memory, and then
//! write values into that block as necessary. Values cannot be freed until the entire block
//! is finished with, when every value is implicitly deallocated at once.
//!
//! # Examples
//!
//! To rewrite the above example `some_function()` using arena allocation provided by this crate,
//! an `Arena` would need to be constructed or passed in to the function. Any transient data allocated
//! by the function would then be allocated from the `Arena`:
//!
//! ```
//! use rotunda::{Arena, handle::Handle};
//!
//! # #[derive(Default)]
//! # struct SomeType {};
//! # #[derive(Default)]
//! # struct OtherType {};
//! # fn use_values(_: &SomeType, _: &OtherType) {}
//! fn some_function(arena: &Arena) {
//!     // Allocate the values as before:
//!     let value = Handle::new_in(&arena, SomeType::default());
//!     let value_2 = Handle::new_in(&arena, OtherType::default());
//!
//!     // Use the values as before:
//!     use_values(&*value, &*value_2);
//!
//!     // The values are dropped here, but the memory backing them in the arena is
//!     // not deallocated until the `Arena` is dropped, or the `Arena::reset()` method
//!     // is called.
//! }
//! ```
//!
//! The `Arena` can be visualised as an array of memory, with an integer marking the current
//! end of the `Arena`. When the `Arena` is first initialised and has a block allocated, then
//! the stack is empty:
//!
//! ```text
//! +-------+-----------------------------------------------------------+
//! |  (H)  |                                                           |
//! +-------+-----------------------------------------------------------+
//! ^       ^
//! Header  Arena end
//! ```
//!
//! Then, when `value` and `value_2` are allocated into the arena, they
//! are written into the memory, and the end is adjusted to point into the
//! next available memory space:
//!
//! ```text
//! +-------+---------------+---------------+--------------------------------------+
//! |  (H)  |     (data)    |     (data)    |                                      |
//! +-------+---------------+---------------+--------------------------------------+
//! ^       ^               ^               ^
//! Header  value           value_2         Arena end
//! ```
//!
//! Then, when `value` and `value_2` are dropped, their `drop` logic will be
//! called, but the backing storage of the `Arena` will not shrink to release
//! the memory; it will remain inaccessible for future allocations:
//!
//! ```text
//! +-------+-------------------------------+--------------------------------------+
//! |  (H)  |xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx|                                      |
//! +-------+-------------------------------+--------------------------------------+
//! ^                                       ^
//! Header                                  Arena end
//! ```
//!
//! When the [`Arena::reset()`] method is called then the memory will be reclaimed,
//! and will be usable for future allocations. Note that no active references are
//! allowed into the `Arena` when [`Arena::reset()`] is called, so there is no
//! danger of referring to dangling memory. When the `Arena` is dropped, the
//! backing memory will be deallocated by the system allocator and returned to
//! the operating system.
//!
//! If a block of memory in the arena is completely used up, then a new one will be
//! allocated. Each block stores a pointer to the next block in its header, so blocks
//! will be accessible. reseting an `Arena` will cause blocks to be stored in a free
//! list so that they can be re-used later, reducing pressure on the memory allocator.
//!
//! # Handle Types
//!
//! The `arena` module comes with a few specialised handle types to make working with `Arena`s
//! more ergonomic. These provide a few useful conveniences:
//!
//! * They have a lifetime which ensures that data allocated from the `Arena` cannot outlive the
//!   `Arena`, preventing use after frees.
//! * They handle `Drop` logic when the handle goes out of scope, preventing resource leaks.
//!
//! ## `Handle`
//!
//! The basic handle type is [`handle::Handle`], which is analogous to a `Box<T>` - it provides unique ownership
//! of an object allocated in an [`Arena`], allows mutation, and drops the object when it goes out of scope.
//!
//! Read more in the [`handle`] module.
//!
//! ## `RcHandle`
//!
//! ## `Buffer`
//!
//! ## `StringBuffer`
//!
//! ## `LinkedList`
//!
//! [`Arena`]: ./struct.Arena.html
//! [`Arena::reset()`]: ./struct.Arena.html#method.reset
//! [`handle::Handle`]: ./handle/struct.Handle.html
//! [`handle`]: ./handle/index.html

extern crate alloc;
#[cfg(any(test, feature = "std"))]
extern crate std;

use crate::blocks::{Block, BlockIter, Blocks, ScopedRestore};
use alloc::alloc::{AllocError, Allocator, Global, Layout, LayoutError, handle_alloc_error};
use core::{
    error::Error as ErrorTrait,
    ffi::c_void,
    fmt,
    iter::FusedIterator,
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit},
    ptr::{self, NonNull},
    str,
};

pub mod buffer;
pub mod handle;
pub mod linked_list;
pub mod rc_handle;
pub mod string_buffer;

mod blocks;
#[cfg(test)]
mod tests;

/// An arena allocator, parameterised by global allocator.
///
/// See the [module documentation](index.html) for more info.
pub struct Arena<A: Allocator = Global> {
    blocks: Blocks,
    alloc: A,
    _boo: PhantomData<*mut c_void>,
}

impl Arena {
    /// Creates a new `Arena` with the default allocator and block size.
    ///
    /// See [`Arena::new_in()`] for more details.
    ///
    /// # Example
    ///
    /// ```
    /// use rotunda::Arena;
    ///
    /// let arena = Arena::new();
    /// ```
    ///
    /// [`Arena::new_in()`]: ./struct.Arena.html#method.new_in
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        const { Self::new_in(Global) }
    }

    /// Creates a new `Arena` with the default allocator and a custom block size
    /// in bytes.
    ///
    /// See [`Arena::with_block_size_in()`] for more details.
    ///
    /// # Panics
    ///
    /// This method will panic if `block_size` is greater than `isize::MAX`, as this is a
    /// requirement rust imposes on its allocator trait.
    ///
    /// # Example
    ///
    /// ```
    /// use rotunda::Arena;
    ///
    /// let block_size = 1024 * 1024 * 3;
    /// let arena = Arena::with_block_size(block_size);
    /// ```
    ///
    /// [`Arena::with_block_size_in()`]: ./struct.Arena.html#method.with_block_size_in
    #[inline]
    #[must_use]
    pub const fn with_block_size(block_size: usize) -> Self {
        Self::with_block_size_in(block_size, Global)
    }
}

impl<A: Allocator> Arena<A> {
    /// Create a new `Arena`, using the provided `allocator` to allocate
    /// backing storage from.
    ///
    /// The default block size is `isize::MAX + 1`. This provides a good
    /// trade-off between blocks which are large enough to store most types
    /// and not require too much reallocation, but not too large that
    /// the allocator may run out of memory when allocating the backing
    /// memory for the `Arena`.
    ///
    /// See [`Arena::with_block_size_in()`] for more details.
    ///
    /// # Notes
    ///
    /// This function requires the `allocator_api` feature to be enabled to
    /// use it.
    ///
    /// # Example
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use rotunda::Arena;
    /// use std::alloc::Global;
    ///
    /// let arena = Arena::new_in(Global);
    /// ```
    ///
    /// [`Arena::with_block_size_in()`]: ./struct.Arena.html#method.with_block_size_in
    #[inline]
    #[must_use]
    pub const fn new_in(allocator: A) -> Self {
        let default_block_size = i16::MAX as usize + 1;
        Self::with_block_size_in(default_block_size, allocator)
    }

    /// Create a new `Arena` with the given `block_size` in bytes, and the provided
    /// `allocator` to allocate backing storage from.
    ///
    /// # Panics
    ///
    /// This method will panic if `block_size` is greater than `isize::MAX`, as this is a
    /// requirement rust imposes on its allocator trait.
    ///
    /// # Example
    ///
    /// ```
    /// #![feature(allocator_api)]
    ///
    /// use rotunda::Arena;
    /// use std::alloc::Global;
    ///
    /// let block_size = 4 * 1024 * 1024;
    /// let arena = Arena::with_block_size_in(block_size, Global);
    /// ```
    #[must_use]
    #[inline]
    pub const fn with_block_size_in(block_size: usize, allocator: A) -> Self {
        Self {
            blocks: Blocks::new(block_size),
            alloc: allocator,
            _boo: PhantomData,
        }
    }

    /// Returns the total number of bytes which can be allocated into the current
    /// block before a new block will need to be allocated.
    ///
    /// If a current block is not available, returns `None`.
    ///
    /// # Example
    ///
    /// ```
    /// # use rotunda::{Arena, handle::Handle};
    /// let block_size = 4 * 1024 * 1024;
    /// let arena = Arena::with_block_size(block_size);
    /// let handle = Handle::new_in(&arena, 54i32);
    /// assert_eq!(arena.curr_block_capacity().unwrap(), block_size - std::mem::size_of::<i32>());
    /// ```
    #[inline]
    #[must_use]
    pub fn curr_block_capacity(&self) -> Option<usize> {
        self.blocks.curr_block_capacity()
    }

    /// Reserves the given number of `num_blocks` into the `Arena`, placing them in the free list.
    ///
    /// Should the current block overflow while servicing allocation requests, the free blocks
    /// can be used without needing to call the allocator.
    ///
    /// # Panics
    ///
    /// If the allocator cannot allocate the blocks, then `handle_alloc_err()` will be called,
    /// which may panic or abort the process.
    ///
    /// # Example
    ///
    /// ```
    /// use rotunda::Arena;
    ///
    /// let arena = Arena::new();
    ///
    /// arena.reserve_blocks(3);
    /// ```
    #[track_caller]
    #[inline]
    pub fn reserve_blocks(&self, num_blocks: usize) {
        let (layout, alloc) = (self.blocks.block_layout(), self.allocator());
        for _ in 0..num_blocks {
            let block = Block::alloc(layout, alloc);
            self.blocks.push_free_block(block);
        }
    }

    /// Tries to reserve the given number of `num_blocks` into the `Arena`, placing them in the free list.
    ///
    /// Should the current block overflow while servicing allocation requests, the free blocks
    /// can be used without needing to call the allocator.
    ///
    /// # Errors
    ///
    /// If the allocator cannot allocate a `Block`, then `AllocError` is returned. If other blocks were allocated
    /// as part of this call, then they will remain in the `Arena`.
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(allocator_api)]
    /// use rotunda::Arena;
    /// # extern crate alloc;
    ///
    /// let arena = Arena::new();
    ///
    /// # fn inner(arena: &Arena) -> Result<(), alloc::alloc::AllocError> {
    /// arena.try_reserve_blocks(3)?;
    /// # Ok(())
    /// # }
    /// # inner(&arena);
    /// ```
    #[inline]
    pub fn try_reserve_blocks(&self, num_blocks: usize) -> Result<(), AllocError> {
        let (layout, alloc) = (self.blocks.block_layout(), self.allocator());
        for _ in 0..num_blocks {
            let block = Block::try_alloc(layout, alloc)?;
            self.blocks.push_free_block(block);
        }
        Ok(())
    }

    /// Returns a pointer into the head of the current block.
    ///
    /// The available space in the current block is returned as the slice length.
    ///
    /// # Examples
    ///
    /// ```
    /// # use core::{mem, ptr};
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let capacity = 3100;
    ///
    /// let arena = Arena::with_block_size(capacity);
    ///
    /// assert!(arena.curr_block_head().is_none());
    ///
    /// arena.force_push_new_block();
    ///
    /// let block = arena.curr_block_head().unwrap();
    /// assert_eq!(block.len(), capacity);
    ///
    /// let data = Handle::new_in(&arena, 24i32);
    /// assert!(ptr::eq(Handle::as_ptr(&data).cast::<u8>(), block.as_ptr() as *mut _ as *mut u8));
    ///
    /// let block = arena.curr_block_head().unwrap();
    /// assert_eq!(block.len(), capacity - mem::size_of::<i32>());
    /// ```
    #[inline]
    pub fn curr_block_head(&self) -> Option<NonNull<[MaybeUninit<u8>]>> {
        let curr_block = self.blocks.curr_block().get()?;

        let (block_size, block_pos) = (self.block_size(), self.blocks.curr_block_pos().get());

        let data = unsafe {
            Block::data_ptr(curr_block, block_size)
                .cast::<MaybeUninit<u8>>()
                .as_ptr()
                .map_addr(|data| data + block_pos)
        };

        NonNull::new(ptr::slice_from_raw_parts_mut(data, block_size - block_pos))
    }

    /// Forces the `Arena` to push all allocations into a new block of memory.
    ///
    /// The current block of memory will be moved to the used list, and will be
    /// inaccessible until the `Arena` is cleared.
    ///
    /// This function will take a block from the free list if it is available, or allocate
    /// a new one otherwise.
    ///
    /// # Panics
    ///
    /// This method will panic if the global allocator is unable to allocate a new block.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rotunda::{Arena, handle::Handle};
    /// # use core::mem;
    ///
    /// let arena = Arena::with_block_size(640);
    ///
    /// let handle_1 = Handle::new_in(&arena, 543);
    /// let handle_2 = Handle::new_in(&arena, 123);
    ///
    /// assert_eq!(
    ///     Handle::as_ptr(&handle_2).addr() - Handle::as_ptr(&handle_1).addr(),
    ///     mem::size_of::<i32>());
    ///
    /// arena.force_push_new_block();
    ///
    /// let handle_3 = Handle::new_in(&arena, 456);
    ///
    /// assert_ne!(
    ///     Handle::as_ptr(&handle_3).addr() - Handle::as_ptr(&handle_2).addr(),
    ///     mem::size_of::<i32>());
    /// ```
    #[track_caller]
    #[inline]
    pub fn force_push_new_block(&self) {
        let _ = unsafe {
            self.get_free_block()
                .map_err(|_| handle_alloc_error(self.blocks.block_layout()))
        };
    }

    /// Forces the `Arena` to push all allocations into a new block of memory.
    ///
    /// The current block of memory will be moved to the used list, and will be
    /// inaccessible until the `Arena` is cleared.
    ///
    /// This function will take a block from the free list if it is available, or allocate
    /// a new one otherwise.
    ///
    /// # Errors
    ///
    /// This method will return an error if the global allocator is unable to allocate a new block.
    #[inline]
    pub fn try_force_push_new_block(&self) -> Result<(), AllocError> {
        unsafe { self.get_free_block().map(|_| ()) }
    }

    /// Checks whether the current block has the capacity to allocate up to `block_capacity_bytes`.
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate alloc;
    /// use rotunda::Arena;
    /// use alloc::alloc::Layout;
    ///
    /// let arena = Arena::with_block_size(25);
    /// arena.force_push_new_block();
    ///
    /// assert!(arena.has_block_capacity(24));
    ///
    /// let layout = Layout::new::<[u8; 5]>();
    /// arena.alloc_raw(layout);
    ///
    /// assert!(!arena.has_block_capacity(24));
    /// ```
    #[inline]
    pub fn has_block_capacity(&self, block_capacity_bytes: usize) -> bool {
        if self.block_size() < block_capacity_bytes {
            return false;
        }

        self.curr_block_capacity()
            .map(|cap| cap >= block_capacity_bytes)
            .unwrap_or_default()
    }

    /// Checks whether the current block has the capacity to allocate up to `block_capacity_bytes`,
    /// and allocates a new block if it doesn't.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate alloc;
    /// use rotunda::Arena;
    /// use alloc::alloc::Layout;
    ///
    /// let arena = Arena::with_block_size(25);
    /// assert_eq!(arena.curr_block_capacity(), None);
    ///
    /// arena.ensure_block_capacity(20);
    /// assert_eq!(arena.curr_block_capacity(), Some(25));
    ///
    /// let _ = arena.alloc_raw(Layout::new::<[u8; 23]>());
    /// assert!(!arena.has_block_capacity(20));
    ///
    /// arena.ensure_block_capacity(20);
    /// assert!(arena.has_block_capacity(20));
    /// ```
    #[inline]
    pub fn ensure_block_capacity(&self, block_capacity_bytes: usize) {
        if self.block_size() < block_capacity_bytes {
            return;
        }

        if !self.has_block_capacity(block_capacity_bytes) {
            self.force_push_new_block();
        }
    }

    /// Resets the `Arena` so that all used blocks are moved into the free list, and
    /// the current block counter is reset to `0`.
    ///
    /// This method does not use the `Arena`'s allocator to perform any memory
    /// deallocation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rotunda::{Arena, handle::Handle};
    /// let mut arena = Arena::new();
    /// {
    ///     let _handle = Handle::new_in(&arena, 23);
    /// }
    /// arena.reset();
    /// ```
    #[inline]
    pub fn reset(&mut self) {
        self.blocks.reset();
    }

    /// Deallocates all blocks in the `Arena`'s free list.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle};
    ///
    /// let mut arena = Arena::new();
    /// arena.reserve_blocks(1);
    ///
    /// assert_eq!(arena.free_blocks().count(), 1);
    ///
    /// arena.trim();
    ///
    /// assert_eq!(arena.free_blocks().count(), 0);
    /// ```
    #[inline]
    pub fn trim(&self) {
        self.blocks.trim(self.allocator());
    }

    /// Deallocates `n` blocks in the `Arena`'s free list.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::Arena;
    ///
    /// let mut arena = Arena::new();
    /// arena.reserve_blocks(5);
    ///
    /// assert_eq!(arena.free_blocks().count(), 5);
    ///
    /// arena.trim_n(2);
    ///
    /// assert_eq!(arena.free_blocks().count(), 3);
    /// ```
    #[track_caller]
    #[inline]
    pub fn trim_n(&self, n: usize) {
        self.blocks.trim_n(n, self.allocator());
    }

    /// Deallocates the currently active block in the `Arena`.
    ///
    /// As this takes an exclusive reference, it is guaranteed that there are
    /// no other active allocations backed by the current block.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::Arena;
    ///
    /// let mut arena = Arena::new();
    ///
    /// let data = arena.alloc_ref(25);
    /// drop(data);
    ///
    /// assert!(arena.curr_block().is_some());
    ///
    /// arena.deallocate_current();
    ///
    /// assert!(arena.curr_block().is_none());
    /// ```
    #[inline]
    pub fn deallocate_current(&mut self) {
        self.blocks.curr_block_pos().set(0);
        self.blocks.deallocate_current(self.allocator());
    }

    /// Runs the given closure within a scope that frees all memory allocated from the arena within
    /// it.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rotunda::{Arena, handle::Handle};
    /// let arena = Arena::new();
    /// let result = arena.with_scope(|arena| {
    ///     let h1 = Handle::new_in(&arena, 5);
    ///     let h2 = Handle::new_in(&arena, 10);
    ///     *h1 + *h2
    /// });
    ///
    /// // `h1` and `h2` have been deallocated here, and the memory in the arena has been reset to
    /// // from before they were allocated, so the space can be re-used.
    ///
    /// assert_eq!(result, 15);
    /// ```
    ///
    /// # Notes
    ///
    /// The lifetime of `T` must be `'static`, which means that the closure cannot return borrowed data,
    /// and the lifetime of `F` is `'static`, which means that data used in the scope must be moved into
    /// the closure.
    ///
    /// If this is too restrictive, you can use [`Arena::with_scope_dynamic()`].
    ///
    /// The arena state is restored using a guard type, so the arena's memory will be restored if the given
    /// closure panics.
    ///
    /// [`Arena::with_scope_dynamic()`]: ./struct.Arena.html#method.with_scope_dynamic
    #[inline]
    pub fn with_scope<T: 'static, F: 'static + FnOnce(&Arena<A>) -> T>(&self, f: F) -> T {
        let _scope = ScopedRestore::new(&self.blocks);
        f(self)
    }

    /// Runs the given closure within a scope that frees all memory allocated from the arena within
    /// it.
    ///
    /// This function is similar to [`Arena::with_scope()`], except that it relaxes the
    /// `'static` lifetime requirement on the return type. This allows the caller to
    /// return borrowed data from the scope closure.
    ///
    /// As the closure taken by the function does not take ownership of items moved into it, it
    /// can freely use items from the caller's scope, and does not have an `Arena` parameter.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rotunda::{Arena, handle::Handle};
    /// # use core::cell::Cell;
    /// let data = vec![Cell::new(23)];
    /// let arena = Arena::new();
    /// let value = unsafe {
    ///     arena.with_scope_dynamic(|| {
    ///         let handle = Handle::new_in(&arena, 25);
    ///         let value = &data[0];
    ///         value.update(|old_val| old_val + *handle);
    ///         value
    ///     })
    /// };
    ///
    /// assert_eq!(value.get(), 48);
    /// ```
    ///
    /// # Safety
    ///
    /// This method is unsafe as it is possible to return a handle to memory which would
    /// be freed in the Arena automatically as the scope exits. This would allow safe code
    /// to overwrite the memory in the `Arena` with a new value while still allowing the 
    /// old `Handle` to be accessed, which could cause memory unsafety.
    ///
    /// To avoid this safety issue, never return data allocated in the arena during the scope.
    ///
    /// ```rust,unsafe
    /// # use rotunda::{Arena, handle::Handle};
    /// let arena = Arena::new();
    /// let handle = unsafe {
    ///     arena.with_scope_dynamic(|| {
    ///         let handle = Handle::new_in(&arena, vec![1, 2, 3, 4, 5]);
    ///         # let mut handle = handle;
    ///         # unsafe { core::ptr::drop_in_place(Handle::as_mut_ptr(&mut handle)); } // Keep miri happy
    ///         handle
    ///     })
    /// };
    ///
    /// // Warning ⚠️: `handle` has simultaneously been leakded and points to uninitialised memory.
    /// // It is undefined behaviour to dereference it in any way (including via non-trivial drop).
    /// # core::mem::forget(handle);
    ///
    /// // Warning ⚠️: This call to `Handle::new()` will use the same memory as which was used in the
    /// // `with_scope_dynamic()` call above.
    /// let handle = Handle::new_in(&arena, [128usize, 255, 300]);
    /// ```
    ///
    /// Note that this includes 'smuggling' handles out through a collection type which
    /// uses the `Arena` as a backing store:
    ///
    /// ```rust,unsafe
    /// # use rotunda::{Arena, buffer::Buffer, handle::Handle};
    /// let arena = Arena::new();
    /// let mut buffer = Buffer::with_capacity_in(&arena, 5);
    /// unsafe {
    ///     arena.with_scope_dynamic(|| {
    ///         let handle = Handle::new_in(&arena, String::from("This is a message!"));
    ///         # let mut handle = handle;
    ///         # unsafe { core::ptr::drop_in_place(Handle::as_mut_ptr(&mut handle)); } // Keep miri happy
    ///         buffer.push(handle);
    ///     });
    /// };
    ///
    /// // Warning ⚠️: `buffer[0]` has simultaneously been leakded and points to uninitialised memory.
    /// // It is undefined behaviour to dereference it in any way (including via non-trivial drop).
    /// # core::mem::forget(buffer.remove(0));
    /// ```
    ///
    /// [`Arena::with_scope()`]: ./struct.Arena.html#method.with_scope
    #[inline]
    pub unsafe fn with_scope_dynamic<T, F: FnMut() -> T>(&self, mut f: F) -> T {
        let _scope = ScopedRestore::new(&self.blocks);
        f()
    }

    /// Returns a reference to the underlying [`Allocator`] type used to allocate/deallocate
    /// memory.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rotunda::Arena;
    /// let arena = Arena::new();
    /// let allocator = arena.allocator();
    /// # let _ = allocator;
    /// ```
    ///
    /// [`Allocator`]: https://doc.rust-lang.org/stable/alloc/alloc/trait.Allocator.html
    #[must_use]
    #[inline]
    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Returns the block size of the `Arena`. This is specified using the [`Arena::with_block_size()`]
    /// and [`Arena::with_block_size_in()`] constructors.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rotunda::Arena;
    /// const BLOCK_SIZE: usize = 6 * 1024 * 1024;
    /// let arena = Arena::with_block_size(BLOCK_SIZE);
    /// assert_eq!(arena.block_size(), BLOCK_SIZE);
    /// ```
    ///
    /// [`Arena::with_block_size()`]:
    /// [`Arena::with_block_size_in()`]:
    #[must_use]
    #[inline]
    pub fn block_size(&self) -> usize {
        self.blocks.block_size()
    }

    /// Returns a new allocation matching `layout` from the current block in the `Arena`.
    ///
    /// # Panics
    ///
    /// This method can panic if:
    ///
    /// * The requested `Layout` can never be satisfied by a block with the size of [`Arena::block_size()`].
    /// * The underlying allocator raises an error which is handled by the [`handle_alloc_error()`] function.
    ///
    /// [`Arena::block_size()`]: ./struct.Arena.html#method.block_size
    /// [`handle_alloc_error()`]: https://doc.rust-lang.org/stable/alloc/alloc/fn.handle_alloc_error.html
    #[track_caller]
    #[must_use]
    pub fn alloc_raw(&self, layout: Layout) -> NonNull<c_void> {
        if layout.size() == 0 {
            return self.alloc_zst(layout.align());
        }

        #[cold]
        #[track_caller]
        #[inline(never)]
        fn insert_fresh_block_fail(layout: &Layout) -> ! {
            panic!(
                "can never insert a type with layout `Layout (size = {}, align = {})` into arena",
                layout.size(),
                layout.align()
            );
        }

        if let Err(e) = self.get_block_for_layout(layout) {
            match e {
                Error::AllocErr(_) => handle_alloc_error(layout),
                _ => insert_fresh_block_fail(&layout),
            }
        }

        unsafe {
            let slot = self.blocks.bump_layout(layout);
            NonNull::new_unchecked(slot.as_ptr())
        }
    }

    /// Returns a new allocation matching `layout` in the `Arena`.
    ///
    /// # Errors
    ///
    /// If the underlying allocator could not service a request, then
    /// an `Error::AllocErr` is returned.
    ///
    /// If the given `layout` could not fit in an empty block, then
    /// `Error::BadLayout` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate alloc;
    /// # use alloc::alloc::Layout;
    /// # use rotunda::Error;
    /// use rotunda::Arena;
    ///
    /// let arena = Arena::with_block_size(4);
    ///
    /// # fn inner(arena: &Arena) -> Result<(), rotunda::Error> {
    /// let data = arena.try_alloc_raw(Layout::new::<u16>()).expect("Will succeed");
    /// let will_error = arena.try_alloc_raw(Layout::new::<u64>())?;
    /// # Ok(())
    /// # }
    /// # let result = inner(&arena);
    /// # let layout = match result.unwrap_err() { Error::BadLayout(l) => l, _ => unreachable!() };
    /// # assert_eq!(layout, Layout::new::<u64>());
    /// ```
    #[inline]
    pub fn try_alloc_raw(&self, layout: Layout) -> Result<NonNull<c_void>, Error> {
        if layout.size() == 0 {
            return Ok(self.alloc_zst(layout.align()));
        }

        self.get_block_for_layout(layout)?;

        unsafe {
            let slot = self.blocks.bump_layout(layout);
            Ok(NonNull::new_unchecked(slot.as_ptr()))
        }
    }

    /// Returns a new zeroed allocation from the current block in the `Arena`.
    ///
    /// If the current block in the `Arena` is full, then a new one will be allocated and used.
    ///
    /// # Panics
    ///
    /// This method can panic if:
    ///
    /// * The requested `Layout` can never be satisfied by a block with the size of [`Arena::block_size()`].
    /// * The underlying allocator raises an error which is handled by the [`handle_alloc_error()`] function.
    ///
    /// [`Arena::block_size()`]: ./struct.Arena.html#method.block_size
    /// [`handle_alloc_error()`]: https://doc.rust-lang.org/stable/alloc/alloc/fn.handle_alloc_error.html
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn alloc_raw_zeroed(&self, layout: Layout) -> NonNull<c_void> {
        let slot = self.alloc_raw(layout);
        unsafe {
            ptr::write_bytes(slot.as_ptr().cast::<u8>(), b'\0', layout.size());
        }
        slot
    }

    /// Allocate the given `value` into this `Arena` and returns an exclusive reference to it.
    ///
    /// The allocated value is wrapped in a [`ManuallyDrop`] to indicate that its `drop` method
    /// will not be called when the value goes out of scope. If the value has a meaningful
    /// [`Drop::drop()`] implementation, then you should call [`ManuallyDrop::drop()`] on it.
    ///
    /// # Errors
    ///
    /// This method can fail if the arena cannot allocate a new block to store the value.
    ///
    /// # Example
    ///
    /// ```
    /// use core::mem::ManuallyDrop;
    /// use rotunda::Arena;
    ///
    /// let mut arena = Arena::new();
    ///
    /// let value = arena.alloc_ref("Hello!");
    /// assert_eq!(**value, "Hello!");
    ///
    /// unsafe {
    ///     ManuallyDrop::drop(value);
    /// }
    /// ```
    ///
    /// [`ManuallyDrop`]: https://doc.rust-lang.org/stable/core/mem/struct.ManuallyDrop.html
    /// [`Drop::drop()`]: https://doc.rust-lang.org/stable/core/ops/trait.Drop.html#tymethod.drop
    /// [`ManuallyDrop::drop()`]: https://doc.rust-lang.org/stable/core/mem/struct.ManuallyDrop.html#method.drop
    #[allow(clippy::mut_from_ref)]
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn alloc_ref<T>(&self, value: T) -> &mut ManuallyDrop<T> {
        let mut slot = self.alloc_raw(Layout::new::<T>()).cast::<ManuallyDrop<T>>();
        unsafe {
            slot.write(ManuallyDrop::new(value));
            slot.as_mut()
        }
    }

    /// Allocate the given `string` into the `Arena`, returning a exclusive reference to the string
    /// in the arena.
    ///
    /// # Panics
    ///
    /// This method will panic if the string cannot be allocated in the `Arena`
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::Arena;
    ///
    /// let arena = Arena::new();
    ///
    /// let message = arena.alloc_str("hello ❤️");
    ///
    /// message.make_ascii_uppercase();
    ///
    /// assert_eq!(message, "HELLO ❤️");
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn alloc_str<S: ?Sized + AsRef<str>>(&self, string: &S) -> &mut str {
        let string = string.as_ref();
        let len = string.len();

        unsafe {
            // @SAFETY: It is always safe to create a layout for any number of bytes.
            let slot = self.alloc_raw(Layout::array::<u8>(len).unwrap_unchecked());

            let slot = slot.as_ptr().cast::<u8>();
            ptr::copy(string.as_ptr(), slot, len);
            let bytes = ptr::slice_from_raw_parts_mut(slot, len);
            &mut *(bytes as *mut str)
        }
    }

    /// Returns a reference to the currently in-use block, if it is available.
    ///
    /// This method requires exclusive access to the `Arena`, so there
    /// cannot be any objects allocated live from the arena.
    ///
    /// # Examples
    ///
    /// ```
    /// # use core::mem::{drop, MaybeUninit};
    /// use rotunda::{Arena, handle::Handle};
    /// let mut arena = Arena::new();
    ///
    /// let no_block = arena.curr_block();
    /// assert!(no_block.is_none());
    ///
    /// let handle = Handle::new_str_in(&arena, "Hello, world!");
    /// drop(handle);
    ///
    /// let block = arena.curr_block().expect("block must be allocated");
    ///
    /// block.fill(MaybeUninit::new(0xcd));
    /// ```
    #[must_use]
    #[inline]
    pub fn curr_block(&mut self) -> Option<&mut [MaybeUninit<u8>]> {
        unsafe {
            self.blocks
                .curr_block()
                .get()
                .map(|block| Block::data_mut(block, self.block_size()))
        }
    }

    /// Returns an iterator over free blocks in the `Arena`.
    ///
    /// This method requires exclusive access to the `Arena`, so there
    /// cannot be any objects allocated live from the arena.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rotunda::{Arena, handle::Handle};
    /// # use core::mem::MaybeUninit;
    /// let mut arena = Arena::new();
    /// # arena.force_push_new_block();
    /// # arena.force_push_new_block();
    ///
    /// for free_block in arena.free_blocks() {
    ///     free_block.fill(MaybeUninit::new(0x00));
    /// }
    /// ```
    #[must_use]
    #[inline]
    pub fn free_blocks(&mut self) -> FreeBlocksMut<'_, A> {
        FreeBlocksMut::new(self)
    }

    /// Returns an iterator over all blocks in the `Arena`.
    ///
    /// This method requires exclusive access to the `Arena`, so there
    /// cannot be any objects allocated live from the arena.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rotunda::{Arena, handle::Handle};
    /// # use core::mem::MaybeUninit;
    /// let mut arena = Arena::new();
    /// # arena.force_push_new_block();
    /// # arena.force_push_new_block();
    ///
    /// for block in arena.all_blocks() {
    ///     block.fill(MaybeUninit::new(0xff));
    /// }
    /// ```
    #[must_use]
    #[inline]
    pub fn all_blocks(&mut self) -> AllBlocksMut<'_, A> {
        AllBlocksMut::new(self)
    }

    #[must_use]
    #[inline]
    unsafe fn get_free_block(&self) -> Result<NonNull<Block>, AllocError> {
        let block = match self.blocks.free_blocks().get() {
            // If we have a free block, grab it
            Some(block) => unsafe {
                let next_free_block = block.as_ref().next.get();
                block.as_ref().next.set(None);

                self.blocks.free_blocks().set(next_free_block);
                block
            },
            // otherwise, alloc another one
            _ => self.alloc_block()?,
        };

        self.blocks.curr_block_pos().set(0);
        if let Some(curr_block) = self.blocks.curr_block().get() {
            self.blocks.push_used_block(curr_block);
        }

        self.blocks.curr_block().set(Some(block));
        Ok(block)
    }

    #[must_use]
    #[inline]
    fn alloc_block(&self) -> Result<NonNull<Block>, AllocError> {
        let layout = self.blocks.block_layout();
        Block::try_alloc(layout, self.allocator())
    }

    #[must_use]
    fn get_block_for_layout(&self, layout: Layout) -> Result<NonNull<Block>, Error> {
        let block = self
            .blocks
            .curr_block()
            .get()
            .filter(|_| self.blocks.can_write_layout(&layout))
            .map(Ok)
            .unwrap_or_else(|| {
                let block = unsafe { self.get_free_block()? };
                if !self.blocks.can_write_layout(&layout) {
                    return Err(Error::BadLayout(layout));
                }

                Ok(block)
            });

        if let Ok(block) = block {
            debug_assert!(ptr::eq(
                block.as_ptr(),
                self.blocks
                    .curr_block()
                    .get()
                    .map(|p| p.as_ptr())
                    .unwrap_or(ptr::null_mut())
            ));
        }

        block
    }

    #[must_use]
    fn alloc_zst(&self, align: usize) -> NonNull<c_void> {
        let mut ptr = NonNull::dangling();
        let offset = ptr.align_offset(align);

        if offset != 0 || offset != usize::MAX {
            ptr = ptr.map_addr(|addr| addr.saturating_add(offset));
        }

        return ptr;
    }
}

impl<A: Allocator + Default> Default for Arena<A> {
    #[inline]
    fn default() -> Self {
        Self::new_in(Default::default())
    }
}

impl<A: Allocator> fmt::Debug for Arena<A> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.blocks.write_debug("Arena", fmtr)
    }
}

unsafe impl<A: Allocator + Sync> Send for Arena<A> {}

impl<A: Allocator> Drop for Arena<A> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.blocks.dealloc_all_memory(self.allocator());
        }
    }
}

/// An iterator type over the free blocks of an [`Arena`].
///
/// See the [`Arena::free_blocks()`] method for more information.
///
/// [`Arena`]: ./struct.Arena.html
/// [`Arena::free_blocks()`]: ./struct.Arena.html#method.free_blocks
pub struct FreeBlocksMut<'a, A: Allocator = Global> {
    arena: &'a mut Arena<A>,
    it: BlockIter,
}

impl<'a, A: Allocator> FreeBlocksMut<'a, A> {
    #[must_use]
    #[inline]
    const fn new(arena: &'a mut Arena<A>) -> Self {
        let it = BlockIter::new(arena.blocks.free_blocks().get());
        Self { arena, it }
    }
}

impl<'a, A: Allocator> Iterator for FreeBlocksMut<'a, A> {
    type Item = &'a mut [MaybeUninit<u8>];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let block = self.it.next()?;

        let data = unsafe {
            let block_size = self.arena.block_size();
            Block::data_mut(block, block_size)
        };

        Some(data)
    }
}

impl<'a, A: Allocator> FusedIterator for FreeBlocksMut<'a, A> {}

impl<'a, A: Allocator> fmt::Debug for FreeBlocksMut<'a, A> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmtr.debug_struct("BlocksMut").finish_non_exhaustive()
    }
}

/// An iterator type over every block of an [`Arena`].
///
/// See the [`Arena::all_blocks()`] method for more information.
///
/// [`Arena`]: ./struct.Arena.html
/// [`Arena::all_blocks()`]: ./struct.Arena.html#method.all_blocks
pub struct AllBlocksMut<'a, A: Allocator = Global> {
    arena: &'a mut Arena<A>,
    curr: Option<NonNull<Block>>,
    free_blocks: BlockIter,
    used_blocks: BlockIter,
}

impl<'a, A: Allocator> AllBlocksMut<'a, A> {
    #[inline]
    const fn new(arena: &'a mut Arena<A>) -> Self {
        let curr = arena.blocks.curr_block().get();

        let free_blocks = BlockIter::new(arena.blocks.free_blocks().get());
        let used_blocks = BlockIter::new(arena.blocks.used_blocks().get());

        Self {
            curr,
            arena,
            free_blocks,
            used_blocks,
        }
    }
}

impl<'a, A: Allocator> Iterator for AllBlocksMut<'a, A> {
    type Item = &'a mut [MaybeUninit<u8>];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.curr
            .take()
            .or_else(|| self.free_blocks.next())
            .or_else(|| self.used_blocks.next())
            .map(|block| {
                let block_size = self.arena.block_size();
                unsafe { Block::data_mut(block, block_size) }
            })
    }
}

impl<'a, A: Allocator> fmt::Debug for AllBlocksMut<'a, A> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmtr.debug_struct("AllBlocksMut").finish_non_exhaustive()
    }
}

/// Represents error types which may be returned while using an `Arena`.
#[derive(Debug)]
pub enum Error {
    /// A `Layout` could not be constructed for a particular type.
    LayoutErr(LayoutError),
    /// The underlying allocator could not service the request.
    AllocErr(AllocError),
    /// The allocation request with the given `Layout` could not be serviced.
    BadLayout(Layout),
}

impl From<LayoutError> for Error {
    #[inline]
    fn from(value: LayoutError) -> Self {
        Self::LayoutErr(value)
    }
}

impl From<AllocError> for Error {
    #[inline]
    fn from(value: AllocError) -> Self {
        Self::AllocErr(value)
    }
}

impl fmt::Display for Error {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::AllocErr(ref e) => fmt::Display::fmt(e, fmtr),
            Self::LayoutErr(ref e) => fmt::Display::fmt(e, fmtr),
            Self::BadLayout(layout) => write!(
                fmtr,
                "Arena cannot allocate a value of size {}, alignment {}",
                layout.size(),
                layout.align()
            ),
        }
    }
}

impl ErrorTrait for Error {
    #[inline]
    fn source(&self) -> Option<&(dyn ErrorTrait + 'static)> {
        match *self {
            Self::AllocErr(ref e) => e.source(),
            Self::LayoutErr(ref e) => e.source(),
            Self::BadLayout(_) => None,
        }
    }
}

pub(crate) type InvariantLifetime<'a, T> = PhantomData<fn(&'a T) -> &'a T>;
