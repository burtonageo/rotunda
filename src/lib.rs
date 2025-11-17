#![no_std]
#![feature(
    alloc_layout_extra,
    allocator_api,
    const_index,
    const_trait_impl,
    derive_coerce_pointee,
    ptr_metadata
)]

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
//! The basic handle type is [`Handle`], which is analogous to a `Box<T>` - it provides unique ownership
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
//! [`Handle`]: ./struct.Handle.html
//! [`handle`]:

extern crate alloc;
#[cfg(any(test, feature = "std"))]
extern crate std;

use crate::blocks::{Block, Blocks, ScopedRestore};
use alloc::alloc::{AllocError, Allocator, Global, Layout};
use core::{
    ffi::c_void,
    fmt,
    marker::PhantomData,
    ptr::{self, NonNull},
};

pub mod buffer;
pub mod handle;
pub mod linked_list;
pub mod rc_handle;
pub mod string_buffer;

mod blocks;
#[cfg(test)]
mod tests;

pub use self::buffer::Buffer;
pub use self::handle::Handle;
pub use self::linked_list::LinkedList;
pub use self::rc_handle::{RcHandle, WeakHandle};
pub use self::string_buffer::StringBuffer;

/// An arena allocator, paramterised by global allocator.
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
    /// # Example
    ///
    /// ```
    /// # use rotunda::{Arena, handle::Handle};
    /// let block_size = 4 * 1024 * 1024;
    /// let arena = Arena::with_block_size(block_size);
    /// let handle = Handle::new_in(&arena, 54i32);
    /// assert_eq!(arena.curr_block_capacity(), block_size - std::mem::size_of::<i32>());
    /// ```
    #[inline]
    #[must_use]
    pub fn curr_block_capacity(&self) -> usize {
        self.blocks.curr_block_capacity()
    }

    #[track_caller]
    #[inline]
    pub fn reserve_blocks(&self, num_blocks: usize) {
        let (layout, alloc) = (self.blocks.block_layout(), self.allocator());
        for _ in 0..num_blocks {
            let block = Block::alloc(layout, alloc);
            self.blocks.push_free_block(block);
        }
    }

    #[inline]
    pub fn try_reserve_blocks(&self, num_blocks: usize) -> Result<(), AllocError> {
        let (layout, alloc) = (self.blocks.block_layout(), self.allocator());
        for _ in 0..num_blocks {
            let block = Block::try_alloc(layout, alloc)?;
            self.blocks.push_free_block(block);
        }
        Ok(())
    }

    /// Forces the `Arena` to push all allocations into a new block of memory. The current
    /// block of memory will be moved to the used list, and will be inaccessible until the
    /// `Arena` is cleared.
    ///
    /// This function will take a block from the free list if it is available, or allocate
    /// a new one otherwise.
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
        let _ = unsafe { self.get_free_block() };
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

    /// Deallocates all blocks in the `Arena`'s free list, and deallocates the current
    /// block if the arena is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rotunda::{Arena, handle::Handle};
    /// let arena = Arena::new();
    /// let _handle = Handle::new_in(&arena, 23);
    /// arena.trim();
    /// ```
    #[inline]
    pub fn trim(&self) {
        self.blocks.trim(self.allocator());
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

    /// This function is similar to [`Arena::with_scope()`], except that it relaxes the
    /// `'static` lifetime requirement on the return type. This allows the caller to
    /// return borrowed data from the scope closure.
    ///
    /// As the closure taken by function does not take ownership of items moved into it, it
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
    /// be freed automatically as the scope exits. This would allow safe code to be exposed
    /// to dangling pointers, which could cause undefined behaviour.
    ///
    /// To avoid this safety issue, never return data allocated in the arena during the scope.
    ///
    /// ```rust,unsafe
    /// # use rotunda::{Arena, handle::Handle};
    /// let arena = Arena::new();
    /// let handle = unsafe {
    ///     arena.with_scope_dynamic(|| {
    ///         Handle::new_in(&arena, 23)
    ///     })
    /// };
    ///
    /// // Warning ⚠️: `handle` points to uninitialised memory here. It is undefined behaviour
    /// // to dereference it in any way (including via non-trivial drop).
    /// # core::mem::forget(handle);
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
    ///         let handle = Handle::new_in(&arena, 21);
    ///         buffer.push(handle);
    ///     });
    /// };
    ///
    /// // Warning ⚠️: `buffer[0]` points to uninitialised memory here. It is undefined behaviour
    /// // to dereference it in any way (including via non-trivial drop).
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

    /// Returns a new allocation from the current block in the `Arena`.
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
        let block = self
            .blocks
            .curr_block()
            .get()
            .filter(|_| self.blocks.can_write_layout(&layout))
            .unwrap_or_else(|| {
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

                let block = unsafe { self.get_free_block() };
                if !self.blocks.can_write_layout(&layout) {
                    insert_fresh_block_fail(&layout);
                }

                block
            });

        let offset = self.blocks.offset_to_align_for(&layout);
        unsafe {
            let start = self.blocks.bump(offset);
            let slot = Block::data_start(block).add(start).cast::<c_void>();
            self.blocks.bump(layout.size());

            NonNull::new_unchecked(slot.as_ptr())
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
    #[must_use]
    #[inline]
    pub fn alloc_raw_zeroed(&self, layout: Layout) -> NonNull<c_void> {
        let slot = self.alloc_raw(layout);
        unsafe {
            ptr::write_bytes(slot.as_ptr().cast::<u8>(), b'\0', layout.size());
        }
        slot
    }

    #[must_use]
    #[inline]
    unsafe fn get_free_block(&self) -> NonNull<Block> {
        let block = match self.blocks.free_blocks().get() {
            // If we have a free block, grab it
            Some(block) => unsafe {
                let next_free_block = block.as_ref().next.get();
                block.as_ref().next.set(None);

                self.blocks.free_blocks().set(next_free_block);
                block
            },
            // otherwise, alloc another one
            _ => self.alloc_block(),
        };

        self.blocks.curr_block_pos().set(0);
        if let Some(curr_block) = self.blocks.curr_block().get() {
            self.blocks.push_used_block(curr_block);
        }

        self.blocks.curr_block().set(Some(block));
        block
    }

    #[must_use]
    #[inline]
    fn alloc_block(&self) -> NonNull<Block> {
        let layout = self.blocks.block_layout();
        Block::alloc(layout, self.allocator())
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
