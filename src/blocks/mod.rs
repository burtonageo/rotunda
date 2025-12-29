// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::alloc::{AllocError, Allocator, Layout, handle_alloc_error};
use core::{
    cell::{Cell, UnsafeCell},
    ffi::c_void,
    fmt,
    marker::{PhantomData, PhantomPinned},
    mem::{self, MaybeUninit, offset_of},
    ptr::{self, NonNull},
    str,
};

pub(super) mod lock;

#[non_exhaustive]
pub(super) struct Blocks {
    block_size: usize,
    curr_block_pos: Cell<usize>,
    curr_block: BlockCellPtr,
    free_blocks: BlockCellPtr,
    used_blocks: BlockCellPtr,
    _priv: (),
}

impl Blocks {
    #[must_use]
    #[inline]
    pub(super) const fn new(block_size: usize) -> Self {
        assert!(block_size <= isize::MAX as usize);
        Self {
            block_size,
            curr_block_pos: Cell::new(0),
            curr_block: Cell::new(None),
            free_blocks: Cell::new(None),
            used_blocks: Cell::new(None),
            _priv: (),
        }
    }

    #[track_caller]
    #[inline]
    pub(super) fn reset(&mut self) {
        self.ensure_unlocked();

        self.curr_block_pos.replace(0);
        let old_used = self.used_blocks.replace(None);
        if let Some(block) = old_used {
            self.push_free_block(block);
        }
    }

    #[track_caller]
    #[inline]
    pub(super) fn reset_all(&mut self) {
        self.ensure_unlocked();
        self.reset();

        if let Some(block) = self.curr_block.replace(None) {
            self.push_free_block(block);
        }
    }

    #[track_caller]
    #[inline]
    pub(super) fn trim(&self, allocator: &dyn Allocator) {
        self.ensure_unlocked();

        let block_layout = self.block_layout();
        unsafe {
            dealloc_blocks(block_layout, &self.free_blocks, allocator);
        }
    }

    #[track_caller]
    pub(super) fn trim_n(&self, n: usize, allocator: &dyn Allocator) {
        self.ensure_unlocked();

        let block_layout = self.block_layout();
        unsafe {
            dealloc_blocks_n(n, block_layout, &self.free_blocks, allocator);
        }
    }

    #[track_caller]
    pub(super) fn deallocate_current(&self, allocator: &dyn Allocator) {
        self.ensure_unlocked();

        debug_assert_eq!(self.curr_block_pos.get(), 0);
        unsafe {
            dealloc_blocks(self.block_layout(), &self.curr_block, allocator);
        }
    }

    #[track_caller]
    #[inline]
    pub(super) const fn push_used_block(&self, block: NonNull<Block>) {
        push_block(&self.used_blocks, block);
    }

    #[track_caller]
    #[inline]
    pub(super) const fn push_free_block(&self, block: NonNull<Block>) {
        push_block(&self.free_blocks, block);
    }

    #[track_caller]
    #[inline]
    pub(super) unsafe fn dealloc_blocks(
        &self,
        block_start: &BlockCellPtr,
        allocator: &dyn Allocator,
    ) {
        self.ensure_unlocked();

        let block_layout = self.block_layout();
        unsafe {
            dealloc_blocks(block_layout, block_start, allocator);
        }
    }

    #[track_caller]
    pub(super) unsafe fn dealloc_all_memory(&self, allocator: &dyn Allocator) {
        self.ensure_unlocked();

        unsafe {
            self.dealloc_blocks(&self.free_blocks, allocator);
            self.dealloc_blocks(&self.used_blocks, allocator);
            self.dealloc_blocks(&self.curr_block, allocator);
        }
    }

    pub(super) fn write_debug(
        &self,
        struct_name: &str,
        fmtr: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        #[repr(transparent)]
        struct DebugPtrChain(BlockCellPtr);

        impl fmt::Debug for DebugPtrChain {
            fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
                let mut list = fmtr.debug_list();

                for ptr in BlockIter(self.0.get()) {
                    list.entry(&ptr);
                }

                list.finish()
            }
        }

        let curr_block = self.curr_block.get();
        let curr_block = curr_block
            .as_ref()
            .map(|block| block as &dyn fmt::Debug)
            .unwrap_or({
                struct Null;
                impl fmt::Debug for Null {
                    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
                        fmtr.write_str("null")
                    }
                }
                &Null
            });

        fmtr.debug_struct(struct_name)
            .field("block_size", &self.block_size)
            .field("curr_block_pos", &self.curr_block_pos.get())
            .field("curr_block", curr_block)
            .field("used_blocks", &DebugPtrChain(self.used_blocks.clone()))
            .field("free_blocks", &DebugPtrChain(self.free_blocks.clone()))
            .finish()
    }

    #[track_caller]
    #[inline]
    pub(super) unsafe fn bump(&self, bytes: usize) -> usize {
        self.ensure_unlocked();

        #[cold]
        fn bump_fail() -> ! {
            panic!("Overflowed block allocated memory");
        }

        self.curr_block_pos.update(|curr| {
            let new_size = curr + bytes;
            if new_size > self.block_size {
                bump_fail();
            }

            new_size
        });
        self.curr_block_pos.get()
    }

    pub(super) unsafe fn bump_layout(&self, layout: Layout) -> NonNull<c_void> {
        let block = self.curr_block.get().unwrap_or(NonNull::dangling());
        let offset = self.offset_to_align_for(&layout);

        unsafe {
            let start = self.bump(offset);
            let slot = Block::data_start(block).add(start).cast::<c_void>();
            self.bump(layout.size());

            slot
        }
    }

    #[allow(unused)]
    #[track_caller]
    #[inline]
    pub(super) unsafe fn unbump(&self, bytes: usize) -> usize {
        self.ensure_unlocked();

        #[cold]
        fn bump_fail() -> ! {
            panic!("Underflowed block allocated memory");
        }

        self.curr_block_pos.update(|curr| {
            let new_size = curr.checked_sub(bytes);
            new_size.unwrap_or_else(|| bump_fail())
        });
        self.curr_block_pos.get()
    }

    #[must_use]
    #[inline]
    pub(super) const fn offset_to_align_for(&self, layout: &Layout) -> usize {
        let align = layout.align();
        let curr = self.curr_block_pos.get();
        let rem = curr % align;
        if rem > 0 { align - rem } else { rem }
    }

    #[must_use]
    #[inline]
    pub(super) const fn can_write_layout(&self, layout: &Layout) -> bool {
        let required = self.offset_to_align_for(layout) + layout.size();
        let arena_cap = self.block_size - self.curr_block_pos.get();
        required <= arena_cap
    }

    #[must_use]
    #[inline]
    pub(super) const fn curr_block_capacity(&self) -> Option<usize> {
        if self.curr_block.get().is_some() {
            Some(self.block_size - self.curr_block_pos.get())
        } else {
            None
        }
    }

    #[must_use]
    #[inline]
    pub(crate) const fn block_layout(&self) -> Layout {
        let data_layout =
            unsafe { Layout::from_size_align_unchecked(self.block_size, mem::align_of::<Block>()) };
        let meta_layout = Layout::new::<Block>();
        match meta_layout.extend(data_layout) {
            Ok((layout, ..)) => layout,
            Err(_) => panic!("bad layout"),
        }
    }

    #[must_use]
    #[inline]
    pub(crate) const fn block_size(&self) -> usize {
        self.block_size
    }

    #[must_use]
    #[inline]
    pub(crate) const fn curr_block_pos(&self) -> &Cell<usize> {
        &self.curr_block_pos
    }

    #[must_use]
    #[inline]
    pub(crate) const fn curr_block(&self) -> &BlockCellPtr {
        &self.curr_block
    }

    #[must_use]
    #[inline]
    pub(crate) const fn free_blocks(&self) -> &BlockCellPtr {
        &self.free_blocks
    }

    #[must_use]
    #[inline]
    pub(crate) const fn used_blocks(&self) -> &BlockCellPtr {
        &self.used_blocks
    }

    #[must_use]
    #[inline]
    pub(crate) fn is_locked(&self) -> bool {
        <Option<NonNull<Block>>>::eq(&self.free_blocks.get(), &Some(lock::LOCKED_PTR))
    }

    pub(crate) unsafe fn lock_unchecked(&self) -> LockData {
        let old_free_blocks = self.free_blocks.replace(Some(lock::LOCKED_PTR));
        let old_used_blocks = self.used_blocks.replace(Some(lock::LOCKED_PTR));
        let prev_in_use = self.curr_block_pos().get();

        LockData {
            old_free_blocks,
            old_used_blocks,
            prev_in_use,
        }
    }

    pub(crate) unsafe fn unlock(&self, lock_data: &LockData) {
        self.free_blocks.replace(lock_data.old_free_blocks);
        self.used_blocks.replace(lock_data.old_used_blocks);
        self.curr_block_pos().set(lock_data.prev_in_use);
    }

    #[inline]
    pub(super) fn is_last_allocation(&self, ptr: NonNull<()>) -> bool {
        self.curr_block
            .get()
            .map(|block| unsafe {
                NonNull::eq(&block.byte_add(self.curr_block_pos.get()), &ptr.cast())
            })
            .unwrap_or(false)
    }

    #[track_caller]
    #[inline]
    fn ensure_unlocked(&self) {
        assert!(!self.is_locked(), "cannot modify Arena while it is locked")
    }
}

#[repr(C)]
pub(super) struct Block {
    pub(super) next: BlockCellPtr,
    data: UnsafeCell<()>,
    _boo: PhantomData<(*mut u8, PhantomPinned)>,
}

impl Block {
    #[must_use]
    #[inline]
    pub(super) unsafe fn data_mut<'a>(
        this: NonNull<Self>,
        len: usize,
    ) -> &'a mut [MaybeUninit<u8>] {
        unsafe { Self::data_ptr(this, len).as_mut() }
    }

    #[must_use]
    #[inline]
    pub(super) unsafe fn data_ptr(this: NonNull<Self>, len: usize) -> NonNull<[MaybeUninit<u8>]> {
        let ptr = this
            .map_addr(|addr| addr.saturating_add(offset_of!(Block, data)))
            .cast::<MaybeUninit<u8>>();
        unsafe { NonNull::new_unchecked(ptr::slice_from_raw_parts_mut(ptr.as_ptr(), len)) }
    }

    #[track_caller]
    #[must_use]
    #[cold]
    #[inline(never)]
    pub(super) fn alloc(block_layout: Layout, allocator: &dyn Allocator) -> NonNull<Block> {
        Self::try_alloc(block_layout, allocator)
            .unwrap_or_else(|_| handle_alloc_error(block_layout))
    }

    #[track_caller]
    #[cold]
    #[inline(never)]
    pub(super) fn try_alloc(
        block_layout: Layout,
        allocator: &dyn Allocator,
    ) -> Result<NonNull<Block>, AllocError> {
        allocator
            .allocate(block_layout)
            .map(NonNull::cast::<Block>)
            .map(|mut block| {
                unsafe {
                    let block = block.as_mut();
                    ptr::write(&raw mut block.next, Cell::new(None));
                }
                block
            })
    }

    #[must_use]
    #[inline]
    pub(super) fn data_start(this: NonNull<Self>) -> NonNull<c_void> {
        this.cast::<c_void>()
            .map_addr(|addr| addr.saturating_add(mem::size_of::<BlockCellPtr>()))
    }
}

#[track_caller]
#[inline(never)]
#[cold]
pub(super) unsafe fn dealloc_blocks(
    block_layout: Layout,
    block_start: &BlockCellPtr,
    allocator: &dyn Allocator,
) {
    unsafe {
        dealloc_blocks_n(usize::MAX, block_layout, block_start, allocator);
    }
}

#[track_caller]
pub(super) unsafe fn dealloc_blocks_n(
    n: usize,
    block_layout: Layout,
    block_start: &BlockCellPtr,
    allocator: &dyn Allocator,
) {
    let mut iter = BlockIter(block_start.get()).enumerate();
    for (i, block) in iter.by_ref() {
        if i >= n {
            block_start.set(Some(block));
            return;
        }

        unsafe {
            allocator.deallocate(block.cast(), block_layout);
        }
    }

    block_start.set(iter.next().map(|tup| tup.1));
}

pub(super) struct ScopedRestore<'a> {
    blocks: &'a Blocks,
    old_block_pos: usize,
    old_curr_block: Option<NonNull<Block>>,
    old_used_blocks: Option<NonNull<Block>>,
}

impl<'a> ScopedRestore<'a> {
    #[must_use]
    pub(super) const fn new(blocks: &'a Blocks) -> Self {
        let old_block_pos = blocks.curr_block_pos.get();
        let old_curr_block = blocks.curr_block.get();
        let old_used_blocks = blocks.used_blocks.replace(None);

        Self {
            blocks,
            old_block_pos,
            old_curr_block,
            old_used_blocks,
        }
    }
}

impl<'a> Drop for ScopedRestore<'a> {
    fn drop(&mut self) {
        self.blocks.curr_block_pos.set(self.old_block_pos);
        if self.blocks.curr_block.get() != self.old_curr_block {
            if let Some(curr_block) = self.blocks.curr_block.get() {
                self.blocks.push_free_block(curr_block);
            }

            self.blocks.curr_block.set(self.old_curr_block);
        }

        let free_block = self.blocks.used_blocks.replace(self.old_used_blocks);
        for block in BlockIter(free_block) {
            if self
                .old_curr_block
                .is_none_or(|old_block| !NonNull::eq(&old_block, &block))
            {
                push_single_block(&self.blocks.free_blocks, block);
            }
        }

        if self.blocks.curr_block.get().is_none() {
            let free = self.blocks.free_blocks.get();
            if let Some(free) = free {
                unsafe {
                    self.blocks.free_blocks.set(free.as_ref().next.get());
                }
            }
            self.blocks.curr_block.set(free);
        }
    }
}

#[repr(transparent)]
pub(crate) struct BlockIter(Option<NonNull<Block>>);

impl BlockIter {
    #[inline(always)]
    pub(crate) const fn new(ptr: Option<NonNull<Block>>) -> Self {
        BlockIter(ptr)
    }
}

impl Iterator for BlockIter {
    type Item = NonNull<Block>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let ptr = self.0?;
        let next = unsafe { ptr.as_ref().next.clone() };
        self.0 = next.get();
        Some(ptr)
    }
}

type BlockCellPtr = Cell<Option<NonNull<Block>>>;

const fn push_single_block(list_head: &BlockCellPtr, block: NonNull<Block>) {
    let old_head = list_head.get();
    unsafe {
        block.as_ref().next.replace(old_head);
    }
    list_head.replace(Some(block));
}

const fn push_block(list_head: &BlockCellPtr, block: NonNull<Block>) {
    let old_head = list_head.get();
    unsafe {
        let mut curr_block = block;
        while let Some(next) = curr_block.as_ref().next.get() {
            curr_block = next;
        }
        curr_block.as_ref().next.replace(old_head);
    }
    list_head.replace(Some(block));
}

#[derive(Clone, Copy)]
pub(crate) struct LockData {
    pub(crate) old_free_blocks: Option<NonNull<Block>>,
    pub(crate) old_used_blocks: Option<NonNull<Block>>,
    pub(crate) prev_in_use: usize,
}
