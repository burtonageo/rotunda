use alloc::alloc::{AllocError, Allocator, Layout, handle_alloc_error};
use core::{
    cell::{Cell, UnsafeCell},
    ffi::c_void,
    fmt,
    marker::{PhantomData, PhantomPinned},
    mem::{self, offset_of},
    ptr::{self, NonNull},
    slice, str,
};

#[non_exhaustive]
pub(super) struct Blocks {
    block_size: usize,
    curr_block_pos: Cell<usize>,
    curr_block: BlockPtr,
    free_blocks: BlockPtr,
    used_blocks: BlockPtr,
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

    #[inline]
    pub(super) const fn reset(&mut self) {
        self.curr_block_pos.replace(0);
        let old_used = self.used_blocks.replace(None);
        if let Some(block) = old_used {
            self.push_free_block(block);
        }
    }

    #[inline]
    pub(super) fn trim(&self, allocator: &dyn Allocator) {
        let block_layout = self.block_layout();
        unsafe {
            dealloc_blocks(block_layout, &self.free_blocks, allocator);

            if self.curr_block_pos.get() == 0 {
                dealloc_blocks(block_layout, &self.curr_block, allocator);
            }
        }
    }

    #[inline]
    pub(super) const fn push_used_block(&self, block: NonNull<Block>) {
        push_block(&self.used_blocks, block);
    }

    #[inline]
    pub(super) const fn push_free_block(&self, block: NonNull<Block>) {
        push_block(&self.free_blocks, block);
    }

    #[inline]
    pub(super) unsafe fn dealloc_blocks(&self, block_start: &BlockPtr, allocator: &dyn Allocator) {
        let block_layout = self.block_layout();
        unsafe {
            dealloc_blocks(block_layout, block_start, allocator);
        }
    }

    pub(super) unsafe fn dealloc_all_memory(&self, allocator: &dyn Allocator) {
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
        struct DebugPtrChain(BlockPtr);

        impl fmt::Debug for DebugPtrChain {
            fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
                let mut list = fmtr.debug_list();

                let mut curr = self.0.get();
                while let Some(ptr) = curr {
                    list.entry(&ptr);
                    curr = unsafe { ptr.as_ref().next.get() };
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
    pub(super) const fn curr_block_capacity(&self) -> usize {
        self.block_size - self.curr_block_pos.get()
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
    pub(crate) const fn curr_block(&self) -> &BlockPtr {
        &self.curr_block
    }

    #[must_use]
    #[inline]
    pub(crate) const fn free_blocks(&self) -> &BlockPtr {
        &self.free_blocks
    }

    #[cfg(test)]
    #[must_use]
    #[inline]
    pub(crate) const fn used_blocks(&self) -> &BlockPtr {
        &self.used_blocks
    }
}

#[repr(C)]
pub(super) struct Block {
    pub(super) next: BlockPtr,
    data: UnsafeCell<()>,
    _boo: PhantomData<(*mut u8, PhantomPinned)>,
}

impl Block {
    #[must_use]
    #[inline]
    pub(super) unsafe fn data(&mut self, len: usize) -> &mut [u8] {
        let ptr = ptr::from_mut(self)
            .map_addr(|addr| addr + offset_of!(Block, data))
            .cast::<u8>();
        unsafe { slice::from_raw_parts_mut(ptr, len) }
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
    #[must_use]
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
            .map_addr(|addr| addr.saturating_add(mem::size_of::<BlockPtr>()))
    }
}

type BlockPtr = Cell<Option<NonNull<Block>>>;

#[track_caller]
#[inline(never)]
#[cold]
pub(super) unsafe fn dealloc_blocks(
    block_layout: Layout,
    block_start: &BlockPtr,
    allocator: &dyn Allocator,
) {
    while let Some(block) = block_start.get() {
        let next = unsafe { &block.as_ref().next };
        block_start.set(next.get());

        unsafe {
            allocator.deallocate(block.cast(), block_layout);
        }
    }

    block_start.set(None);
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

        let mut free_block = self.blocks.used_blocks.replace(self.old_used_blocks);
        while let Some(block) = free_block {
            free_block = unsafe { block.as_ref().next.get() };
            if Some(block) != self.old_curr_block {
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

const fn push_single_block(list_head: &BlockPtr, block: NonNull<Block>) {
    let old_head = list_head.get();
    unsafe {
        block.as_ref().next.replace(old_head);
    }
    list_head.replace(Some(block));
}

const fn push_block(list_head: &BlockPtr, block: NonNull<Block>) {
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
