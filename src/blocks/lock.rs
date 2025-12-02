use core::{ptr::{self, NonNull}, usize};
use alloc::alloc::Allocator;
use crate::{Arena, blocks::Block};

pub(crate) const LOCKED_PTR: NonNull<Block> = unsafe {
    NonNull::new_unchecked(ptr::without_provenance_mut(usize::MAX))
};

pub(crate) struct BlockLock<'a, A: Allocator> {
    arena: &'a Arena<A>,
    old_free_blocks: Option<NonNull<Block>>,
    old_used_blocks: Option<NonNull<Block>>,
    prev_in_use: usize,
} 

impl<'a, A: Allocator> BlockLock<'a, A> {
    #[inline]
    pub(crate) unsafe fn lock(arena: &'a Arena<A>) -> Self {
        let old_free_blocks = arena.blocks.free_blocks.replace(Some(LOCKED_PTR));
        let old_used_blocks = arena.blocks.used_blocks.replace(Some(LOCKED_PTR));
        let prev_in_use = arena.blocks.curr_block_pos().get();

        Self {
            arena,
            old_free_blocks,
            old_used_blocks,
            prev_in_use,
        }
    }
}

impl<'a, A: Allocator> Drop for BlockLock<'a, A> {
    #[inline]
    fn drop(&mut self) {
        self.arena.blocks.free_blocks.replace(self.old_free_blocks);
        self.arena.blocks.used_blocks.replace(self.old_used_blocks);
        self.arena.blocks.curr_block_pos().set(self.prev_in_use);
    }
}
