use crate::{
    Arena,
    blocks::{Block, LockData},
};
use alloc::alloc::Allocator;
use core::ptr::{self, NonNull};

pub(crate) const LOCKED_PTR: NonNull<Block> =
    unsafe { NonNull::new_unchecked(ptr::without_provenance_mut(usize::MAX)) };

pub(crate) struct BlockLock<'a, A: Allocator> {
    arena: &'a Arena<A>,
    lock_data: LockData,
}

impl<'a, A: Allocator> BlockLock<'a, A> {
    #[inline]
    pub(crate) fn lock(arena: &'a Arena<A>) -> Self {
        arena.blocks.ensure_unlocked();
        unsafe { Self::lock_unchecked(arena) }
    }

    #[allow(unused)]
    #[inline]
    pub(crate) fn try_lock(arena: &'a Arena<A>) -> Option<Self> {
        if !arena.blocks.is_locked() {
            unsafe { Some(Self::lock_unchecked(arena)) }
        } else {
            None
        }
    }

    #[inline]
    unsafe fn lock_unchecked(arena: &'a Arena<A>) -> Self {
        let lock_data = unsafe { arena.blocks.lock_unchecked() };
        Self { arena, lock_data }
    }
}

impl<'a, A: Allocator> Drop for BlockLock<'a, A> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.arena.blocks.unlock(&self.lock_data);
        }
    }
}
