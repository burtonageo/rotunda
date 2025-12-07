use crate::{
    Arena, buf,
    buffer::Buffer,
    handle::Handle,
    linked_list::LinkedList,
    rc_handle::{RcHandle, WeakHandle},
    string_buffer::StringBuffer,
};
use core::{mem::ManuallyDrop, panic::AssertUnwindSafe, sync::atomic::AtomicUsize};
use std::{
    alloc::{Allocator, Layout, System},
    iter::Extend,
    mem, panic,
    ptr::{self, NonNull},
    rc::Rc,
    sync::atomic::{AtomicU32, Ordering as AtomicOrdering},
};

#[test]
fn test_arena_alloc() {
    let arena = Arena::new();

    let twenty = arena.alloc_ref(20);
    assert_eq!(**twenty, 20);
    unsafe {
        ManuallyDrop::drop(twenty);
    }

    let five = Handle::new_in(&arena, 5);
    assert!(Handle::as_ptr(&five).is_aligned());
    assert_eq!(*five, 5);

    let ten = Handle::new_in(&arena, 10);
    assert!(Handle::as_ptr(&ten).is_aligned());
    assert_eq!(*ten, 10);

    let fifteen = Handle::new_in(&arena, *five + *ten as usize);
    assert!(Handle::as_ptr(&fifteen).is_aligned());
    assert_eq!(*fifteen, 15);

    let mut value = Handle::new_in(&arena, 21);
    let new_value = Handle::replace(&mut value, 42);
    assert_eq!(new_value, 21);
    assert_eq!(*value, 42);

    const MESSAGE: &'_ str = "Hello!";
    let message = Handle::new_str_in(&arena, MESSAGE);
    assert_eq!(&*message, MESSAGE);
}

#[test]
fn test_arena_clear() {
    let mut arena = Arena::new();

    let p0 = {
        let handle = Handle::new_in(&arena, 21u8);
        Handle::as_ptr(&handle).addr()
    };

    arena.reset();

    let p1 = {
        let handle = Handle::new_in(&arena, 21u8);
        Handle::as_ptr(&handle).addr()
    };

    arena.reset();

    assert_eq!(p0, p1);

    let mut arena = Arena::with_block_size(mem::size_of::<usize>());
    let p0 = Handle::new_in(&arena, 0usize);
    let p1 = Handle::new_in(&arena, 1usize);
    let p2 = Handle::new_in(&arena, 1usize);
    drop((p0, p1, p2));

    arena.reset();

    let mut num_free_blocks = 0;
    let mut curr_block = arena.blocks.free_blocks().get();
    while let Some(block) = curr_block {
        num_free_blocks += 1;
        curr_block = unsafe { block.as_ref().next.get() };
    }

    assert_eq!(num_free_blocks, 2);
    assert!(arena.blocks.curr_block().get().is_some());
}

#[test]
fn test_arena_blocks() {
    use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};
    static DROP_COUNT: AtomicU32 = AtomicU32::new(0);

    #[derive(Debug, Eq, PartialEq)]
    struct CountDrops(u32);

    impl Drop for CountDrops {
        fn drop(&mut self) {
            DROP_COUNT.fetch_add(1, AtomicOrdering::SeqCst);
        }
    }

    const SIZE_OF_COUNTDROPS: usize = mem::size_of::<CountDrops>();

    let mut arena = Arena::with_block_size(SIZE_OF_COUNTDROPS);

    {
        let h0 = Handle::new_in(&arena, CountDrops(21));
        let h1 = Handle::new_in(&arena, CountDrops(21));

        assert_eq!(&*h0, &*h1);
        assert_ne!(
            Handle::as_ptr(&h1).addr(),
            Handle::as_ptr(&h0).addr() + SIZE_OF_COUNTDROPS
        );
        assert_eq!(DROP_COUNT.load(AtomicOrdering::SeqCst), 0);
    }

    assert_eq!(DROP_COUNT.load(AtomicOrdering::SeqCst), 2);
    arena.reset();
    assert_eq!(DROP_COUNT.load(AtomicOrdering::SeqCst), 2);
}

#[test]
fn test_arena_slice() {
    use std::convert::identity;
    const LEN: usize = 210;

    let arena = Arena::new();

    let array = Handle::new_slice_with_fn_in(&arena, LEN, identity);
    assert_eq!(array.len(), LEN);

    for (idx, elem) in array.iter().enumerate() {
        assert_eq!(idx, *elem);
    }

    let (lhs, rhs) = Handle::split_at(array, LEN / 2);
    for (i, val) in lhs.iter().enumerate() {
        assert_eq!(*val, i);
    }

    for (i, val) in rhs.iter().enumerate() {
        assert_eq!(*val, i + (LEN / 2));
    }

    let mut handle = Handle::<[i32]>::default();
    assert!(handle.is_empty());

    handle = Handle::<[i32]>::new_slice_from_iter_in(&arena, [1, 2, 3, 4, 5]);
    assert_eq!(&*handle, &[1, 2, 3, 4, 5]);

    handle = Handle::empty();
    assert!(handle.is_empty());

    let handle = Handle::<str>::default();
    assert!(handle.is_empty());
}

#[test]
fn test_zst() {
    let arena = Arena::new();

    let mut buf = Buffer::with_capacity_in(&arena, 91);
    for _ in 0..21 {
        buf.push(Handle::new_in(&arena, ()));
    }

    assert_eq!(buf.len(), 21);

    let first = buf.first().unwrap();
    let addr = Handle::as_ptr(first).addr();

    for item in buf.iter().skip(1) {
        assert_eq!(Handle::as_ptr(item).addr(), addr);
    }
}

#[test]
fn test_trim() {
    let mut arena = Arena::with_block_size(mem::size_of::<i32>());

    {
        let handle = Handle::new_in(&arena, 21);
        let handle_2 = Handle::new_in(&arena, 32);

        assert_eq!(*handle, 21);
        assert_eq!(*handle_2, 32);
    }

    assert_ne!(arena.blocks.used_blocks().get(), None);
    assert_eq!(arena.blocks.free_blocks().get(), None);

    arena.reset();
    assert_eq!(arena.blocks.used_blocks().get(), None);
    assert_ne!(arena.blocks.free_blocks().get(), None);

    arena.trim();
    assert_eq!(arena.blocks.free_blocks().get(), None);

    const N: usize = 150;
    for i in 0..N {
        arena.reserve_blocks(N);

        arena.trim_n(i);
        assert_eq!(arena.free_blocks().count(), N.saturating_sub(i));

        arena.trim();
    }
}

#[test]
fn test_scoped() {
    let arena = Arena::with_block_size(mem::size_of::<u64>());

    let handle = Handle::new_in(&arena, 25i32);
    let handle_2 = Handle::new_in(&arena, 25i64);

    let handle_3_ptr = arena.with_scope(|arena| {
        let handle_3 = Handle::new_in(&arena, 45i64);
        Handle::as_ptr(&handle_3)
    });

    arena.with_scope(move |arena| {
        let handle_4 = Handle::new_in(&arena, 5i64);
        let handle_4_ptr = Handle::as_ptr(&handle_4);

        assert!(ptr::eq(handle_3_ptr, handle_4_ptr));
    });

    arena.with_scope(|arena| {
        let first = Handle::new_in(&arena, 255i32);
        let second = Handle::new_in(&arena, 132u64);

        unsafe {
            arena.with_scope_dynamic(|| {
                let value = Handle::new_in(&arena, *first as u64 + *second);
                assert_eq!(*value, 387);
                let _value = Handle::new_in(&arena, 510u16);
                arena.reserve_blocks(2);
            });
        }

        let _third = Handle::new_in(&arena, 432);
    });

    assert_eq!(*handle, 25);
    assert_eq!(*handle_2, 25);

    drop(handle);
    drop(handle_2);
    drop(arena);

    let arena = Arena::new();
    const SIZE: usize = 124;
    let _value = Handle::new_in(&arena, [0u8; SIZE]);

    assert_eq!(arena.blocks.curr_block_pos().get(), SIZE);
    arena.with_scope(|arena| {
        let _slice_handle = Handle::new_slice_splat_in(&arena, SIZE, [0u8; 8]);
        assert_eq!(
            arena.blocks.curr_block_pos().get(),
            SIZE + (SIZE * mem::size_of::<[u8; 8]>())
        );
        arena.reserve_blocks(2);
    });

    assert_eq!(arena.blocks.curr_block_pos().get(), SIZE);

    let mut value = Handle::new_uninit_in(&arena);
    unsafe {
        arena.with_scope_dynamic(|| {
            let val_1 = Handle::new_in(&arena, 2);
            let val_2 = Handle::new_in(&arena, 4);
            value.as_mut().write(*val_1 + *val_2);
        });
    }

    let value = unsafe { Handle::assume_init(value) };
    let _data = Handle::new_in(&arena, [0i32; 16]);
    assert_eq!(*value, 6);
}

#[test]
fn test_multithreaded() {
    use std::thread;

    let arena = Arena::new();
    let value = Handle::new_in(&arena, 21);

    let _ = thread::scope(|scope| {
        let value_2 = Handle::new_in(&arena, 21);
        scope.spawn({
            let value = &value;
            move || {
                assert_eq!(**value + *value_2, 42);
            }
        });
    });

    assert_eq!(*value, 21);
    drop(value);

    let send_arena = thread::spawn(|| {
        let _arena = arena;
    });

    send_arena.join().unwrap();
}

#[test]
fn test_rc() {
    use std::string::String;
    let arena = Arena::new();

    arena.with_scope(|arena| {
        use std::convert::identity;

        fn take_rc(hndl: RcHandle<'_, i32>) {
            assert_eq!(*hndl, 21);
        }

        let sv_1 = RcHandle::new_in(&arena, 21);
        take_rc(sv_1.clone());
        assert_eq!(*sv_1, 21);

        const LEN: usize = 102;
        let sv_2 = RcHandle::new_slice_from_fn_in(&arena, LEN, identity);
        assert_eq!(sv_2.len(), LEN);
        for (i, item) in sv_2.iter().enumerate() {
            assert_eq!(i, *item);
        }
    });

    arena.with_scope(|arena| {
        let mut weak = WeakHandle::<'_, i32>::new();
        {
            let ptr = weak.clone().into_raw();
            let weak_2 = unsafe { WeakHandle::from_raw(ptr) };
            assert_eq!(weak_2.upgrade(), None);
        }

        let mut weak_2 = WeakHandle::new();
        assert_eq!(WeakHandle::ref_count(&weak), 0);

        {
            let value = RcHandle::new_in(&arena, 32);
            weak = RcHandle::downgrade(&value);
            assert_eq!(WeakHandle::ref_count(&weak), 1);

            assert!(WeakHandle::ptr_eq(&weak, &value));
            assert!(RcHandle::ptr_eq(&value, &weak));
            assert!(RcHandle::ptr_eq(&value, &value));
            assert!(WeakHandle::ptr_eq(&weak, &weak));

            assert_eq!(weak.upgrade().map(|hndl| *hndl), Some(32));
            let value_2 = RcHandle::new_in(&arena, 21);

            assert_eq!(WeakHandle::ref_count(&weak_2), 0);
            weak_2 = RcHandle::downgrade(&value_2);
            assert_eq!(WeakHandle::ref_count(&weak_2), 1);

            assert!(!WeakHandle::ptr_eq(&weak_2, &value));
            assert!(!RcHandle::ptr_eq(&value_2, &weak));
            assert!(!RcHandle::ptr_eq(&value, &value_2));
            assert!(!WeakHandle::ptr_eq(&weak, &weak_2));

            let inner = RcHandle::try_unwrap(value_2).expect("not unique");
            assert_eq!(inner, 21);
        }

        assert_eq!(WeakHandle::ref_count(&weak), 0);
        assert!(WeakHandle::upgrade(&weak).is_none());

        assert_eq!(WeakHandle::ref_count(&weak_2), 0);
        assert!(WeakHandle::upgrade(&weak_2).is_none());
    });

    static DROP_COUNT: AtomicU32 = AtomicU32::new(0);

    #[derive(Debug, Eq, PartialEq)]
    struct CountDrops(u32);

    impl Drop for CountDrops {
        fn drop(&mut self) {
            DROP_COUNT.fetch_add(1, AtomicOrdering::SeqCst);
        }
    }

    arena.with_scope(|arena| {
        let sv_3 = RcHandle::new_in(&arena, CountDrops(21));
        assert_eq!(DROP_COUNT.load(AtomicOrdering::SeqCst), 0);

        let values = Handle::new_slice_with_fn_in(&arena, 21, |_| sv_3.clone());

        assert_eq!(DROP_COUNT.load(AtomicOrdering::SeqCst), 0);
        drop(values);
        assert_eq!(DROP_COUNT.load(AtomicOrdering::SeqCst), 0);

        drop(sv_3);
        assert_eq!(DROP_COUNT.load(AtomicOrdering::SeqCst), 1);
    });

    arena.with_scope(|arena| {
        let rc_handle = RcHandle::new_in(&arena, String::from("Hello!"));
        let weak = RcHandle::downgrade(&rc_handle);

        {
            let handle_2 = rc_handle.clone();
            assert!(RcHandle::try_into_handle(handle_2).is_err());
        }

        {
            let handle = RcHandle::into_handle(rc_handle).unwrap();
            assert_eq!(*handle, "Hello!");
            assert!(weak.upgrade().is_none());
        }

        assert!(weak.upgrade().is_none());
    });
}

#[test]
fn test_dyn() {
    use std::{fmt::Display, string::ToString};

    let arena = Arena::new();

    const VALUE: usize = 203;
    let value = Handle::new_in(&arena, VALUE) as Handle<'_, dyn Display>;
    let shared = RcHandle::new_in(&arena, VALUE) as RcHandle<'_, dyn Display>;

    assert_eq!(value.to_string(), shared.to_string());

    let value = Handle::new_in(&arena, [VALUE; VALUE]) as Handle<'_, [usize]>;
    let shared = RcHandle::new_in(&arena, [VALUE; VALUE]) as RcHandle<'_, [usize]>;

    assert_eq!(*value, *shared);
}

#[test]
fn test_buffer() {
    let arena = Arena::new();

    let mut buf = Buffer::with_capacity_in(&arena, 10);
    assert_eq!(buf.pop(), None);
    buf.extend([1, 2, 3, 4, 5].into_iter());
    assert_eq!(buf.try_remove(5), None);

    let value = buf.remove(2);
    assert_eq!(value, 3);
    assert_eq!(buf.len(), 4);
    assert_eq!(&*buf, &[1, 2, 4, 5]);

    assert_eq!(buf.pop(), Some(5));

    buf.extend([5, 6, 7, 8].into_iter());
    assert_eq!(buf.swap_remove(0), Some(1));
    assert_eq!(&*buf, [8, 2, 4, 5, 6, 7]);

    buf.clear();
    assert_eq!(buf.len(), 0);
    assert_eq!(buf.pop(), None);
    assert_eq!(buf.try_remove(0), None);

    let buffer = buf!([0u8; 5] in arena);
    for item in &buffer {
        assert_eq!(*item, 0);
    }

    let buffer = buf!(in arena; [0usize, 1, 2, 3, 4, 5]);
    for (i, item) in buffer.iter().enumerate() {
        assert_eq!(i, *item);
    }

    let starting_len = buffer.len();
    let mut iter = buffer.into_iter();
    assert_eq!(iter.as_slice(), &[0, 1, 2, 3, 4, 5]);
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next_back(), Some(5));
    assert_eq!(iter.as_slice(), &[2, 3, 4]);

    let mut buffer = iter.into_buffer();
    assert_eq!(&*buffer, &[2, 3, 4]);
    assert_eq!(buffer.capacity(), starting_len);
    assert_eq!(buffer.len(), 3);

    buffer.truncate(2);
    assert_eq!(&*buffer, &[2, 3]);

    drop(buffer);

    let mut buffer = Buffer::with_capacity_in(&arena, 5);
    let handle = Handle::new_in(&arena, 25);
    buffer.extend_from_slice_copy(&[1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(&*buffer, &[1, 2, 3, 4, 5]);
    assert_eq!(*handle, 25);

    buffer.clear();
    buffer.extend_from_slice_copy(&[1, 2]);
    buffer.extend_from_slice_copy(&[3, 4]);
    assert_eq!(&*buffer, &[1, 2, 3, 4]);

    buffer.clear();
    buffer.extend_from_slice(&[1, 2]);
    buffer.extend_from_slice(&[3, 4]);
    buffer.extend_from_slice(&[5, 6]);
    assert_eq!(&*buffer, &[1, 2, 3, 4, 5]);

    let mut buffer = Buffer::<i32>::new();
    let handle = Handle::new_in(&arena, 42);

    if *handle == 42 {
        buffer = Buffer::with_capacity_in(&arena, 40);
        buffer.push(1804);
    }

    assert_eq!(buffer.capacity(), 40);
    assert_eq!(*handle, 42);

    buffer.clear();
    buffer.push(45);

    let mut into_iter = buffer.into_iter();
    assert_eq!(into_iter.next(), Some(45));
    assert_eq!(into_iter.next(), None);

    let buffer = into_iter.into_buffer();
    assert_eq!(buffer.len(), 0);
}

#[test]
#[should_panic]
fn test_buf_overflow() {
    const CAP: usize = 10;
    let arena = Arena::new();
    let mut buf = Buffer::with_capacity_in(&arena, CAP);

    let data = [0i32; CAP + 1];
    buf.extend(data.iter().copied());
}

#[test]
fn test_custom_alloc() {
    struct CustomAllocator(System);

    unsafe impl Allocator for CustomAllocator {
        #[inline]
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
            self.0.allocate(layout)
        }

        #[inline]
        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            unsafe { self.0.deallocate(ptr, layout) }
        }
    }

    let arena = Arena::<CustomAllocator>::new_in(CustomAllocator(System));
    let value = Handle::new_in(&arena, 5usize);
    assert_eq!(*value, 5);
}

#[test]
fn test_string_buffer() {
    let arena = Arena::new();

    arena.with_scope(|arena| {
        let mut string_buf = StringBuffer::with_capacity_in_arena(200, &arena);

        string_buf.push_str("Lorem ipsum");
        string_buf.push_str(" dolor sit");
        string_buf.push_str(" amet");

        assert_eq!(&*string_buf, "Lorem ipsum dolor sit amet");

        string_buf.clear();

        string_buf.push_str("Café 世界");
        assert_eq!(&*string_buf, "Café 世界");
    });
}

#[test]
fn test_list() {
    let arena = Arena::new();

    arena.with_scope(|arena| {
        let mut list = LinkedList::new(&arena);
        list.extend([1, 2, 3, 4, 5]);
        list.swap(0, 1);
        assert_eq!(list, [2, 1, 3, 4, 5]);

        list.swap(2, 3);
        assert_eq!(list, [2, 1, 4, 3, 5]);
        list.swap(3, 4);
        assert_eq!(list, [2, 1, 4, 5, 3]);
    });

    arena.with_scope(|arena| {
        let mut list = LinkedList::new(&arena);
        list.extend([1usize, 2, 3, 4, 5]);

        for (i, elem) in list.iter().copied().enumerate() {
            assert_eq!(i + 1, elem);
        }

        list.insert(1, 23);

        assert_eq!(list.get(0), Some(&1));
        assert_eq!(list.get(1), Some(&23));
        assert_eq!(list.get(2), Some(&2));
        assert_eq!(list.get(4), Some(&4));
        assert_eq!(list.get(list.len() - 1), list.back());

        let removed = list.remove(1);
        assert_eq!(removed.as_deref(), Some(&23));
        assert_eq!(list.get(0), Some(&1));
        assert_eq!(list.get(1), Some(&2));
        assert_eq!(list.get(4), Some(&5));

        list.clear();
        assert!(list.is_empty());
    });

    arena.with_scope(|arena| {
        let mut list = LinkedList::new(arena);
        const DATA: &'static [usize] = &[1, 2, 3, 4];
        list.extend(DATA.into_iter().copied());
        list.reverse();
        list.reverse();
        assert_eq!(list, DATA);
    });

    arena.with_scope(|arena| {
        let mut list = LinkedList::new(&arena);
        list.extend([1usize, 3, 5]);
        list.swap(0, 2);

        assert_eq!(list.front(), Some(&5));
        assert_eq!(list.back(), Some(&1));

        let back = list.pop_back();
        assert_eq!(back.as_deref(), Some(&1));
        assert_eq!(list, [5usize, 3]);
        list.reverse();
        assert_eq!(list, [3usize, 5]);

        let front = list.pop_front();
        assert_eq!(front.as_deref(), Some(&3));
    });

    arena.with_scope(|arena| {
        let mut list = LinkedList::new(&arena);
        const NUM: usize = 15;

        let mut data = (0usize..=NUM).collect::<alloc::vec::Vec<_>>();
        list.extend(data.iter().cloned());

        {
            let back = list.back();
            let front = list.front();

            let last = list.iter().next_back();
            let first = list.iter().rev().nth(NUM);

            assert_eq!(back, last);
            assert_eq!(front, first);
        }

        for (x, y) in list.iter().rev().zip(data.iter().rev()) {
            assert_eq!(x, y);
        }

        list.reverse();
        data.reverse();

        assert_eq!(&list, &data[..]);
        for (lhs, rhs) in list.iter().zip(data.iter()) {
            assert_eq!(lhs, rhs);
        }

        assert_eq!(list.get(NUM - 1), Some(&1));
        assert_eq!(list.front(), Some(&NUM));
        assert_eq!(list.back(), Some(&0));

        list.reverse();
        list.reverse();

        assert_eq!(&list, &data[..]);
    });

    arena.with_scope(|arena| {
        let mut list = LinkedList::new(&arena);
        list.extend([1, 2, 3, 4, 5]);

        let reversed = list.iter().rev().cloned().collect::<std::vec::Vec<_>>();
        assert_eq!(reversed.as_slice(), &[5, 4, 3, 2, 1]);
    });
}

#[test]
fn test_list_split() {
    let arena = Arena::new();

    const LIST_LEN: usize = 25;
    for i in 0..=LIST_LEN {
        arena.with_scope(move |arena| {
            let mut list = LinkedList::from_iter_in(arena, 1..=LIST_LEN);
            let end = list.split_off(i);

            if i > 0 {
                assert_eq!(list, &(1..=i).collect::<alloc::vec::Vec<_>>()[..]);
            } else {
                assert!(list.is_empty());
            }
            assert_eq!(end, &(i + 1..=LIST_LEN).collect::<alloc::vec::Vec<_>>()[..]);
        });
    }
}

#[test]
fn test_list_pop() {
    let arena = Arena::new();

    arena.with_scope(|arena| {
        let mut list = LinkedList::new(arena);
        list.push_front(21);
        list.push_front(32);

        let front = list.pop_front().unwrap();
        assert_eq!(*front, 32);
    });
}

#[test]
fn test_list_retain() {
    let arena = Arena::new();

    let mut list = LinkedList::from_iter_in(&arena, 0..10);
    list.retain(|elem| elem % 2 == 0);

    assert_eq!(list, [0, 2, 4, 6, 8]);
}

#[test]
fn test_growable_buffer() {
    let arena = Arena::with_block_size(5 * mem::size_of::<i32>());

    let buffer = Buffer::with_growable_in(&arena, |buffer| {
        assert!(buffer.max_capacity() == 5);

        buffer.reserve(4);
        buffer.extend([25, 42, 180]);
        assert_eq!(buffer.as_slice(), &[25, 42, 180]);
    });

    assert_eq!(buffer, [25, 42, 180]);
    assert_eq!(buffer.capacity(), 4);

    static COUNT_DROPS: AtomicUsize = AtomicUsize::new(0);
    #[derive(Debug)]
    struct CountDrops(#[allow(unused)] u8);

    impl Drop for CountDrops {
        fn drop(&mut self) {
            COUNT_DROPS.fetch_add(1, AtomicOrdering::SeqCst);
        }
    }

    let prev_head = arena.curr_block_head().unwrap();
    let buffer_result = Buffer::try_with_growable_in(&arena, |buffer| {
        assert_eq!(buffer.max_capacity(), 4);
        buffer.extend([CountDrops(0), CountDrops(0)]);

        buffer.clear();
        assert_eq!(COUNT_DROPS.load(AtomicOrdering::SeqCst), 2);
        assert_eq!(buffer.len(), 0);

        buffer.extend([CountDrops(0), CountDrops(0)]);

        Err(())
    });

    assert_eq!(COUNT_DROPS.load(AtomicOrdering::SeqCst), 4);
    buffer_result.unwrap_err();

    let curr_head = arena.curr_block_head().unwrap();
    assert!(ptr::eq(prev_head.as_ptr(), curr_head.as_ptr()));

    let buffer_result =
        Buffer::<'_, u64>::try_with_growable_in(&arena, |buffer| buffer.try_reserve(50));
    assert!(buffer_result.is_err(),);

    let arena = Arena::new();
    let buffer = Buffer::with_growable_in(&arena, |buffer| {
        buffer.reserve(4);
        buffer.extend([1usize, 2]);
        buffer.shrink_to_fit();
    });

    assert_eq!(buffer.capacity(), 2);

    let arena = Rc::new(Arena::new());
    let arena_2 = Rc::clone(&arena);

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        Buffer::with_growable_in(arena.as_ref(), move |buffer| {
            buffer.push(24);
            buffer.push(51);
            let handle = Handle::new_in(arena_2.as_ref(), 45);

            // None of this should be executed due to the above statement `panic`ing.
            buffer.push(64);
            assert_eq!(handle, &45);
            let _handle = Handle::new_in(arena_2.as_ref(), 62);
        })
    }));

    assert!(result.is_err());

    let data = alloc::vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    let buffer = Buffer::with_growable_in(&arena, |buf| {
        buf.extend(data.iter().take(5).copied());
    });

    let buffer_2 = Buffer::with_growable_in(&arena, |buf| {
        buf.extend(data.iter().skip(5).take(5).copied());
    });

    assert_eq!(buffer.as_slice(), [1, 2, 3, 4, 5]);
    assert_eq!(buffer_2.as_slice(), [6, 7, 8, 9, 10]);
}
