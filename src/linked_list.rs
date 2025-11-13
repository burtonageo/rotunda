use crate::{Arena, handle::Handle};
use alloc::alloc::{Allocator, Global, Layout};
use core::{
    fmt,
    iter::{DoubleEndedIterator, FusedIterator, Iterator},
    marker::PhantomData,
    mem,
    ptr::{self, NonNull},
};

pub struct LinkedList<'a, T: 'a, A: Allocator = Global> {
    head: NonNull<Node<T>>,
    tail: NonNull<Node<T>>,
    len: usize,
    arena: &'a Arena<A>,
    _boo: PhantomData<(T, fn(&'a Arena) -> &'a Arena)>,
}

// A LinkedList can be sent to other threads if the type within is
// thread-safe - it is guaranteed to drop before the arena is
// deallocated thanks to borrowing rules.
unsafe impl<'a, T: Send> Send for LinkedList<'a, T> {}
unsafe impl<'a, T: Sync> Sync for LinkedList<'a, T> {}

impl<'a, T: 'a, A: Allocator> LinkedList<'a, T, A> {
    #[must_use]
    #[inline]
    pub const fn new(arena: &'a Arena<A>) -> Self {
        Self {
            head: NonNull::dangling(),
            tail: NonNull::dangling(),
            len: 0,
            arena,
            _boo: PhantomData,
        }
    }

    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    #[inline]
    pub fn arena(&self) -> &Arena<A> {
        &self.arena
    }

    #[inline]
    pub fn push_front(&mut self, value: T) {
        self.insert(0, value);
    }

    #[must_use]
    #[inline]
    pub fn push_front_mut(&mut self, value: T) -> &mut T {
        self.insert_mut(0, value)
    }

    #[inline]
    pub fn pop_front(&mut self) -> Option<T> {
        self.remove(0)
    }

    #[inline]
    pub fn push_back(&mut self, value: T) {
        self.insert(self.len, value);
    }

    #[must_use]
    #[inline]
    pub fn push_back_mut(&mut self, value: T) -> &mut T {
        self.insert_mut(self.len, value)
    }

    #[inline]
    pub fn pop_back(&mut self) -> Option<T> {
        self.remove(self.len.saturating_sub(1))
    }

    #[inline]
    pub fn insert(&mut self, index: usize, value: T) {
        let _ = self.insert_mut(index, value);
    }

    #[track_caller]
    #[must_use]
    #[inline]
    pub fn insert_mut(&mut self, index: usize, value: T) -> &mut T {
        assert!(index <= self.len, "index out of bounds");

        let mut node_ptr = self
            .arena
            .alloc_raw(Layout::new::<Node<T>>())
            .cast::<Node<T>>();

        unsafe {
            ptr::write(&raw mut node_ptr.as_mut().next, NonNull::dangling());
            ptr::write(&raw mut node_ptr.as_mut().prev, NonNull::dangling());
            ptr::write(&raw mut node_ptr.as_mut().data, value);
        }

        self.insert_node(index, node_ptr);
        unsafe { &mut node_ptr.as_mut().data }
    }

    #[inline]
    pub fn remove(&mut self, index: usize) -> Option<T> {
        self.remove_node(index)
            .map(|node| unsafe { ptr::read(&node.as_ref().data) })
    }

    #[inline]
    pub fn swap(&mut self, mut first_index: usize, mut second_index: usize) {
        assert!(first_index < self.len, "index out of bounds");
        assert!(second_index < self.len, "index out of bounds");

        if first_index == second_index {
            return;
        } else if first_index > second_index {
            mem::swap(&mut first_index, &mut second_index);
        }

        let second_node = unsafe { self.remove_node(second_index).unwrap_unchecked() };
        self.insert_node(first_index, second_node);

        let first_node = unsafe { self.remove_node(first_index + 1).unwrap_unchecked() };
        self.insert_node(second_index, first_node);
    }

    #[inline]
    pub fn reverse(&mut self) {
        let n = self.len / 2;

        for i in 0..n {
            let inv = self.len.saturating_sub(1).saturating_sub(i);
            self.swap(i, inv);
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        let arena = self.arena;
        let _ = mem::replace(self, Self::new(arena));
    }

    #[must_use]
    #[inline]
    pub fn front(&self) -> Option<&T> {
        if !self.is_empty() {
            unsafe { Some(&self.head.as_ref().data) }
        } else {
            None
        }
    }

    #[must_use]
    #[inline]
    pub fn front_mut(&mut self) -> Option<&mut T> {
        if !self.is_empty() {
            unsafe { Some(&mut self.head.as_mut().data) }
        } else {
            None
        }
    }

    #[must_use]
    #[inline]
    pub fn back(&self) -> Option<&T> {
        if !self.is_empty() {
            unsafe { Some(&self.tail.as_ref().data) }
        } else {
            None
        }
    }

    #[must_use]
    #[inline]
    pub fn back_mut(&mut self) -> Option<&mut T> {
        if !self.is_empty() {
            unsafe { Some(&mut self.tail.as_mut().data) }
        } else {
            None
        }
    }

    #[must_use]
    #[inline]
    pub fn get(&'_ self, index: usize) -> Option<&'_ T> {
        self.get_node(index)
            .map(|node| unsafe { &node.as_ref().data })
    }

    #[must_use]
    #[inline]
    pub fn get_mut(&'_ mut self, index: usize) -> Option<&'_ mut T> {
        self.get_node(index)
            .map(|mut node| unsafe { &mut node.as_mut().data })
    }

    #[allow(unused, dead_code)]
    #[track_caller]
    #[inline]
    pub fn retain<F: FnMut(&T) -> bool>(&mut self, pred: F) {
        todo!()
    }

    #[must_use]
    #[inline]
    pub fn iter(&self) -> Iter<'a, T> {
        Iter {
            iter: NodeIter::new(&self),
            _boo: PhantomData,
        }
    }

    #[must_use]
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'a, T> {
        IterMut {
            iter: NodeIter::new(&self),
            _boo: PhantomData,
        }
    }

    #[must_use]
    #[inline]
    fn get_node(&self, index: usize) -> Option<NonNull<Node<T>>> {
        if index >= self.len {
            return None;
        }

        unsafe { Some(self.get_node_unchecked(index)) }
    }

    #[must_use]
    #[inline]
    unsafe fn get_node_unchecked(&self, index: usize) -> NonNull<Node<T>> {
        let mut iter = NodeIter::new(self);
        // @TODO(George): Iter from the end of the list if index >= self.len / 2
        let node = iter.nth(index);

        unsafe { node.unwrap_unchecked() }
    }

    #[track_caller]
    #[inline]
    fn insert_node(&mut self, index: usize, mut node_ptr: NonNull<Node<T>>) {
        match index {
            0 => {
                if !self.is_empty() {
                    let mut old_head = self.head;
                    unsafe {
                        node_ptr.as_mut().next = old_head;
                        old_head.as_mut().prev = node_ptr;
                    }
                } else {
                    self.tail = node_ptr;
                }

                self.head = node_ptr;
            }
            _ if index >= self.len => {
                unsafe {
                    let mut old_tail = self.tail;
                    node_ptr.as_mut().prev = old_tail;
                    old_tail.as_mut().next = node_ptr;
                }

                self.tail = node_ptr;
            }
            _ => unsafe {
                let mut before_node_ptr = self.get_node_unchecked(index.saturating_sub(1));
                let before_node = before_node_ptr.as_mut();

                let mut after_node_ptr = before_node.next;
                let after_node = after_node_ptr.as_mut();

                before_node.next = node_ptr;
                after_node.prev = node_ptr;

                let node = node_ptr.as_mut();

                node.prev = before_node_ptr;
                node.next = after_node_ptr;
            },
        }

        self.len = self.len.checked_add(1).expect("list overflow");
    }

    #[inline]
    fn remove_node(&mut self, index: usize) -> Option<NonNull<Node<T>>> {
        let mut node = self.get_node(index)?;
        unsafe {
            let node = node.as_mut();
            self.len = self.len.checked_sub(1).expect("list underflow");

            if self.len > 0 {
                node.prev.as_mut().next = node.next;
            }

            if self.len < index {
                node.next.as_mut().prev = node.prev;
            }

            if index == 0 {
                self.head = node.next;
            } else if index == self.len {
                self.tail = node.prev;
            }
        }

        Some(node)
    }
}

impl<'a, T: 'a, A: Allocator> Drop for LinkedList<'a, T, A> {
    #[inline]
    fn drop(&mut self) {
        for item in self {
            unsafe {
                ptr::drop_in_place(item);
            }
        }
    }
}

impl<'a, T: fmt::Debug, A: Allocator> fmt::Debug for LinkedList<'a, T, A> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmtr.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, T: 'a, A: Allocator> From<&'a Arena<A>> for LinkedList<'a, T, A> {
    #[inline]
    fn from(arena: &'a Arena<A>) -> Self {
        Self::new(arena)
    }
}

impl<'a, T: 'a + PartialEq, A: Allocator> PartialEq for LinkedList<'a, T, A> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }

        self.iter()
            .zip(other.iter())
            .all(|(lhs, rhs)| PartialEq::eq(lhs, rhs))
    }
}

impl<'a, T: 'a + Eq, A: Allocator> Eq for LinkedList<'a, T, A> {}

impl<'a, T: 'a + PartialEq, A: Allocator> PartialEq<[T]> for LinkedList<'a, T, A> {
    #[inline]
    fn eq(&self, other: &[T]) -> bool {
        if self.len != other.len() {
            return false;
        }

        self.iter()
            .zip(other.iter())
            .all(|(lhs, rhs)| PartialEq::eq(lhs, rhs))
    }
}

impl<'a, T: 'a + PartialEq, A: Allocator, const N: usize> PartialEq<[T; N]>
    for LinkedList<'a, T, A>
{
    #[inline]
    fn eq(&self, other: &[T; N]) -> bool {
        PartialEq::eq(self, &other[..])
    }
}

impl<'a, T: 'a, A: Allocator> IntoIterator for &'_ LinkedList<'a, T, A> {
    type IntoIter = Iter<'a, T>;
    type Item = &'a T;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: 'a, A: Allocator> IntoIterator for &'_ mut LinkedList<'a, T, A> {
    type IntoIter = IterMut<'a, T>;
    type Item = &'a mut T;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'a, T: 'a + Unpin, A: Allocator> IntoIterator for LinkedList<'a, T, A> {
    type IntoIter = IntoIter<'a, T, A>;
    type Item = Handle<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter { list: self }
    }
}

impl<'a, T: 'a, A: Allocator> Extend<T> for LinkedList<'a, T, A> {
    #[track_caller]
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.push_back(item);
        }
    }
}

pub struct Iter<'a, T> {
    iter: NodeIter<T>,
    _boo: PhantomData<&'a Node<T>>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|node| unsafe { &node.as_ref().data })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T> ExactSizeIterator for Iter<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, T> FusedIterator for Iter<'a, T> {}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter
            .next_back()
            .map(|node| unsafe { &node.as_ref().data })
    }
}

impl<'a, T> Clone for Iter<'a, T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
            _boo: PhantomData,
        }
    }
}

impl<'a, T: fmt::Debug> fmt::Debug for Iter<'a, T> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.iter, fmtr)
    }
}

pub struct IterMut<'a, T> {
    iter: NodeIter<T>,
    _boo: PhantomData<&'a mut Node<T>>,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|mut node| unsafe { &mut node.as_mut().data })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T> ExactSizeIterator for IterMut<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, T> FusedIterator for IterMut<'a, T> {}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter
            .next_back()
            .map(|mut node| unsafe { &mut node.as_mut().data })
    }
}

impl<'a, T: fmt::Debug> fmt::Debug for IterMut<'a, T> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.iter, fmtr)
    }
}

pub struct IntoIter<'a, T: 'a, A: Allocator> {
    list: LinkedList<'a, T, A>,
}

impl<'a, T: 'a, A: Allocator> IntoIter<'a, T, A> {
    #[must_use]
    #[inline]
    pub fn into_list(self) -> LinkedList<'a, T, A> {
        self.list
    }
}

impl<'a, T: 'a + fmt::Debug, A: Allocator> fmt::Debug for IntoIter<'a, T, A> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.list, fmtr)
    }
}

impl<'a, T: 'a, A: Allocator> Iterator for IntoIter<'a, T, A> {
    type Item = Handle<'a, T>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.list.remove_node(0).map(|mut node| unsafe {
            self.list.len -= 1;
            Handle::from_raw(&mut node.as_mut().data)
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<'a, T: 'a, A: Allocator> ExactSizeIterator for IntoIter<'a, T, A> {
    #[inline]
    fn len(&self) -> usize {
        self.list.len
    }
}

impl<'a, T: 'a, A: Allocator> FusedIterator for IntoIter<'a, T, A> {}

impl<'a, T: 'a, A: Allocator> DoubleEndedIterator for IntoIter<'a, T, A> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.list
            .remove_node(self.list.len.saturating_sub(1))
            .map(|mut node| unsafe { Handle::from_raw(&mut node.as_mut().data) })
    }
}

struct NodeIter<T> {
    head: NonNull<Node<T>>,
    tail: NonNull<Node<T>>,
    len: usize,
}

impl<T> Clone for NodeIter<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for NodeIter<T> {}

impl<T> NodeIter<T> {
    #[must_use]
    #[inline]
    const fn new<'a, A: Allocator>(list: &LinkedList<'a, T, A>) -> Self {
        Self {
            head: list.head,
            tail: list.tail,
            len: list.len,
        }
    }
}

impl<T> Iterator for NodeIter<T> {
    type Item = NonNull<Node<T>>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            return None;
        }

        let head = self.head;

        unsafe {
            self.head = head.as_ref().next;
        }

        self.len -= 1;

        Some(head)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<T> ExactSizeIterator for NodeIter<T> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

impl<T> DoubleEndedIterator for NodeIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            return None;
        }

        let tail = self.tail;

        unsafe {
            self.tail = tail.as_ref().prev;
        }

        self.len -= 1;

        Some(tail)
    }
}

impl<T: fmt::Debug> fmt::Debug for NodeIter<T> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmtr.debug_list().entries(*self).finish()
    }
}

#[allow(dead_code, unused)]
struct CursorMut<'a, T> {
    head: Option<NonNull<Node<T>>>,
    tail: Option<NonNull<Node<T>>>,
    curr: Option<NonNull<Node<T>>>,
    len: usize,
    _boo: PhantomData<&'a mut Node<T>>,
}

#[allow(dead_code, unused)]
impl<'a, T> CursorMut<'a, T> {
    fn move_next(&mut self) {}

    fn move_prev(&mut self) {}

    fn current(&mut self) -> Option<&mut T> {
        todo!()
    }

    fn remove_current(&mut self) -> Option<Handle<'a, T>> {
        todo!()
    }
}

struct Node<T> {
    next: NonNull<Node<T>>,
    prev: NonNull<Node<T>>,
    data: T,
}
