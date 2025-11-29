//! A doubly-linked owned list of nodes.
//!
//! The `LinkedList` allows pushing and popping elements at either end of the list in constant time.

use crate::{Arena, handle::Handle};
use alloc::alloc::{Allocator, Global, Layout};
use core::{
    fmt,
    iter::{DoubleEndedIterator, FusedIterator, Iterator},
    marker::PhantomData,
    mem::{self, offset_of},
    ptr::{self, NonNull},
};

/// A doubly-linked list type, backed by an [`Arena`].
///
/// See the [module documentation] for more info.
///
/// [`Arena`]: ../struct.Arena.html
/// [module documentation]: ./index.html
pub struct LinkedList<'a, T: 'a, A: Allocator = Global> {
    head: NonNull<Node<T>>,
    tail: NonNull<Node<T>>,
    len: usize,
    arena: &'a Arena<A>,
    _boo: PhantomData<(T, fn(&'a Arena<A>) -> &'a Arena<A>)>,
}

// A LinkedList can be sent to other threads if the type within is
// thread-safe - it is guaranteed to drop before the arena is
// deallocated thanks to borrowing rules.
unsafe impl<'a, T: Send> Send for LinkedList<'a, T> {}
unsafe impl<'a, T: Sync> Sync for LinkedList<'a, T> {}

impl<'a, T: 'a, A: Allocator> LinkedList<'a, T, A> {
    /// Create an empty `LinkedList` backed by the given `Arena`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, linked_list::LinkedList};
    /// let arena = Arena::new();
    ///
    /// let linked_list = LinkedList::<i32>::new(&arena);
    /// ```
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
    pub fn new_from_iter_in<I: IntoIterator<Item = T>>(arena: &'a Arena<A>, iter: I) -> Self {
        let mut list = Self::new(arena);
        list.extend(iter);
        list
    }

    /// Returns the number of elements in the `LinkedList`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    /// let mut linked_list = LinkedList::new(&arena);
    ///
    /// linked_list.push_back(24);
    ///
    /// assert_eq!(linked_list.len(), 1);
    /// ```
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if there are no elements in the `LinkedList`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    /// let mut linked_list = LinkedList::new(&arena);
    ///
    /// assert!(linked_list.is_empty());
    ///
    /// linked_list.push_front(25);
    ///
    /// assert!(!linked_list.is_empty());
    /// ```
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a reference to the underlying `Arena` which is used to allocate from.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    /// let linked_list = LinkedList::<usize>::new(&arena);
    ///
    /// let list_arena = linked_list.arena();
    /// assert!(core::ptr::eq(&arena, list_arena));
    /// ```
    #[must_use]
    #[inline]
    pub fn arena(&self) -> &Arena<A> {
        &self.arena
    }

    /// Pushes the given `value` to the front of the `LinkedList`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    /// let mut linked_list = LinkedList::new(&arena);
    ///
    /// linked_list.push_front(Handle::new_str_in(&arena, "Test!"));
    ///
    /// assert_eq!(linked_list.front().map(|handle| handle.as_ref()), Some("Test!"));
    /// ```
    #[inline]
    pub fn push_front(&mut self, value: T) {
        self.insert(0, value);
    }

    /// Pushes the given `value` to the front of the `LinkedList`, returning a mutable
    /// reference to the newly added value.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    /// let mut linked_list = LinkedList::new(&arena);
    ///
    /// let value = linked_list.push_front_mut(25);
    /// assert_eq!(*value, 25);
    ///
    /// *value = 31;
    /// # drop(value);
    ///
    /// assert_eq!(linked_list.front(), Some(&31));
    /// ```
    #[must_use]
    #[inline]
    pub fn push_front_mut(&mut self, value: T) -> &mut T {
        self.insert_mut(0, value)
    }

    /// Removes the first element in the `LinkedList`.
    ///
    /// If the list is empty, this method returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    /// let mut linked_list = LinkedList::new(&arena);
    ///
    /// linked_list.push_front(2);
    /// linked_list.push_front(3);
    ///
    /// assert_eq!(*linked_list.pop_front().unwrap(), 3);
    /// assert_eq!(*linked_list.pop_front().unwrap(), 2);
    /// assert_eq!(linked_list.pop_front(), None);
    /// ```
    #[inline]
    pub fn pop_front(&'_ mut self) -> Option<Handle<'a, T>> {
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

    /// Removes the last element in the `LinkedList`.
    ///
    /// If the list is empty, this method returns `None`.
    ///
    /// ```
    ///  use rotunda::{Arena, handle::Handle, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    /// let mut linked_list = LinkedList::new(&arena);
    ///
    /// linked_list.push_back(2);
    /// linked_list.push_back(3);
    ///
    /// assert_eq!(*linked_list.pop_back().unwrap(), 3);
    /// assert_eq!(*linked_list.pop_back().unwrap(), 2);
    /// assert_eq!(linked_list.pop_back(), None);
    /// ```
    #[inline]
    pub fn pop_back(&'_ mut self) -> Option<Handle<'a, T>> {
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
    pub fn remove(&'_ mut self, index: usize) -> Option<Handle<'a, T>> {
        self.remove_node_by_index(index)
            .map(|node| unsafe { Node::into_handle(node) })
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

        let (first_node, second_node) = unsafe {
            (
                self.get_node_unchecked(first_index),
                self.get_node_unchecked(second_index),
            )
        };

        unsafe {
            self.swap_nodes(first_node, first_index, second_node, second_index);
        }
    }

    #[inline]
    pub fn reverse(&mut self) {
        let len = self.len();
        if len <= 1 {
            return;
        }

        let n = len / 2;

        let mut first = self.head;
        let mut second = self.tail;

        for first_idx in 0..n {
            let second_idx = len.saturating_sub(first_idx + 1);

            unsafe {
                let (first_next, second_next) = (first.as_mut().next, second.as_mut().prev);

                self.swap_nodes(first, first_idx, second, second_idx);

                first = first_next;
                second = second_next;
            }
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        let arena = self.arena;
        let _ = mem::replace(self, Self::new(arena));
    }

    #[inline]
    pub fn split_off(&mut self, index: usize) -> LinkedList<'a, T, A> {
        let len = self.len;
        assert!(index <= len);

        if index == 0 {
            return mem::replace(self, LinkedList::new(self.arena));
        } else if index == len {
            return LinkedList::new(self.arena);
        }

        let node = unsafe { self.get_node_unchecked(index) };

        let mut list = LinkedList::new(&self.arena);

        unsafe {
            let (mut split_off_head, split_off_tail) = (node, self.tail);

            let mut new_tail = split_off_head.as_ref().prev;

            self.tail = new_tail;
            new_tail.as_mut().next = NonNull::dangling();
            self.len = index;

            split_off_head.as_mut().prev = NonNull::dangling();
            list.head = split_off_head;
            list.tail = split_off_tail;
            list.len = len - index;
        }

        list
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

        let node = if index >= self.len / 2 {
            iter.rev().nth(self.len().saturating_sub(index + 1))
        } else {
            iter.nth(index)
        };

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
    fn remove_node_by_index(&mut self, index: usize) -> Option<NonNull<Node<T>>> {
        let node = self.get_node(index)?;
        unsafe { Some(self.remove_node(node, index)) }
    }

    #[inline]
    unsafe fn remove_node(&mut self, mut node: NonNull<Node<T>>, index: usize) -> NonNull<Node<T>> {
        unsafe {
            let node = node.as_mut();
            self.len = self.len.checked_sub(1).expect("list underflow");

            if self.len > 0 && index > 0 {
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

        node
    }

    unsafe fn swap_nodes(
        &mut self,
        mut first_node: NonNull<Node<T>>,
        first_index: usize,
        mut second_node: NonNull<Node<T>>,
        second_index: usize,
    ) {
        debug_assert!(!ptr::eq(first_node.as_ptr(), NonNull::dangling().as_ptr()));
        debug_assert!(!ptr::eq(second_node.as_ptr(), NonNull::dangling().as_ptr()));

        let (first_prev, mut first_next) = unsafe {
            let prev = if first_index > 0 {
                Some(first_node.as_ref().prev)
            } else {
                None
            };

            (prev, first_node.as_ref().next)
        };

        let (mut second_prev, second_next) = unsafe {
            let next = if second_index < (self.len - 1) {
                Some(second_node.as_ref().next)
            } else {
                None
            };

            (second_node.as_ref().prev, next)
        };

        unsafe {
            if first_index.abs_diff(second_index) == 1 {
                first_node.as_mut().prev = second_node;
                second_node.as_mut().next = first_node;
            } else {
                first_node.as_mut().prev = second_prev;
                second_prev.as_mut().next = first_node;

                second_node.as_mut().next = first_next;
                first_next.as_mut().prev = second_node;
            }
        }

        unsafe {
            if let Some(mut first_prev) = first_prev {
                debug_assert!(!ptr::eq(first_prev.as_ptr(), NonNull::dangling().as_ptr()));

                second_node.as_mut().prev = first_prev;
                first_prev.as_mut().next = second_node;
            } else {
                self.head = second_node;
            }

            if let Some(mut second_next) = second_next {
                debug_assert!(!ptr::eq(second_next.as_ptr(), NonNull::dangling().as_ptr()));

                first_node.as_mut().next = second_next;
                second_next.as_mut().prev = first_node;
            } else {
                self.tail = first_node;
            }
        }
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

impl<'a, T, A: Allocator> From<&'a Arena<A>> for LinkedList<'a, T, A> {
    #[inline]
    fn from(arena: &'a Arena<A>) -> Self {
        Self::new(arena)
    }
}

impl<'a, T: PartialEq, A: Allocator> PartialEq for LinkedList<'a, T, A> {
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

impl<'a, T: Eq, A: Allocator> Eq for LinkedList<'a, T, A> {}

impl<'a, T: PartialEq, A: Allocator> PartialEq<[T]> for LinkedList<'a, T, A> {
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

impl<'a, 's, T: PartialEq, A: Allocator> PartialEq<&'s [T]> for LinkedList<'a, T, A> {
    #[inline]
    fn eq(&self, other: &&'s [T]) -> bool {
        PartialEq::eq(self, &other[..])
    }
}

impl<'a, T: PartialEq, A: Allocator, const N: usize> PartialEq<[T; N]> for LinkedList<'a, T, A> {
    #[inline]
    fn eq(&self, other: &[T; N]) -> bool {
        PartialEq::eq(self, &other[..])
    }
}

impl<'a, 's, T: PartialEq, A: Allocator, const N: usize> PartialEq<&'s [T; N]>
    for LinkedList<'a, T, A>
{
    #[inline]
    fn eq(&self, other: &&'s [T; N]) -> bool {
        PartialEq::eq(self, &other[..])
    }
}

impl<'a, T, A: Allocator> IntoIterator for &'_ LinkedList<'a, T, A> {
    type IntoIter = Iter<'a, T>;
    type Item = &'a T;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, A: Allocator> IntoIterator for &'_ mut LinkedList<'a, T, A> {
    type IntoIter = IterMut<'a, T>;
    type Item = &'a mut T;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'a, T: Unpin, A: Allocator> IntoIterator for LinkedList<'a, T, A> {
    type IntoIter = IntoIter<'a, T, A>;
    type Item = Handle<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter { list: self }
    }
}

impl<'a, T, A: Allocator> Extend<T> for LinkedList<'a, T, A> {
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
        self.list
            .remove_node_by_index(0)
            .map(|node| unsafe { Node::into_handle(node) })
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
            .remove_node_by_index(self.list.len.saturating_sub(1))
            .map(|node| unsafe { Node::into_handle(node) })
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

impl<T> Node<T> {
    #[must_use]
    #[inline]
    unsafe fn into_handle<'a>(node: NonNull<Node<T>>) -> Handle<'a, T> {
        unsafe {
            Handle::from_raw(
                node.as_ptr()
                    .map_addr(|addr| addr + offset_of!(Node<T>, data))
                    .cast::<T>(),
            )
        }
    }
}
