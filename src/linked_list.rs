#![warn(missing_docs, clippy::missing_safety_doc)]

//! A doubly-linked owned list of nodes, backed by an `Arena`.
//!
//! The `LinkedList` allows pushing and popping elements at either end of the list in constant time.

use crate::{Arena, InvariantLifetime, handle::Handle};
use alloc::alloc::{Allocator, Global, Layout};
use core::{
    fmt,
    iter::{DoubleEndedIterator, FusedIterator, Iterator},
    marker::PhantomData,
    mem::{self, offset_of},
    ptr::{self, NonNull},
};
#[cfg(feature = "serde")]
use serde_core::{Serialize, Serializer};

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
    _boo: PhantomData<(T, InvariantLifetime<'a, Arena<A>>)>,
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

    /// Create a new `LinkedList` containing the contents of the given iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, linked_list::LinkedList};
    /// let arena = Arena::new();
    ///
    /// let linked_list = LinkedList::from_iter_in(&arena, [1, 2, 3, 4, 5].into_iter());
    /// assert_eq!(linked_list, &[1, 2, 3, 4, 5]);
    /// ```
    #[must_use]
    #[inline]
    pub fn from_iter_in<I: IntoIterator<Item = T>>(arena: &'a Arena<A>, iter: I) -> Self {
        let mut list = Self::new(arena);
        list.extend(iter);
        list
    }

    /// Create a new `LinkedList` from the given function.
    ///
    /// The function `f()` will be called `len` times with the index of each
    /// element as the parameter, and the results will be collected into
    /// the returned `LinkedList`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle, linked_list::LinkedList};
    /// let arena = Arena::new();
    ///
    /// let linked_list = LinkedList::from_fn_in(&arena, 5, |elem| {
    ///     Handle::new_str_in(&arena, &format!("{}", elem))
    /// });
    ///
    /// assert_eq!(linked_list.get(0).unwrap(), "0");
    /// assert_eq!(linked_list.get(1).unwrap(), "1");
    /// assert_eq!(linked_list.get(2).unwrap(), "2");
    /// assert_eq!(linked_list.get(3).unwrap(), "3");
    /// assert_eq!(linked_list.get(4).unwrap(), "4");
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn from_fn_in<F: FnMut(usize) -> T>(arena: &'a Arena<A>, len: usize, mut f: F) -> Self {
        let mut list = LinkedList::new(arena);
        for i in 0..len {
            list.push_back(f(i));
        }
        list
    }

    /// Returns the number of elements in the `LinkedList`.
    ///
    /// Note that a `LinkedList` stores its length as a field, so this
    /// function returns in constant time.
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
    /// Note that a `LinkedList` stores its length as a field, so this
    /// function returns in constant time.
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
        self.arena
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

    /// Appends the given `value` to the end of the `LinkedList`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, handle::Handle, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    /// let mut linked_list = LinkedList::new(&arena);
    ///
    /// linked_list.push_back(Handle::new_str_in(&arena, "Message"));
    ///
    /// assert_eq!(linked_list.front().map(|handle| handle.as_ref()), Some("Message"));
    /// ```
    #[inline]
    pub fn push_back(&mut self, value: T) {
        self.insert(self.len, value);
    }

    /// Pushes the given `value` to the back of the `LinkedList`, returning a mutable
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
    /// let value = linked_list.push_back_mut(42);
    /// assert_eq!(*value, 42);
    ///
    /// *value = 155;
    /// # drop(value);
    ///
    /// assert_eq!(linked_list.back(), Some(&155));
    /// ```
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
    ///  use rotunda::{Arena, linked_list::LinkedList};
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

    /// Insert the given `value` into the `LinkedList` at position `index`.
    ///
    /// # Panics
    ///
    /// This method will panic if `index` is greater than or equal to `self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut list = LinkedList::from_iter_in(&arena, [0, 2]);
    ///
    /// list.insert(1, 1);
    ///
    /// assert_eq!(&list, &[0, 1, 2]);
    /// ```
    #[track_caller]
    #[inline]
    pub fn insert(&mut self, index: usize, value: T) {
        let _ = self.insert_mut(index, value);
    }

    /// Insert the given `value` into the `LinkedList` at position `index`.
    ///
    /// A mutable reference to the newly-inserted `value` is returned.
    ///
    /// # Panics
    ///
    /// This method will panic if `index` is greater than or equal to `self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut list = LinkedList::from_iter_in(&arena, [1, 3, 4]);
    ///
    /// let item = list.insert_mut(1, 1);
    /// assert_eq!(*item, 1);
    ///
    /// *item = 2;
    ///
    /// assert_eq!(&list, &[1, 2, 3, 4]);
    /// ```
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

    /// Removes the element at `index` from the list.
    ///
    /// The removed element is returned as a `Handle` to its value in the `Arena`.
    /// If the `index` is out of bounds, then this method will return `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut list = LinkedList::from_iter_in(&arena, [1, 2, 3, 4]);
    ///
    /// let value = list.remove(1);
    ///
    /// assert_eq!(value.unwrap(), &2);
    /// assert_eq!(&list, &[1, 3, 4]);
    /// ```
    #[inline]
    pub fn remove(&'_ mut self, index: usize) -> Option<Handle<'a, T>> {
        self.remove_node_by_index(index)
            .map(|node| unsafe { Node::into_handle(node) })
    }

    /// Swap the elements at `first_index` and `second_index` in the `LinkedList`.
    ///
    /// # Panics
    ///
    /// This method will panic if either `first_index` or `second_index` are greater than
    /// or equal to `self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut list = LinkedList::from_iter_in(&arena, [1, 2, 3, 4]);
    ///
    /// list.swap(0, 3);
    ///
    /// assert_eq!(&list, &[4, 2, 3, 1]);
    /// ```
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

    /// Reverse the elements in the `LinkedList`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut list = LinkedList::from_iter_in(&arena, [1, 2, 3, 4]);
    ///
    /// list.reverse();
    ///
    /// assert_eq!(&list, &[4, 3, 2, 1]);
    /// ```
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

    /// Remove all elements in the `LinkedList`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut list = LinkedList::from_iter_in(&arena, [1, 2, 3, 4]);
    ///
    /// list.clear();
    ///
    /// assert_eq!(&list, &[]);
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        let _ = self.split_off(0);
    }

    /// Splits the list at `index`.
    ///
    /// Elements after `index` are in the returned `LinkedList`,
    /// while elements before `index` are retained in `self`.
    ///
    /// # Panics
    ///
    /// This method will panic if `index` is greater than or equal to
    /// `self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut list = LinkedList::from_fn_in(&arena, 5, |idx| idx * 2);
    ///
    /// let rhs = list.split_off(3);
    ///
    /// assert_eq!(&list, &[0, 2, 4]);
    /// assert_eq!(&rhs, &[6, 8]);
    /// ```
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

        let mut list = LinkedList::new(self.arena);

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

    /// Returns a reference to the first element in the `LinkedList`, or `None` if the list is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    ///
    /// let list = LinkedList::from_iter_in(&arena, [1, 2, 3, 4]);
    ///
    /// assert_eq!(list.front(), Some(&1));
    ///
    /// let list = LinkedList::<i32>::new(&arena);
    ///
    /// assert_eq!(list.front(), None);
    /// ```
    #[must_use]
    #[inline]
    pub fn front(&self) -> Option<&T> {
        if !self.is_empty() {
            unsafe { Some(Node::data_ptr(self.head).as_ref()) }
        } else {
            None
        }
    }

    /// Returns a mutable reference to the first element in the `LinkedList`, or `None` if the list is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut list = LinkedList::from_iter_in(&arena, [1, 2, 3, 4]);
    ///
    /// let front = list.front_mut().unwrap();
    /// assert_eq!(front, &1);
    /// *front = 25;
    ///
    /// assert_eq!(&list, &[25, 2, 3, 4]);
    ///
    /// let mut list = LinkedList::<i32>::new(&arena);
    ///
    /// assert_eq!(list.front_mut(), None);
    /// ```
    #[must_use]
    #[inline]
    pub fn front_mut(&mut self) -> Option<&mut T> {
        if !self.is_empty() {
            unsafe { Some(Node::data_ptr(self.head).as_mut()) }
        } else {
            None
        }
    }

    /// Returns a reference to the last element in the `LinkedList`, or `None` if the list is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    ///
    /// let list = LinkedList::from_iter_in(&arena, [1, 2, 3, 4]);
    ///
    /// assert_eq!(list.back(), Some(&4));
    ///
    /// let list = LinkedList::<i32>::new(&arena);
    ///
    /// assert_eq!(list.back(), None);
    /// ```
    #[must_use]
    #[inline]
    pub fn back(&self) -> Option<&T> {
        if !self.is_empty() {
            unsafe { Some(Node::data_ptr(self.tail).as_ref()) }
        } else {
            None
        }
    }

    /// Returns a mutable reference to the last element in the `LinkedList`, or `None` if the list is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut list = LinkedList::from_iter_in(&arena, [1, 2, 3, 4]);
    ///
    /// let back = list.back_mut().unwrap();
    /// assert_eq!(back, &4);
    /// *back = 25;
    ///
    /// assert_eq!(&list, &[1, 2, 3, 25]);
    ///
    /// let mut list = LinkedList::<i32>::new(&arena);
    ///
    /// assert_eq!(list.back_mut(), None);
    /// ```
    #[must_use]
    #[inline]
    pub fn back_mut(&mut self) -> Option<&mut T> {
        if !self.is_empty() {
            unsafe { Some(Node::data_ptr(self.tail).as_mut()) }
        } else {
            None
        }
    }

    /// Returns a reference to the `index`th node of the `LinkedList` if it exists,
    /// or `None` if it is out of bounds.
    ///
    /// This method may iterate over the `LinkedList` from the end if `index` is in the second
    /// half of the `LinkedList`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    ///
    /// let list = LinkedList::from_iter_in(&arena, [1, 2, 3, 4, 5]);
    ///
    /// assert_eq!(list.get(2), Some(&3));
    /// assert_eq!(list.get(4), Some(&5));
    /// assert_eq!(list.get(6), None);
    /// ```
    #[must_use]
    #[inline]
    pub fn get(&'_ self, index: usize) -> Option<&'_ T> {
        self.get_node(index)
            .map(|node| unsafe { Node::data_ptr(node).as_ref() })
    }

    /// Returns a mutable reference to the `index`th node of the `LinkedList` if it exists,
    /// or `None` if it is out of bounds.
    ///
    /// This method may iterate over the `LinkedList` from the end if `index` is in the second
    /// half of the `LinkedList`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut list = LinkedList::from_iter_in(&arena, [1, 2, 3, 4, 5]);
    ///
    /// assert_eq!(list.get_mut(2).unwrap(), &3);
    /// assert_eq!(list.get_mut(4).unwrap(), &5);
    /// assert_eq!(list.get_mut(6), None);
    ///
    /// *list.get_mut(0).unwrap() = 6;
    ///
    /// assert_eq!(&list, &[6, 2, 3, 4, 5]);
    /// ```
    #[must_use]
    #[inline]
    pub fn get_mut(&'_ mut self, index: usize) -> Option<&'_ mut T> {
        self.get_node(index)
            .map(|mut node| unsafe { &mut node.as_mut().data })
    }

    /// Retains only the elements specified by `pred`.
    ///
    /// Removes all elements where `pred(element)` returns `false`. This method
    /// operates in-place, and preserves the order of the `LinkedList`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rotunda::{Arena, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut list = LinkedList::from_iter_in(&arena, (0..=10usize));
    ///
    /// assert_eq!(&list, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    ///
    /// list.retain(|elem| elem.is_multiple_of(2));
    /// assert_eq!(&list, &[0, 2, 4, 6, 8, 10]);
    /// ```
    #[track_caller]
    #[inline]
    pub fn retain<F: FnMut(&T) -> bool>(&mut self, mut pred: F) {
        let mut node_iter = NodeIter::new(self);
        let mut i = 0;
        while let Some(node) = node_iter.next() {
            let should_drop = unsafe {
                let node_ref = &node.as_ref().data;
                !pred(node_ref)
            };

            if should_drop {
                unsafe {
                    let _ = Node::into_handle(self.remove_node(node, i));
                }
            } else {
                i += 1;
            }
        }
    }

    /// Returns an immutable iterator over the elements of the `LinkedList`.
    ///
    /// # Example
    ///
    /// ```
    /// use rotunda::{Arena, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    ///
    /// let list = LinkedList::from_iter_in(&arena, [5, 6, 7]);
    ///
    /// let mut iter = list.iter();
    ///
    /// assert_eq!(iter.next(), Some(&5));
    /// assert_eq!(iter.next_back(), Some(&7));
    /// assert_eq!(iter.next(), Some(&6));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[must_use]
    #[inline]
    pub fn iter(&self) -> Iter<'a, T> {
        Iter {
            iter: NodeIter::new(self),
            _boo: PhantomData,
        }
    }

    /// Returns a mutable iterator over the elements of the `LinkedList`.
    ///
    /// # Example
    ///
    /// ```
    /// use rotunda::{Arena, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    ///
    /// let mut list = LinkedList::from_iter_in(&arena, [5, 6, 7]);
    ///
    /// let mut iter = list.iter_mut();
    ///
    /// assert_eq!(iter.next().unwrap(), &5);
    /// assert_eq!(iter.next_back().unwrap(), &7);
    /// *iter.next().unwrap() = 8;
    /// assert_eq!(iter.next(), None);
    ///
    /// assert_eq!(&list, &[5, 8, 7]);
    /// ```
    #[must_use]
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'a, T> {
        IterMut {
            iter: NodeIter::new(self),
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

impl<'a, T, A: Allocator> Drop for LinkedList<'a, T, A> {
    #[inline]
    fn drop(&mut self) {
        for node in NodeIter::new(self) {
            let data = Node::data_ptr(node).as_ptr();

            unsafe {
                ptr::drop_in_place(data);
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

impl<'a, T: 'a, A: Allocator> From<IntoIter<'a, T, A>> for LinkedList<'a, T, A> {
    #[inline]
    fn from(value: IntoIter<'a, T, A>) -> Self {
        value.into_list()
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

impl<'a, T, A: Allocator> IntoIterator for LinkedList<'a, T, A> {
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

impl<'a, 't, T: Copy, A: Allocator> Extend<&'t T> for LinkedList<'a, T, A> {
    #[track_caller]
    #[inline]
    fn extend<I: IntoIterator<Item = &'t T>>(&mut self, iter: I) {
        Extend::extend(self, iter.into_iter().copied())
    }
}

impl<'a, 't, T: Copy, A: Allocator> Extend<&'t [T]> for LinkedList<'a, T, A> {
    #[track_caller]
    #[inline]
    fn extend<I: IntoIterator<Item = &'t [T]>>(&mut self, iter: I) {
        for item in iter {
            Extend::extend(self, item.into_iter().copied())
        }
    }
}

#[cfg(feature = "serde")]
impl<'a, T: Serialize> Serialize for LinkedList<'a, T> {
    #[inline]
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.iter())
    }
}

/// Immutable iterator over a `LinkedList`.
///
/// This method is created by the [`iter()`] method on [`LinkedList`]
///
/// [`iter()`]: ./struct.LinkedList.html#method.iter
/// [`LinkedList`]: ./struct.LinkedList.html
pub struct Iter<'a, T> {
    iter: NodeIter<T>,
    _boo: PhantomData<&'a Node<T>>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|node| unsafe { Node::data_ptr(node).as_ref() })
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
            .map(|node| unsafe { Node::data_ptr(node).as_ref() })
    }
}

impl<'a, T> Clone for Iter<'a, T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            iter: self.iter,
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

/// Mutable iterator over a `LinkedList`.
///
/// This method is created by the [`iter_mut()`] method on [`LinkedList`]
///
/// [`iter_mut()`]: ./struct.LinkedList.html#method.iter_mut
/// [`LinkedList`]: ./struct.LinkedList.html
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
            .map(|node| unsafe { Node::data_ptr(node).as_mut() })
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
            .map(|node| unsafe { Node::data_ptr(node).as_mut() })
    }
}

impl<'a, T: fmt::Debug> fmt::Debug for IterMut<'a, T> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.iter, fmtr)
    }
}

/// A by-value iterator over a [`LinkedList`].
///
/// [`LinkedList`]: ./struct.LinkedList.html
pub struct IntoIter<'a, T: 'a, A: Allocator> {
    list: LinkedList<'a, T, A>,
}

impl<'a, T: 'a, A: Allocator> IntoIter<'a, T, A> {
    /// Consumes the `IntoIter`, returning the underlying `LinkedList`.
    ///
    /// # Example
    ///
    /// ```
    /// use rotunda::{Arena, linked_list::LinkedList};
    ///
    /// let arena = Arena::new();
    ///
    /// let list = LinkedList::from_iter_in(&arena, [1, 2, 3, 4, 5]);
    ///
    /// let mut iter = list.into_iter();
    ///
    /// let _ = iter.next();
    /// let _ = iter.next();
    /// let _ = iter.next_back();
    ///
    /// let list = iter.into_list();
    ///
    /// assert_eq!(&list, &[3, 4]);
    /// ```
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
        self.list.pop_front()
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
        self.list.pop_back()
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

impl<T> FusedIterator for NodeIter<T> {}

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
    fn data_ptr(this: NonNull<Node<T>>) -> NonNull<T> {
        this.map_addr(|addr| addr.saturating_add(offset_of!(Node<T>, data)))
            .cast::<T>()
    }

    #[must_use]
    #[inline]
    unsafe fn into_handle<'a>(node: NonNull<Node<T>>) -> Handle<'a, T> {
        unsafe { Handle::from_raw(Node::data_ptr(node).as_ptr()) }
    }
}
