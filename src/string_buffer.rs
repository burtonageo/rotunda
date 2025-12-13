#![allow(missing_docs, clippy::missing_safety_doc)]

//! A UTF8-encoded, growable string. Backed by an `Arena`.

use crate::{
    Arena,
    buffer::{Buffer, TryExtendError},
    handle::Handle,
};
use alloc::alloc::Allocator;
use core::{
    borrow::{Borrow, BorrowMut},
    fmt,
    hash::Hash,
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    ptr, str,
};
#[cfg(feature = "serde")]
use serde_core::{Serialize, Serializer};

#[derive(Default)]
pub struct StringBuffer<'a> {
    inner: Buffer<'a, u8>,
}

impl<'a> StringBuffer<'a> {
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self {
            inner: const { Buffer::new() },
        }
    }

    #[must_use]
    #[inline]
    pub const unsafe fn from_handle(handle: Handle<'a, [MaybeUninit<u8>]>) -> Self {
        Self {
            inner: unsafe { Buffer::from_raw_parts(handle, 0) },
        }
    }

    #[track_caller]
    #[inline]
    #[must_use]
    pub fn with_capacity_in_arena<A: Allocator>(capacity: usize, arena: &'a Arena<A>) -> Self {
        unsafe { Self::from_handle(Handle::new_slice_uninit_in(arena, capacity)) }
    }

    #[track_caller]
    #[inline]
    #[must_use]
    pub fn new_in<S: AsRef<str>, A: Allocator>(arena: &'a Arena<A>, s: &S) -> Self {
        let s = s.as_ref();
        let mut buf = StringBuffer::with_capacity_in_arena(s.len(), arena);
        buf.push_str(s);
        buf
    }

    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    #[must_use]
    #[inline]
    pub const fn is_full(&self) -> bool {
        self.inner.is_full()
    }

    #[must_use]
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    #[must_use]
    #[inline]
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        self.inner.spare_capacity_mut()
    }

    #[inline]
    pub const unsafe fn set_len(&mut self, new_len: usize) {
        unsafe {
            self.inner.set_len(new_len);
        }
    }

    #[inline]
    pub const unsafe fn set_capacity(&mut self, new_capacity: usize) {
        unsafe {
            self.inner.set_capacity(new_capacity);
        }
    }

    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        if new_len < self.len() {
            assert!(self.as_str().is_char_boundary(new_len));
            self.inner.truncate(new_len);
        }
    }

    #[inline]
    pub fn try_push_str<'s, S: ?Sized + AsRef<str>>(
        &mut self,
        string: &'s S,
    ) -> Result<(), TryExtendError<str::Bytes<'s>>> {
        let string = string.as_ref();
        self.inner.try_extend(string.bytes())
    }

    #[inline]
    pub fn push_str<S: ?Sized + AsRef<str>>(&mut self, string: &S) {
        self.inner.extend(string.as_ref().bytes());
    }

    #[inline]
    pub fn push_char(&mut self, ch: char) {
        let mut ch_bytes = [0u8; char::MAX_LEN_UTF8];
        self.push_str(ch.encode_utf8(&mut ch_bytes));
    }

    #[must_use]
    #[inline]
    pub const fn as_str(&self) -> &str {
        unsafe { str::from_utf8_unchecked(self.inner.as_slice()) }
    }

    #[must_use]
    #[inline]
    pub const fn as_mut_str(&mut self) -> &mut str {
        unsafe { str::from_utf8_unchecked_mut(self.inner.as_mut_slice()) }
    }

    #[must_use]
    #[inline]
    pub fn into_str_handle(self) -> Handle<'a, str> {
        let len = self.len();
        let handle = self.inner.into_slice_handle();
        let slice_ptr = Handle::into_raw(handle);
        let string = ptr::from_raw_parts_mut(slice_ptr as *mut u8, len);
        unsafe { Handle::from_raw(string) }
    }

    #[must_use]
    #[inline]
    pub const fn as_bytes(&self) -> &[u8] {
        self.as_str().as_bytes()
    }

    #[must_use]
    #[inline]
    pub const fn is_char_boundary(&self, idx: usize) -> bool {
        if idx == 0 {
            return true;
        }

        let str_len = self.as_str().len();
        if idx >= str_len {
            idx == str_len
        } else {
            // inlined from `char::is_utf8_char_boundary()`, as that's a private method
            (self.as_bytes()[idx] as i8) >= -0x40
        }
    }
}

impl<'a> Deref for StringBuffer<'a> {
    type Target = str;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl<'a> DerefMut for StringBuffer<'a> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_str()
    }
}

impl<'a> Borrow<str> for StringBuffer<'a> {
    #[inline]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl<'a> BorrowMut<str> for StringBuffer<'a> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut str {
        self.as_mut_str()
    }
}

impl<'a> AsRef<str> for StringBuffer<'a> {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl<'a> AsMut<str> for StringBuffer<'a> {
    #[inline]
    fn as_mut(&mut self) -> &mut str {
        self.as_mut_str()
    }
}

impl<'a> fmt::Debug for StringBuffer<'a> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), fmtr)
    }
}

impl<'a> fmt::Display for StringBuffer<'a> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), fmtr)
    }
}

impl<'a> fmt::Write for StringBuffer<'a> {
    #[inline]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        StringBuffer::try_push_str(self, s).map_err(|_| fmt::Error)
    }
}

impl<'a> PartialEq for StringBuffer<'a> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(self.as_str(), other.as_str())
    }
}

impl<'a> PartialEq<str> for StringBuffer<'a> {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        PartialEq::eq(self.as_str(), other)
    }
}

impl<'a> PartialEq<Handle<'_, str>> for StringBuffer<'a> {
    #[inline]
    fn eq(&self, other: &Handle<str>) -> bool {
        PartialEq::eq(self.as_str(), other.as_ref())
    }
}

impl<'a> Eq for StringBuffer<'a> {}

impl<'a> Hash for StringBuffer<'a> {
    #[inline]
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl<'a> PartialOrd for StringBuffer<'a> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> PartialOrd<str> for StringBuffer<'a> {
    #[inline]
    fn partial_cmp(&self, other: &str) -> Option<core::cmp::Ordering> {
        Some(self.as_str().cmp(other))
    }
}

impl<'a> PartialOrd<Handle<'_, str>> for StringBuffer<'a> {
    #[inline]
    fn partial_cmp(&self, other: &Handle<'_, str>) -> Option<core::cmp::Ordering> {
        Some(self.as_str().cmp(other.as_ref()))
    }
}

impl<'a> Ord for StringBuffer<'a> {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_str().cmp(other.as_str())
    }
}

#[cfg(feature = "serde")]
impl<'a> Serialize for StringBuffer<'a> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        <str as Serialize>::serialize(self.as_ref(), serializer)
    }
}
