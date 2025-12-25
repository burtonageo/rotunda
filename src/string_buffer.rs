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
    error::Error as ErrorTrait,
    hash::Hash,
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    ptr, str::{self, Utf8Error},
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

    #[inline]
    pub const fn from_utf8(bytes: Buffer<'a, u8>) -> Result<Self, FromUtf8Error<'a>> {
        match str::from_utf8(bytes.as_slice()) {
            Ok(_) => unsafe { Ok(Self::from_utf8_unchecked(bytes)) },
            Err(error) => Err(FromUtf8Error::new(bytes, error)),
        }
    }

    #[inline]
    pub const unsafe fn from_utf8_unchecked(bytes: Buffer<'a, u8>) -> Self {
        Self { inner: bytes }
    }

    #[inline]
    pub fn into_bytes(self) -> Buffer<'a, u8> {
        self.inner
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
    pub fn split_at_spare_capacity(self) -> (StringBuffer<'a>, Buffer<'a, MaybeUninit<u8>>) {
        let (string, spare_cap) = self.inner.split_at_spare_capacity();
        unsafe {
            (StringBuffer::from_utf8_unchecked(Buffer::from(string)), Buffer::from(spare_cap))
        }
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
        let bytes = ptr::slice_from_raw_parts_mut(slice_ptr as *mut u8, len);
        unsafe { Handle::from_raw(bytes as *mut str) }
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

impl<'a> AsRef<[u8]> for StringBuffer<'a> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
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
        PartialEq::<str>::eq(self.as_str(), other.as_ref())
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

impl<'a> TryFrom<Buffer<'a, u8>> for StringBuffer<'a> {
    type Error = FromUtf8Error<'a>;
    #[inline]
    fn try_from(value: Buffer<'a, u8>) -> Result<Self, Self::Error> {
        StringBuffer::from_utf8(value)
    }
}

impl<'a> TryFrom<Handle<'a, [u8]>> for StringBuffer<'a> {
    type Error = FromUtf8Error<'a>;
    #[inline]
    fn try_from(value: Handle<'a, [u8]>) -> Result<Self, Self::Error> {
        TryFrom::try_from(Buffer::from_slice_handle(value))
    }
}

impl<'a> From<StringBuffer<'a>> for Buffer<'a, u8> {
    #[inline]
    fn from(value: StringBuffer<'a>) -> Self {
        value.into_bytes()
    }
}

impl<'a> From<StringBuffer<'a>> for Handle<'a, [u8]> {
    #[inline]
    fn from(value: StringBuffer<'a>) -> Self {
        value.into_bytes().into_slice_handle()
    }
}


#[cfg(feature = "serde")]
impl<'a> Serialize for StringBuffer<'a> {
    #[inline]
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        <str as Serialize>::serialize(self.as_ref(), serializer)
    }
}

#[derive(Debug)]
pub struct FromUtf8Error<'a> {
    bytes: Buffer<'a, u8>,
    error: Utf8Error,
}

impl<'a> FromUtf8Error<'a> {
    #[inline]
    pub(crate) const fn new(bytes: Buffer<'a, u8>, error: Utf8Error) -> Self {
        Self { bytes, error }
    }

    #[must_use]
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[must_use]
    #[inline]
    pub fn into_bytes(self) -> Buffer<'a, u8> {
        self.bytes
    }

    #[must_use]
    #[inline]
    pub const fn utf8_error(&self) -> &Utf8Error {
        &self.error
    }
}

impl<'a> fmt::Display for FromUtf8Error<'a> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.utf8_error(), fmtr)
    }
}

impl<'a> ErrorTrait for FromUtf8Error<'a> {
    #[inline]
    fn source(&self) -> Option<&(dyn ErrorTrait + 'static)> {
        Some(&self.error)
    }
}
