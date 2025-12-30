// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, clippy::missing_safety_doc)]

//! A UTF8-encoded, growable string. Backed by an `Arena`.

use crate::{
    Arena,
    buffer::{Buffer, GrowableBuffer, TryExtendError, TryReserveError},
    handle::Handle,
};
use alloc::alloc::{Allocator, Global};
use core::{
    borrow::{Borrow, BorrowMut},
    cmp,
    error::Error as ErrorTrait,
    fmt,
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    ptr,
    str::{self, Utf8Error},
};
#[cfg(feature = "serde")]
use serde_core::{Serialize, Serializer};

#[derive(Default)]
pub struct StringBuffer<'a, A: Allocator = Global> {
    inner: Buffer<'a, u8, A>,
}

impl<'a, A: Allocator> StringBuffer<'a, A> {
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self {
            inner: const { Buffer::new() },
        }
    }

    #[must_use]
    #[inline]
    pub const unsafe fn from_handle(handle: Handle<'a, [MaybeUninit<u8>], A>) -> Self {
        Self {
            inner: unsafe { Buffer::from_raw_parts(handle, 0) },
        }
    }

    #[inline]
    pub const fn from_str_handle(handle: Handle<'a, str, A>) -> Self {
        Self {
            inner: Buffer::from_slice_handle(Handle::into_bytes(handle)),
        }
    }

    #[inline]
    pub fn try_with_growable_in<E, F>(arena: &'a Arena<A>, f: F) -> Result<Self, E>
    where
        A: Allocator,
        F: for<'buf> FnOnce(&'buf mut GrowableStringBuffer<'a, A>) -> Result<(), E>,
    {
        let buffer = Buffer::try_with_growable_in(arena, |buf| {
            let buf = unsafe { &mut *ptr::from_mut(buf).cast::<GrowableStringBuffer<'a, A>>() };
            f(buf)
        });

        buffer.map(|buffer| unsafe { Self::from_utf8_unchecked(buffer) })
    }

    #[inline]
    pub fn with_growable<F>(arena: &'a Arena<A>, f: F) -> Self
    where
        F: for<'buf> FnOnce(&'buf mut GrowableStringBuffer<'a, A>),
    {
        let buffer = Buffer::with_growable_in(arena, |buf| {
            let buf = unsafe { &mut *ptr::from_mut(buf).cast::<GrowableStringBuffer<'a, A>>() };
            f(buf)
        });

        unsafe { Self::from_utf8_unchecked(buffer) }
    }

    #[inline]
    pub const fn from_utf8(bytes: Buffer<'a, u8, A>) -> Result<Self, FromUtf8Error<'a, A>> {
        match str::from_utf8(bytes.as_slice()) {
            Ok(_) => unsafe { Ok(Self::from_utf8_unchecked(bytes)) },
            Err(error) => Err(FromUtf8Error::new(bytes, error)),
        }
    }

    #[inline]
    pub const unsafe fn from_utf8_unchecked(bytes: Buffer<'a, u8, A>) -> Self {
        Self { inner: bytes }
    }

    #[inline]
    pub fn into_bytes(self) -> Buffer<'a, u8, A> {
        self.inner
    }

    #[track_caller]
    #[inline]
    #[must_use]
    pub fn with_capacity_in_arena(capacity: usize, arena: &'a Arena<A>) -> Self {
        unsafe { Self::from_handle(Handle::new_slice_uninit_in(arena, capacity)) }
    }

    #[track_caller]
    #[inline]
    #[must_use]
    pub fn new_in<S: AsRef<str>>(arena: &'a Arena<A>, s: &S) -> Self {
        let s = s.as_ref();
        let mut buf = StringBuffer::with_capacity_in_arena(s.len(), arena);
        buf.push_str(s);
        buf
    }

    #[inline]
    pub fn split_at_spare_capacity(self) -> (Self, Buffer<'a, MaybeUninit<u8>, A>) {
        let (string, spare_cap) = self.inner.split_at_spare_capacity();
        unsafe {
            (
                StringBuffer::from_utf8_unchecked(Buffer::from(string)),
                Buffer::from(spare_cap),
            )
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
        let mut ch_bytes = [0u8; 4];
        self.push_str(ch.encode_utf8(&mut ch_bytes));
    }

    #[must_use]
    #[inline]
    pub fn pop(&mut self) -> Option<char> {
        let ch = self.as_str().chars().next_back()?;
        let new_len = self.inner.len() - ch.len_utf8();

        unsafe {
            self.set_len(new_len);
        }

        Some(ch)
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
    pub fn into_str_handle(self) -> Handle<'a, str, A> {
        let len = self.len();
        let handle = self.inner.into_slice_handle();
        let slice_ptr = Handle::into_raw(handle);
        let bytes = ptr::slice_from_raw_parts_mut(slice_ptr as *mut u8, len);
        unsafe { Handle::from_raw_with_alloc(bytes as *mut str) }
    }
}

impl<'a, A: Allocator> Extend<char> for StringBuffer<'a, A> {
    #[track_caller]
    #[inline]
    fn extend<I: IntoIterator<Item = char>>(&mut self, iter: I) {
        for item in iter {
            self.push_char(item);
        }
    }
}

impl<'a, 'c, A: Allocator> Extend<&'c char> for StringBuffer<'a, A> {
    #[track_caller]
    #[inline]
    fn extend<I: IntoIterator<Item = &'c char>>(&mut self, iter: I) {
        Extend::extend(self, iter.into_iter().copied());
    }
}

impl<'a, 's, A: Allocator> Extend<&'s str> for StringBuffer<'a, A> {
    #[track_caller]
    #[inline]
    fn extend<I: IntoIterator<Item = &'s str>>(&mut self, iter: I) {
        for item in iter {
            self.push_str(item);
        }
    }
}

impl<'a, A: Allocator> Deref for StringBuffer<'a, A> {
    type Target = str;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl<'a, A: Allocator> DerefMut for StringBuffer<'a, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_str()
    }
}

impl<'a, A: Allocator> Borrow<str> for StringBuffer<'a, A> {
    #[inline]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl<'a, A: Allocator> BorrowMut<str> for StringBuffer<'a, A> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut str {
        self.as_mut_str()
    }
}

impl<'a, A: Allocator> AsRef<str> for StringBuffer<'a, A> {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl<'a, A: Allocator> AsMut<str> for StringBuffer<'a, A> {
    #[inline]
    fn as_mut(&mut self) -> &mut str {
        self.as_mut_str()
    }
}

impl<'a, A: Allocator> AsRef<[u8]> for StringBuffer<'a, A> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl<'a, A: Allocator> fmt::Debug for StringBuffer<'a, A> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), fmtr)
    }
}

impl<'a, A: Allocator> fmt::Display for StringBuffer<'a, A> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), fmtr)
    }
}

impl<'a, A: Allocator> fmt::Write for StringBuffer<'a, A> {
    #[inline]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        StringBuffer::try_push_str(self, s).map_err(|_| fmt::Error)
    }
}

impl<'a, A: Allocator> PartialEq for StringBuffer<'a, A> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(self.as_str(), other.as_str())
    }
}

impl<'a, A: Allocator> PartialEq<str> for StringBuffer<'a, A> {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        PartialEq::eq(self.as_str(), other)
    }
}

impl<'a, A: Allocator, A2: Allocator> PartialEq<Handle<'_, str, A2>> for StringBuffer<'a, A> {
    #[inline]
    fn eq(&self, other: &Handle<'_, str, A2>) -> bool {
        PartialEq::<str>::eq(self.as_str(), other.as_ref())
    }
}

impl<'a, A: Allocator> Eq for StringBuffer<'a, A> {}

impl<'a, A: Allocator> Hash for StringBuffer<'a, A> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl<'a> PartialOrd for StringBuffer<'a> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> PartialOrd<str> for StringBuffer<'a> {
    #[inline]
    fn partial_cmp(&self, other: &str) -> Option<cmp::Ordering> {
        Some(self.as_str().cmp(other))
    }
}

impl<'a> PartialOrd<Handle<'_, str>> for StringBuffer<'a> {
    #[inline]
    fn partial_cmp(&self, other: &Handle<'_, str>) -> Option<cmp::Ordering> {
        Some(self.as_str().cmp(other.as_ref()))
    }
}

impl<'a> Ord for StringBuffer<'a> {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
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

impl<'a> From<Handle<'a, str>> for StringBuffer<'a> {
    #[inline]
    fn from(value: Handle<'a, str>) -> Self {
        StringBuffer::from_str_handle(value)
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

#[repr(transparent)]
pub struct GrowableStringBuffer<'a, A: Allocator = Global> {
    inner: GrowableBuffer<'a, u8, A>,
}

impl<'a, A: Allocator> GrowableStringBuffer<'a, A> {
    #[must_use]
    #[inline]
    pub const fn max_capacity(&self) -> usize {
        self.inner.max_capacity()
    }

    #[must_use]
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    #[inline]
    pub const fn is_full(&self) -> bool {
        self.inner.is_full()
    }

    #[must_use]
    #[inline]
    pub const fn has_capacity(&self, required_capacity: usize) -> bool {
        self.inner.has_capacity(required_capacity)
    }

    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner.try_reserve(additional)
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }

    #[inline]
    pub fn try_push_char(&mut self, value: char) -> Result<(), char> {
        let mut bytes = [0u8; 4];
        let string = value.encode_utf8(&mut bytes);

        match self.inner.try_extend(string.bytes()) {
            Ok(()) => Ok(()),
            Err(_) => Err(value),
        }
    }

    #[inline]
    pub fn try_push_str<'s, S: 's + ?Sized + AsRef<str>>(
        &mut self,
        string: &'s S,
    ) -> Result<(), &'s str> {
        let string = string.as_ref();
        match self.inner.try_extend(string.bytes()) {
            Ok(()) => Ok(()),
            Err(_) => Err(string),
        }
    }

    #[track_caller]
    #[inline]
    pub fn push_char(&mut self, value: char) {
        match self.try_push_char(value) {
            Ok(_) => (),
            Err(_) => panic!("No space for char in this buffer"),
        }
    }

    #[track_caller]
    #[inline]
    pub fn push_str<S: ?Sized + AsRef<str>>(&mut self, string: &S) {
        match self.try_push_str(string) {
            Ok(_) => (),
            Err(_) => panic!("No space for string in this buffer"),
        }
    }

    #[must_use]
    #[inline]
    pub fn pop(&mut self) -> Option<char> {
        let ch = self.as_str().chars().next_back()?;
        let new_len = self.inner.len() - ch.len_utf8();

        unsafe {
            self.set_len(new_len);
        }

        Some(ch)
    }

    #[inline]
    pub const unsafe fn set_len(&mut self, new_len: usize) {
        unsafe {
            self.inner.set_len(new_len);
        }
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit();
    }

    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear();
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
    pub const fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        self.inner.spare_capacity_mut()
    }
}

impl<'a, A: Allocator> Extend<char> for GrowableStringBuffer<'a, A> {
    #[track_caller]
    #[inline]
    fn extend<I: IntoIterator<Item = char>>(&mut self, iter: I) {
        for item in iter {
            self.push_char(item);
        }
    }
}

impl<'a, 'c, A: Allocator> Extend<&'c char> for GrowableStringBuffer<'a, A> {
    #[track_caller]
    #[inline]
    fn extend<I: IntoIterator<Item = &'c char>>(&mut self, iter: I) {
        Extend::extend(self, iter.into_iter().copied());
    }
}

impl<'a, 's, A: Allocator> Extend<&'s str> for GrowableStringBuffer<'a, A> {
    #[track_caller]
    #[inline]
    fn extend<I: IntoIterator<Item = &'s str>>(&mut self, iter: I) {
        for item in iter {
            self.push_str(item);
        }
    }
}

impl<'a, A: Allocator> AsRef<str> for GrowableStringBuffer<'a, A> {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl<'a, A: Allocator> AsMut<str> for GrowableStringBuffer<'a, A> {
    #[inline]
    fn as_mut(&mut self) -> &mut str {
        self.as_mut_str()
    }
}

impl<'a, A: Allocator> AsRef<[u8]> for GrowableStringBuffer<'a, A> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl<'a, A: Allocator> Borrow<str> for GrowableStringBuffer<'a, A> {
    #[inline]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl<'a, A: Allocator> BorrowMut<str> for GrowableStringBuffer<'a, A> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut str {
        self.as_mut_str()
    }
}

impl<'a, A: Allocator> Deref for GrowableStringBuffer<'a, A> {
    type Target = str;
    #[inline]
    fn deref(&self) -> &<Self as Deref>::Target {
        self.as_str()
    }
}

impl<'a, A: Allocator> DerefMut for GrowableStringBuffer<'a, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut <Self as Deref>::Target {
        self.as_mut_str()
    }
}

pub struct FromUtf8Error<'a, A: Allocator = Global> {
    bytes: Buffer<'a, u8, A>,
    error: Utf8Error,
}

impl<'a, A: Allocator> fmt::Debug for FromUtf8Error<'a, A> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmtr.debug_struct("FromUtf8Error")
            .field("bytes", &self.bytes)
            .field("error", &self.error)
            .finish()
    }
}

impl<'a, A: Allocator> FromUtf8Error<'a, A> {
    #[inline]
    pub(crate) const fn new(bytes: Buffer<'a, u8, A>, error: Utf8Error) -> Self {
        Self { bytes, error }
    }

    #[must_use]
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[must_use]
    #[inline]
    pub fn into_bytes(self) -> Buffer<'a, u8, A> {
        self.bytes
    }

    #[must_use]
    #[inline]
    pub const fn utf8_error(&self) -> &Utf8Error {
        &self.error
    }
}

impl<'a, A: Allocator> fmt::Display for FromUtf8Error<'a, A> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.utf8_error(), fmtr)
    }
}

impl<'a, A: Allocator> ErrorTrait for FromUtf8Error<'a, A> {
    #[inline]
    fn source(&self) -> Option<&(dyn ErrorTrait + 'static)> {
        Some(&self.error)
    }
}
