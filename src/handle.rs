use crate::{
    Arena, buffer::Buffer, rc_handle::RcHandle, string_buffer::StringBuffer,
    buf,
};
use alloc::alloc::{Allocator, Layout};
use core::{
    any::Any,
    borrow::{Borrow, BorrowMut},
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    hint::assert_unchecked,
    iter::IntoIterator,
    marker::{CoercePointee, PhantomData, Unpin},
    mem::{self, ManuallyDrop, MaybeUninit},
    ops::{Deref, DerefMut},
    pin::Pin,
    ptr::{self, NonNull, Pointee, Thin},
    str,
};
#[cfg(feature = "std")]
use std::io::{self, BufRead, Read, Write};

/// An owned, mutable pointer to some memory backed by an [`Arena`], analogous to
/// [`Box<T>`].
///
/// See the [module documentation] for more information.
///
/// [`Arena`]: ../struct.Arena.html
/// [`Box<T>`]: https://doc.rust-lang.org/stable/std/boxed/struct.Box.html
/// [module documentation]: ./index.html
#[repr(transparent)]
#[derive(CoercePointee)]
pub struct Handle<'a, T: ?Sized> {
    pub(crate) ptr: NonNull<T>,
    _boo: PhantomData<(&'a Arena, T)>,
}

const _: () = assert!(mem::size_of::<Handle<()>>() == mem::size_of::<NonNull<()>>());
const _: () = assert!(mem::size_of::<Option<Handle<()>>>() == mem::size_of::<Handle<()>>());
const _: () = assert!(mem::align_of::<Handle<()>>() == mem::align_of::<NonNull<()>>());

// A handle can be sent to other threads if the type within is
// thread-safe - it is guaranteed to drop before the arena is
// deallocated thanks to borrowing rules.
unsafe impl<'a, T: ?Sized + Send> Send for Handle<'a, T> {}
unsafe impl<'a, T: ?Sized + Sync> Sync for Handle<'a, T> {}

impl<'a, T> Handle<'a, T> {
    /// Create a new `Handle` containing the given `value`.
    ///
    /// # Example
    ///
    /// ```
    /// # use rotunda::{Arena, handle::Handle};
    /// let arena = Arena::new();
    /// let handle = Handle::new_in(&arena, 156);
    /// # let _ = handle;
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_in<A: Allocator>(arena: &'a Arena<A>, value: T) -> Self {
        let handle = Handle::new_uninit_in(arena);
        Handle::init(handle, value)
    }

    /// Converts the handle into a `Handle<[T]>` with a slice length of `1`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rotunda::{Arena, handle::Handle};
    /// let arena = Arena::new();
    /// let handle = Handle::new_in(&arena, 'c');
    /// let slice_handle = Handle::into_slice(handle);
    /// assert_eq!(&*slice_handle, &['c']);
    /// ```
    #[must_use]
    #[inline]
    pub const fn into_slice(this: Self) -> Handle<'a, [T]> {
        let slice = NonNull::slice_from_raw_parts(this.ptr, 1);
        let _this = ManuallyDrop::new(this);
        Handle {
            ptr: slice,
            _boo: PhantomData,
        }
    }
}

impl<'a, T: Default> Handle<'a, T> {
    /// Create a new `Handle` in `arena`, containing the default value of `T`.
    ///
    /// # Example
    ///
    /// ```
    /// # use rotunda::{Arena, handle::Handle};
    /// let arena = Arena::new();
    /// let handle = Handle::<Vec<i32>>::new_default_in(&arena);
    /// assert_eq!(&*handle, &<Vec<i32>>::default());
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_default_in<A: Allocator>(arena: &'a Arena<A>) -> Self {
        Handle::new_in(arena, Default::default())
    }
}

impl<'a, T> Handle<'a, MaybeUninit<T>> {
    /// Create a new `Handle` in `arena`, containing an uninitialized `MaybeUninit<T>`.
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_uninit_in<A: Allocator>(arena: &'a Arena<A>) -> Self {
        let layout = Layout::new::<T>().pad_to_align();
        let ptr = arena.alloc_raw(layout).cast::<MaybeUninit<T>>();
        unsafe { Handle::from_raw(ptr.as_ptr()) }
    }

    /// Create a new `Handle` in `arena`, containing a zeroed `MaybeUninit<T>`.
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_uninit_zeroed_in<A: Allocator>(arena: &'a Arena<A>) -> Self {
        let layout = Layout::new::<T>().pad_to_align();
        let ptr = arena.alloc_raw_zeroed(layout).cast::<MaybeUninit<T>>();
        unsafe { Handle::from_raw(ptr.as_ptr()) }
    }
}

impl<'a, T, const N: usize> Handle<'a, [MaybeUninit<T>; N]> {
    #[track_caller]
    #[inline]
    #[must_use]
    pub fn new_array_uninit_in<A: Allocator>(arena: &'a Arena<A>) -> Self {
        let handle = Handle::<'_, MaybeUninit<[T; N]>>::new_uninit_in(arena);
        Handle::<'_, MaybeUninit<[T; N]>>::transpose_inner_uninit(handle)
    }

    #[track_caller]
    #[inline]
    #[must_use]
    pub fn new_array_uninit_zeroed_in<A: Allocator>(arena: &'a Arena<A>) -> Self {
        let handle = Handle::<'_, MaybeUninit<[T; N]>>::new_uninit_zeroed_in(arena);
        Handle::<'_, MaybeUninit<[T; N]>>::transpose_inner_uninit(handle)
    }
}

impl<'a, T> Handle<'a, [MaybeUninit<T>]> {
    #[track_caller]
    #[inline]
    #[must_use]
    pub fn new_slice_uninit_in<A: Allocator>(arena: &'a Arena<A>, slice_len: usize) -> Self {
        let type_layout = Layout::new::<T>();
        let (array_layout, ..) = Layout::repeat(&type_layout, slice_len).expect("size overflow");

        let ptr = {
            let ptr = arena.alloc_raw(array_layout).cast::<MaybeUninit<T>>();
            NonNull::slice_from_raw_parts(ptr, slice_len)
        };

        unsafe { Handle::from_raw(ptr.as_ptr()) }
    }
}

impl<'a, T, const N: usize> Handle<'a, [T; N]> {
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_array_from_fn_in<A: Allocator, F: FnMut(usize) -> T>(
        arena: &'a Arena<A>,
        f: F,
    ) -> Self {
        let buffer = Buffer::from_fn_in(arena, N, f);
        unsafe { Handle::into_array_unchecked::<N>(buffer.into_slice_handle()) }
    }
}

impl<'a, T: Copy, const N: usize> Handle<'a, [T; N]> {
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_array_splat_in<A: Allocator>(arena: &'a Arena<A>, value: T) -> Self {
        let buffer = buf!([value; N] in arena);
        unsafe { Handle::into_array_unchecked::<N>(buffer.into_slice_handle()) }
    }
}

impl<'a, T, const N: usize> Handle<'a, MaybeUninit<[T; N]>> {
    #[must_use]
    #[inline]
    pub fn transpose_inner_uninit(self) -> Handle<'a, [MaybeUninit<T>; N]> {
        unsafe { mem::transmute(self) }
    }
}

impl<'a, T, const N: usize> Handle<'a, [MaybeUninit<T>; N]> {
    #[must_use]
    #[inline]
    pub fn transpose_outer_uninit(self) -> Handle<'a, MaybeUninit<[T; N]>> {
        unsafe { mem::transmute(self) }
    }

    #[must_use]
    #[inline]
    pub const unsafe fn assume_init_array(self) -> Handle<'a, [T; N]> {
        unsafe { mem::transmute(self) }
    }
}

impl<'a, T: ?Sized> Handle<'a, T> {
    #[must_use]
    #[inline]
    pub const fn into_raw(this: Self) -> *mut T {
        let raw = this.ptr.as_ptr();
        let _this = ManuallyDrop::new(this);
        raw
    }

    #[inline]
    pub const unsafe fn from_raw(raw: *mut T) -> Self {
        let ptr = unsafe { NonNull::new_unchecked(raw) };

        Self {
            ptr,
            _boo: PhantomData,
        }
    }

    #[inline]
    #[must_use]
    pub const fn as_ptr(this: &Self) -> *const T {
        this.ptr.as_ptr().cast_const()
    }

    #[inline]
    #[must_use]
    pub const fn as_mut_ptr(this: &mut Self) -> *mut T {
        this.ptr.as_ptr()
    }

    #[inline]
    #[must_use]
    pub const fn into_pin(this: Self) -> Pin<Self> {
        unsafe { Pin::new_unchecked(this) }
    }
}

impl<'a, T: Unpin> Handle<'a, T> {
    #[inline]
    pub const fn into_inner(this: Self) -> T {
        let inner = unsafe { this.ptr.read() };
        let _this = ManuallyDrop::new(this);
        inner
    }

    #[inline]
    pub const fn extract_inner(this: Self) -> (T, Handle<'a, MaybeUninit<T>>) {
        let inner = unsafe { this.ptr.read() };
        let handle = unsafe { Handle::from_raw(this.ptr.cast::<MaybeUninit<T>>().as_ptr()) };

        let _this = ManuallyDrop::new(this);
        (inner, handle)
    }

    #[inline]
    pub const fn replace(this: &mut Self, mut value: T) -> T {
        unsafe {
            mem::swap(this.ptr.as_mut(), &mut value);
        }
        value
    }
}

impl<'a> Handle<'a, dyn Any> {
    #[inline]
    pub fn downcast<T: Any>(self) -> Result<Handle<'a, T>, Handle<'a, dyn Any>> {
        if self.is::<T>() {
            unsafe { Ok(Self::downcast_unchecked::<T>(self)) }
        } else {
            Err(self)
        }
    }

    #[must_use]
    #[inline]
    pub unsafe fn downcast_unchecked<T: Any>(self) -> Handle<'a, T> {
        unsafe {
            let ptr = Self::into_raw(self);
            Handle::from_raw(ptr as *mut T)
        }
    }
}

impl<'a> Handle<'a, dyn Any + Send> {
    #[inline]
    pub fn downcast<T: Any>(self) -> Result<Handle<'a, T>, Handle<'a, dyn Any + Send>> {
        if self.is::<T>() {
            unsafe { Ok(Self::downcast_unchecked::<T>(self)) }
        } else {
            Err(self)
        }
    }

    #[must_use]
    #[inline]
    pub unsafe fn downcast_unchecked<T: Any>(self) -> Handle<'a, T> {
        unsafe {
            let ptr = Self::into_raw(self);
            Handle::from_raw(ptr as *mut T)
        }
    }
}

impl<'a> Handle<'a, dyn Any + Send + Sync> {
    #[inline]
    pub fn downcast<T: Any>(self) -> Result<Handle<'a, T>, Handle<'a, dyn Any + Send + Sync>> {
        if self.is::<T>() {
            unsafe { Ok(Self::downcast_unchecked::<T>(self)) }
        } else {
            Err(self)
        }
    }

    #[must_use]
    #[inline]
    pub unsafe fn downcast_unchecked<T: Any>(self) -> Handle<'a, T> {
        unsafe {
            let ptr = Self::into_raw(self);
            Handle::from_raw(ptr as *mut T)
        }
    }
}

impl<'a, T: ?Sized + Pointee> Handle<'a, T> {
    #[must_use]
    #[inline]
    pub const unsafe fn from_raw_parts(
        ptr: *mut impl Thin,
        metadata: <T as Pointee>::Metadata,
    ) -> Self {
        let ptr = ptr::from_raw_parts_mut(ptr, metadata);
        unsafe { Handle::from_raw(ptr) }
    }
}

impl<'a, T> Handle<'a, [T]> {
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_slice_with_fn_in<A: Allocator, F: FnMut(usize) -> T>(
        arena: &'a Arena<A>,
        slice_len: usize,
        mut f: F,
    ) -> Self {
        let mut buf = Buffer::with_capacity_in(arena, slice_len);
        for i in 0..slice_len {
            unsafe {
                buf.push_unchecked(f(i));
            }
        }

        buf.into_slice_handle()
    }

    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_slice_from_iter_in<A: Allocator, I: IntoIterator<Item = T>>(
        arena: &'a Arena<A>,
        iter: I,
    ) -> Self
    where
        I::IntoIter: ExactSizeIterator,
    {
        Buffer::new_in(arena, iter).into_slice_handle()
    }

    #[inline]
    pub const fn empty() -> Self {
        unsafe { Handle::from_raw_parts(ptr::dangling_mut::<T>(), 0) }
    }

    #[must_use]
    #[inline]
    pub const fn into_buffer(this: Self) -> Buffer<'a, T> {
        Buffer::from_slice_handle(this)
    }

    #[track_caller]
    #[must_use]
    #[inline]
    pub const fn split_at(this: Self, mid: usize) -> (Handle<'a, [T]>, Handle<'a, [T]>) {
        if Self::check_split(&this, mid) {
            unsafe { Self::split_at_unchecked(this, mid) }
        } else {
            panic!("mid > len")
        }
    }

    #[must_use]
    #[inline]
    pub const fn split_at_checked(
        this: Self,
        mid: usize,
    ) -> Result<(Handle<'a, [T]>, Handle<'a, [T]>), Handle<'a, [T]>> {
        if Self::check_split(&this, mid) {
            unsafe { Ok(Self::split_at_unchecked(this, mid)) }
        } else {
            Err(this)
        }
    }

    #[must_use]
    #[inline]
    pub const unsafe fn split_at_unchecked(
        this: Self,
        mid: usize,
    ) -> (Handle<'a, [T]>, Handle<'a, [T]>) {
        let len = unsafe { this.ptr.as_ref().len() };
        unsafe {
            assert_unchecked(mid <= len);
        }

        let ptr = Self::into_raw(this) as *mut T;

        let lhs = ptr::slice_from_raw_parts_mut(ptr, mid);
        let rhs = unsafe { ptr::slice_from_raw_parts_mut(ptr.add(mid), len.unchecked_sub(mid)) };

        unsafe { (Handle::from_raw(lhs), Handle::from_raw(rhs)) }
    }

    #[must_use]
    #[inline]
    pub const fn transpose_into_uninit(this: Self) -> Handle<'a, [MaybeUninit<T>]> {
        let ptr = unsafe { NonNull::new_unchecked(this.ptr.as_ptr() as *mut [MaybeUninit<T>]) };
        let _this = ManuallyDrop::new(this);

        Handle {
            ptr,
            _boo: PhantomData,
        }
    }

    #[inline]
    #[must_use]
    const fn check_split(this: &Self, mid: usize) -> bool {
        mid <= this.ptr.len()
    }

    #[inline]
    pub(crate) const unsafe fn set_len(this: &mut Self, new_len: usize) {
        let ptr = this.ptr.as_ptr() as *mut T;
        let ptr = unsafe { NonNull::new_unchecked(ptr::from_raw_parts_mut(ptr, new_len)) };

        this.ptr = ptr;
    }

    #[must_use]
    #[inline]
    pub(crate) unsafe fn into_array_unchecked<const N: usize>(this: Self) -> Handle<'a, [T; N]> {
        let ptr = Self::into_raw(this) as *mut [T; N];
        unsafe { Handle::from_raw(ptr) }
    }
}

impl<'a, T: Copy> Handle<'a, [T]> {
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_slice_splat_in<A: Allocator>(
        arena: &'a Arena<A>,
        slice_len: usize,
        value: T,
    ) -> Self {
        let mut buf = Buffer::with_capacity_in(arena, slice_len);
        for _ in 0..slice_len {
            unsafe {
                buf.push_unchecked(value);
            }
        }
        buf.into_slice_handle()
    }
}

impl<'a, T> Handle<'a, MaybeUninit<T>> {
    #[inline]
    pub const unsafe fn assume_init(this: Self) -> Handle<'a, T> {
        let ptr = this.ptr.cast();
        let _this = ManuallyDrop::new(this);
        Handle {
            ptr,
            _boo: PhantomData,
        }
    }

    #[inline]
    pub const fn init(this: Self, value: T) -> Handle<'a, T> {
        unsafe {
            ptr::write(this.ptr.as_ptr(), MaybeUninit::new(value));
            Self::assume_init(this)
        }
    }
}

impl<'a, T> Handle<'a, [MaybeUninit<T>]> {
    #[inline]
    pub const unsafe fn assume_init_slice(mut this: Self) -> Handle<'a, [T]> {
        let len = this.ptr.len();
        let ptr = unsafe { this.ptr.as_mut().as_mut_ptr() as *mut T };
        let _this = ManuallyDrop::new(this);

        let ptr = unsafe { NonNull::slice_from_raw_parts(NonNull::new_unchecked(ptr), len) };

        Handle {
            ptr,
            _boo: PhantomData,
        }
    }
}

impl<'a> Handle<'a, str> {
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn new_str_in<S: ?Sized + AsRef<str>, A: Allocator>(
        arena: &'a Arena<A>,
        string: &S,
    ) -> Self {
        fn inner<'a, A: Allocator>(arena: &'a Arena<A>, string: &str) -> Handle<'a, str> {
            let len = string.len();
            let hndl = Handle::<'_, [MaybeUninit<u8>]>::new_slice_uninit_in(arena, len);
            let data = Handle::into_raw(hndl) as *mut MaybeUninit<u8> as *mut _;

            unsafe {
                ptr::copy_nonoverlapping(string.as_ptr(), data, string.len());

                let ptr = ptr::from_raw_parts_mut(data, len);
                Handle::from_raw(ptr)
            }
        }

        inner(arena, string.as_ref())
    }

    #[must_use]
    #[inline]
    pub const fn empty_str() -> Self {
        unsafe { Handle::from_raw_parts(ptr::dangling_mut::<u8>(), 0) }
    }
}

impl<'a, T> Default for Handle<'a, [T]> {
    #[inline]
    fn default() -> Self {
        Handle::empty()
    }
}

impl<'a> Default for Handle<'a, str> {
    #[inline]
    fn default() -> Self {
        Self::empty_str()
    }
}

impl<'a, T: ?Sized + fmt::Debug> fmt::Debug for Handle<'a, T> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.as_ref(), fmtr)
    }
}

impl<'a, T: ?Sized + fmt::Display> fmt::Display for Handle<'a, T> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.as_ref(), fmtr)
    }
}

impl<'a, T: ?Sized> fmt::Pointer for Handle<'a, T> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, fmtr)
    }
}

impl<'a, T: ?Sized> AsRef<T> for Handle<'a, T> {
    #[inline]
    fn as_ref(&self) -> &T {
        unsafe { self.ptr.as_ref() }
    }
}

impl<'a, T: ?Sized> AsMut<T> for Handle<'a, T> {
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        unsafe { self.ptr.as_mut() }
    }
}

impl<'a, T: ?Sized> Borrow<T> for Handle<'a, T> {
    #[inline]
    fn borrow(&self) -> &T {
        self.as_ref()
    }
}

impl<'a, T: ?Sized> BorrowMut<T> for Handle<'a, T> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut T {
        self.as_mut()
    }
}

impl<'sl: 'a, 'a, T: 'a> IntoIterator for &'sl Handle<'a, [T]> {
    type Item = &'a T;
    type IntoIter = <&'a [T] as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        AsRef::<[T]>::as_ref(self).into_iter()
    }
}

impl<'sl: 'a, 'a, T: 'a> IntoIterator for &'sl mut Handle<'a, [T]> {
    type Item = &'a mut T;
    type IntoIter = <&'a mut [T] as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        AsMut::<[T]>::as_mut(self).into_iter()
    }
}

impl<'a, T> IntoIterator for Handle<'a, [T]> {
    type Item = T;
    type IntoIter = <Buffer<'a, T> as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Buffer::from_slice_handle(self).into_iter()
    }
}

impl<'a, T> From<Buffer<'a, T>> for Handle<'a, [T]> {
    #[inline]
    fn from(value: Buffer<'a, T>) -> Self {
        value.into_slice_handle()
    }
}

impl<'a> From<StringBuffer<'a>> for Handle<'a, str> {
    #[inline]
    fn from(value: StringBuffer<'a>) -> Self {
        value.into_str_handle()
    }
}

impl<'a, T: ?Sized> Drop for Handle<'a, T> {
    #[inline]
    fn drop(&mut self) {
        let p = self.ptr.as_ptr();
        unsafe {
            ptr::drop_in_place(p);
        }

        // cleanup of backing allocation happens in `Arena::reset()` or `Arena::drop()`.
    }
}

impl<'a, T: ?Sized> TryFrom<RcHandle<'a, T>> for Handle<'a, T> {
    type Error = RcHandle<'a, T>;
    #[inline]
    fn try_from(value: RcHandle<'a, T>) -> Result<Self, Self::Error> {
        RcHandle::try_into_handle(value)
    }
}

impl<'a, T, const N: usize> TryFrom<Handle<'a, [T]>> for Handle<'a, [T; N]> {
    type Error = Handle<'a, [T]>;
    #[inline]
    fn try_from(value: Handle<'a, [T]>) -> Result<Self, Self::Error> {
        if value.len() == N {
            unsafe {
                let ptr = Handle::into_raw(value).cast::<[T; N]>();
                Ok(Handle::from_raw(ptr))
            }
        } else {
            Err(value)
        }
    }
}

impl<'a, T: ?Sized> Deref for Handle<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<'a, T: ?Sized> DerefMut for Handle<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}

impl<'a, T: ?Sized + PartialEq> PartialEq for Handle<'a, T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_ref().eq(other.as_ref())
    }
}

impl<'a, T: ?Sized + PartialEq> PartialEq<T> for Handle<'a, T> {
    #[inline]
    fn eq(&self, other: &T) -> bool {
        self.as_ref().eq(other)
    }
}

impl<'a, T: ?Sized + Eq> Eq for Handle<'a, T> {}

impl<'a, T: ?Sized + PartialOrd> PartialOrd for Handle<'a, T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_ref().partial_cmp(other.as_ref())
    }
}

impl<'a, T: ?Sized + PartialOrd> PartialOrd<T> for Handle<'a, T> {
    #[inline]
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        self.as_ref().partial_cmp(other)
    }
}

impl<'a, T: ?Sized + Ord> Ord for Handle<'a, T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_ref().cmp(other.as_ref())
    }
}

impl<'a, T: ?Sized + Hash> Hash for Handle<'a, T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state)
    }
}

#[cfg(feature = "std")]
impl<'a, R: ?Sized + Read> Read for Handle<'a, R> {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.as_mut().read(buf)
    }
}

#[cfg(feature = "std")]
impl<'a, R: ?Sized + BufRead> BufRead for Handle<'a, R> {
    #[inline]
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.as_mut().fill_buf()
    }

    #[inline]
    fn consume(&mut self, amount: usize) {
        self.as_mut().consume(amount)
    }
}

#[cfg(feature = "std")]
impl<'a, W: ?Sized + Write> Write for Handle<'a, W> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.as_mut().write(buf)
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        self.as_mut().flush()
    }
}
