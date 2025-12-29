# Rotunda.

A fast, featureful arena allocator for Rust.

## Arena allocation

In many workloads, the lifetime of a heap allocation is scoped to a particular function call.
For example:

```rust
# #[derive(Default)]
# struct SomeType {};
# #[derive(Default)]
# struct OtherType {};
# fn use_values(_: &SomeType, _: &OtherType) {}
fn some_function() {
    let value = Box::new(SomeType::default());
    let value_2 = Box::new(OtherType::default());

    // Use `value` and `value_2` in the body of `some_function()`
    // ...
    use_values(&*value, &*value_2);

    // `value` and `value_2` are dropped and deallocated here as the function returns.
}
```

If many values need to be allocated and deallocated for a function body, this can be
inefficient as the heap allocator needs to do work to allocate and deallocate each
individual value.

To compensate for this, many strategies can be used so that values are allocated from
a single block of memory, reducing the work that the memory allocator needs to do.
One of these strategies is called arena allocation.

The idea behind arena allocation is to preallocate a large block of memory, and then
write values into that block as necessary. Values cannot be freed until the entire block
is finished with, when every value is implicitly deallocated at once.

# Examples

To rewrite the above example `some_function()` using arena allocation provided by this crate,
an `Arena` would need to be constructed or passed in to the function. Any transient data allocated
by the function would then be allocated from the `Arena`:

```rust
use rotunda::{Arena, handle::Handle};

# #[derive(Default)]
# struct SomeType {};
# #[derive(Default)]
# struct OtherType {};
# fn use_values(_: &SomeType, _: &OtherType) {}
fn some_function(arena: &Arena) {
    // Allocate the values as before:
    let value = Handle::new_in(&arena, SomeType::default());
    let value_2 = Handle::new_in(&arena, OtherType::default());

    // Use the values as before:
    use_values(&*value, &*value_2);

    // The values are dropped here, but the memory backing them in the arena is
    // not deallocated until the `Arena` is dropped, or the `Arena::reset()` method
    // is called.
}
```

The `Arena` can be visualised as an array of memory, with an integer marking the current
end of the `Arena`. When the `Arena` is first initialised and has a block allocated, then
the stack is empty:

```text
+-------+-----------------------------------------------------------+
|  (H)  |                                                           |
+-------+-----------------------------------------------------------+
^       ^
Header  Arena end
```

Then, when `value` and `value_2` are allocated into the arena, they
are written into the memory, and the end is adjusted to point into the
next available memory space:

```text
+-------+---------------+---------------+--------------------------------------+
|  (H)  |     (data)    |     (data)    |                                      |
+-------+---------------+---------------+--------------------------------------+
^       ^               ^               ^
Header  value           value_2         Arena end
```

Then, when `value` and `value_2` are dropped, their `drop` logic will be
called, but the backing storage of the `Arena` will not shrink to release
the memory; it will remain inaccessible for future allocations:

```text
+-------+-------------------------------+--------------------------------------+
|  (H)  |xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx|                                      |
+-------+-------------------------------+--------------------------------------+
^                                       ^
Header                                  Arena end
```

When the [`Arena::reset()`] method is called then the memory will be reclaimed,
and will be usable for future allocations. Note that no active references are
allowed into the `Arena` when [`Arena::reset()`] is called, so there is no
danger of referring to dangling memory. When the `Arena` is dropped, the
backing memory will be deallocated by the system allocator and returned to
the operating system.

If a block of memory in the arena is completely used up, then a new one will be
allocated. Each block stores a pointer to the next block in its header, so blocks
will be accessible. reseting an `Arena` will cause blocks to be stored in a free
list so that they can be re-used later, reducing pressure on the memory allocator.

## Handle Types

The `arena` module comes with a few specialised handle types to make working with `Arena`s
more ergonomic. These provide a few useful conveniences:

* They have a lifetime which ensures that data allocated from the `Arena` cannot outlive the
  `Arena`, preventing use after frees.
* They handle `Drop` logic when the handle goes out of scope, preventing resource leaks.

### [`Handle`]

The basic handle type is [`handle::Handle`], which is analogous to a `Box<T>` - it provides unique ownership
of an object allocated in an [`Arena`], allows mutation, and drops the object when it goes out of scope.

Read more in the [`handle`] module.

### [`RcHandle`] and [`WeakHandle`]

These are reference counted shared ownership handle types, analogous to `Rc<T>` and `Weak<T>`.

Read more in the [`rc_handle`] module.

### [`Buffer`]

An owned, growable buffer of elements backed by an allocation into an `Arena`. The [`Buffer`] has
a fixed maximum size which cannot be changed in the `Arena`.

Read more in the [`buffer`] module.

## [`StringBuffer`]

An owned, growable buffer containing utf8 encoded bytes.

Read more in the [`string_buffer`] module.

### [`LinkedList`]

A linked list of nodes, backed by an `Arena`. This allows an ordered collection of elements backed by
an `Arena` without requiring contiguous space for all elements in the `Arena`.

Read more in the [`linked_list`] module.

## Features

This crate can be customised with a few optional features:

### `allocator-api2`

This feature uses the `allocator-api2` crate for its definitions of the `Allocator` trait
and other supporting APIs. This allows `rotunda` to be built on the `stable` and `beta`
rust compilers.

It is required that at least one of either this feature or the `nightly` feature are enabled so that
an allocator API is provided.

### `nightly`

This feature enables usage of nightly features in `rotunda`, such as [`CoercePointee`]. It also
allows using the `Allocator` trait and supporting APIs from [`alloc::alloc`]. Note that this feature
will supersede `allocator-api2` (i.e. `alloc::alloc::Allocator` will be used over `allocator_api2::alloc::Allocator`)
if both are enabled.

### `std`

This feature enables integration with `std` traits, such as [`std::io::Read`].

### `serde`

This feature allows the contents of handle types to be serialized transparently using [`serde`].

[`Arena`]: https://docs.rs/rotunda/latest/rotunda/struct.Arena.html
[`Arena::reset()`]: https://docs.rs/rotunda/latest/rotunda/struct.Arena.html#method.reset
[`handle::Handle`]: https://docs.rs/rotunda/latest/rotunda/handle/struct.Handle.html
[`handle`]: https://docs.rs/rotunda/latest/rotunda/handle/index.html
[`Handle`]: https://docs.rs/rotunda/latest/rotunda/handle/struct.Handle.html
[`WeakHandle`]: https://docs.rs/rotunda/latest/rotunda/rc_handle/struct.WeakHandle.html
[`RcHandle`]: https://docs.rs/rotunda/latest/rotunda/rc_handle/struct.RcHandle.html
[`rc_handle`]: https://docs.rs/rotunda/latest/rotunda/rc_handle/index.html
[`Buffer`]: https://docs.rs/rotunda/latest/rotunda/buffer/struct.Buffer.html
[`StringBuffer`]: https://docs.rs/rotunda/latest/rotunda/string_buffer/struct.StringBuffer.html
[`string_buffer`]: https://docs.rs/rotunda/latest/rotunda/string_buffer/index.html
[`LinkedList`]: https://docs.rs/rotunda/latest/rotunda/linked_list/struct.LinkedList.html
[`linked_list`]: https://docs.rs/rotunda/latest/rotunda/linked_list/index.html
[`CoercePointee`]: https://doc.rust-lang.org/stable/core/marker/derive.CoercePointee.html
[`alloc::alloc`]: https://doc.rust-lang.org/stable/core/alloc/alloc/index.html
[`std::io::Read`]: https://doc.rust-lang.org/stable/std/io/trait.Read.html
[`serde`]: https://docs.rs/serde/latest
