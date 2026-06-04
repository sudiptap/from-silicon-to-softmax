---
title: "Lesson 3 — Rust for Systems Programming"
date: "2026-06-03"
module: "bare-metal"
order: 3
tags: ["rust", "systems", "ownership", "borrowing", "no-gc", "abi"]
author: "Sudipta Pathak"
prerequisites: ["01-cpu-mental-model", "02-memory-hierarchy"]
---

# Lesson 3 — Rust for Systems Programming

## Why this matters

We're about to write a matrix multiply that gets 100× faster across four lessons. The language we use needs to let us see — and control — allocations, layout, copying, and SIMD. C and C++ let you do all of that; they also let you be subtly wrong about all of it. Python hides all of it; you can't see what you're not allowed to touch. Rust occupies the useful middle: it forces you to be honest about lifetimes and ownership, refuses to compile code that's silently wrong, and produces machine code competitive with C.

That's the practical pitch. The bigger reason: in ML systems work, you'll regularly read code in Rust — `candle`, `burn`, `tokenizers`, parts of `vllm` adjacent tooling, much of `safetensors`, the entire `ggml`/`llama.cpp` Rust ecosystem, `mistral.rs`, and most of the Rust-on-GPU frontier (Module 3 closes with this). Being able to read and write Rust comfortably is no longer optional in this corner of the stack. One small project — the matmul we're about to build — is enough to make the rest readable.

This lesson is not a Rust tutorial. It's the minimum mental model an ML systems person needs, framed around what's relevant for the rest of this module.

## Concept

Four ideas carry most of the weight.

### 1. Ownership: there is exactly one owner of every value

In Rust, every value has a single owner. When the owner goes out of scope, the value is freed. There is no garbage collector. There is no reference counting (unless you explicitly opt in with `Rc`/`Arc`). The compiler can prove at compile time when each allocation gets dropped, because the rules of ownership make it deterministic.

```rust
{
    let v = vec![1, 2, 3];       // v owns the allocation
    // ... use v ...
}                                 // v's scope ends; allocation is freed here
```

For a systems person, this is *liberating*. You know precisely when memory is freed. There are no GC pauses. There is no reference counting overhead by default. The allocation pattern of your program is exactly what the source code says.

When you "give" a value to a function:

```rust
fn consume(v: Vec<i64>) { /* ... */ }

let v = vec![1, 2, 3];
consume(v);
// v is no longer usable here; ownership moved into `consume`
```

This is the **move**. Conceptually, ownership transferred. Physically, what the compiler does is usually nothing — moves are typically zero-cost; the compiler just stops tracking the old name and starts tracking the new one.

### 2. Borrowing: temporary access without giving up ownership

If `consume` would have used `v` and discarded it, taking ownership is fine. If you want `consume` to look at `v` and let you use `v` afterwards, you **borrow**.

```rust
fn read_only(v: &[i64]) { /* ... */ }       // immutable borrow
fn modify(v: &mut [i64]) { /* ... */ }      // mutable borrow

let mut v = vec![1, 2, 3];
read_only(&v);          // pass shared reference; v still usable
modify(&mut v);         // pass exclusive reference; v still usable after
```

The rule the compiler enforces: at any given time, you have either **any number of immutable borrows or exactly one mutable borrow**. Never both. This is the "aliasing XOR mutability" rule, and it's why Rust programs are immune to whole classes of memory-safety bugs that haunt C++.

For performance work, this rule has a quiet bonus: the compiler can assume references don't alias, and emit better code. C compilers have to assume two `int*` arguments might point to the same memory unless you say otherwise (`restrict`); Rust's `&mut T` is automatically `restrict`.

### 3. No hidden costs: what you write is what you get

In Rust, `vec![0; 1_000_000]` allocates a million zeros. Once. On the heap. You can see it. You can control it. There's no surprise copy if you pass the `Vec` around by reference. There's no surprise reallocation unless you `push` past capacity. The cost model maps directly onto what the code says.

This matters for our matmul because we need to allocate the result matrix exactly once, reuse buffers across iterations, and never copy a row when we mean to view a slice of it. Rust makes all four of these natural and visible.

### 4. Zero-cost abstractions

Iterators, generics, traits — Rust's high-level features compile down to the same machine code you'd write by hand. `xs.iter().map(|x| x * 2).sum::<i64>()` produces, after optimization, the same instructions as the manual `for` loop. This is unusual: in most languages, "elegant" comes with a tax. In Rust, the tax is usually paid at the compiler.

You can — and we will — write idiomatic Rust for performance code without giving up the ability to read what the compiler produced.

## Code walkthrough

A scaffold for the matmul we'll grow over the next several lessons. Even at this stage, it shows the systems mindset.

```rust
// matmul.rs

/// A row-major matrix view backed by a flat buffer.
pub struct Mat<'a> {
    pub rows: usize,
    pub cols: usize,
    pub data: &'a [f32],
}

impl<'a> Mat<'a> {
    pub fn at(&self, i: usize, j: usize) -> f32 {
        debug_assert!(i < self.rows && j < self.cols);
        self.data[i * self.cols + j]
    }
}

pub struct MatMut<'a> {
    pub rows: usize,
    pub cols: usize,
    pub data: &'a mut [f32],
}

impl<'a> MatMut<'a> {
    pub fn at_mut(&mut self, i: usize, j: usize) -> &mut f32 {
        debug_assert!(i < self.rows && j < self.cols);
        &mut self.data[i * self.cols + j]
    }
}

/// Multiply A (m x k) by B (k x n) into C (m x n), row-major.
/// Naive triple loop — the version we'll beat.
pub fn matmul_naive(a: &Mat, b: &Mat, c: &mut MatMut) {
    assert_eq!(a.cols, b.rows);
    assert_eq!(a.rows, c.rows);
    assert_eq!(b.cols, c.cols);

    for i in 0..a.rows {
        for j in 0..b.cols {
            let mut acc = 0.0f32;
            for kk in 0..a.cols {
                acc += a.at(i, kk) * b.at(kk, j);
            }
            *c.at_mut(i, j) += acc;
        }
    }
}
```

A few things worth noticing in this small piece of code.

**Lifetimes (`'a`)** tell the compiler that the `Mat` borrowing `data` cannot outlive the buffer it borrows from. We never have to think about it — the compiler enforces it. We get pointer-like ergonomics with no risk of dangling references.

**Flat buffer, manual indexing.** We're explicit about row-major layout: `data[i * cols + j]`. A 2D Rust vector (`Vec<Vec<f32>>`) would be tempting but disastrous for performance — each row would be a separate heap allocation, with pointer indirection per access and no spatial locality between rows. The flat buffer is the cache-friendly shape.

**`debug_assert!` not `assert!`** for bounds. Real Rust slices already bounds-check; we want the option of removing our checks in release builds when needed, while keeping them in debug builds where bugs are cheap.

**Mutable access through `&mut self`.** We can't accidentally have two threads mutating the same matrix; the compiler refuses. We can't pass a `Mat` and a `MatMut` to the same buffer; the compiler refuses (aliasing-XOR-mutability). The safety properties are free; we get them by writing normal code.

### Reading what the compiler produced

A useful habit: look at the assembly of a small function. Rust makes this approachable with [Compiler Explorer](https://godbolt.org/) (set the language to Rust) or `cargo-show-asm` locally. Drop the inner loop of `matmul_naive` in, set the optimization level to `-O`, and notice:

- The compiler turns `f32` loads into vectorized loads on platforms that support it.
- The inner loop accumulator stays in a register.
- The `debug_assert!` disappears entirely in release mode.

That last point — code present in source but absent in machine code — is what "zero-cost" actually means. You'll see this often.

## Mental model & pitfalls

The single sentence to keep: **Rust forces you to make memory decisions explicit at compile time; in return, it generates exactly the machine code you described.**

Common pitfalls when coming from Python / numpy:

- **Reaching for `clone()` to silence the borrow checker.** This compiles, but it does a heap copy. In numeric inner loops, this is suicide. The compiler is usually pointing at a real ownership issue; fix the design, not the symptom.
- **Wrapping things in `Rc<RefCell<T>>` because that's what "made it work."** Almost never needed in numeric code. If you're reaching for `RefCell` in a matmul, step back; you've structured ownership wrong.
- **Using `Vec<Vec<f32>>` for matrices.** As above — every row is a separate heap allocation. Use a flat `Vec<f32>` plus explicit row stride.
- **Allocating in the inner loop.** Look out for `String`, `Vec::new()`, `format!`, or any `.collect()` inside hot loops. Each is a heap call.

Common pitfalls when coming from C++:

- **Writing constructor-heavy types** — Rust's `Drop` is fine, but the C++ habit of RAII'ing every resource into a type-with-destructor is overkill for numeric buffers. Plain `Vec` is the right tool.
- **Using `unsafe` for "performance"** — almost never necessary. The cases where `unsafe` is genuinely faster are narrow (raw SIMD intrinsics, FFI, hand-tuned data structures). For a matmul kernel, safe Rust on `-O3` is within a percent or two of `unsafe`.

## Hands-on (at home)

Set up the project we'll grow across the rest of the module.

```bash
mkdir matmul-from-scratch && cd matmul-from-scratch
cargo init --bin
```

Edit `Cargo.toml` to enable a release profile that's worth using:

```toml
[profile.release]
opt-level = 3
debug = true        # keep symbols for profiling
lto = "thin"
codegen-units = 1   # better codegen at the cost of compile time
```

`src/main.rs`:

```rust
use std::time::Instant;

mod matmul;
use matmul::{Mat, MatMut, matmul_naive};

fn make_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    // Tiny deterministic RNG to keep results reproducible.
    let mut s = seed;
    (0..rows * cols).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (u32::MAX as f32) - 0.5
    }).collect()
}

fn bench<F: Fn()>(name: &str, gflops_factor: f64, mut f: F) {
    let _ = (&mut f, &f); // shut up about unused mut
    f();
    let t0 = Instant::now();
    let runs = 5;
    for _ in 0..runs {
        f();
    }
    let dt = t0.elapsed().as_secs_f64() / runs as f64;
    let gflops = gflops_factor / dt / 1e9;
    println!("{name:>20}: {:>7.2} ms   {gflops:>6.2} GFLOPs", dt * 1e3);
}

fn main() {
    let m = 512usize;
    let k = 512usize;
    let n = 512usize;

    let a_buf = make_matrix(m, k, 1);
    let b_buf = make_matrix(k, n, 2);
    let mut c_buf = vec![0.0f32; m * n];

    let a = Mat { rows: m, cols: k, data: &a_buf };
    let b = Mat { rows: k, cols: n, data: &b_buf };
    let mut c = MatMut { rows: m, cols: n, data: &mut c_buf };

    // 2 * m * n * k float ops in a matmul.
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    bench("matmul_naive", flops, || {
        // Reset C each run so we measure the multiplication, not accumulation.
        for x in c.data.iter_mut() { *x = 0.0; }
        matmul_naive(&a, &b, &mut c);
    });
}
```

Add `src/matmul.rs` with the `Mat`, `MatMut`, and `matmul_naive` from the code walkthrough.

Run:

```bash
cargo run --release
```

You should see something like:

```
        matmul_naive:   85.20 ms     1.57 GFLOPs
```

Numbers will vary across machines. On Apple M-series you might see 1–3 GFLOPs. On a recent Intel/AMD desktop you might see 2–5 GFLOPs. The peak for one core of either is in the **50–100+ GFLOPs** range. We're at 1–5% of peak. That's the gap we'll close.

Take a screenshot or save the number. It's our baseline. Every lesson from here to the end of the module is a multiplier on this.

## Further reading

- *The Rust Book* (doc.rust-lang.org/book) — chapters 4 (ownership), 10 (generics/traits/lifetimes), and 16 (concurrency) are the relevant ones for this module. Skip the rest for now.
- *Rust for Rustaceans* (Jon Gjengset) — the advanced book. Worth picking up after you've written a few hundred lines of Rust.
- `cargo-show-asm` (`cargo install cargo-show-asm`) — print the assembly for any function in your crate. Indispensable for performance work.
- [Compiler Explorer](https://godbolt.org) — the same thing in a browser, useful for sharing.

Next lesson: the naive matmul. We measure it, understand exactly why it's slow given what we now know about caches, and set up the next four lessons of beating it.
