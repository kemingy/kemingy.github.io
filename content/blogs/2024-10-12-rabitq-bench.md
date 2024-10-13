+++
title = "Improve an algorithm performance step by step"
[taxonomies]
categories = ["Technology", "Rust"]
+++

Recently, I've been working on a new approximate nearest neighbor search algorithm called [RaBitQ](https://arxiv.org/abs/2405.12497). The author has already provided a [C++ implementation](https://github.com/gaoj0017/RaBitQ) that runs quite fast. I tried to [rewrite it in Rust](https://github.com/kemingy/rabitq) (yet another RiiR). But I found that my implementation is much slower than the original one. Here is how I improve the performance step by step.

<!-- more -->

## Prepare the environment

### Datasets

The most important thing is to have some reasonable datasets. Since the paper already demonstrate some results on the `sift_dim128_1m_l2` and `gist_dim960_1m_l2` datasets, 128 and 960 dimensions are typical and 1_000_000 vectors should be sufficient for benchmark purpose. I decided to use them as well. The datasets can be downloaded from [here](http://corpus-texmex.irisa.fr/). (Yes, I know this site doesn't have TLS and it only provides FTP downloads).

The format used by these datasets is called `fvecs/ivecs`, which is a common vector format:

```python
| dim (4 bytes) | vector (4 * dim bytes) |
| dim (4 bytes) | vector (4 * dim bytes) |
...
| dim (4 bytes) | vector (4 * dim bytes) |
```

You can get the read/write script from my [gist](https://gist.github.com/kemingy/2f503fcfff86b9e0197e975c02359157).

### Profiling tool

I use [samply](https://github.com/mstange/samply) to profile the Rust code. It has a nice integration with the [Firefox Profiler](https://profiler.firefox.com/). You can also share the profiling results with others by uploading them to the cloud. Here is [an example of the C++ version profiling on GIST](https://share.firefox.dev/3Y4Hppz). The FlameGraph and CallTree are the most common views. Remember to grant the performance event permission and increase the `mlock` limit:

```bash
echo '1' | sudo tee /proc/sys/kernel/perf_event_paranoid
sudo sysctl kernel.perf_event_mlock_kb=2048
```

The [GodBolt](https://godbolt.org/) compiler explorer is also useful for comparing the assembly function code between C++ and Rust.

### Cargo profile

To include the debug information in the release build, you can add another profile to the `Cargo.toml`:

```toml
[profile.perf]
inherits = "release"
debug = true
codegen-units = 16
```

The compiling cost and runtime speed can greatly affect the profiling user experience.

- `cargo build` has a faster compile speed, but the code may be slower than pure Python
- `cargo build --release` runs fast but it might take a long time to compile

For benchmarking, we have no choice but to use the `opt-level = 3`.

I saw some advice to use the following settings:

```toml
codegen-units = 1
lto = "fat"
panic = "abort"
```

In my case, this only slows down the compilation speed and doesn't improve the performance at all.

### Benchmark tool

[Criterion](https://github.com/bheisler/criterion.rs) is a good statistics-driven benchmark tool. I create another [repo](https://github.com/kemingy/rs_bench) to store all the related benchmark code. It turns out that I should put them in the same repo.

One thing to note is that the benchmark results are not very stable. I have seen **`±10%`** differences without modifying the code. If you're using your laptop, this could be even worse since the CPU might be underclocked due to the high temperature.

I suggest to benchmark the function with several different parameters. In this case, I use different vector dimensions. If the results for all the dimensions are positive, it usually means that the improvement is effective.

### Metrics

Remember to add some metrics from the start. Many bugs and performance issues can be found by checking the metrics. I use `AtomicU64` directly since the current requirements are simple. I may switch to the [Prometheus metrics](https://github.com/prometheus/client_rust) later.

Note that too many metrics/logging/traces can also affect the performance. So be careful when adding them.

### Resources

During the benchmark, I noticed that the end-to-end QPS is extremely unstable. I could get a **15%** improvement or deterioration the nex day morning without recompiling the code. Then I found that the CPUs are not completely idle as I have VSCode + Rust Analyzer, it seems they don't consume much CPU but they do affect the benchmark results heavily. Even though I'm using [Intel Core i7-13700K](https://www.intel.com/content/www/us/en/products/sku/230500/intel-core-i713700k-processor-30m-cache-up-to-5-40-ghz/specifications.html), which has 8 performance cores and 8 efficient cores, also the program is single-threaded.

I use [`taskset`](https://www.man7.org/linux/man-pages/man1/taskset.1.html) to bind the process to a specific CPU. This way it won't be affected by mixed cores scheduling.

Note that Intel Core 13th/14th CPUs are affected by the instability problem due to the extremely high voltage. I have fixed this in the BIOS.

Cloud VMs may not be affected by the CPU temperature, but the cloud providers may have their own CPU throttling and overbooking policies.

## Step by Step Improvement

### Start with an naive implementation

My [first release](https://github.com/kemingy/rabitq/tree/dbfd54bd5d739b0729dc28e6fbd8d5413b019561) implemented the RaBitQ algorithm based on an algebra library called [nalgebra](https://docs.rs/nalgebra). The main reason is that I need to use the QR decomposition to obtain the orthogonal matrix, which is the key step in the RaBitQ algorithm. Also, a mature linear algebra library provides many useful functions for manipulating the matrix and vectors, making it easier for me to implement the algorithm. Imagine that implementing an algorithm involving matrix multiplication, projection and decomposition in Python without `numpy`, it's a nightmare.

I thought that the performance should be good since `nalgebra` is optimized for such kind of scenarios. But the benchmark shows that is much slower than I expected. I guess reimplementing it in `numpy` would be much faster :(

According to the [profiling](https://share.firefox.dev/3AwiVNR), there are lots of `f32::clone()` calls. It takes about 33% of the total time, or 44% if you focus on the `query_one` function. This reminds me that I can preallocate the memory for some vectors and reuse it in the iteration, a very common trick. So instead of using `(x - y).norm_squared()`, I need to pre-declare another vector that stores the result of `(x - y)`, which ends up being `x.sub_to(y, &mut z); z.norm_squared()`. See the [commit 23f9aff](https://github.com/kemingy/rabitq/commit/23f9aff4c8b3303c0a03ac9a7472ada8cc915a3b).

Like most of the algebra libraries, it stores the matrix in the column-major order, which means iterating over the column could be faster than over the row. It's a bit annoying because I have to transpose the matrix before the iteration, and not all the vector/matrix multiplications can detect the dimension mismatch error (`1 x dyn` or `dyn x 1`) during compilation.

### CPU target

RaBitQ uses the binary dot product distance to estimate the approximate distance, which is computed by:

```rust
fn binary_dot_product(x: &[u64], y: &[u64]) -> u32 {
    assert_eq!(x.len(), y.len());
    let mut res = 0;
    for i in 0..x.len() {
        res += (x[i] & y[i]).count_ones();
    }
    res
}
```

Here the [`u64::count_ones()`](https://doc.rust-lang.org/std/primitive.u64.html#method.count_ones) would use intrinsics directly, I thought. It turns out that I still need to enable the `popcnt` feature during the compilation. This could be done by using the `RUSTFLAGS="-C target-feature=+popcnt"`, but I prefer `RUSTFLAGS="-C target-cpu=native"`, which enables all the CPU features supported by the current CPU, but also makes the binary non-portable, which is fine for now. The following sections also require this `env` to enable the AVX2 features.

You can use the following command to check your CPU features:

```bash
rustc --print=cfg -C target-cpu=native | rg target_feature
```

### SIMD

The key function for the nearest neighbor search is the distance function, which in this case is the Euclidean distance. We usually use the L2 square distance to avoid the square root computation. The naive implementation is as follows:

```rust
{
    y.sub_to(x, &mut residual);
    residual.norm_squared()
}
```

After the profiling, I found that it still has `f32::clone()`. By checking the source code of `nalgebra`, I found that there are many `clone` for some reasons I don't know. I decide to write the SIMD by hand. Fortunately, [hnswlib](https://github.com/nmslib/hnswlib) (a popular HNSW implementation) already implements [this](https://github.com/nmslib/hnswlib/blob/master/hnswlib/space_l2.h).

This eliminates the `f32::clone()` in the distance computation and improves the QPS by **28%** for SIFT. Check the [commit 5f82fcc](https://github.com/kemingy/rabitq/commit/5f82fccf8b39964ef1f66e9927fb126fd6886765).

My CPU doesn't support AVX512, so I use the AVX2 version. You can check the [Steam Hardware Stats](https://store.steampowered.com/hwsurvey/), it lists the SIMD support in the "*Other Settings*". **100%** users have SSE3, **94.61%** users have AVX2, only **13.06%** users have AVX512F. Of course this statistic is biased, most of the cloud Intel CPUs have AVX512 support, game players cannot represent all the users.

To use SIMD, the most useful guide is the [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#). It's better to download the website as the online experience is not good. Remember to check the "**latency**" and "**throughput**" of the intrinsics, otherwise your code may be slower than the normal version.

Another resource is the [x86 Intrinsics Cheat Sheet](https://db.in.tum.de/~finis/x86%20intrinsics%20cheat%20sheet%20v1.0.pdf). This is good for newbies like me.

[@ashvardanian](https://github.com/ashvardanian) has a [post](https://ashvardanian.com/posts/simsimd-faster-scipy/#tails-of-the-past-the-significance-of-masked-loads) about the "mask load" that solves the tail elements problem (requires AVX512).

To make the code work on other platforms:

```rust
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
{
    if is_x86_feature_detected!("avx2") {
        // AVX2 version
    } else {
        // normal version
    }
}
```

There are some useful crates for writing better `cfg` for the SIMD, let's keep it simple for now.

### More SIMD

SIMD is like a hammer, now I need to find more nails in the code.

- rewrite the `binarize_vector` function with AVX2 in [commit f114fc1](https://github.com/kemingy/rabitq/commit/f114fc1ec58686596ade0df02a96fcf04b0bf828) improves the QPS by **32%** for GIST.

~~Compared to the original C++ version, this implementation is also branchless.~~ When enabling `opt-level=3`, this can be optimied by the compiler. See the [assembly](https://godbolt.org/z/hjP5qjabz).

```diff
- let shift = if (i / 32) % 2 == 0 { 32 } else { 0 };
+ let shift = ((i >> 5) & 1) << 5;
```

### Scalar quantization

To eliminate more `f32::clone()` in the code, I decided to replace more `nalgebra` functions with the manual implementation. The `min` and `max` functions are the most common ones. The `nalgebra` version is like this:

```rust
let lower_bound = residual.min();
let upper_bound = residual.max();
```

This can be done by:

```rust
fn min_max(vec: &[f32]) -> (f32, f32) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for v in vec.iter() {
        if *v < min {
            min = *v;
        }
        if *v > max {
            max = *v;
        }
    }
    (min, max)
}
```

I used to use `f32::min()` and `f32::max()` because they are convenient. But for non-(asc/desc) vectors, `if` has a better performance.

Instead of iterating through the vector several times in a function chain and computing the scalar quantization with sum in different iterations:

```rust
let y_scaled = residual.add_scalar(-lower_bound) * one_over_delta + &self.rand_bias;
let y_quantized = y_scaled.map(|v| v.to_u8().expect("convert to u8 error"));
let scalar_sum = y_quantized.iter().fold(0u32, |acc, &v| acc + v as u32);
```

We can do this in one loop:

```rust
{
    let mut sum = 0u32;
    for i in 0..vec.len() {
        let q = ((vec[i] - lower_bound) * multiplier + bias[i]) as u8;
        quantized[i] = q;
        sum += q as u32;
    }
    sum
}
```

For scalar quantization, we are sure that the `f32` can be converted to `u8`, so we can use `as u8` instead of `to_u8().unwrap()`.

The [commit af39c1c](https://github.com/kemingy/rabitq/commit/af39c1ce47eb8ea32e11f47b99548e77846397ea) & [commit d2d51b0](https://github.com/kemingy/rabitq/commit/d2d51b0785f0234df4d83a60eea96a36486a1120) improved the QPS by **31%** for GIST.

The following part can also be rewritten with SIMD, which improves the QPS by **12%** for GIST:

- min/max: [commit c97be68](https://github.com/kemingy/rabitq/commit/c97be68c13c7b4498b564afe3de2a1f6d8bca5ce) & [commit e5a4af0](https://github.com/kemingy/rabitq/commit/e5a4af05433bf724da6902d34a745b4b2bdefd8d)
- scalar quantization: [commit 28efe09](https://github.com/kemingy/rabitq/commit/28efe097a46696bb1a5469db22e500bafdc04514)

I also tried replacing `tr_mul` with SIMD, which is a vector projection. It turns out that `nalgebra` uses [`BLAS`](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) here, so the performance stays the same.

### Yet another algebra crate: faer

I found another Rust algebra crate called [faer](https://github.com/sarah-quinones/faer-rs) while investigating the `f32::clone()` problem. It's optimized with lots of SIMD and provides better row/column iteration performance. The QR decomposition is also much faster than `nalgebra`. This [commit 0411821](https://github.com/kemingy/rabitq/commit/04118219d28bd0d43594c98c71e752faa81ff79d) makes the training part faster.

Also, I can now use these vectors as a normal slice without the `ColRef` or `RowRef` wrapper after [commit 0d969bd](https://github.com/kemingy/rabitq/commit/0d969bdcfb331f87e938e043e01acc648e1cf963).

I have to admit that if I used `faer` from the beginning, I could avoid lots of troubles. Anyway, I learned a lot from this experience.

### Binary dot product

I thought `popcnt` already solved the binary dot product, but the [FlameGraph](https://share.firefox.dev/3Yk3Ok8) shows that `count_ones()` only takes 7% of the `binary_dot_product`. Although the AVX512 has the `vpopcntq` instruction, I would prefer to use the AVX2 simulation since it's more common.

[This](https://github.com/komrad36/popcount/blob/master/popcnt.h) is a good reference for the `popcnt` implementation with AVX2. The [commit edabd4a](https://github.com/kemingy/rabitq/commit/edabd4a64c5b8ea2637b5332105638edf16afa7c) re-implement this in Rust which improves the QPS by **11%** for GIST. This trick only works when the vector has more than 256 dimensions, which means 256 bits for the binary representation.

### Inline

The [#[inline]](https://doc.rust-lang.org/reference/attributes/codegen.html#the-inline-attribute) attribute should be used with caution. Adding this attribute to all the SIMD functions improves the QPS by **5%** for GIST.

### IO

I need to add some background information here.

The current implementation is based on the IVF algorithm, which will uses [*k*-means](https://en.wikipedia.org/wiki/K-means_clustering) to cluster the vectors and stores the centroids in memory. The query vector is only compared to the clusters with smaller `l2_squared_distance(query, centroid)`.

There is a parameter called `n_probe` that controls how many nearest clusters will be probed. A large `n_probe` will increase the recall but decrease the QPS.

RaBitQ uses the binary dot product to estimate the approximate distance. If it's smaller than the threshold, it will re-rank with the original L2 squared distance and update the threshold accordingly.

Previously, I used [`slice::select_nth_unstable`](https://doc.rust-lang.org/std/primitive.slice.html#method.select_nth_unstable) which only selects the n-nearest but doesn't sort them in order. Going through the clusters that are far away from the query will increase the re-ranking ratio, which requires more L2 squared distance computation. Re-sorting the selected n-th clusters improved the QPS by **4%** for GIST.

Another trick is to sort the vectors in each cluster by their distance to the centroids, this [commit ea13ebc](https://github.com/kemingy/rabitq/commit/ea13ebca46257d7c2e22250fe02a481e7681f0a9) also improved the QPS by **4%** for GIST.

There are some metadata used to estimate the approximate distance for each vector:

- factor_ip: f32
- factor_ppc: f32
- error: f32
- x_c_distance_square: f32

Previously I use 4 `Vec<f32>` to store them, which is not IO friendly, since the calculation requires `vector[i]` for each of them. By combining them into one `struct` in [commit bb440e3](https://github.com/kemingy/rabitq/commit/bb440e3e8b150f590523eaa77e7c62165a5ee764), the QPS improved by **2.5%** for GIST. This works well because it's 4xf32, so I can use the C representation directly:

```rust
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[repr(C)]
struct Factor {
    factor_ip: f32,
    factor_ppc: f32,
    error_bound: f32,
    center_distance_square: f32,
}
```

Unfortunately, `faer` doesn't support u64 vectors. So I have to store the vector binary representation in `Vec<Vec<u64>>`. By changing it to `Vec<u64>` in [commit 48236b2](https://github.com/kemingy/rabitq/commit/48236b23069db92bdb741fc6693e126b52c397ce), the QPS improved by **2%** for GIST.

### Const generics

The C++ version uses the template to generate the code for different dimensions. This feature is also available in Rust. I didn't try it because re-compiling the code for different dimensions might only be possible for specific use cases, like inside a company that only has a few fixed dimensions. For the public library, it's better to provide a general solution so users don't have to re-compile it by themselves.

### Other tools

There is a [bounds-check-cookbook](https://github.com/Shnatsel/bounds-check-cookbook/) which provides several examples of how to eliminate the boundary checking in safe Rust.

I tried [PGO](https://doc.rust-lang.org/rustc/profile-guided-optimization.html) and [BOLT](https://github.com/llvm/llvm-project/tree/main/bolt) but didn't get any improvement.

Switching to [jemalloc](https://github.com/tikv/jemallocator) or [mimalloc](https://github.com/microsoft/mimalloc) doesn't improve the performance either.

## Conclusion

- SIMD is awesome when it's used properly
- IO is also important, especially for the large datasets

The current performance is the same as the C++ version for dataset GIST. While I use more SIMD, the C++ version uses const generics.

## References

- [Algorithmica / HPC](https://en.algorithmica.org/hpc/algorithms/matmul/)
