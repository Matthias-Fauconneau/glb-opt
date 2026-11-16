# glb-opt

A pure-Rust command-line tool that simplifies triangle meshes in binary glTF files. It uses
[`meshoptimizer-rs`](https://crates.io/crates/meshoptimizer-rs) for mesh simplification,
[`gltf`](https://crates.io/crates/gltf) to decode mesh data, and [`clap`](https://crates.io/crates/clap)
for command-line parsing. It does not require a C++ toolchain.

```text
cargo run --release -- input.glb output.glb --ratio 0.5 --error 0.01
```

Add `--preserve-colors` to include the `COLOR_0` vertex attribute in the simplification metric
for primitives that have vertex colors. This helps preserve color detail; primitives without
`COLOR_0` continue to use position-only simplification.

`--ratio` is the fraction of triangles to target (default `0.5`). `--error` sets the maximum
normalized geometric error (default `0.01`). The simplifier may keep more triangles than the
target to stay within that error. Vertex attribute values are preserved while streams are
reordered and unused vertices removed. Non-triangle primitives and primitives without positions are
left untouched. The CLI optimizes triangle order for the vertex cache, calls
`meshoptimizer-rs` `optimize_vertex_fetch`, compacts referenced vertex records, and repacks buffer
views. It applies the same fetch order to primitive attributes and morph target attributes.
Sparse vertex accessors are not supported for compaction.

The input must be a GLB 2.0 file with one embedded buffer. GLB parsing and serialization use the
`gltf` crate. The output is also GLB 2.0.
