# bitnet_lut.ea — LUT-based ternary matmul kernel

## Problem

The existing `bitnet_i2s.ea` kernels use multiply-accumulate (`maddubs_i32` on x86,
`vdot_i32` on ARM) to compute ternary weight x int8 activation dot products one row
at a time. BitNet's T-MAC/TL1/TL2 approach replaces multiplication with table lookups,
processing many weight rows in parallel through activation-derived tables.

The per-row `i2_dot_i8` API cannot capture this advantage — the LUT win comes from
row parallelism: one `shuffle_bytes` processes 16 weight rows simultaneously. This
requires a matrix-level API.

eabitnet needs a `lut_matmul` kernel for this. The `shuffle_bytes` intrinsic is now
available in eacompute (cross-platform: SSSE3 `pshufb` on x86, NEON `tbl` on ARM).

## Design

### Algorithm: 1-weight-per-lookup with row parallelism

For each activation position, build a 3-entry table from that activation value:
`table[0] = -a`, `table[1] = 0`, `table[2] = +a` (ternary {-1, 0, +1} mapped to
{0, 1, 2}).

Then for 16 weight rows at once: load 16 packed weight bytes (one per row), extract
the ternary value for this activation position, and use it as a `shuffle_bytes` index
into the table. One shuffle processes 16 rows in parallel.

**Why 1-weight-per-lookup (not 2 or 4):**
- 1 ternary weight x i8 activation: range [-127, 127]. Fits in i8.
- 2 ternary weights x i8 activations: range [-254, 254]. Overflows i8.
- No nibble-split needed. Simpler, correct, and the parallelism across rows is
  where the real throughput comes from.

**Weight encoding:** Same 2-bit packed format as `pack_ternary_row`. 4 weights per
byte: bits 7:6, 5:4, 3:2, 1:0. Values {0, 1, 2}. The 4 weights in a byte correspond
to 4 consecutive activation positions (not the QK=128 interleaved layout used by i2s).

### Table construction in Eä

Build a 3-entry lookup table from a single i8 activation value `a`:
- Lane 0: `-a` (weight=0 → ternary -1)
- Lane 1: `0` (weight=1 → ternary 0)
- Lane 2: `+a` (weight=2 → ternary +1)
- Lanes 3..15: `0` (unused, zeroed for safety)

Construction using available Eä intrinsics:
```
let zero: u8x16 = splat(0)
let pos: u8x16 = splat(a)          // all lanes = a (bit pattern)
let neg: u8x16 = zero .- pos       // all lanes = -a (wrapping)
// Lane 0 = neg, lane 2 = pos, rest = 0
// Use shuffle to place: shuffle(neg, [0,16,16,...]) puts neg[0] in lane 0, zero elsewhere
// Then shuffle(pos, [16,16,2,16,...]) puts pos[0] in lane 2
// Add both: table = neg_placed .+ pos_placed
```

Alternative: use `select` with comparison masks to place values in the right lanes.
Exact construction determined at implementation time — multiple approaches work.

### Inner loop structure

```
for each group of 4 activation positions (j, j+1, j+2, j+3):
    build table_a0 from activations[j]
    build table_a1 from activations[j+1]
    build table_a2 from activations[j+2]
    build table_a3 from activations[j+3]

    for each chunk of 16 weight rows (i..i+15):
        // One packed byte per row, at column offset j/4
        load 16 packed weight bytes into u8x16

        // Extract ternary index for each activation position
        indices_g0 = (packed .>> splat(6)) .& splat(3)   // bits 7:6 → a[j]
        indices_g1 = (packed .>> splat(4)) .& splat(3)   // bits 5:4 → a[j+1]
        indices_g2 = (packed .>> splat(2)) .& splat(3)   // bits 3:2 → a[j+2]
        indices_g3 = packed .& splat(3)                   // bits 1:0 → a[j+3]

        // Lookup: 16 rows x 1 activation per shuffle
        r0: u8x16 = shuffle_bytes(table_a0, indices_g0)
        r1: u8x16 = shuffle_bytes(table_a1, indices_g1)
        r2: u8x16 = shuffle_bytes(table_a2, indices_g2)
        r3: u8x16 = shuffle_bytes(table_a3, indices_g3)

        // Widen EACH result to f32x4 (sign-extends via widen_i8_f32x4)
        // then accumulate into f32x4 row accumulators.
        // Process 4 lanes at a time (4 x f32x4 = 16 rows).
        // widen_i8_f32x4 accepts u8x16 and sign-extends when called
        // as widen_i8_f32x4 (unsigned=false path in eacompute codegen).
        acc_0_3 = acc_0_3 .+ widen_i8_f32x4(r0) .+ widen_i8_f32x4(r1)
                          .+ widen_i8_f32x4(r2) .+ widen_i8_f32x4(r3)
        // Repeat for lanes 4..7, 8..11, 12..15 (shift u8x16 down,
        // or use shuffle to extract next 4 bytes, then widen)
```

Each `shuffle_bytes` call processes 16 rows x 1 activation = 16 multiply-accumulates.
Four shuffles per packed weight byte = 64 effective multiply-accumulates.

Widening happens per-shuffle-result (not after summing) to avoid i8 overflow.
f32 accumulation is exact for integer values within 2^24 — BitNet dot product
sums are far below this threshold.

### API

Two functions in a single cross-platform file (no `#[cfg]`):

```
// Matrix-vector multiply: n_rows weight rows x one activation vector
// weights: n_rows * (n_cols/4) packed bytes (2-bit ternary, pack_ternary_row output)
// activations: n_cols i8 values
// out: n_rows i32 raw dot products
export func lut_matmul(
    weights: *restrict u8,
    activations: *restrict i8,
    out scores: *mut i32 [cap: n_rows],
    n_rows: i32,
    n_cols: i32
)

// Tail case: fewer than 16 rows, scalar cross-platform fallback.
// Uses per-weight scalar multiply-accumulate (no platform-specific intrinsics).
export func lut_matmul_tail(
    weights: *restrict u8,
    activations: *restrict i8,
    out scores: *mut i32 [cap: n_rows],
    n_rows: i32,
    n_cols: i32
)
```

No `build_lut_tables` function. Tables are ephemeral — built on the stack from
activations inside the hot loop. Weight format is unchanged from `pack_ternary_row`.

### Accumulation strategy

`shuffle_bytes` returns u8x16 with signed values in u8 lanes (range [-127, 127]).

**Widening path:** `widen_i8_f32x4` sign-extends u8x16 → i32 → f32x4. The function
name (not input type) controls sign extension in eacompute codegen. The typechecker
accepts both i8x16 and u8x16 as input.

**Per-shuffle widening:** Each shuffle result is widened to f32x4 immediately and
accumulated in f32x4 accumulators. No summing in u8 lanes (would overflow after 2+
additions).

**Row accumulators:** 16 rows = 4 x f32x4 accumulators. After all activation columns
processed, convert to i32 via `to_i32(reduce_add(...))` or store via narrowing.

### Cross-platform

The kernel uses only cross-platform operations:
- `shuffle_bytes(u8x16, u8x16) -> u8x16` (SSSE3/NEON)
- `widen_i8_f32x4(u8x16) -> f32x4` (sign-extending)
- Bitwise: `.>>`, `.&`
- Arithmetic: `.+`, `.-`
- `splat`, `load`, `store`, `select`, `shuffle`
- `reduce_add` for final row sums

No `#[cfg]` needed. Single `.ea` file works on both x86 and aarch64.
First eabitnet kernel without a platform gate — all intrinsics used are cross-platform.

### Alignment requirements

- `n_cols` must be divisible by 4 (4 weights per packed byte)
- `n_rows` should be divisible by 16 for full LUT throughput; caller invokes
  `lut_matmul` for the first `n_rows & ~15` rows, then `lut_matmul_tail` for remainder
- Weight matrix is row-major: row i starts at byte offset `i * (n_cols / 4)`

### eaclaw integration

Patch ggml's `ggml_compute_forward_mul_mat` for ternary layers to call `lut_matmul`:
- ggml dispatches matmul at the matrix level — `lut_matmul` maps directly
- Weight format: use `pack_ternary_row` at model load (already exists)
- No intermediate table storage needed
- Caller splits: `lut_matmul` for rows 0..`n_rows&~15`, `lut_matmul_tail` for rest
- `eaclaw --model bitnet-3b` config selects the eabitnet backend

### Files

| File | Purpose |
|------|---------|
| `kernels/bitnet_lut.ea` | `lut_matmul` + `lut_matmul_tail` |
| `tests/test_lut.c` | End-to-end tests |

### Test plan

- Known vectors: all-zero weights, all-one activations, alternating +1/-1 weights
- Activation extremes: -128, 127 (note: -128 negated wraps to -128 in i8, still valid)
- Cross-verify: same inputs through `lut_matmul` and row-by-row scalar reference
  must produce identical results
- Sizes: 16x128, 16x512, 64x1024, 128x4096 (n_rows x n_cols)
- Tail case: n_rows = 17, 33 (verify lut_matmul + lut_matmul_tail combination)
- Benchmark: compare throughput against calling `i2_dot_i8` in a loop

### Constraints

- All eacompute hard rules apply: <500 lines per file, end-to-end tested, no fakes
- SIMD throughout in `lut_matmul` — no scalar fallback
- `lut_matmul_tail` uses scalar cross-platform ops (no `maddubs_i32`/`vdot_i32`)
- Returns raw sums; caller applies ternary offset correction + scale (same as i2s)
