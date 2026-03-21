# bitnet_lut.ea Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a LUT-based ternary matmul kernel that processes 16 weight rows in parallel via `shuffle_bytes`, with a weight preparation function and scalar tail handler.

**Architecture:** Activation values are broadcast into 3-entry lookup tables (`[-a, 0, +a]`). Packed ternary weight indices (0/1/2) from 16 rows are used as `shuffle_bytes` indices into these tables, producing 16 partial sums per shuffle. A `prepare_lut_weights` function transposes the weight matrix at model load so that weight bytes from 16 different rows are contiguous for vector loads. Accumulation uses `widen_i8_f32x4` (sign-extending) into f32x4 accumulators.

**Tech Stack:** Eä (`.ea` compiled via eacompute), C (tests), Make

**Spec:** `docs/superpowers/specs/2026-03-21-bitnet-lut-design.md`

**Key references:**
- Eä intrinsics: https://petlukk.github.io/eacompute/reference/intrinsics.html
- Existing kernel patterns: `kernels/bitnet_i2s.ea`, `kernels/bitnet_quant.ea`
- Existing test patterns: `tests/test_i2s.c`, `tests/test_quant.c`
- Autoresearch examples: `eacompute/autoresearch/kernels/gather_lut/`, `dot_u8i8/`, `matmul/`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `kernels/bitnet_lut.ea` | Three exported functions: `prepare_lut_weights`, `lut_matmul`, `lut_matmul_tail` |
| `tests/test_lut.c` | C test harness: scalar reference, correctness tests, benchmark |
| `Makefile` | Add `test_lut` build + run targets |

All functions in a single `.ea` file (no `#[cfg]`, cross-platform). Total kernel code should be well under 500 lines.

---

## Important Implementation Notes

### Weight layout

The existing `pack_ternary_row` packs 4 consecutive ternary values per byte: `v0<<6 | v1<<4 | v2<<2 | v3`. For the LUT kernel, `prepare_lut_weights` must transpose from row-major packed bytes to column-interleaved format where 16 bytes from 16 different rows (at the same column position) are contiguous.

**Row-major layout** (input to `prepare_lut_weights`):
```
Row 0: [byte_0, byte_1, ..., byte_{cols/4-1}]
Row 1: [byte_0, byte_1, ..., byte_{cols/4-1}]
...
```

**Column-interleaved layout** (output, consumed by `lut_matmul`):
```
Col 0, rows 0-15:  [row0_byte0, row1_byte0, ..., row15_byte0]
Col 0, rows 16-31: [row16_byte0, row17_byte0, ..., row31_byte0]
Col 1, rows 0-15:  [row0_byte1, row1_byte1, ..., row15_byte1]
...
```

Offset formula: `col * n_rows + group * 16 + row_within_group`

### Table construction in Eä

Build a 3-entry lookup table from activation value `a` using `select` and comparison:
```
let lane_idx: u8x16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]u8x16
let zvec: u8x16 = splat(0)
let pos: u8x16 = splat(a)
let neg: u8x16 = zvec .- pos
let table: u8x16 = select(lane_idx .== splat(0), neg, select(lane_idx .== splat(2), pos, zvec))
```

Lane 0 = `-a`, lane 1 = `0`, lane 2 = `+a`, lanes 3-15 = `0`.

### Byte extraction for widening

`widen_i8_f32x4` takes lower 4 bytes of u8x16. To process all 16 rows, use compile-time `shuffle` to rotate bytes down:
```
// Lanes 0-3 (rows 0-3): direct
let f0: f32x4 = widen_i8_f32x4(result)
// Lanes 4-7 (rows 4-7): shuffle down
let r4: u8x16 = shuffle(result, [4,5,6,7, 0,1,2,3, 0,1,2,3, 0,1,2,3])
let f1: f32x4 = widen_i8_f32x4(r4)
// Lanes 8-11 (rows 8-11)
let r8: u8x16 = shuffle(result, [8,9,10,11, 0,1,2,3, 0,1,2,3, 0,1,2,3])
let f2: f32x4 = widen_i8_f32x4(r8)
// Lanes 12-15 (rows 12-15)
let r12: u8x16 = shuffle(result, [12,13,14,15, 0,1,2,3, 0,1,2,3, 0,1,2,3])
let f3: f32x4 = widen_i8_f32x4(r12)
```

### Activation pointer type

`shuffle_bytes` requires u8x16. Activations are logically i8 but the table construction works on the byte's bit pattern. Declare the activation parameter as `*restrict u8` — the caller passes the same pointer (i8 and u8 are bit-identical). The subtraction `splat(0) .- splat(a)` in u8 gives the correct two's complement negation.

---

### Task 1: Scaffold `prepare_lut_weights` with test

**Files:**
- Create: `kernels/bitnet_lut.ea`
- Create: `tests/test_lut.c`
- Modify: `Makefile`

- [ ] **Step 1: Write the scalar reference and test for `prepare_lut_weights` in C**

Create `tests/test_lut.c`:

```c
// test_lut.c — Validates LUT matmul kernel against scalar reference

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

extern void prepare_lut_weights(
    const uint8_t *src, uint8_t *dst, int32_t n_rows, int32_t n_cols);
extern void lut_matmul(
    const uint8_t *weights, const uint8_t *activations,
    int32_t *scores, int32_t n_rows, int32_t n_cols);
extern void lut_matmul_tail(
    const uint8_t *weights, const int8_t *activations,
    int32_t *scores, int32_t n_rows, int32_t n_cols);

#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"

// Scalar reference for prepare_lut_weights
// Transposes row-major packed bytes to column-interleaved:
// 16 consecutive bytes = same column byte from 16 consecutive rows
static void ref_prepare(const uint8_t *src, uint8_t *dst,
                        int n_rows, int n_cols) {
    int stride = n_cols / 4;
    int n_groups = n_rows / 16;
    for (int col = 0; col < stride; col++) {
        for (int g = 0; g < n_groups; g++) {
            for (int r = 0; r < 16; r++) {
                dst[col * n_rows + g * 16 + r] =
                    src[(g * 16 + r) * stride + col];
            }
        }
    }
}

// Scalar reference for ternary dot product (operates on row-major packed weights)
// Used for both lut_matmul verification (on column-interleaved data) and tail.
static void ref_ternary_matmul_rowmajor(const uint8_t *packed, const uint8_t *act,
                                        int32_t *scores, int n_rows, int n_cols) {
    int stride = n_cols / 4;
    memset(scores, 0, n_rows * sizeof(int32_t));

    for (int row = 0; row < n_rows; row++) {
        for (int col = 0; col < stride; col++) {
            int act_base = col * 4;
            int8_t a0 = (int8_t)act[act_base];
            int8_t a1 = (int8_t)act[act_base + 1];
            int8_t a2 = (int8_t)act[act_base + 2];
            int8_t a3 = (int8_t)act[act_base + 3];

            uint8_t p = packed[row * stride + col];
            uint8_t w0 = (p >> 6) & 3;
            uint8_t w1 = (p >> 4) & 3;
            uint8_t w2 = (p >> 2) & 3;
            uint8_t w3 = p & 3;

            // ternary: 0→-1, 1→0, 2→+1
            scores[row] += ((int32_t)w0 - 1) * (int32_t)a0;
            scores[row] += ((int32_t)w1 - 1) * (int32_t)a1;
            scores[row] += ((int32_t)w2 - 1) * (int32_t)a2;
            scores[row] += ((int32_t)w3 - 1) * (int32_t)a3;
        }
    }
}

// Pack ternary {-1,0,+1} → {0,1,2}, 4 consecutive values per byte
static void pack_row_major(const int8_t *ternary, uint8_t *packed,
                           int n_rows, int n_cols) {
    int stride = n_cols / 4;
    for (int row = 0; row < n_rows; row++) {
        for (int col = 0; col < n_cols; col += 4) {
            int base = row * n_cols + col;
            uint8_t v0 = (uint8_t)(ternary[base]     + 1);
            uint8_t v1 = (uint8_t)(ternary[base + 1]  + 1);
            uint8_t v2 = (uint8_t)(ternary[base + 2]  + 1);
            uint8_t v3 = (uint8_t)(ternary[base + 3]  + 1);
            packed[row * stride + col / 4] = (v0 << 6) | (v1 << 4) | (v2 << 2) | v3;
        }
    }
}

static int8_t rand_ternary(void) {
    return (int8_t)((rand() % 3) - 1);
}

static int8_t rand_i8(void) {
    return (int8_t)((rand() % 256) - 128);
}

static int test_prepare(int n_rows, int n_cols) {
    printf("  prepare %dx%d ... ", n_rows, n_cols);

    int stride = n_cols / 4;
    int total_bytes = n_rows * stride;

    uint8_t *src = malloc(total_bytes);
    for (int i = 0; i < total_bytes; i++) src[i] = (uint8_t)(rand() & 0xFF);

    uint8_t *expected = calloc(total_bytes, 1);
    uint8_t *got      = calloc(total_bytes, 1);

    ref_prepare(src, expected, n_rows, n_cols);
    prepare_lut_weights(src, got, n_rows, n_cols);

    int ok = (memcmp(expected, got, total_bytes) == 0);
    printf("%s\n", ok ? GREEN "PASS" RESET : RED "FAIL" RESET);

    if (!ok) {
        for (int i = 0; i < total_bytes; i++) {
            if (expected[i] != got[i]) {
                printf("    first diff at [%d]: expected=%d got=%d\n",
                       i, expected[i], got[i]);
                break;
            }
        }
    }

    free(src); free(expected); free(got);
    return ok;
}

static int test_matmul(int n_rows, int n_cols) {
    printf("  matmul %dx%d ... ", n_rows, n_cols);

    int stride = n_cols / 4;

    // Generate random ternary weights and activations
    int8_t *ternary = malloc(n_rows * n_cols);
    uint8_t *act = malloc(n_cols);
    for (int i = 0; i < n_rows * n_cols; i++) ternary[i] = rand_ternary();
    for (int i = 0; i < n_cols; i++) act[i] = (uint8_t)rand_i8();

    // Pack row-major
    uint8_t *packed = calloc(n_rows * stride, 1);
    pack_row_major(ternary, packed, n_rows, n_cols);

    // Prepare (transpose)
    uint8_t *prepared = calloc(n_rows * stride, 1);
    prepare_lut_weights(packed, prepared, n_rows, n_cols);

    // Reference: compute expected from row-major packed weights
    int32_t *expected = calloc(n_rows, sizeof(int32_t));
    ref_ternary_matmul_rowmajor(packed, act, expected, n_rows, n_cols);

    // Kernel under test
    int32_t *got = calloc(n_rows, sizeof(int32_t));
    int main_rows = (n_rows / 16) * 16;
    int tail_rows = n_rows - main_rows;

    if (main_rows > 0) {
        lut_matmul(prepared, act, got, main_rows, n_cols);
    }
    if (tail_rows > 0) {
        // Tail uses row-major packed weights (not transposed)
        lut_matmul_tail(packed + main_rows * stride, (const int8_t *)act,
                        got + main_rows, tail_rows, n_cols);
    }

    int ok = 1;
    for (int i = 0; i < n_rows; i++) {
        if (expected[i] != got[i]) {
            printf("%sFAIL%s at row %d: expected=%d got=%d\n",
                   RED, RESET, i, expected[i], got[i]);
            ok = 0;
            break;
        }
    }
    if (ok) printf("%sPASS%s\n", GREEN, RESET);

    free(ternary); free(act); free(packed);
    free(prepared); free(expected); free(got);
    return ok;
}

static double bench_matmul(int n_rows, int n_cols, int iters) {
    int stride = n_cols / 4;

    uint8_t *packed = malloc(n_rows * stride);
    for (int i = 0; i < n_rows * stride; i++) packed[i] = (uint8_t)(rand() & 0xFF);

    uint8_t *prepared = malloc(n_rows * stride);
    prepare_lut_weights(packed, prepared, n_rows, n_cols);

    uint8_t *act = malloc(n_cols);
    for (int i = 0; i < n_cols; i++) act[i] = (uint8_t)rand_i8();

    int32_t *scores = calloc(n_rows, sizeof(int32_t));

    // Warmup
    for (int i = 0; i < 100; i++) lut_matmul(prepared, act, scores, n_rows, n_cols);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++) {
        lut_matmul(prepared, act, scores, n_rows, n_cols);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double ns = (elapsed / iters) * 1e9;

    free(packed); free(prepared); free(act); free(scores);
    return ns;
}

int main(void) {
    srand(42);

    printf("=== BitNet LUT matmul kernel tests ===\n\n");
    int pass = 0, total = 0;

    // prepare_lut_weights
    printf("prepare_lut_weights:\n");
    int prep_sizes[][2] = {{16,128}, {32,256}, {64,512}, {128,1024}};
    for (int i = 0; i < 4; i++) {
        total++;
        if (test_prepare(prep_sizes[i][0], prep_sizes[i][1])) pass++;
    }

    // lut_matmul (n_rows divisible by 16)
    printf("\nlut_matmul (aligned):\n");
    int mat_sizes[][2] = {{16,128}, {16,512}, {64,1024}, {128,4096}};
    for (int i = 0; i < 4; i++) {
        total++;
        if (test_matmul(mat_sizes[i][0], mat_sizes[i][1])) pass++;
    }

    // lut_matmul + tail (n_rows NOT divisible by 16)
    printf("\nlut_matmul + tail:\n");
    int tail_sizes[][2] = {{17,128}, {33,512}, {65,1024}};
    for (int i = 0; i < 3; i++) {
        total++;
        if (test_matmul(tail_sizes[i][0], tail_sizes[i][1])) pass++;
    }

    // Edge cases
    printf("\nEdge cases:\n");
    // All-zero weights (all ternary 0 → packed value 0x55 = 01010101)
    {
        printf("  all-zero weights 16x128 ... ");
        int n_rows = 16, n_cols = 128;
        int stride = n_cols / 4;
        uint8_t *packed = malloc(n_rows * stride);
        memset(packed, 0x55, n_rows * stride);  // all weight=1 → ternary 0
        uint8_t *prepared = malloc(n_rows * stride);
        prepare_lut_weights(packed, prepared, n_rows, n_cols);
        uint8_t *act = malloc(n_cols);
        for (int i = 0; i < n_cols; i++) act[i] = (uint8_t)rand_i8();
        int32_t scores[16] = {0};
        lut_matmul(prepared, act, scores, n_rows, n_cols);
        int ok = 1;
        for (int i = 0; i < n_rows; i++) {
            if (scores[i] != 0) { ok = 0; break; }
        }
        total++;
        if (ok) { pass++; printf("%sPASS%s\n", GREEN, RESET); }
        else printf("%sFAIL%s\n", RED, RESET);
        free(packed); free(prepared); free(act);
    }

    // Activation extremes: all 127
    {
        printf("  act=127 weights=+1 16x128 ... ");
        int n_rows = 16, n_cols = 128;
        int stride = n_cols / 4;
        uint8_t *packed = malloc(n_rows * stride);
        memset(packed, 0xAA, n_rows * stride);  // all weight=2 → ternary +1
        uint8_t *prepared = malloc(n_rows * stride);
        prepare_lut_weights(packed, prepared, n_rows, n_cols);
        uint8_t act[128];
        memset(act, 127, n_cols);
        int32_t scores[16] = {0};
        lut_matmul(prepared, act, scores, n_rows, n_cols);
        int ok = 1;
        for (int i = 0; i < n_rows; i++) {
            if (scores[i] != 127 * n_cols) { ok = 0; break; }
        }
        total++;
        if (ok) { pass++; printf("%sPASS%s\n", GREEN, RESET); }
        else { printf("%sFAIL%s expected=%d got=%d\n", RED, RESET, 127*n_cols, scores[0]); }
        free(packed); free(prepared);
    }

    // Activation -128 (edge: -(-128) wraps to -128 in i8)
    {
        printf("  act=-128 weights=+1 16x128 ... ");
        int n_rows = 16, n_cols = 128;
        int stride = n_cols / 4;
        uint8_t *packed = malloc(n_rows * stride);
        memset(packed, 0xAA, n_rows * stride);  // all weight=2 → ternary +1
        uint8_t *prepared = malloc(n_rows * stride);
        prepare_lut_weights(packed, prepared, n_rows, n_cols);
        uint8_t act_buf[128];
        memset(act_buf, 0x80, n_cols);  // 0x80 = -128 as i8
        int32_t scores[16] = {0};
        lut_matmul(prepared, act_buf, scores, n_rows, n_cols);
        // (+1) * (-128) * 128 cols = -16384
        int ok = 1;
        for (int i = 0; i < n_rows; i++) {
            if (scores[i] != -128 * n_cols) { ok = 0; break; }
        }
        total++;
        if (ok) { pass++; printf("%sPASS%s\n", GREEN, RESET); }
        else { printf("%sFAIL%s expected=%d got=%d\n", RED, RESET, -128*n_cols, scores[0]); }
        free(packed); free(prepared);
    }

    // Alternating +1/-1 weights
    {
        printf("  alternating +1/-1 16x128 ... ");
        int n_rows = 16, n_cols = 128;
        int stride = n_cols / 4;
        int8_t *ternary = malloc(n_rows * n_cols);
        for (int i = 0; i < n_rows * n_cols; i++)
            ternary[i] = (i % 2 == 0) ? 1 : -1;
        uint8_t *packed = calloc(n_rows * stride, 1);
        pack_row_major(ternary, packed, n_rows, n_cols);
        uint8_t *prepared = malloc(n_rows * stride);
        prepare_lut_weights(packed, prepared, n_rows, n_cols);
        uint8_t act_buf[128];
        for (int i = 0; i < n_cols; i++) act_buf[i] = (uint8_t)(int8_t)1;
        int32_t expected[16] = {0};
        ref_ternary_matmul_rowmajor(packed, act_buf, expected, n_rows, n_cols);
        int32_t scores[16] = {0};
        lut_matmul(prepared, act_buf, scores, n_rows, n_cols);
        int ok = 1;
        for (int i = 0; i < n_rows; i++) {
            if (scores[i] != expected[i]) { ok = 0; break; }
        }
        total++;
        if (ok) { pass++; printf("%sPASS%s\n", GREEN, RESET); }
        else { printf("%sFAIL%s expected=%d got=%d\n", RED, RESET, expected[0], scores[0]); }
        free(ternary); free(packed); free(prepared);
    }

    printf("\n%d/%d tests passed\n", pass, total);

    // Benchmark
    if (pass == total) {
        printf("\n=== Benchmark (lut_matmul) ===\n");
        struct { int rows, cols, iters; } bench[] = {
            {64, 1024, 500000},
            {128, 4096, 100000},
            {256, 4096, 50000},
        };
        for (int i = 0; i < 3; i++) {
            double ns = bench_matmul(bench[i].rows, bench[i].cols, bench[i].iters);
            double gops = (double)bench[i].rows * bench[i].cols / ns;
            printf("  %dx%d: %.1f ns/call  (%.2f Gop/s)\n",
                   bench[i].rows, bench[i].cols, ns, gops);
        }
    }

    return pass == total ? 0 : 1;
}
```

- [ ] **Step 2: Write `prepare_lut_weights` in Eä**

Create `kernels/bitnet_lut.ea`:

```
// bitnet_lut.ea — LUT-based ternary matmul via shuffle_bytes
//
// Cross-platform (x86 SSSE3 + ARM NEON). No #[cfg] needed.
//
// Three functions:
// 1. prepare_lut_weights: transpose row-major packed bytes to column-interleaved
// 2. lut_matmul: SIMD matmul processing 16 rows in parallel via shuffle_bytes
// 3. lut_matmul_tail: scalar fallback for remainder rows (< 16)

// Transpose packed weight bytes from row-major to column-interleaved format.
// Column-interleaved: 16 consecutive bytes = same column from 16 consecutive rows.
// This enables contiguous vector loads in lut_matmul.
// n_rows must be divisible by 16. n_cols must be divisible by 4.
export func prepare_lut_weights(
    src: *restrict u8,
    out dst: *mut u8 [cap: n_rows * n_cols / 4],
    n_rows: i32,
    n_cols: i32
) {
    let stride: i32 = n_cols / 4
    let n_groups: i32 = n_rows / 16

    let mut col: i32 = 0
    while col < stride {
        let mut g: i32 = 0
        while g < n_groups {
            let out_base: i32 = col * n_rows + g * 16
            let row_base: i32 = g * 16

            let mut r: i32 = 0
            while r < 16 {
                dst[out_base + r] = src[(row_base + r) * stride + col]
                r = r + 1
            }

            g = g + 1
        }
        col = col + 1
    }
}
```

- [ ] **Step 3: Add build target to Makefile**

Add to `Makefile` test target:

```makefile
test: kernels
	$(CC) $(CFLAGS) tests/test_i2s.c -L$(LIB) -lbitnet_i2s -o $(BUILD)/test_i2s -lm
	$(CC) $(CFLAGS) tests/test_quant.c -L$(LIB) -lbitnet_quant -o $(BUILD)/test_quant -lm
	$(CC) $(CFLAGS) tests/test_lut.c -L$(LIB) -lbitnet_lut -o $(BUILD)/test_lut -lm
	LD_LIBRARY_PATH=$(LIB) $(BUILD)/test_i2s
	LD_LIBRARY_PATH=$(LIB) $(BUILD)/test_quant
	LD_LIBRARY_PATH=$(LIB) $(BUILD)/test_lut
```

- [ ] **Step 4: Build kernels and run prepare_lut_weights tests only**

Run: `cd /home/peter/projects/eabitnet && make kernels`

The kernel should compile. If it fails, fix Eä syntax issues.

Then build and run the test (only prepare tests will pass since matmul isn't implemented yet — the test will crash when it reaches those, but prepare tests come first):

Run: `make test` (expect prepare tests to pass, matmul tests to fail/crash since lut_matmul is a stub)

Alternative: temporarily comment out matmul tests, run only prepare tests.

- [ ] **Step 5: Commit**

```bash
git add kernels/bitnet_lut.ea tests/test_lut.c Makefile
git commit -m "feat: add prepare_lut_weights and test scaffold for LUT matmul"
```

---

### Task 2: Implement `lut_matmul_tail` (scalar fallback)

**Files:**
- Modify: `kernels/bitnet_lut.ea`

This is the simple scalar version that handles remainder rows. Implement it first because the test harness uses it for tail cases, and it validates the algorithm before SIMD.

- [ ] **Step 1: Add `lut_matmul_tail` to `bitnet_lut.ea`**

Append to `kernels/bitnet_lut.ea`:

```
// Scalar fallback for rows not divisible by 16.
// Takes row-major packed weights (NOT column-interleaved).
// n_cols must be divisible by 4.
export func lut_matmul_tail(
    weights: *restrict u8,
    activations: *restrict i8,
    out scores: *mut i32 [cap: n_rows],
    n_rows: i32,
    n_cols: i32
) {
    let stride: i32 = n_cols / 4

    let mut row: i32 = 0
    while row < n_rows {
        let mut sum: i32 = 0
        let mut col: i32 = 0
        while col < stride {
            let packed: u8 = weights[row * stride + col]
            let act_base: i32 = col * 4

            let w0: i32 = to_i32((packed / 64) % 4)
            let w1: i32 = to_i32((packed / 16) % 4)
            let w2: i32 = to_i32((packed / 4) % 4)
            let w3: i32 = to_i32(packed % 4)

            // activations is *i8, so to_i32 sign-extends correctly
            let a0: i32 = to_i32(activations[act_base])
            let a1: i32 = to_i32(activations[act_base + 1])
            let a2: i32 = to_i32(activations[act_base + 2])
            let a3: i32 = to_i32(activations[act_base + 3])

            // ternary: 0→-1, 1→0, 2→+1
            sum = sum + (w0 - 1) * a0 + (w1 - 1) * a1 + (w2 - 1) * a2 + (w3 - 1) * a3

            col = col + 1
        }
        scores[row] = sum
        row = row + 1
    }
}
```

**Note on `to_i32`:** Eä scalar loads from u8 pointer return u8. We need i32 for arithmetic. But `to_i32` may not accept u8 — check the Eä type docs. If it doesn't, use scalar arithmetic that stays in u8 and cast only at the multiply step. Alternatively, the activation values need sign-extension from u8 (which contains an i8 bit pattern). For values 128-255, `to_i32` on u8 gives 128-255 (wrong — should be -128 to -1). Workaround: `if a0 > 127 { a0 = a0 - 256 }` after the to_i32 conversion. Or declare activations as `*restrict i8` here — the scalar indexing then returns i8, which `to_i32` sign-extends correctly. Use `*restrict i8` for the tail function since it doesn't use shuffle_bytes.

- [ ] **Step 2: Build and verify tail-only tests compile**

Run: `cd /home/peter/projects/eabitnet && make kernels`

Fix any Eä compilation issues.

- [ ] **Step 3: Commit**

```bash
git add kernels/bitnet_lut.ea
git commit -m "feat: add lut_matmul_tail scalar fallback"
```

---

### Task 3: Implement `lut_matmul` (SIMD hot path)

**Files:**
- Modify: `kernels/bitnet_lut.ea`

This is the core LUT kernel. Processes 16 rows in parallel using `shuffle_bytes`.

- [ ] **Step 1: Add `lut_matmul` to `bitnet_lut.ea`**

Insert between `prepare_lut_weights` and `lut_matmul_tail`:

```
// LUT-based matmul: 16 weight rows processed in parallel per shuffle_bytes.
// Takes column-interleaved weights (output of prepare_lut_weights).
// n_rows must be divisible by 16. n_cols must be divisible by 4.
// activations: n_cols u8 values (i8 bit patterns stored as u8).
// Returns raw dot products in scores (caller applies offset correction + scale).
export func lut_matmul(
    weights: *restrict u8,
    activations: *restrict u8,
    out scores: *mut i32 [cap: n_rows],
    n_rows: i32,
    n_cols: i32
) {
    let stride: i32 = n_cols / 4
    let n_groups: i32 = n_rows / 16
    let lane_idx: u8x16 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]u8x16
    let zvec: u8x16 = splat(0)
    let two: u8x16 = splat(2)
    let s6: u8x16 = splat(6)
    let s4: u8x16 = splat(4)
    let s2: u8x16 = splat(2)
    let mask3: u8x16 = splat(3)

    let mut g: i32 = 0
    while g < n_groups {
        // Initialize 4 f32x4 accumulators for 16 rows
        let mut acc0: f32x4 = splat(0.0)
        let mut acc1: f32x4 = splat(0.0)
        let mut acc2: f32x4 = splat(0.0)
        let mut acc3: f32x4 = splat(0.0)

        let mut col: i32 = 0
        while col < stride {
            // 4 activation values for this column group
            let act_base: i32 = col * 4
            let a0: u8x16 = splat(activations[act_base])
            let a1: u8x16 = splat(activations[act_base + 1])
            let a2: u8x16 = splat(activations[act_base + 2])
            let a3: u8x16 = splat(activations[act_base + 3])

            // Build 4 lookup tables: [-a, 0, +a, 0, ..., 0]
            let neg0: u8x16 = zvec .- a0
            let t0: u8x16 = select(lane_idx .== zvec, neg0,
                             select(lane_idx .== two, a0, zvec))
            let neg1: u8x16 = zvec .- a1
            let t1: u8x16 = select(lane_idx .== zvec, neg1,
                             select(lane_idx .== two, a1, zvec))
            let neg2: u8x16 = zvec .- a2
            let t2: u8x16 = select(lane_idx .== zvec, neg2,
                             select(lane_idx .== two, a2, zvec))
            let neg3: u8x16 = zvec .- a3
            let t3: u8x16 = select(lane_idx .== zvec, neg3,
                             select(lane_idx .== two, a3, zvec))

            // Load 16 packed weight bytes (one per row in this group)
            let packed: u8x16 = load(weights, col * n_rows + g * 16)

            // Extract ternary indices for each activation position
            let idx0: u8x16 = (packed .>> s6) .& mask3
            let idx1: u8x16 = (packed .>> s4) .& mask3
            let idx2: u8x16 = (packed .>> s2) .& mask3
            let idx3: u8x16 = packed .& mask3

            // Lookup: 16 rows x 1 activation per shuffle
            let r0: u8x16 = shuffle_bytes(t0, idx0)
            let r1: u8x16 = shuffle_bytes(t1, idx1)
            let r2: u8x16 = shuffle_bytes(t2, idx2)
            let r3: u8x16 = shuffle_bytes(t3, idx3)

            // Widen each result to f32x4 (sign-extending via widen_i8_f32x4)
            // and accumulate. Process 4 lanes at a time.

            // Rows 0-3 (lower 4 bytes)
            acc0 = acc0 .+ widen_i8_f32x4(r0) .+ widen_i8_f32x4(r1)
                        .+ widen_i8_f32x4(r2) .+ widen_i8_f32x4(r3)

            // Rows 4-7
            let r0_4: u8x16 = shuffle(r0, [4,5,6,7,0,1,2,3,0,1,2,3,0,1,2,3])
            let r1_4: u8x16 = shuffle(r1, [4,5,6,7,0,1,2,3,0,1,2,3,0,1,2,3])
            let r2_4: u8x16 = shuffle(r2, [4,5,6,7,0,1,2,3,0,1,2,3,0,1,2,3])
            let r3_4: u8x16 = shuffle(r3, [4,5,6,7,0,1,2,3,0,1,2,3,0,1,2,3])
            acc1 = acc1 .+ widen_i8_f32x4(r0_4) .+ widen_i8_f32x4(r1_4)
                        .+ widen_i8_f32x4(r2_4) .+ widen_i8_f32x4(r3_4)

            // Rows 8-11
            let r0_8: u8x16 = shuffle(r0, [8,9,10,11,0,1,2,3,0,1,2,3,0,1,2,3])
            let r1_8: u8x16 = shuffle(r1, [8,9,10,11,0,1,2,3,0,1,2,3,0,1,2,3])
            let r2_8: u8x16 = shuffle(r2, [8,9,10,11,0,1,2,3,0,1,2,3,0,1,2,3])
            let r3_8: u8x16 = shuffle(r3, [8,9,10,11,0,1,2,3,0,1,2,3,0,1,2,3])
            acc2 = acc2 .+ widen_i8_f32x4(r0_8) .+ widen_i8_f32x4(r1_8)
                        .+ widen_i8_f32x4(r2_8) .+ widen_i8_f32x4(r3_8)

            // Rows 12-15
            let r0_12: u8x16 = shuffle(r0, [12,13,14,15,0,1,2,3,0,1,2,3,0,1,2,3])
            let r1_12: u8x16 = shuffle(r1, [12,13,14,15,0,1,2,3,0,1,2,3,0,1,2,3])
            let r2_12: u8x16 = shuffle(r2, [12,13,14,15,0,1,2,3,0,1,2,3,0,1,2,3])
            let r3_12: u8x16 = shuffle(r3, [12,13,14,15,0,1,2,3,0,1,2,3,0,1,2,3])
            acc3 = acc3 .+ widen_i8_f32x4(r0_12) .+ widen_i8_f32x4(r1_12)
                        .+ widen_i8_f32x4(r2_12) .+ widen_i8_f32x4(r3_12)

            col = col + 1
        }

        // Store results: convert f32x4 accumulators to i32 and write
        let base_row: i32 = g * 16
        scores[base_row]      = to_i32(acc0[0])
        scores[base_row + 1]  = to_i32(acc0[1])
        scores[base_row + 2]  = to_i32(acc0[2])
        scores[base_row + 3]  = to_i32(acc0[3])
        scores[base_row + 4]  = to_i32(acc1[0])
        scores[base_row + 5]  = to_i32(acc1[1])
        scores[base_row + 6]  = to_i32(acc1[2])
        scores[base_row + 7]  = to_i32(acc1[3])
        scores[base_row + 8]  = to_i32(acc2[0])
        scores[base_row + 9]  = to_i32(acc2[1])
        scores[base_row + 10] = to_i32(acc2[2])
        scores[base_row + 11] = to_i32(acc2[3])
        scores[base_row + 12] = to_i32(acc3[0])
        scores[base_row + 13] = to_i32(acc3[1])
        scores[base_row + 14] = to_i32(acc3[2])
        scores[base_row + 15] = to_i32(acc3[3])

        g = g + 1
    }
}
```

**Key things to watch:**
- `widen_i8_f32x4` accepts u8x16 input (confirmed in eacompute typechecker: `Type::I8 | Type::U8`)
- `shuffle` uses array literal syntax: `shuffle(vec, [4,5,6,7,...])`
- `lane_idx .== zvec` produces a boolean vector — verify `select` works with it
- The `two` and `s2` variables both equal `splat(2)` but serve different purposes (comparison vs shift) — can be shared if Eä allows it
- f32 accumulation is exact for integers within 2^24; BitNet sums stay well below this

- [ ] **Step 2: Build and run full test suite**

Run: `cd /home/peter/projects/eabitnet && make clean && make`

All tests should pass. Debug any failures.

- [ ] **Step 3: Commit**

```bash
git add kernels/bitnet_lut.ea
git commit -m "feat: add lut_matmul SIMD kernel — 16-row parallel via shuffle_bytes"
```

---

### Task 4: Update spec and run full validation

**Files:**
- Modify: `SPEC.md`

- [ ] **Step 1: Update SPEC.md kernel status table**

Add `bitnet_lut.ea` entry to the kernel status table and update the remaining work section:

```markdown
| `bitnet_lut.ea` | `prepare_lut_weights`, `lut_matmul`, `lut_matmul_tail` | N/N | Done (cross-platform) |
```

Mark `bitnet_lut.ea` as ✅ done in the architecture diagram.
Remove from "Remaining work > Kernels" section.
Remove `shuffle_bytes` from "Remaining work > Compiler" section.

- [ ] **Step 2: Run full test suite one more time**

Run: `cd /home/peter/projects/eabitnet && make clean && make`

All tests (i2s, quant, lut) must pass.

- [ ] **Step 3: Commit**

```bash
git add SPEC.md
git commit -m "docs: update SPEC.md — bitnet_lut kernel complete"
```

---

### Task 5: Line count and final check

- [ ] **Step 1: Verify kernel file stays under 500 lines**

Run: `wc -l kernels/bitnet_lut.ea`

Must be under 500. If over, split `prepare_lut_weights` into `kernels/bitnet_lut_prep.ea`.

- [ ] **Step 2: Verify test file stays under 500 lines**

Run: `wc -l tests/test_lut.c`

If over 500, this is less critical (test files vs kernel code), but still good practice.
