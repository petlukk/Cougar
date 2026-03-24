// test_q4k_dot.c — Validates Q4_K × Q8_K dot product kernel against scalar reference
//
// Build & run:
//   EA=/root/dev/eacompute/target/release/ea
//   $EA kernels/q4k_dot.ea --lib -o build/lib/libq4k_dot.so
//   gcc -O2 -Wall tests/test_q4k_dot.c -Lbuild/lib -lq4k_dot -o build/test_q4k_dot -lm
//   LD_LIBRARY_PATH=build/lib ./build/test_q4k_dot

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

extern float q4k_dot_q8k(
    const uint8_t *q4,
    const int8_t *q8,
    const int32_t *bsums,
    const uint8_t *scales,
    const uint8_t *mins,
    int32_t n_blocks,
    float d,
    float dmin
);

extern void q4k_dot_q8k_4row(
    const uint8_t *rw0, const uint8_t *rw1,
    const uint8_t *rw2, const uint8_t *rw3,
    const int8_t *q8,
    const int32_t *bsums,
    const uint8_t *sc0, const uint8_t *sc1,
    const uint8_t *sc2, const uint8_t *sc3,
    const uint8_t *mn0, const uint8_t *mn1,
    const uint8_t *mn2, const uint8_t *mn3,
    float *scores,
    int32_t n_blocks,
    float d0, float d1, float d2, float d3,
    float dm0, float dm1, float dm2, float dm3
);

#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"

// Scalar reference for Q4_K × Q8_K dot product
static float ref_q4k_dot_q8k(
    const uint8_t *q4, const int8_t *q8, const int32_t *bsums,
    const uint8_t *scales, const uint8_t *mins,
    int n_blocks, float d, float dmin
) {
    float result = 0.0f;
    for (int blk = 0; blk < n_blocks; blk++) {
        int sumi = 0, summs = 0;
        for (int j = 0; j < 4; j++) {
            int s0 = 0, s1 = 0;
            for (int k = 0; k < 32; k++) {
                s0 += (q4[blk*128 + j*32 + k] & 0xF) * q8[blk*256 + j*64 + k];
                s1 += (q4[blk*128 + j*32 + k] >> 4)  * q8[blk*256 + j*64 + 32 + k];
            }
            sumi += s0 * scales[blk*8 + 2*j] + s1 * scales[blk*8 + 2*j+1];
        }
        for (int j = 0; j < 8; j++) {
            summs += mins[blk*8 + j] * (bsums[blk*16 + 2*j] + bsums[blk*16 + 2*j+1]);
        }
        result += d * (float)sumi - dmin * (float)summs;
    }
    return result;
}

// Compute bsums from q8 values (sum of 16 consecutive i8 -> i32)
static void compute_bsums(const int8_t *q8, int32_t *bsums, int n_blocks) {
    for (int blk = 0; blk < n_blocks; blk++) {
        for (int g = 0; g < 16; g++) {
            int32_t sum = 0;
            for (int k = 0; k < 16; k++) {
                sum += (int32_t)q8[blk*256 + g*16 + k];
            }
            bsums[blk*16 + g] = sum;
        }
    }
}

static int check_close(const char *label, float expected, float got, float rel_tol) {
    float diff = fabsf(expected - got);
    float denom = fabsf(expected);
    if (denom < 1e-6f) denom = 1.0f;  // absolute check for near-zero
    float rel = diff / denom;
    if (rel > rel_tol && diff > 0.5f) {
        printf("%sFAIL%s %s: expected=%f got=%f (rel_err=%.6f)\n",
               RED, RESET, label, expected, got, rel);
        return 0;
    }
    return 1;
}

// ===== Test cases =====

static int test_known_small(void) {
    printf("  q4k_dot known_small (1 block) ... ");

    // 1 super-block = 256 elements
    uint8_t q4[128];      // packed nibbles
    int8_t q8[256];       // Q8_K activations
    uint8_t scales[8];
    uint8_t mins[8];

    // Fill with simple known pattern
    // q4 nibbles: low=3, high=5 -> packed byte = (5 << 4) | 3 = 0x53
    memset(q4, 0x53, 128);

    // q8: all 2
    memset(q8, 2, 256);

    // scales: all 1
    memset(scales, 1, 8);

    // mins: all 0 (no correction)
    memset(mins, 0, 8);

    int32_t bsums[16];
    compute_bsums(q8, bsums, 1);

    float d = 1.0f;
    float dmin = 0.0f;

    float expected = ref_q4k_dot_q8k(q4, q8, bsums, scales, mins, 1, d, dmin);
    float got = q4k_dot_q8k(q4, q8, bsums, scales, mins, 1, d, dmin);

    if (!check_close("result", expected, got, 0.0001f)) return 0;
    printf("%sPASS%s (expected=%f got=%f)\n", GREEN, RESET, expected, got);
    return 1;
}

static int test_two_blocks(void) {
    printf("  q4k_dot two_blocks ... ");

    srand(123);
    int nb = 2;
    uint8_t q4[256];     // 128 * 2
    int8_t q8[512];      // 256 * 2
    uint8_t scales[16];  // 8 * 2
    uint8_t mins[16];    // 8 * 2

    for (int i = 0; i < 256; i++) q4[i] = rand() & 0xFF;
    for (int i = 0; i < 512; i++) q8[i] = (int8_t)((rand() % 255) - 127);
    for (int i = 0; i < 16; i++) scales[i] = (rand() % 63) + 1;
    for (int i = 0; i < 16; i++) mins[i] = rand() % 63;

    int32_t bsums[32];
    compute_bsums(q8, bsums, nb);

    float d = 0.00125f;
    float dmin = 0.0008f;

    float expected = ref_q4k_dot_q8k(q4, q8, bsums, scales, mins, nb, d, dmin);
    float got = q4k_dot_q8k(q4, q8, bsums, scales, mins, nb, d, dmin);

    if (!check_close("result", expected, got, 0.0001f)) return 0;
    printf("%sPASS%s (expected=%f got=%f)\n", GREEN, RESET, expected, got);
    return 1;
}

static int test_zero_weights(void) {
    printf("  q4k_dot zero_weights ... ");

    // All nibbles = 0, so dot product = 0, result = -dmin * summs
    uint8_t q4[128];
    int8_t q8[256];
    uint8_t scales[8];
    uint8_t mins[8];

    memset(q4, 0, 128);
    for (int i = 0; i < 256; i++) q8[i] = (int8_t)(i % 10 + 1);
    memset(scales, 5, 8);
    memset(mins, 3, 8);

    int32_t bsums[16];
    compute_bsums(q8, bsums, 1);

    float d = 1.0f;
    float dmin = 0.5f;

    float expected = ref_q4k_dot_q8k(q4, q8, bsums, scales, mins, 1, d, dmin);
    float got = q4k_dot_q8k(q4, q8, bsums, scales, mins, 1, d, dmin);

    if (!check_close("result", expected, got, 0.0001f)) return 0;
    // Verify it's negative (only -dmin*summs contribution)
    if (got >= 0.0f) {
        printf("%sFAIL%s expected negative result, got=%f\n", RED, RESET, got);
        return 0;
    }
    printf("%sPASS%s (expected=%f got=%f)\n", GREEN, RESET, expected, got);
    return 1;
}

static int test_uniform_nibbles(void) {
    printf("  q4k_dot uniform_nibble8 ... ");

    // All nibbles = 8 -> packed = (8<<4)|8 = 0x88
    uint8_t q4[128];
    memset(q4, 0x88, 128);

    // q8: all 1
    int8_t q8[256];
    memset(q8, 1, 256);

    uint8_t scales[8];
    uint8_t mins[8];
    memset(scales, 2, 8);
    memset(mins, 1, 8);

    int32_t bsums[16];
    compute_bsums(q8, bsums, 1);

    float d = 0.01f;
    float dmin = 0.005f;

    float expected = ref_q4k_dot_q8k(q4, q8, bsums, scales, mins, 1, d, dmin);
    float got = q4k_dot_q8k(q4, q8, bsums, scales, mins, 1, d, dmin);

    if (!check_close("result", expected, got, 0.0001f)) return 0;
    printf("%sPASS%s (expected=%f got=%f)\n", GREEN, RESET, expected, got);
    return 1;
}

static int test_random_large(void) {
    printf("  q4k_dot random_8blocks ... ");

    srand(42);
    int nb = 8;
    uint8_t *q4 = malloc(128 * nb);
    int8_t *q8 = malloc(256 * nb);
    uint8_t *scales = malloc(8 * nb);
    uint8_t *mins = malloc(8 * nb);
    int32_t *bsums = malloc(16 * nb * sizeof(int32_t));

    for (int i = 0; i < 128 * nb; i++) q4[i] = rand() & 0xFF;
    for (int i = 0; i < 256 * nb; i++) q8[i] = (int8_t)((rand() % 255) - 127);
    for (int i = 0; i < 8 * nb; i++) scales[i] = rand() % 64;
    for (int i = 0; i < 8 * nb; i++) mins[i] = rand() % 64;
    compute_bsums(q8, bsums, nb);

    float d = 0.002f;
    float dmin = 0.001f;

    float expected = ref_q4k_dot_q8k(q4, q8, bsums, scales, mins, nb, d, dmin);
    float got = q4k_dot_q8k(q4, q8, bsums, scales, mins, nb, d, dmin);

    int ok = check_close("result", expected, got, 0.0001f);
    if (ok) printf("%sPASS%s (expected=%f got=%f)\n", GREEN, RESET, expected, got);

    free(q4); free(q8); free(scales); free(mins); free(bsums);
    return ok;
}

static int test_4row(void) {
    printf("  q4k_dot_4row ... ");

    srand(99);
    int nb = 4;
    int q4_sz = 128 * nb;
    int sc_sz = 8 * nb;

    uint8_t *rw[4], *sc[4], *mn[4];
    for (int r = 0; r < 4; r++) {
        rw[r] = malloc(q4_sz);
        sc[r] = malloc(sc_sz);
        mn[r] = malloc(sc_sz);
        for (int i = 0; i < q4_sz; i++) rw[r][i] = rand() & 0xFF;
        for (int i = 0; i < sc_sz; i++) sc[r][i] = rand() % 64;
        for (int i = 0; i < sc_sz; i++) mn[r][i] = rand() % 64;
    }

    int8_t *q8 = malloc(256 * nb);
    for (int i = 0; i < 256 * nb; i++) q8[i] = (int8_t)((rand() % 255) - 127);

    int32_t *bsums = malloc(16 * nb * sizeof(int32_t));
    compute_bsums(q8, bsums, nb);

    float d[4]  = {0.002f, 0.003f, 0.001f, 0.0015f};
    float dm[4] = {0.001f, 0.0015f, 0.0005f, 0.0008f};

    // Reference: call single-row 4 times
    float expected[4];
    for (int r = 0; r < 4; r++) {
        expected[r] = ref_q4k_dot_q8k(rw[r], q8, bsums, sc[r], mn[r], nb, d[r], dm[r]);
    }

    // Kernel: 4-row variant
    float got[4];
    q4k_dot_q8k_4row(
        rw[0], rw[1], rw[2], rw[3],
        q8, bsums,
        sc[0], sc[1], sc[2], sc[3],
        mn[0], mn[1], mn[2], mn[3],
        got, nb,
        d[0], d[1], d[2], d[3],
        dm[0], dm[1], dm[2], dm[3]
    );

    int ok = 1;
    for (int r = 0; r < 4; r++) {
        char label[32];
        snprintf(label, sizeof(label), "row[%d]", r);
        if (!check_close(label, expected[r], got[r], 0.0001f)) ok = 0;
    }

    if (ok) {
        printf("%sPASS%s (", GREEN, RESET);
        for (int r = 0; r < 4; r++)
            printf("r%d: exp=%f got=%f%s", r, expected[r], got[r], r < 3 ? ", " : "");
        printf(")\n");
    }

    for (int r = 0; r < 4; r++) { free(rw[r]); free(sc[r]); free(mn[r]); }
    free(q8); free(bsums);
    return ok;
}

static int test_4row_matches_single(void) {
    printf("  q4k_dot_4row vs single-row ... ");

    srand(77);
    int nb = 2;
    int q4_sz = 128 * nb;
    int sc_sz = 8 * nb;

    uint8_t *rw[4], *sc[4], *mn[4];
    for (int r = 0; r < 4; r++) {
        rw[r] = malloc(q4_sz);
        sc[r] = malloc(sc_sz);
        mn[r] = malloc(sc_sz);
        for (int i = 0; i < q4_sz; i++) rw[r][i] = rand() & 0xFF;
        for (int i = 0; i < sc_sz; i++) sc[r][i] = rand() % 64;
        for (int i = 0; i < sc_sz; i++) mn[r][i] = rand() % 64;
    }

    int8_t *q8 = malloc(256 * nb);
    for (int i = 0; i < 256 * nb; i++) q8[i] = (int8_t)((rand() % 255) - 127);

    int32_t *bsums = malloc(16 * nb * sizeof(int32_t));
    compute_bsums(q8, bsums, nb);

    float d[4]  = {0.01f, 0.02f, 0.005f, 0.015f};
    float dm[4] = {0.005f, 0.01f, 0.003f, 0.007f};

    // Single-row kernel results
    float single[4];
    for (int r = 0; r < 4; r++) {
        single[r] = q4k_dot_q8k(rw[r], q8, bsums, sc[r], mn[r], nb, d[r], dm[r]);
    }

    // 4-row kernel results
    float multi[4];
    q4k_dot_q8k_4row(
        rw[0], rw[1], rw[2], rw[3],
        q8, bsums,
        sc[0], sc[1], sc[2], sc[3],
        mn[0], mn[1], mn[2], mn[3],
        multi, nb,
        d[0], d[1], d[2], d[3],
        dm[0], dm[1], dm[2], dm[3]
    );

    int ok = 1;
    for (int r = 0; r < 4; r++) {
        char label[32];
        snprintf(label, sizeof(label), "row[%d]", r);
        if (!check_close(label, single[r], multi[r], 0.0001f)) ok = 0;
    }

    if (ok) printf("%sPASS%s\n", GREEN, RESET);

    for (int r = 0; r < 4; r++) { free(rw[r]); free(sc[r]); free(mn[r]); }
    free(q8); free(bsums);
    return ok;
}

int main(void) {
    printf("=== Q4_K × Q8_K dot product kernel tests ===\n\n");
    int pass = 0, total = 0;

    printf("q4k_dot_q8k (single-row):\n");
    total++; if (test_known_small())     pass++;
    total++; if (test_two_blocks())      pass++;
    total++; if (test_zero_weights())    pass++;
    total++; if (test_uniform_nibbles()) pass++;
    total++; if (test_random_large())    pass++;

    printf("\nq4k_dot_q8k_4row:\n");
    total++; if (test_4row())                pass++;
    total++; if (test_4row_matches_single()) pass++;

    printf("\n%d/%d tests passed\n", pass, total);
    return pass == total ? 0 : 1;
}
