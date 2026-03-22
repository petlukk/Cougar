#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/eabitnet.h"

static int tests_passed = 0;
static int tests_total = 0;

#define CHECK(name, cond) do { \
    tests_total++; \
    if (cond) { tests_passed++; printf("  PASS: %s\n", name); } \
    else { printf("  FAIL: %s\n", name); } \
} while(0)

#define CLOSE(a, b, tol) (fabsf((a) - (b)) < (tol))

static void test_identity(void) {
    // x=[1,0,0,0,...], rows=identity-ish → out[r] = rows[r][0]
    int dim = 8, n = 4;
    float x[] = {1,0,0,0,0,0,0,0};
    float rows[] = {
        2,0,0,0,0,0,0,0,
        0,3,0,0,0,0,0,0,
        0,0,4,0,0,0,0,0,
        0,0,0,5,0,0,0,0
    };
    float out[4];
    tiled_dot_4row(x, rows, out, dim, n);
    CHECK("identity_r0", CLOSE(out[0], 2.0f, 1e-5f));
    CHECK("identity_r1", CLOSE(out[1], 0.0f, 1e-5f));
    CHECK("identity_r2", CLOSE(out[2], 0.0f, 1e-5f));
    CHECK("identity_r3", CLOSE(out[3], 0.0f, 1e-5f));
}

static void test_uniform(void) {
    int dim = 16, n = 8;
    float *x = malloc(dim * sizeof(float));
    float *rows = malloc(n * dim * sizeof(float));
    float out[8];
    for (int i = 0; i < dim; i++) x[i] = 1.0f;
    for (int i = 0; i < n * dim; i++) rows[i] = 1.0f;
    tiled_dot_4row(x, rows, out, dim, n);
    int ok = 1;
    for (int i = 0; i < n; i++)
        if (!CLOSE(out[i], (float)dim, 1e-3f)) ok = 0;
    CHECK("uniform_16x8", ok);
    free(x); free(rows);
}

static void test_large(void) {
    // Realistic: dim=2560, n_rows=1024
    int dim = 2560, n = 1024;
    float *x = malloc(dim * sizeof(float));
    float *rows = malloc((size_t)n * dim * sizeof(float));
    float *out = malloc(n * sizeof(float));
    srand(42);
    for (int i = 0; i < dim; i++) x[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < n * dim; i++) rows[i] = (float)rand() / RAND_MAX - 0.5f;
    tiled_dot_4row(x, rows, out, dim, n);
    // Verify against scalar reference
    int ok = 1;
    for (int r = 0; r < n; r++) {
        float ref = 0;
        for (int d = 0; d < dim; d++) ref += x[d] * rows[r * dim + d];
        if (!CLOSE(out[r], ref, 1e-1f)) { ok = 0; break; }
    }
    CHECK("large_2560x1024", ok);
    free(x); free(rows); free(out);
}

static void test_tail_rows(void) {
    // n_rows=5 → 4 tiled + 1 tail
    int dim = 8, n = 5;
    float x[] = {1,1,1,1,1,1,1,1};
    float rows[40];
    for (int r = 0; r < n; r++)
        for (int d = 0; d < dim; d++)
            rows[r * dim + d] = (float)(r + 1);
    float out[5];
    tiled_dot_4row(x, rows, out, dim, n);
    int ok = 1;
    for (int r = 0; r < n; r++)
        if (!CLOSE(out[r], (float)(r + 1) * dim, 1e-3f)) ok = 0;
    CHECK("tail_5rows", ok);
}

int main(void) {
    printf("test_output:\n");
    test_identity();
    test_uniform();
    test_large();
    test_tail_rows();
    printf("\n%d/%d passed\n", tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}
