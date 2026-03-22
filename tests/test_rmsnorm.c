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

static void ref_rmsnorm(const float *x, const float *w, float *out, int n, float eps) {
    float sumsq = 0.0f;
    for (int i = 0; i < n; i++) sumsq += x[i] * x[i];
    float rms = sqrtf(sumsq / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * w[i] / rms;
}

static void test_known_vector(void) {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float out[8], ref[8];
    ref_rmsnorm(x, w, ref, 8, 1e-5f);
    rmsnorm_f32(x, w, out, 8, 1e-5f);
    int ok = 1;
    for (int i = 0; i < 8; i++)
        if (!CLOSE(out[i], ref[i], 1e-4f)) ok = 0;
    CHECK("known_8_unit_weight", ok);
}

static void test_with_weights(void) {
    float x[] = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f};
    float w[] = {0.5f, 2.0f, 0.5f, 2.0f, 0.5f, 2.0f, 0.5f, 2.0f};
    float out[8], ref[8];
    ref_rmsnorm(x, w, ref, 8, 1e-5f);
    rmsnorm_f32(x, w, out, 8, 1e-5f);
    int ok = 1;
    for (int i = 0; i < 8; i++)
        if (!CLOSE(out[i], ref[i], 1e-4f)) ok = 0;
    CHECK("weighted_8", ok);
}

static void test_uniform(void) {
    int n = 16;
    float x[16], w[16], out[16], ref[16];
    for (int i = 0; i < n; i++) { x[i] = 3.0f; w[i] = 1.0f; }
    ref_rmsnorm(x, w, ref, n, 1e-5f);
    rmsnorm_f32(x, w, out, n, 1e-5f);
    int ok = 1;
    for (int i = 0; i < n; i++)
        if (!CLOSE(out[i], ref[i], 1e-4f)) ok = 0;
    CHECK("uniform_16", ok);
}

static void test_large_dim(void) {
    int n = 2560;
    float *x = malloc(n * sizeof(float));
    float *w = malloc(n * sizeof(float));
    float *out = malloc(n * sizeof(float));
    float *ref = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        x[i] = sinf((float)i * 0.01f);
        w[i] = 1.0f + 0.1f * cosf((float)i * 0.02f);
    }
    ref_rmsnorm(x, w, ref, n, 1e-5f);
    rmsnorm_f32(x, w, out, n, 1e-5f);
    int ok = 1;
    for (int i = 0; i < n; i++)
        if (!CLOSE(out[i], ref[i], 1e-3f)) ok = 0;
    CHECK("large_2560", ok);
    free(x); free(w); free(out); free(ref);
}

static void test_near_zero(void) {
    float x[] = {1e-10f, -1e-10f, 1e-10f, -1e-10f, 1e-10f, -1e-10f, 1e-10f, -1e-10f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float out[8], ref[8];
    ref_rmsnorm(x, w, ref, 8, 1e-5f);
    rmsnorm_f32(x, w, out, 8, 1e-5f);
    int ok = 1;
    for (int i = 0; i < 8; i++)
        if (!CLOSE(out[i], ref[i], 1e-2f)) ok = 0;
    CHECK("near_zero_eps_stabilized", ok);
}

int main(void) {
    printf("test_rmsnorm:\n");
    test_known_vector();
    test_with_weights();
    test_uniform();
    test_large_dim();
    test_near_zero();
    printf("\n%d/%d passed\n", tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}
