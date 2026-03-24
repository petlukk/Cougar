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

// Reference implementation
static void ref_silu_mul(const float *gate, const float *up, float *out, int n) {
    for (int i = 0; i < n; i++) {
        float s = 1.0f / (1.0f + expf(-gate[i]));
        out[i] = gate[i] * s * up[i];
    }
}

static void test_basic_values(void) {
    float gate[] = {0.5f, 1.0f, 2.0f, 3.0f};
    float up[]   = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];
    float expected[4];

    ref_silu_mul(gate, up, expected, 4);
    silu_mul_f32(gate, up, out, 4);

    int ok = 1;
    for (int i = 0; i < 4; i++) {
        if (!CLOSE(out[i], expected[i], 1e-5f)) {
            ok = 0;
            printf("    [i=%d] got %f, expected %f\n", i, out[i], expected[i]);
        }
    }
    CHECK("basic_values", ok);
}

static void test_zero_gate(void) {
    float gate[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up[]   = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4];

    silu_mul_f32(gate, up, out, 4);

    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    int ok = 1;
    for (int i = 0; i < 4; i++) {
        if (fabsf(out[i]) > 1e-6f) {
            ok = 0;
            printf("    [i=%d] got %f, expected 0\n", i, out[i]);
        }
    }
    CHECK("zero_gate", ok);
}

static void test_negative_values(void) {
    float gate[] = {-1.0f, -2.0f, -3.0f, -4.0f};
    float up[]   = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];
    float expected[4];

    ref_silu_mul(gate, up, expected, 4);
    silu_mul_f32(gate, up, out, 4);

    // For large negative values, silu should be small but non-zero
    int ok = 1;
    for (int i = 0; i < 4; i++) {
        if (!CLOSE(out[i], expected[i], 1e-5f)) {
            ok = 0;
            printf("    [i=%d] got %f, expected %f\n", i, out[i], expected[i]);
        }
    }
    CHECK("negative_values", ok);
}

static void test_large_positive(void) {
    float gate[] = {5.0f, 10.0f, 50.0f, 100.0f};
    float up[]   = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];
    float expected[4];

    ref_silu_mul(gate, up, expected, 4);
    silu_mul_f32(gate, up, out, 4);

    // For large positive values, sigmoid(x) ≈ 1, so silu(x) ≈ x
    int ok = 1;
    for (int i = 0; i < 4; i++) {
        if (!CLOSE(out[i], expected[i], 1e-4f)) {
            ok = 0;
            printf("    [i=%d] got %f, expected %f\n", i, out[i], expected[i]);
        }
    }
    CHECK("large_positive", ok);
}

static void test_mixed_values(void) {
    float gate[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    float up[]   = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float out[6];
    float expected[6];

    ref_silu_mul(gate, up, expected, 6);
    silu_mul_f32(gate, up, out, 6);

    int ok = 1;
    for (int i = 0; i < 6; i++) {
        if (!CLOSE(out[i], expected[i], 1e-5f)) {
            ok = 0;
            printf("    [i=%d] got %f, expected %f\n", i, out[i], expected[i]);
        }
    }
    CHECK("mixed_values", ok);
}

static void test_large_array(void) {
    int n = 256;
    float *gate = malloc(n * sizeof(float));
    float *up = malloc(n * sizeof(float));
    float *out = malloc(n * sizeof(float));
    float *expected = malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        gate[i] = sinf((float)i * 0.02f) * 3.0f;
        up[i] = cosf((float)i * 0.015f) + 1.5f;
    }

    ref_silu_mul(gate, up, expected, n);
    silu_mul_f32(gate, up, out, n);

    int ok = 1;
    for (int i = 0; i < n; i++) {
        if (!CLOSE(out[i], expected[i], 1e-4f)) {
            ok = 0;
            printf("    [i=%d] got %f, expected %f\n", i, out[i], expected[i]);
            break;
        }
    }
    CHECK("large_array_256", ok);

    free(gate);
    free(up);
    free(out);
    free(expected);
}

int main(void) {
    printf("test_silu:\n");
    test_basic_values();
    test_zero_gate();
    test_negative_values();
    test_large_positive();
    test_mixed_values();
    test_large_array();
    printf("\n%d/%d passed\n", tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}
