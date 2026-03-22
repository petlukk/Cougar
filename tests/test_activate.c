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

static void test_positive_gate(void) {
    float gate[] = {2.0f, 3.0f, 1.0f, 0.5f, 2.0f, 3.0f, 1.0f, 0.5f};
    float up[]   = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float out[8];
    squared_relu_mul_f32(gate, up, out, 8);
    CHECK("pos_gate_0", CLOSE(out[0], 4.0f, 1e-5f));
    CHECK("pos_gate_1", CLOSE(out[1], 9.0f, 1e-5f));
    CHECK("pos_gate_2", CLOSE(out[2], 1.0f, 1e-5f));
    CHECK("pos_gate_3", CLOSE(out[3], 0.25f, 1e-5f));
}

static void test_negative_gate(void) {
    float gate[] = {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f};
    float up[]   = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
    float out[8];
    squared_relu_mul_f32(gate, up, out, 8);
    int ok = 1;
    for (int i = 0; i < 8; i++)
        if (fabsf(out[i]) > 1e-6f) ok = 0;
    CHECK("negative_gate_zero", ok);
}

static void test_zero_gate(void) {
    float gate[8] = {0};
    float up[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float out[8];
    squared_relu_mul_f32(gate, up, out, 8);
    int ok = 1;
    for (int i = 0; i < 8; i++)
        if (fabsf(out[i]) > 1e-6f) ok = 0;
    CHECK("zero_gate", ok);
}

static void test_mixed(void) {
    float gate[] = {2.0f, -1.0f, 3.0f, -2.0f, 0.0f, 1.0f, -0.5f, 4.0f};
    float up[]   = {1.0f,  1.0f, 2.0f,  2.0f, 5.0f, 3.0f,  3.0f, 0.5f};
    float expected[] = {4.0f, 0.0f, 18.0f, 0.0f, 0.0f, 3.0f, 0.0f, 8.0f};
    float out[8];
    squared_relu_mul_f32(gate, up, out, 8);
    int ok = 1;
    for (int i = 0; i < 8; i++)
        if (!CLOSE(out[i], expected[i], 1e-4f)) ok = 0;
    CHECK("mixed_8", ok);
}

static void test_large(void) {
    int n = 6912;
    float *gate = malloc(n * sizeof(float));
    float *up = malloc(n * sizeof(float));
    float *out = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        gate[i] = sinf((float)i * 0.01f);
        up[i] = 1.0f;
    }
    squared_relu_mul_f32(gate, up, out, n);
    int ok = 1;
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        float expected = g > 0 ? g * g : 0.0f;
        if (!CLOSE(out[i], expected, 1e-3f)) ok = 0;
    }
    CHECK("large_6912", ok);
    free(gate); free(up); free(out);
}

int main(void) {
    printf("test_activate:\n");
    test_positive_gate();
    test_negative_gate();
    test_zero_gate();
    test_mixed();
    test_large();
    printf("\n%d/%d passed\n", tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}
