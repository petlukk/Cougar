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

static void build_freqs(float *freqs, int head_dim, int position) {
    for (int i = 0; i < head_dim / 2; i++) {
        float theta = (float)position / powf(10000.0f, 2.0f * (float)i / (float)head_dim);
        freqs[2 * i] = cosf(theta);
        freqs[2 * i + 1] = sinf(theta);
    }
}

static void test_position_zero(void) {
    int hd = 8, nh = 1;
    float q[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float k[8] = {8, 7, 6, 5, 4, 3, 2, 1};
    float q_orig[8], k_orig[8];
    for (int i = 0; i < 8; i++) { q_orig[i] = q[i]; k_orig[i] = k[i]; }
    float freqs[8];
    build_freqs(freqs, hd, 0);
    rope_f32(q, k, freqs, hd, nh);
    int ok = 1;
    for (int i = 0; i < 8; i++)
        if (!CLOSE(q[i], q_orig[i], 1e-5f) || !CLOSE(k[i], k_orig[i], 1e-5f)) ok = 0;
    CHECK("position_zero_identity", ok);
}

static void test_known_rotation(void) {
    int hd = 8, nh = 1;
    float q[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    float k[8] = {0, 1, 0, 0, 0, 0, 0, 0};
    float freqs[8] = {0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    rope_f32(q, k, freqs, hd, nh);
    CHECK("q_rotated", CLOSE(q[0], 0.0f, 1e-5f) && CLOSE(q[1], 1.0f, 1e-5f));
    CHECK("k_rotated", CLOSE(k[0], -1.0f, 1e-5f) && CLOSE(k[1], 0.0f, 1e-5f));
}

static void test_multi_head(void) {
    int hd = 8, nh = 4;
    int total = hd * nh;
    float *q = calloc(total, sizeof(float));
    float *k = calloc(total, sizeof(float));
    for (int i = 0; i < total; i++) { q[i] = 1.0f; k[i] = 1.0f; }
    float freqs[8];
    build_freqs(freqs, hd, 42);
    rope_f32(q, k, freqs, hd, nh);
    int ok = 1;
    for (int h = 1; h < nh; h++) {
        for (int i = 0; i < hd; i++) {
            if (!CLOSE(q[h * hd + i], q[i], 1e-5f)) ok = 0;
            if (!CLOSE(k[h * hd + i], k[i], 1e-5f)) ok = 0;
        }
    }
    CHECK("multi_head_same_rotation", ok);
    free(q); free(k);
}

static void test_preserves_norm(void) {
    int hd = 80, nh = 1;
    float q[80], k[80];
    for (int i = 0; i < 80; i++) { q[i] = sinf((float)i); k[i] = cosf((float)i); }
    float q_norm_before = 0, k_norm_before = 0;
    for (int i = 0; i < 80; i++) {
        q_norm_before += q[i] * q[i];
        k_norm_before += k[i] * k[i];
    }
    float freqs[80];
    build_freqs(freqs, hd, 100);
    rope_f32(q, k, freqs, hd, nh);
    float q_norm_after = 0, k_norm_after = 0;
    for (int i = 0; i < 80; i++) {
        q_norm_after += q[i] * q[i];
        k_norm_after += k[i] * k[i];
    }
    CHECK("preserves_q_norm", CLOSE(q_norm_before, q_norm_after, 1e-2f));
    CHECK("preserves_k_norm", CLOSE(k_norm_before, k_norm_after, 1e-2f));
}

int main(void) {
    printf("test_rope:\n");
    test_position_zero();
    test_known_rotation();
    test_multi_head();
    test_preserves_norm();
    printf("\n%d/%d passed\n", tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}
