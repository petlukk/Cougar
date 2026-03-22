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

static void test_scores_single_token(void) {
    int hd = 8;
    float q[] = {1, 0, 0, 0, 0, 0, 0, 0};
    float k[] = {2, 0, 0, 0, 0, 0, 0, 0};
    float out[1];
    attn_scores_f32(q, k, out, hd, 1, 0.5f);
    CHECK("single_token_score", CLOSE(out[0], 1.0f, 1e-5f));
}

static void test_scores_multi_token(void) {
    int hd = 4, seq = 3;
    float q[] = {1, 1, 1, 1};
    float k[] = {1,0,0,0,  0,1,0,0,  0,0,1,0};
    float out[3];
    attn_scores_f32(q, k, out, hd, seq, 1.0f);
    CHECK("multi_token_0", CLOSE(out[0], 1.0f, 1e-5f));
    CHECK("multi_token_1", CLOSE(out[1], 1.0f, 1e-5f));
    CHECK("multi_token_2", CLOSE(out[2], 1.0f, 1e-5f));
}

static void test_weighted_sum_single(void) {
    int hd = 4;
    float scores[] = {1.0f};
    float v[] = {1, 2, 3, 4};
    float out[4] = {0};
    attn_weighted_sum_f32(scores, v, out, hd, 1);
    int ok = 1;
    for (int i = 0; i < 4; i++)
        if (!CLOSE(out[i], v[i], 1e-5f)) ok = 0;
    CHECK("wsum_single", ok);
}

static void test_weighted_sum_uniform(void) {
    int hd = 4, seq = 2;
    float scores[] = {0.5f, 0.5f};
    float v[] = {2,0,0,0,  0,2,0,0};
    float out[4] = {0};
    attn_weighted_sum_f32(scores, v, out, hd, seq);
    CHECK("wsum_avg_0", CLOSE(out[0], 1.0f, 1e-5f));
    CHECK("wsum_avg_1", CLOSE(out[1], 1.0f, 1e-5f));
}

static void test_large_dim(void) {
    int hd = 80, seq = 64;
    float *q = malloc(hd * sizeof(float));
    float *k = malloc(seq * hd * sizeof(float));
    float *scores = malloc(seq * sizeof(float));
    for (int i = 0; i < hd; i++) q[i] = 1.0f;
    for (int t = 0; t < seq; t++)
        for (int i = 0; i < hd; i++)
            k[t * hd + i] = (i == 0) ? 1.0f : 0.0f;
    float scale = 1.0f / sqrtf(80.0f);
    attn_scores_f32(q, k, scores, hd, seq, scale);
    int ok = 1;
    for (int t = 0; t < seq; t++)
        if (!CLOSE(scores[t], scale, 1e-4f)) ok = 0;
    CHECK("large_80x64_scores", ok);
    free(q); free(k); free(scores);
}

static void test_weighted_sum_hd80(void) {
    int hd = 80, seq = 3;
    float scores[] = {0.5f, 0.3f, 0.2f};
    float *v = calloc(seq * hd, sizeof(float));
    float *out = calloc(hd, sizeof(float));
    for (int i = 0; i < hd; i++) v[i] = 1.0f;
    for (int i = 0; i < hd; i++) v[hd + i] = 2.0f;
    for (int i = 0; i < hd; i++) v[2*hd + i] = 3.0f;
    attn_weighted_sum_f32(scores, v, out, hd, seq);
    int ok = 1;
    for (int i = 0; i < hd; i++)
        if (!CLOSE(out[i], 1.7f, 1e-4f)) ok = 0;
    CHECK("wsum_hd80_tail", ok);
    free(v); free(out);
}

int main(void) {
    printf("test_attention:\n");
    test_scores_single_token();
    test_scores_multi_token();
    test_weighted_sum_single();
    test_weighted_sum_uniform();
    test_large_dim();
    test_weighted_sum_hd80();
    printf("\n%d/%d passed\n", tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}
