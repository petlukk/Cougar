use super::*;

// ── f16_to_f32 ─────────────────────────────────────────────────────────

#[test]
fn f16_one() {
    // 0x3C00 = 1.0
    assert_eq!(f16_to_f32(0x3C00), 1.0f32);
}

#[test]
fn f16_two() {
    // 0x4000 = 2.0
    assert_eq!(f16_to_f32(0x4000), 2.0f32);
}

#[test]
fn f16_neg_one() {
    // 0xBC00 = -1.0
    assert_eq!(f16_to_f32(0xBC00), -1.0f32);
}

#[test]
fn f16_pos_zero() {
    // 0x0000 = +0.0
    let v = f16_to_f32(0x0000);
    assert_eq!(v, 0.0f32);
    assert!(v.is_sign_positive());
}

#[test]
fn f16_neg_zero() {
    // 0x8000 = -0.0
    let v = f16_to_f32(0x8000);
    assert_eq!(v, 0.0f32);
    assert!(v.is_sign_negative());
}

#[test]
fn f16_max_normal() {
    // 0x7BFF = 65504.0 (max normal f16)
    assert_eq!(f16_to_f32(0x7BFF), 65504.0f32);
}

#[test]
fn f16_smallest_normal() {
    // 0x0400 = 2^-14 ≈ 6.103515625e-5
    let expected = 2.0f32.powi(-14);
    let got = f16_to_f32(0x0400);
    assert!((got - expected).abs() < 1e-10, "got {got}, expected {expected}");
}

#[test]
fn f16_smallest_subnormal() {
    // 0x0001 = 2^-24 ≈ 5.960464e-8
    let expected = 2.0f32.powi(-24);
    let got = f16_to_f32(0x0001);
    assert!((got - expected).abs() < 1e-30, "got {got}, expected {expected}");
}

// ── embed_f16_lookup ───────────────────────────────────────────────────

#[test]
fn embed_f16_lookup_basic() {
    // Build a fake embedding table: 3 tokens × 4 dims
    // Encoded as u16 (f16) values, stored as raw bytes.
    // Token 1: [1.0, 2.0, -1.0, 0.0] → [0x3C00, 0x4000, 0xBC00, 0x0000]
    let token1: [u16; 4] = [0x3C00, 0x4000, 0xBC00, 0x0000];
    let token0: [u16; 4] = [0x0000; 4];
    let token2: [u16; 4] = [0x7BFF, 0x0400, 0x0001, 0x8000];

    // Lay out as bytes: token0, token1, token2
    let mut raw_bytes = Vec::with_capacity(3 * 4 * 2);
    for &v in token0.iter().chain(token1.iter()).chain(token2.iter()) {
        raw_bytes.extend_from_slice(&v.to_ne_bytes());
    }

    let hidden_dim = 4usize;
    let mut out = vec![0.0f32; hidden_dim];

    embed_f16_lookup(raw_bytes.as_ptr(), 1, &mut out, hidden_dim);

    assert_eq!(out[0], 1.0f32,  "dim 0 should be 1.0");
    assert_eq!(out[1], 2.0f32,  "dim 1 should be 2.0");
    assert_eq!(out[2], -1.0f32, "dim 2 should be -1.0");
    assert_eq!(out[3], 0.0f32,  "dim 3 should be 0.0");

    // Also verify token 2
    embed_f16_lookup(raw_bytes.as_ptr(), 2, &mut out, hidden_dim);
    assert_eq!(out[0], 65504.0f32, "dim 0 of token 2 should be 65504");
    assert!((out[1] - 2.0f32.powi(-14)).abs() < 1e-10, "dim 1 of token 2 should be 2^-14");
}

// ── FFI-dependent matmul tests ─────────────────────────────────────────
//
// These tests require the shared libraries in build/lib.
// Run with: LD_LIBRARY_PATH=build/lib RUSTFLAGS="-L build/lib" cargo test
//
// Ternary weight encoding (BitNet b1.58):
//   2-bit value 0 (0b00) → weight -1
//   2-bit value 1 (0b01) → weight  0
//   2-bit value 2 (0b10) → weight +1
//
// Four weights pack into one byte using the x86 group layout:
//   bits 7:6 = position i in group 0 (positions 0..31 per 32-byte block)
//   bits 5:4 = position i in group 1 (positions 32..63)
//   bits 3:2 = position i in group 2 (positions 64..95)
//   bits 1:0 = position i in group 3 (positions 96..127)
//
// So: byte 0xAA = 10101010b → each 2-bit group = 0b10 = +1
//     byte 0x55 = 01010101b → each 2-bit group = 0b01 =  0
//     byte 0x00 = 00000000b → each 2-bit group = 0b00 = -1
//
// Kernel computes raw = sum(g[i] * act[i]) where g[i] = w[i] + 1 ∈ {0, 1, 2}.
// Caller corrects: correct = raw - act_sum, where act_sum = sum(act[i]).
// This works because g[i] = w[i] + 1, so:
//   raw = sum((w[i]+1) * act[i]) = correct + sum(act[i]) = correct + act_sum.
// Final output: out[i] = (raw[i] - act_sum) * (act_scale / 127.0) * weight_scale.
//
// The kernel processes in blocks of 128 activations (n must be a multiple of 128).

/// Helper: pack a slice of ternary weights {-1, 0, +1} into the BitNet i2 format.
/// in_dim must be a multiple of 128.
fn pack_ternary_weights(ternary: &[i8], in_dim: usize) -> Vec<u8> {
    let n_blocks = in_dim / 128;
    let mut packed = vec![0u8; n_blocks * 32];
    for blk in 0..n_blocks {
        for i in 0..32 {
            let base = blk * 128;
            let g0 = (ternary[base + i] + 1) as u8;       // position i      (bits 7:6)
            let g1 = (ternary[base + i + 32] + 1) as u8;  // position i + 32 (bits 5:4)
            let g2 = (ternary[base + i + 64] + 1) as u8;  // position i + 64 (bits 3:2)
            let g3 = (ternary[base + i + 96] + 1) as u8;  // position i + 96 (bits 1:0)
            packed[blk * 32 + i] = (g0 << 6) | (g1 << 4) | (g2 << 2) | g3;
        }
    }
    packed
}

/// Test correctness of `ternary_matmul_mt_n` with known weights and activations.
#[test]
fn test_ternary_matmul_identity() {
    let in_dim: usize = 128;
    let out_dim: usize = 4;

    let ternary_weights: Vec<i8> = vec![1i8; out_dim * in_dim];
    let mut weight_rows: Vec<Vec<u8>> = Vec::new();
    for r in 0..out_dim {
        let row_slice = &ternary_weights[r * in_dim..(r + 1) * in_dim];
        weight_rows.push(pack_ternary_weights(row_slice, in_dim));
    }
    let row_bytes = in_dim / 4;
    let mut weight: Vec<u8> = Vec::with_capacity(out_dim * row_bytes);
    for r in 0..out_dim {
        weight.extend_from_slice(&weight_rows[r]);
    }

    let act: Vec<i8> = vec![1i8; in_dim];
    let act_sum: i32 = act.iter().map(|&v| v as i32).sum();
    let act_scale = 127.0f32;
    let weight_scale = 1.0f32;

    let mut out = vec![0.0f32; out_dim];
    let pool = crate::threadpool::ThreadPool::new();

    ternary_matmul_mt_n(
        weight.as_ptr(), act.as_ptr(),
        act_scale, act_sum, weight_scale,
        &mut out, out_dim, in_dim,
        pool.thread_count(), &pool,
    );

    let expected = 128.0f32;
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - expected).abs() < 1e-3,
            "row {i}: expected {expected}, got {v}"
        );
    }
}

/// Smoke test: `ternary_matmul_qkv` runs without crashing and produces finite output.
#[test]
fn test_ternary_matmul_qkv_runs() {
    let in_dim: usize = 128;
    let out_dim_q: usize = 8;
    let out_dim_kv: usize = 4;

    let make_weights = |out_dim: usize| -> Vec<u8> {
        let ternary: Vec<i8> = vec![1i8; out_dim * in_dim];
        let row_bytes = in_dim / 4;
        let mut w = Vec::with_capacity(out_dim * row_bytes);
        for r in 0..out_dim {
            w.extend_from_slice(&pack_ternary_weights(&ternary[r * in_dim..(r + 1) * in_dim], in_dim));
        }
        w
    };

    let w_q = make_weights(out_dim_q);
    let w_k = make_weights(out_dim_kv);
    let w_v = make_weights(out_dim_kv);

    let act: Vec<i8> = vec![1i8; in_dim];
    let act_sum: i32 = act.iter().map(|&v| v as i32).sum();
    let act_scale = 127.0f32;

    let mut out_q = vec![0.0f32; out_dim_q];
    let mut out_k = vec![0.0f32; out_dim_kv];
    let mut out_v = vec![0.0f32; out_dim_kv];

    let pool = crate::threadpool::ThreadPool::new();

    ternary_matmul_qkv(
        w_q.as_ptr(), 1.0f32, &mut out_q, out_dim_q,
        w_k.as_ptr(), 1.0f32, &mut out_k, out_dim_kv,
        w_v.as_ptr(), 1.0f32, &mut out_v,
        act.as_ptr(), act_scale, act_sum, in_dim,
        &pool,
    );

    for (i, &v) in out_q.iter().enumerate() {
        assert!(v.is_finite(), "out_q[{i}] is not finite: {v}");
        assert!(v != 0.0, "out_q[{i}] unexpectedly zero");
    }
    for (i, &v) in out_k.iter().enumerate() {
        assert!(v.is_finite(), "out_k[{i}] is not finite: {v}");
        assert!(v != 0.0, "out_k[{i}] unexpectedly zero");
    }
    for (i, &v) in out_v.iter().enumerate() {
        assert!(v.is_finite(), "out_v[{i}] is not finite: {v}");
        assert!(v != 0.0, "out_v[{i}] unexpectedly zero");
    }
}

/// Smoke test: `ternary_matmul_parallel_pair` runs and produces finite, non-zero output.
#[test]
fn test_ternary_matmul_parallel_pair_runs() {
    let in_dim: usize = 128;
    let out_dim: usize = 8;

    let make_weights = || -> Vec<u8> {
        let ternary: Vec<i8> = vec![1i8; out_dim * in_dim];
        let row_bytes = in_dim / 4;
        let mut w = Vec::with_capacity(out_dim * row_bytes);
        for r in 0..out_dim {
            w.extend_from_slice(&pack_ternary_weights(&ternary[r * in_dim..(r + 1) * in_dim], in_dim));
        }
        w
    };

    let w_a = make_weights();
    let w_b = make_weights();

    let act: Vec<i8> = vec![1i8; in_dim];
    let act_sum: i32 = act.iter().map(|&v| v as i32).sum();
    let act_scale = 127.0f32;

    let mut out_a = vec![0.0f32; out_dim];
    let mut out_b = vec![0.0f32; out_dim];

    let pool = crate::threadpool::ThreadPool::new();

    ternary_matmul_parallel_pair(
        w_a.as_ptr(), 1.0f32,
        w_b.as_ptr(), 1.0f32,
        act.as_ptr(), act_scale, act_sum,
        &mut out_a, &mut out_b,
        out_dim, in_dim,
        &pool,
    );

    for (i, &v) in out_a.iter().enumerate() {
        assert!(v.is_finite(), "out_a[{i}] is not finite: {v}");
        assert!(v != 0.0, "out_a[{i}] unexpectedly zero");
    }
    for (i, &v) in out_b.iter().enumerate() {
        assert!(v.is_finite(), "out_b[{i}] is not finite: {v}");
        assert!(v != 0.0, "out_b[{i}] unexpectedly zero");
    }
}

/// Test `i8_output_matmul_mt` with a small known case.
#[test]
fn test_i8_output_matmul() {
    let hidden_dim: usize = 16;
    let vocab_size: usize = 4;

    let embed_i8: Vec<u8> = vec![129u8; vocab_size * hidden_dim];
    let row_scales: Vec<f32> = vec![127.0f32; vocab_size];
    let x: Vec<f32> = vec![1.0f32; hidden_dim];
    let mut out = vec![0.0f32; vocab_size];

    let pool = crate::threadpool::ThreadPool::new();

    i8_output_matmul_mt(
        &embed_i8, &row_scales,
        &x, &mut out,
        vocab_size, hidden_dim,
        &pool,
    );

    let expected = 16.0f32;
    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "out[{i}] is not finite: {v}");
        assert!(
            (v - expected).abs() < 0.05,
            "row {i}: expected ~{expected}, got {v}"
        );
    }
}
