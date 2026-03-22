mod ffi;
mod forward;
mod gguf;
mod model;
mod tokenizer;

use forward::InferenceState;
use model::BitNetModel;
use tokenizer::Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut model_path = None;
    let mut prompt = None;
    let mut max_tokens: usize = 128;
    let mut temperature: f32 = 0.0;
    let mut max_seq_len: usize = 2048;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = Some(args[i].as_str());
            }
            "--prompt" => {
                i += 1;
                prompt = Some(args[i].as_str());
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args[i].parse().expect("invalid --max-tokens");
            }
            "--temperature" => {
                i += 1;
                temperature = args[i].parse().expect("invalid --temperature");
            }
            "--max-seq-len" => {
                i += 1;
                max_seq_len = args[i].parse().expect("invalid --max-seq-len");
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let model_path = model_path.unwrap_or_else(|| {
        eprintln!("Usage: eabitnet --model <path.gguf> --prompt <text> [--max-tokens N] [--temperature T]");
        std::process::exit(1);
    });
    let prompt_text = prompt.unwrap_or_else(|| {
        eprintln!("Usage: eabitnet --model <path.gguf> --prompt <text>");
        std::process::exit(1);
    });

    // Open GGUF
    let gguf = match gguf::GgufFile::open(model_path) {
        Ok(gf) => {
            eprintln!(
                "GGUF v{}: {} tensors, {} metadata keys",
                gf.version,
                gf.tensors.len(),
                gf.metadata.len(),
            );
            gf
        }
        Err(e) => {
            eprintln!("Failed to open GGUF: {e}");
            std::process::exit(1);
        }
    };

    // Build tokenizer
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap_or_else(|e| {
        eprintln!("Failed to build tokenizer: {e}");
        std::process::exit(1);
    });

    // Load model
    let model = BitNetModel::from_gguf(&gguf).unwrap_or_else(|e| {
        eprintln!("Failed to load model: {e}");
        std::process::exit(1);
    });

    eprintln!(
        "Model: {} layers, {} hidden, {} heads ({} kv), {} head_dim, {} kv_dim, {} ffn, {} vocab",
        model.n_layers, model.hidden_dim, model.n_heads, model.n_kv_heads,
        model.head_dim, model.kv_dim, model.ffn_dim, model.vocab_size,
    );
    // Debug: check rope dimension count and weight stats
    if let Some(rd) = gguf.get_u32("bitnet-b1.58.rope.dimension_count") {
        eprintln!("rope.dimension_count = {rd}");
    }
    // Decode first 128 weights of Q layer 0 to check encoding
    if let Some(q_data) = gguf.tensor_data("blk.0.attn_q.weight") {
        let mut counts = [0u32; 4]; // count of 0, 1, 2, 3 values
        let check_bytes = q_data.len().min(1000) - 32; // skip scale
        for &b in &q_data[..check_bytes] {
            counts[((b >> 6) & 3) as usize] += 1;
            counts[((b >> 4) & 3) as usize] += 1;
            counts[((b >> 2) & 3) as usize] += 1;
            counts[(b & 3) as usize] += 1;
        }
        eprintln!("Q weight 2-bit value counts (first {check_bytes} bytes): 0={} 1={} 2={} 3={}", counts[0], counts[1], counts[2], counts[3]);
    }

    eprintln!("rope_theta={} rms_eps={}", model.rope_theta, model.rms_eps);

    // Validation: manually decode first row of Q weight and compute dot product
    {
        // Get the raw I2_S Q weight data (repacked already)
        let q_data = unsafe { std::slice::from_raw_parts(model.layers[0].wq, model.hidden_dim * model.hidden_dim / 4) };
        // Decode first row (first hidden_dim/4 = 640 bytes → 2560 ternary values)
        let in_dim = model.hidden_dim;
        let row_bytes = in_dim / 4;
        let mut ternary = vec![0i8; in_dim];
        for byte_idx in 0..row_bytes {
            let b = q_data[byte_idx];
            // Our encoding: 0→-1, 1→0, 2→+1
            let decode = |v: u8| -> i8 { match v { 0 => -1, 1 => 0, 2 => 1, _ => 0 } };
            let g0 = (b >> 6) & 3;
            let g1 = (b >> 4) & 3;
            let g2 = (b >> 2) & 3;
            let g3 = b & 3;
            // Groups map to activations: g0→[0..31], g1→[32..63], g2→[64..95], g3→[96..127]
            // Within a QK=128 block: byte_idx within block maps to element within group
            let block = byte_idx / 32;
            let within = byte_idx % 32;
            let base = block * 128;
            ternary[base + within] = decode(g0);
            ternary[base + 32 + within] = decode(g1);
            ternary[base + 64 + within] = decode(g2);
            ternary[base + 96 + within] = decode(g3);
        }
        // Count ternary values
        let n_neg = ternary.iter().filter(|&&v| v == -1).count();
        let n_zero = ternary.iter().filter(|&&v| v == 0).count();
        let n_pos = ternary.iter().filter(|&&v| v == 1).count();
        eprintln!("Q row 0 ternary: -1={n_neg} 0={n_zero} +1={n_pos}");
        eprintln!("Q row 0 first 20: {:?}", &ternary[..20]);
    }

    // Manual dot product for row 0 of Q
    {
        // Get embedding for BOS token
        let mut embed = vec![0.0f32; model.hidden_dim];
        let row = unsafe { std::slice::from_raw_parts(model.embed_weight.add(tokenizer.bos_id as usize * model.hidden_dim * 2) as *const u16, model.hidden_dim) };
        for i in 0..model.hidden_dim {
            let h = row[i];
            let sign = ((h >> 15) & 1) as u32;
            let exp = ((h >> 10) & 0x1f) as u32;
            let frac = (h & 0x3ff) as u32;
            embed[i] = if exp == 0 && frac == 0 { 0.0 } else if exp == 0 { 0.0 } else { f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)) };
        }
        // RMSNorm
        let norm_w = unsafe { std::slice::from_raw_parts(model.layers[0].attn_norm, model.hidden_dim) };
        let mut sumsq = 0.0f32;
        for &v in &embed { sumsq += v * v; }
        let inv_rms = 1.0 / (sumsq / embed.len() as f32 + model.rms_eps).sqrt();
        let normed: Vec<f32> = embed.iter().zip(norm_w).map(|(&x, &w)| x * w * inv_rms).collect();
        // Quantize
        let amax = normed.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let inv_scale = 127.0 / amax;
        let quant_i8: Vec<i8> = normed.iter().map(|&x| (x * inv_scale).round().clamp(-127.0, 127.0) as i8).collect();
        let act_sum: i32 = quant_i8.iter().map(|&x| x as i32).sum();
        // Manual ternary dot product for row 0
        let q_data = unsafe { std::slice::from_raw_parts(model.layers[0].wq, model.hidden_dim * model.hidden_dim / 4) };
        let in_dim = model.hidden_dim;
        let row_bytes = in_dim / 4;
        let mut ternary = vec![0i8; in_dim];
        for byte_idx in 0..row_bytes {
            let b = q_data[byte_idx];
            let decode = |v: u8| -> i8 { match v { 0 => -1, 1 => 0, 2 => 1, _ => 0 } };
            let g0 = (b >> 6) & 3;
            let g1 = (b >> 4) & 3;
            let g2 = (b >> 2) & 3;
            let g3 = b & 3;
            let block = byte_idx / 32;
            let within = byte_idx % 32;
            let base = block * 128;
            ternary[base + within] = decode(g0);
            ternary[base + 32 + within] = decode(g1);
            ternary[base + 64 + within] = decode(g2);
            ternary[base + 96 + within] = decode(g3);
        }
        let manual_dot: i32 = ternary.iter().zip(quant_i8.iter()).map(|(&w, &a)| w as i32 * a as i32).sum();
        let manual_result = manual_dot as f32 * (amax / 127.0) * model.layers[0].wq_scale;
        // Kernel result via offset correction
        let raw_dot: i32 = unsafe { crate::ffi::i2_dot_i8(q_data.as_ptr(), quant_i8.as_ptr(), in_dim as i32) };
        let kernel_result = (raw_dot - act_sum) as f32 * (amax / 127.0) * model.layers[0].wq_scale;
        eprintln!("Manual ternary dot row 0: {manual_dot} → {manual_result:.4}");
        eprintln!("Kernel raw dot row 0: {raw_dot} offset-corrected: {} → {kernel_result:.4}", raw_dot - act_sum);
        eprintln!("Match: {}", (manual_result - kernel_result).abs() < 0.01);
    }

    // Debug: print layer 0 weight scales
    let l0 = &model.layers[0];
    eprintln!("L0 scales: wq={:.6} wk={:.6} wv={:.6} wo={:.6} gate={:.6} up={:.6} down={:.6}",
        l0.wq_scale, l0.wk_scale, l0.wv_scale, l0.wo_scale,
        l0.w_gate_scale, l0.w_up_scale, l0.w_down_scale);

    // Encode prompt (prepend BOS)
    let mut tokens = vec![tokenizer.bos_id];
    tokens.extend(tokenizer.encode(prompt_text));
    eprintln!("Prompt: {} tokens", tokens.len());

    // Generate
    let output = InferenceState::generate(
        &model,
        &tokens,
        max_tokens,
        temperature,
        tokenizer.eos_id,
        max_seq_len,
    );

    // Decode generated tokens (skip prompt)
    let generated = &output[tokens.len()..];
    let text = tokenizer.decode(generated);
    println!("{text}");
}
