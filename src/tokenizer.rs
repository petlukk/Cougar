//! Byte-level BPE tokenizer parsed from GGUF vocab metadata.
//!
//! Handles GPT-2 byte encoding where "bad" bytes (0-32, 127-160, 173) are
//! shifted to unicode 256+. Token strings in the GGUF use this encoding;
//! we reverse it at load time so the vocab stores raw bytes.

use std::collections::HashMap;
use crate::gguf::{GgufFile, MetaValue};

pub struct Tokenizer {
    vocab: Vec<Vec<u8>>,
    token_to_id: HashMap<Vec<u8>, u32>,
    scores: Vec<f32>,
    pub bos_id: u32,
    pub eos_id: u32,
}

/// Parse a byte-fallback token like "<0x41>" into the byte value 0x41.
fn parse_byte_token(s: &str) -> Option<u8> {
    if s.len() == 6 && s.starts_with("<0x") && s.ends_with('>') {
        u8::from_str_radix(&s[3..5], 16).ok()
    } else {
        None
    }
}

/// Reverse the GPT-2 byte-to-unicode mapping for a single character.
/// Returns the raw byte value, or None if the character isn't in the GPT-2 mapping.
fn gpt2_unicode_to_byte(cp: u32) -> Option<u8> {
    // "Good" bytes: identity mapping (printable ASCII + Latin-1 supplement ranges)
    if (33..=126).contains(&cp) || (161..=172).contains(&cp) || (174..=255).contains(&cp) {
        return Some(cp as u8);
    }
    // "Bad" bytes: remapped to 256+ in order: 0-32, 127-160, 173
    if cp >= 256 {
        let idx = (cp - 256) as u8;
        if idx <= 32 { return Some(idx); }           // 0-32
        if idx <= 32 + 34 { return Some(idx - 33 + 127); } // 127-160
        if idx == 32 + 34 + 1 { return Some(173); }  // 173
    }
    None
}

/// Decode a GPT-2 encoded token string to raw bytes.
/// Falls back to raw UTF-8 bytes for characters outside the GPT-2 mapping
/// (e.g. special tokens like <|begin_of_text|>).
fn gpt2_decode_str(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    let mut all_mapped = true;
    for c in s.chars() {
        if let Some(b) = gpt2_unicode_to_byte(c as u32) {
            out.push(b);
        } else {
            all_mapped = false;
            break;
        }
    }
    if all_mapped {
        out
    } else {
        // Not a GPT-2 encoded token (e.g. special token) — use raw UTF-8
        s.as_bytes().to_vec()
    }
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let tokens_arr = gguf.metadata.get("tokenizer.ggml.tokens")
            .ok_or("missing tokenizer.ggml.tokens")?;
        let scores_arr = gguf.metadata.get("tokenizer.ggml.scores")
            .ok_or("missing tokenizer.ggml.scores")?;

        let token_strs = match tokens_arr {
            MetaValue::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr {
                    match v {
                        MetaValue::Str(s) => out.push(s.as_str()),
                        _ => return Err("tokens array contains non-string".into()),
                    }
                }
                out
            }
            _ => return Err("tokenizer.ggml.tokens is not an array".into()),
        };

        let scores = match scores_arr {
            MetaValue::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr {
                    match v {
                        MetaValue::F32(f) => out.push(*f),
                        _ => return Err("scores array contains non-f32".into()),
                    }
                }
                out
            }
            _ => return Err("tokenizer.ggml.scores is not an array".into()),
        };

        if token_strs.len() != scores.len() {
            return Err(format!(
                "tokens ({}) and scores ({}) length mismatch",
                token_strs.len(), scores.len()
            ));
        }

        let mut vocab = Vec::with_capacity(token_strs.len());
        let mut token_to_id = HashMap::with_capacity(token_strs.len());

        for (i, tok_str) in token_strs.iter().enumerate() {
            let bytes = if let Some(b) = parse_byte_token(tok_str) {
                vec![b]
            } else {
                gpt2_decode_str(tok_str)
            };
            token_to_id.insert(bytes.clone(), i as u32);
            vocab.push(bytes);
        }

        let bos_id = gguf.get_u32("tokenizer.ggml.bos_token_id").unwrap_or(1);
        let eos_id = gguf.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(2);

        Ok(Tokenizer { vocab, token_to_id, scores, bos_id, eos_id })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        if bytes.is_empty() {
            return Vec::new();
        }

        // Initialize: each byte maps to its single-byte token
        let mut tokens: Vec<u32> = Vec::with_capacity(bytes.len());
        for &b in bytes {
            let id = self.token_to_id.get(&vec![b]).copied();
            match id {
                Some(id) => tokens.push(id),
                None => {
                    // Should not happen if vocab has all byte tokens, but be safe
                    tokens.push(0);
                }
            }
        }

        // BPE merge loop: O(n^2) per merge, fine for short prompts
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = usize::MAX;

            for i in 0..tokens.len().saturating_sub(1) {
                let mut merged = self.vocab[tokens[i] as usize].clone();
                merged.extend_from_slice(&self.vocab[tokens[i + 1] as usize]);
                if let Some(&merge_id) = self.token_to_id.get(&merged) {
                    let score = self.scores[merge_id as usize];
                    if score > best_score {
                        best_score = score;
                        best_idx = i;
                    }
                }
            }

            if best_idx == usize::MAX {
                break;
            }

            // Perform the merge at best_idx
            let mut merged = self.vocab[tokens[best_idx] as usize].clone();
            merged.extend_from_slice(&self.vocab[tokens[best_idx + 1] as usize]);
            let merge_id = self.token_to_id[&merged];
            tokens[best_idx] = merge_id;
            tokens.remove(best_idx + 1);
        }

        tokens
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if (id as usize) < self.vocab.len() {
                bytes.extend_from_slice(&self.vocab[id as usize]);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn byte_vocab() -> Tokenizer {
        let mut vocab = Vec::new();
        let mut scores = Vec::new();
        for b in 0u8..=255 {
            vocab.push(vec![b]);
            scores.push(0.0);
        }
        Tokenizer {
            token_to_id: vocab.iter().enumerate()
                .map(|(i, v)| (v.clone(), i as u32)).collect(),
            vocab,
            scores,
            bos_id: 0,
            eos_id: 0,
        }
    }

    #[test]
    fn test_decode_single_bytes() {
        let tok = byte_vocab();
        assert_eq!(tok.decode(&[72, 101, 108, 108, 111]), "Hello");
    }

    #[test]
    fn test_encode_single_bytes() {
        let tok = byte_vocab();
        let ids = tok.encode("Hi");
        assert_eq!(ids, vec![72, 105]);
    }

    #[test]
    fn test_encode_with_merge() {
        let mut vocab: Vec<Vec<u8>> = Vec::new();
        let mut scores: Vec<f32> = Vec::new();
        for b in 0u8..=255 {
            vocab.push(vec![b]);
            scores.push(0.0);
        }
        vocab.push(b"ab".to_vec());
        scores.push(100.0);

        let tok = Tokenizer {
            token_to_id: vocab.iter().enumerate()
                .map(|(i, v)| (v.clone(), i as u32)).collect(),
            vocab,
            scores,
            bos_id: 0,
            eos_id: 0,
        };
        let ids = tok.encode("abc");
        assert_eq!(ids, vec![256, 99]);
    }

    #[test]
    fn test_encode_chained_merges() {
        let mut vocab: Vec<Vec<u8>> = Vec::new();
        let mut scores: Vec<f32> = Vec::new();
        for b in 0u8..=255 {
            vocab.push(vec![b]);
            scores.push(0.0);
        }
        // "ab" at 256, score 100
        vocab.push(b"ab".to_vec());
        scores.push(100.0);
        // "abc" at 257, score 200
        vocab.push(b"abc".to_vec());
        scores.push(200.0);

        let tok = Tokenizer {
            token_to_id: vocab.iter().enumerate()
                .map(|(i, v)| (v.clone(), i as u32)).collect(),
            vocab,
            scores,
            bos_id: 0,
            eos_id: 0,
        };
        // Should first merge "ab" (score 100), then merge "ab"+"c" -> "abc" (score 200)
        let ids = tok.encode("abcd");
        assert_eq!(ids, vec![257, 100]); // "abc"=257, 'd'=100
    }

    #[test]
    fn test_encode_empty() {
        let tok = byte_vocab();
        assert_eq!(tok.encode(""), Vec::<u32>::new());
    }

    #[test]
    fn test_decode_empty() {
        let tok = byte_vocab();
        assert_eq!(tok.decode(&[]), "");
    }

    #[test]
    fn test_roundtrip() {
        let mut vocab: Vec<Vec<u8>> = Vec::new();
        let mut scores: Vec<f32> = Vec::new();
        for b in 0u8..=255 {
            vocab.push(vec![b]);
            scores.push(0.0);
        }
        vocab.push(b"he".to_vec());
        scores.push(50.0);
        vocab.push(b"ll".to_vec());
        scores.push(40.0);

        let tok = Tokenizer {
            token_to_id: vocab.iter().enumerate()
                .map(|(i, v)| (v.clone(), i as u32)).collect(),
            vocab,
            scores,
            bos_id: 0,
            eos_id: 0,
        };
        let text = "hello";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_parse_byte_token() {
        assert_eq!(parse_byte_token("<0x41>"), Some(0x41));
        assert_eq!(parse_byte_token("<0xFF>"), Some(0xFF));
        assert_eq!(parse_byte_token("<0x00>"), Some(0x00));
        assert_eq!(parse_byte_token("hello"), None);
        assert_eq!(parse_byte_token("<0xGG>"), None);
    }

    #[test]
    fn test_highest_score_wins() {
        // When two merges are possible, the one with higher score wins
        let mut vocab: Vec<Vec<u8>> = Vec::new();
        let mut scores: Vec<f32> = Vec::new();
        for b in 0u8..=255 {
            vocab.push(vec![b]);
            scores.push(0.0);
        }
        // "ab" at 256, score 50
        vocab.push(b"ab".to_vec());
        scores.push(50.0);
        // "bc" at 257, score 100 (higher)
        vocab.push(b"bc".to_vec());
        scores.push(100.0);

        let tok = Tokenizer {
            token_to_id: vocab.iter().enumerate()
                .map(|(i, v)| (v.clone(), i as u32)).collect(),
            vocab,
            scores,
            bos_id: 0,
            eos_id: 0,
        };
        // "abc": "bc" has higher score, so merge bc first -> a, bc
        let ids = tok.encode("abc");
        assert_eq!(ids, vec![97, 257]); // 'a'=97, "bc"=257
    }
}
