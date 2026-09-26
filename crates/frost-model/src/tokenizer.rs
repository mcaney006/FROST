//! Minimal BERT-uncased WordPiece tokenizer over the model's real vocab.txt.
//! Matches nomic's bert-base-uncased config (do_lower_case=true). Accent
//! stripping is omitted (ASCII-clean inputs unaffected) — noted as an
//! approximation vs the HF `tokenizers` normalizer.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    cls: u32,
    sep: u32,
    unk: u32,
    max_word_chars: usize,
}

impl Tokenizer {
    pub fn load(vocab_path: &Path) -> Result<Tokenizer> {
        let text = fs::read_to_string(vocab_path).with_context(|| format!("read {}", vocab_path.display()))?;
        let mut vocab = HashMap::new();
        for (i, line) in text.lines().enumerate() {
            vocab.insert(line.to_string(), i as u32);
        }
        let get = |t: &str| *vocab.get(t).unwrap_or_else(|| panic!("vocab missing {t}"));
        let (cls, sep, unk) = (get("[CLS]"), get("[SEP]"), get("[UNK]"));
        Ok(Tokenizer { vocab, cls, sep, unk, max_word_chars: 100 })
    }

    /// Encode text into ids with [CLS] .. [SEP], truncated to `max_tokens`
    /// (including the two special tokens).
    pub fn encode(&self, text: &str, max_tokens: usize) -> Vec<u32> {
        let mut ids = vec![self.cls];
        let body_cap = max_tokens.saturating_sub(2);
        'outer: for basic in basic_tokenize(text) {
            for piece in self.wordpiece(&basic) {
                if ids.len() - 1 >= body_cap { break 'outer; }
                ids.push(piece);
            }
        }
        ids.push(self.sep);
        ids
    }

    fn wordpiece(&self, token: &str) -> Vec<u32> {
        let chars: Vec<char> = token.chars().collect();
        if chars.len() > self.max_word_chars {
            return vec![self.unk];
        }
        let mut out = Vec::new();
        let mut start = 0;
        while start < chars.len() {
            let mut end = chars.len();
            let mut cur: Option<u32> = None;
            while start < end {
                let sub: String = chars[start..end].iter().collect();
                let cand = if start == 0 { sub } else { format!("##{sub}") };
                if let Some(&id) = self.vocab.get(&cand) {
                    cur = Some(id);
                    break;
                }
                end -= 1;
            }
            match cur {
                Some(id) => { out.push(id); start = end; }
                None => return vec![self.unk], // any unmatchable piece -> whole token UNK
            }
        }
        out
    }
}

/// Lowercase, split on whitespace, then split punctuation into its own tokens.
fn basic_tokenize(text: &str) -> Vec<String> {
    let lowered = text.to_lowercase();
    let mut out = Vec::new();
    for ws_tok in lowered.split_whitespace() {
        let mut cur = String::new();
        for ch in ws_tok.chars() {
            if is_punct(ch) {
                if !cur.is_empty() { out.push(std::mem::take(&mut cur)); }
                out.push(ch.to_string());
            } else {
                cur.push(ch);
            }
        }
        if !cur.is_empty() { out.push(cur); }
    }
    out
}

fn is_punct(c: char) -> bool {
    (c.is_ascii_punctuation()) || (!c.is_alphanumeric() && !c.is_whitespace() && !c.is_control())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn vocab_path() -> PathBuf {
        PathBuf::from(std::env::var("HOME").unwrap())
            .join("Library/Application Support/FROST/models/nomic-embed-text-v1.5/vocab.txt")
    }

    #[test]
    fn wordpiece_known_tokenizations() {
        let p = vocab_path();
        if !p.exists() { eprintln!("SKIP: vocab not present"); return; }
        let t = Tokenizer::load(&p).unwrap();
        // "unaffable" -> un ##aff ##able is the classic BERT example
        let ids = t.encode("unaffable", 32);
        // [CLS] ... [SEP]; body should be 3 pieces
        assert_eq!(ids.len(), 5, "un ##aff ##able -> 3 pieces + CLS/SEP: {ids:?}");
        assert_eq!(ids[0], t.cls);
        assert_eq!(*ids.last().unwrap(), t.sep);
        // punctuation splits
        let ids2 = t.encode("dogs, cats!", 32);
        assert!(ids2.len() >= 6, "expect dogs , cats ! : {ids2:?}");
    }

    #[test]
    fn truncation_respects_cap() {
        let p = vocab_path();
        if !p.exists() { return; }
        let t = Tokenizer::load(&p).unwrap();
        let long = "word ".repeat(100);
        let ids = t.encode(&long, 16);
        assert_eq!(ids.len(), 16, "must truncate to cap incl specials");
        assert_eq!(*ids.last().unwrap(), t.sep);
    }
}
