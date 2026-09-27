//! Pinned-asset tokenizer for the generator.
//!
//! Wraps the HF `tokenizers` crate over the checkpoint's `tokenizer.json` (byte-level BPE).
//! Special/control tokens are NEVER parsed out of text: content is encoded with
//! special-token matching disabled and control ids are inserted by the template code.
//! Their ids are verified against the pinned vocabulary at load, so a swapped
//! tokenizer fails loudly instead of silently emitting wrong control tokens.

use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("tokenizer.json: {0}")]
    Load(String),
    #[error("control token {name} has id {found:?}, expected {expected}")]
    ControlId { name: &'static str, expected: u32, found: Option<u32> },
    #[error("encode: {0}")]
    Encode(String),
}

/// Control token ids of the Mistral v13 (Tekken) vocabulary, verified at load.
pub mod ctl {
    pub const UNK: u32 = 0;
    pub const BOS: u32 = 1;
    pub const EOS: u32 = 2;
    pub const INST: u32 = 3;
    pub const INST_END: u32 = 4;
    pub const AVAILABLE_TOOLS: u32 = 5;
    pub const AVAILABLE_TOOLS_END: u32 = 6;
    pub const TOOL_RESULTS: u32 = 7;
    pub const TOOL_RESULTS_END: u32 = 8;
    pub const TOOL_CALLS: u32 = 9;
    pub const PAD: u32 = 11;
    pub const SYSTEM_PROMPT: u32 = 17;
    pub const SYSTEM_PROMPT_END: u32 = 18;
    pub const ARGS: u32 = 32;
    pub const CALL_ID: u32 = 33;
    pub const THINK: u32 = 34;
    pub const THINK_END: u32 = 35;
    pub const NAMED: &[(&str, u32)] = &[
        ("<unk>", UNK), ("<s>", BOS), ("</s>", EOS), ("[INST]", INST), ("[/INST]", INST_END),
        ("[AVAILABLE_TOOLS]", AVAILABLE_TOOLS), ("[/AVAILABLE_TOOLS]", AVAILABLE_TOOLS_END),
        ("[TOOL_RESULTS]", TOOL_RESULTS), ("[/TOOL_RESULTS]", TOOL_RESULTS_END), ("[TOOL_CALLS]", TOOL_CALLS),
        ("<pad>", PAD), ("[SYSTEM_PROMPT]", SYSTEM_PROMPT), ("[/SYSTEM_PROMPT]", SYSTEM_PROMPT_END),
        ("[ARGS]", ARGS), ("[CALL_ID]", CALL_ID), ("[THINK]", THINK), ("[/THINK]", THINK_END),
    ];
    /// Ids below this are reserved control tokens in this vocabulary.
    pub const FIRST_TEXT_ID: u32 = 1000;
}

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    char_to_byte: HashMap<char, u8>,
    pub vocab_size: usize,
}

impl Tokenizer {
    pub fn load(path: &Path) -> Result<Tokenizer, TokenizerError> {
        let mut inner = tokenizers::Tokenizer::from_file(path).map_err(|e| TokenizerError::Load(e.to_string()))?;
        // Text that looks like "[INST]" stays text. Control ids come only from the template.
        inner.set_encode_special_tokens(false);
        for (name, expected) in ctl::NAMED {
            let found = inner.token_to_id(name);
            if found != Some(*expected) { return Err(TokenizerError::ControlId { name, expected: *expected, found }); }
        }
        let vocab_size = inner.get_vocab_size(true);
        Ok(Tokenizer { inner, char_to_byte: byte_level_char_map(), vocab_size })
    }

    /// Encode plain text (no BOS, no control tokens).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        if text.is_empty() { return Ok(Vec::new()); }
        let enc = self.inner.encode(text, false).map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode a complete id sequence to text (control tokens skipped).
    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids, true).unwrap_or_default()
    }

    /// Ids 0..1000 are the vocabulary's reserved control tokens (verified against the pinned file).
    pub fn is_control(&self, id: u32) -> bool { id < ctl::FIRST_TEXT_ID }

    /// Raw bytes of one non-control token (byte-level BPE inverse mapping).
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        match self.inner.id_to_token(id) {
            Some(tok) => tok.chars().map(|c| self.char_to_byte.get(&c).copied()).map(|b| b.unwrap_or(b'?')).collect(),
            None => Vec::new(),
        }
    }

    pub fn detokenizer(&self) -> Detokenizer { Detokenizer { buf: Vec::new() } }
}

/// Incremental UTF-8 detokenizer: emits only complete characters, holds partial sequences.
pub struct Detokenizer { buf: Vec<u8> }

impl Detokenizer {
    /// Append one token's bytes and return the newly completed text (possibly empty).
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.buf.extend_from_slice(bytes);
        let take = match std::str::from_utf8(&self.buf) {
            Ok(_) => self.buf.len(),
            Err(e) => match e.error_len() {
                None => e.valid_up_to(),                    // incomplete tail: keep waiting
                Some(bad) => e.valid_up_to() + bad,         // invalid bytes: flush lossy through them
            },
        };
        let out = String::from_utf8_lossy(&self.buf[..take]).into_owned();
        self.buf.drain(..take);
        out
    }
    /// Flush whatever remains (lossy) at end of generation.
    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.buf).into_owned();
        self.buf.clear();
        out
    }
}

/// GPT-2 byte-level BPE alphabet: printable bytes map to themselves, the rest to U+0100+.
fn byte_level_char_map() -> HashMap<char, u8> {
    let mut bs: Vec<u16> = (b'!' as u16..=b'~' as u16).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n = 0u32;
    for b in 0u16..256 {
        if !bs.contains(&b) { bs.push(b); cs.push(256 + n); n += 1; }
    }
    bs.iter().zip(cs).map(|(&b, c)| (char::from_u32(c).expect("valid"), b as u8)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_level_map_is_a_bijection_over_all_bytes() {
        let m = byte_level_char_map();
        assert_eq!(m.len(), 256);
        let mut seen = [false; 256];
        for (_, &b) in &m { assert!(!seen[b as usize]); seen[b as usize] = true; }
        assert_eq!(m[&'a'], b'a');
        assert_eq!(m[&'\u{0120}'], b' '); // 'Ġ' is the space byte
    }

    #[test]
    fn detokenizer_holds_partial_utf8_and_flushes_invalid() {
        let mut d = Detokenizer { buf: Vec::new() };
        let euro = "€".as_bytes(); // E2 82 AC
        assert_eq!(d.push(&euro[..1]), "");
        assert_eq!(d.push(&euro[1..2]), "");
        assert_eq!(d.push(&euro[2..]), "€");
        assert_eq!(d.push(b"ok"), "ok");
        assert_eq!(d.push(&[0xFF]), "\u{FFFD}");
        assert_eq!(d.push(&euro[..2]), "");
        assert_eq!(d.finish(), "\u{FFFD}");
    }
}
