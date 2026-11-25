//! Byte-level BPE, compatible with the Hugging Face `tokenizers` model the
//! training code builds (train/trit_train/tokenizer.py).
//!
//! The pre-tokenizer regex there,
//! `'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}| ?[^\s\p{L}\p{N}]+|\s+`,
//! is reimplemented below as a hand-written scanner so no regex engine is
//! needed. Chunks are then merged with BPE by merge rank; merges never cross
//! chunk boundaries.

use std::collections::HashMap;

use crate::format::File;

pub struct Tokenizer {
    tokens: Vec<Vec<u8>>,
    merges: HashMap<(u32, u32), (u32, u32)>,
    byte_ids: [u32; 256],
    eos: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Letter,
    Number,
    Space,
    Other,
}

fn class(c: char) -> Class {
    if c.is_alphabetic() {
        Class::Letter
    } else if c.is_numeric() {
        Class::Number
    } else if c.is_whitespace() {
        Class::Space
    } else {
        Class::Other
    }
}

const CONTRACTIONS: [&str; 7] = ["s", "t", "re", "ve", "m", "ll", "d"];

/// Split `text` into pre-tokenizer chunks (byte ranges).
pub fn pretokenize(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < text.len() {
        let rest = &text[i..];
        let mut chars = rest.chars();
        let c = chars.next().unwrap();
        let run = |s: &str, cls: Class| -> usize {
            s.char_indices()
                .find(|&(_, ch)| class(ch) != cls)
                .map_or(s.len(), |(j, _)| j)
        };
        let len = if c == '\''
            && let Some(k) = CONTRACTIONS.iter().find(|k| rest[1..].starts_with(*k))
        {
            1 + k.len()
        } else {
            // ` ?X`: an optional single leading space joins the next chunk.
            let (lead, body) = match (c, chars.next()) {
                (' ', Some(n)) if class(n) != Class::Space => (1, n),
                _ => (0, c),
            };
            let s = &rest[lead..];
            match class(body) {
                Class::Letter => lead + run(s, Class::Letter),
                Class::Number => lead + body.len_utf8(),
                Class::Other => lead + run(s, Class::Other),
                Class::Space => run(rest, Class::Space),
            }
        };
        out.push(&text[i..i + len]);
        i += len;
    }
    out
}

impl Tokenizer {
    pub fn from_file(f: &File<'_>) -> Self {
        let tokens: Vec<Vec<u8>> = f.tokens.iter().map(|t| t.to_vec()).collect();
        let mut byte_ids = [u32::MAX; 256];
        for (id, t) in tokens.iter().enumerate() {
            if t.len() == 1 && id as u32 != f.config.eos_id {
                byte_ids[t[0] as usize] = id as u32;
            }
        }
        let merges = f
            .merges
            .iter()
            .enumerate()
            .map(|(rank, &(a, b, m))| ((a as u32, b as u32), (rank as u32, m as u32)))
            .collect();
        Self {
            tokens,
            merges,
            byte_ids,
            eos: f.config.eos_id,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn eos(&self) -> u32 {
        self.eos
    }

    /// Raw bytes of a token (may be part of a UTF-8 sequence). EOS decodes to nothing.
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        if id == self.eos {
            return &[];
        }
        self.tokens.get(id as usize).map_or(&[], Vec::as_slice)
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&i| self.token_bytes(i).iter().copied())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut word = Vec::new();
        for chunk in pretokenize(text) {
            word.clear();
            word.extend(chunk.bytes().map(|b| self.byte_ids[b as usize]));
            self.merge(&mut word);
            out.extend_from_slice(&word);
        }
        out
    }

    /// Repeatedly apply the lowest-ranked merge present, all occurrences left to right.
    fn merge(&self, word: &mut Vec<u32>) {
        loop {
            let best = word
                .windows(2)
                .filter_map(|p| {
                    self.merges
                        .get(&(p[0], p[1]))
                        .map(|&(rank, m)| (rank, p[0], p[1], m))
                })
                .min();
            let Some((_, a, b, m)) = best else { return };
            let mut w = 0;
            let mut r = 0;
            while r < word.len() {
                if r + 1 < word.len() && word[r] == a && word[r + 1] == b {
                    word[w] = m;
                    r += 2;
                } else {
                    word[w] = word[r];
                    r += 1;
                }
                w += 1;
            }
            word.truncate(w);
        }
    }
}

/// Turns a stream of token bytes into complete UTF-8 text, holding back an
/// incomplete trailing character until its remaining bytes arrive.
#[derive(Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let s = s.to_owned();
                self.pending.clear();
                s
            }
            Err(e) if e.error_len().is_none() => {
                // Incomplete sequence at the end: emit what is valid, keep the rest.
                let n = e.valid_up_to();
                let s = String::from_utf8_lossy(&self.pending[..n]).into_owned();
                self.pending.drain(..n);
                s
            }
            Err(_) => {
                let s = String::from_utf8_lossy(&self.pending).into_owned();
                self.pending.clear();
                s
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretokenizer_matches_the_regex_on_hand_cases() {
        let cases: &[(&str, &[&str])] = &[
            ("Once upon a time", &["Once", " upon", " a", " time"]),
            ("didn't they'll", &["didn", "'t", " they", "'ll"]),
            ("3 big 12", &["3", " big", " 1", "2"]),
            ("hi!! ok", &["hi", "!!", " ok"]),
            ("a  b", &["a", "  ", "b"]),
            ("end.\n\nNew", &["end", ".", "\n\n", "New"]),
            ("x \n y", &["x", " \n ", "y"]),
            ("'x", &["'", "x"]),
            (" 'tis", &[" '", "tis"]),
        ];
        for (text, want) in cases {
            assert_eq!(pretokenize(text), *want, "{text:?}");
        }
    }
}
