//! Load and run trit's ternary TinyStories model.
//!
//! ```no_run
//! use trit_model::{Generator, Model, Sampler};
//! use trit_kernels::Layout;
//!
//! let bytes = std::fs::read("models/tinystories-15m.trit").unwrap();
//! let model = Model::load(&bytes, Layout::I2).unwrap();
//! let mut g = Generator::new(&model, Sampler::new(0.8, 0.9, 1));
//! g.prompt("Once upon a time");
//! while let Some(text) = g.next_text() {
//!     print!("{text}");
//! }
//! ```

pub mod format;
pub mod math;
pub mod model;
pub mod sampler;
pub mod tokenizer;

pub use format::{Config, Error};
pub use model::{Model, State};
pub use sampler::{Rng, Sampler};
pub use tokenizer::{Tokenizer, Utf8Stream};

/// Prompt-then-sample loop over one sequence.
pub struct Generator<'m> {
    pub model: &'m Model,
    pub state: State,
    pub sampler: Sampler,
    next: Option<u32>,
    utf8: Utf8Stream,
    /// Tokens produced so far (not counting the prompt).
    pub generated: usize,
}

impl<'m> Generator<'m> {
    pub fn new(model: &'m Model, sampler: Sampler) -> Self {
        Self {
            model,
            state: model.new_state(),
            sampler,
            next: None,
            utf8: Utf8Stream::default(),
            generated: 0,
        }
    }

    /// Start a new story: EOS (the separator every training story follows), then the prompt.
    /// Returns the number of prompt tokens.
    pub fn prompt(&mut self, text: &str) -> usize {
        self.state.reset();
        self.generated = 0;
        self.utf8 = Utf8Stream::default();
        let mut ids = vec![self.model.config.eos_id];
        ids.extend(self.model.tokenizer.encode(text));
        let max = self.model.config.max_seq_len - 1;
        ids.truncate(max);
        for &id in &ids {
            self.model.forward(&mut self.state, id);
        }
        self.next = Some(self.sampler.sample(self.state.logits()));
        ids.len() - 1
    }

    /// The next token id, or None at end of story or a full context.
    pub fn next_token(&mut self) -> Option<u32> {
        let tok = self.next.take()?;
        if tok == self.model.config.eos_id {
            return None;
        }
        self.generated += 1;
        if self.state.pos < self.model.config.max_seq_len {
            self.model.forward(&mut self.state, tok);
            self.next = Some(self.sampler.sample(self.state.logits()));
        }
        Some(tok)
    }

    /// Like [`Self::next_token`], decoded; may be empty while a multi-byte character is incomplete.
    pub fn next_text(&mut self) -> Option<String> {
        let tok = self.next_token()?;
        Some(self.utf8.push(self.model.tokenizer.token_bytes(tok)))
    }
}
