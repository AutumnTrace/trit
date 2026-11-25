# trit

A 1.58-bit language model, end to end: a small transformer with every weight in {-1, 0, +1}, trained on TinyStories, run by packed ternary kernels in Rust, and compiled to WebAssembly to write stories in the browser.

Work in progress. `train/` holds the PyTorch side; it starts with the tokenizer and the data pipeline.
