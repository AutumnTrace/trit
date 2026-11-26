# models

`tinystories-15m.trit` is the model the CLI and the browser demo use:
8 layers, width 384, 13,851,648 ternary weights and a 4,096 x 384 int8
embedding, in the format described in [docs/format.md](../docs/format.md)
(version 2: the 396 feed-forward units training switched off were dropped
at export).

It was produced by the commands in the README's "Train it yourself"
section; the training logs and summaries are in
[train/results](../train/results).

**Licence.** The weights were trained on
[TinyStoriesV2](https://huggingface.co/datasets/roneneldan/TinyStories)
(Eldan and Li, 2023), which is published under the Community Data License
Agreement, Sharing, version 1.0. The weights are distributed under the same
terms. The code that made and runs them is MIT.
