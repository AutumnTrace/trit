# The `.trit` file

One file holds everything the engine needs: the shape of the network, the
tokenizer and the weights. It is written by
[`train/trit_train/fileformat.py`](../train/trit_train/fileformat.py) and read
by [`crates/trit-model/src/format.rs`](../crates/trit-model/src/format.rs).
All integers are little-endian.

```
offset  size
0       64        header
64      ...       vocabulary: for each token id, u8 length + bytes
...     6 * M     merges: M x (u16 left, u16 right, u16 merged)
...     ...       tensor table
...     ...       tensor data, each tensor 64-byte aligned
```

## Header

| offset | type | field |
|---|---|---|
| 0 | `[u8; 4]` | magic `TRIT` |
| 4 | u32 | version, 1 |
| 8 | u32 | vocabulary size |
| 12 | u32 | model width `dim` |
| 16 | u32 | layers |
| 20 | u32 | attention heads |
| 24 | u32 | feed-forward width |
| 28 | u32 | context length |
| 32 | u32 | reserved (0) |
| 36 | u32 | reserved (0) |
| 40 | f32 | RoPE θ |
| 44 | f32 | RMSNorm ε |
| 48 | u32 | id of `<\|endoftext\|>` |
| 52 | u32 | number of merges M |
| 56 | u32 | number of tensors |
| 60 | 4 bytes | zero padding |

## Tokenizer

The vocabulary lists each token's raw bytes in id order. Byte-level BPE means
the first merges build on 256 single-byte tokens, so any input can be encoded;
`<|endoftext|>` is stored as its literal text and never produced by encoding.
Merges are in rank order: lower ranks were learned first and apply first.

## Tensor table

Each entry:

| type | field |
|---|---|
| u8 + bytes | name |
| u8 | dtype: 0 = f32, 1 = int8, 2 = ternary |
| u8 | number of dimensions (1 or 2) |
| u32 per dimension | shape, rows first |
| f32 | scale: γ for ternary tensors, otherwise 1 |
| u64 | absolute offset of the data |
| u64 | data length in bytes |

## Ternary data

Five weights per byte, packed in base 3. For a row of trits
`t[0], t[1], ...`, byte `i` holds columns `5i .. 5i + 4` as

```
v = (t[5i] + 1) + 3 (t[5i+1] + 1) + 9 (t[5i+2] + 1) + 27 (t[5i+3] + 1) + 81 (t[5i+4] + 1)
```

so `v` is in `0..243`. Each row is padded with zero weights to a multiple of
five columns and rows follow each other with no further padding. That is
log2(3) ≈ 1.58 bits of information stored in 1.6 bits.

This is a storage format only. On load, the engine unpacks each matrix and
repacks it into one of the kernel layouts described in
[how-it-works.md](how-it-works.md#memory-layouts).

## Tensors in a v1 file

| name | dtype | shape |
|---|---|---|
| `embed.weight` | int8 | vocab x dim |
| `embed.scale` | f32 | vocab (one dequantization factor per row) |
| `layers.{i}.attn_norm` | f32 | dim |
| `layers.{i}.wq`, `wk`, `wv`, `wo` | ternary | dim x dim |
| `layers.{i}.attn_sub_norm` | f32 | dim |
| `layers.{i}.ffn_norm` | f32 | dim |
| `layers.{i}.w_up` | ternary | ffn x dim |
| `layers.{i}.ffn_sub_norm` | f32 | ffn |
| `layers.{i}.w_down` | ternary | dim x ffn |
| `norm` | f32 | dim |

The embedding doubles as the output layer (the weights are tied).
