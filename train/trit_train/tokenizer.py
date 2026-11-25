"""Byte-level BPE tokenizer for TinyStories.

The pre-tokenizer is a single regex that the Rust side reimplements by hand
(crates/trit-model/src/tokenizer.rs), so it deliberately avoids look-around:

    '(?:s|t|re|ve|m|ll|d) | ?\\p{L}+ | ?\\p{N} | ?[^\\s\\p{L}\\p{N}]+ | \\s+

Contractions first, then words with an optional leading space, single digits,
runs of punctuation, and runs of whitespace. Each chunk is mapped to bytes and
merged with BPE; merges never cross chunk boundaries.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

EOS = "<|endoftext|>"
PRETOKENIZE_PATTERN = r"'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}| ?[^\s\p{L}\p{N}]+|\s+"


def build_empty() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(PRETOKENIZE_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    return tok


def iter_stories(path: Path, max_bytes: int | None = None) -> Iterator[str]:
    """Yield stories from a TinyStoriesV2 text file (separated by <|endoftext|>)."""
    read = 0
    buf: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            read += len(line)
            if line.strip() == EOS:
                story = "".join(buf).strip()
                buf.clear()
                if story:
                    yield story
                if max_bytes is not None and read >= max_bytes:
                    return
            else:
                buf.append(line)
    story = "".join(buf).strip()
    if story:
        yield story


def train(text_path: Path, vocab_size: int, sample_bytes: int) -> Tokenizer:
    tok = build_empty()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=True,
    )
    tok.train_from_iterator(iter_stories(text_path, sample_bytes), trainer=trainer)
    return tok


_BYTE_DECODER: dict[str, int] | None = None


def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode table used by the ByteLevel pre-tokenizer."""
    global _BYTE_DECODER
    if _BYTE_DECODER is None:
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        _BYTE_DECODER = {chr(c): b for b, c in zip(bs, cs, strict=True)}
    return _BYTE_DECODER


def token_bytes(tok: Tokenizer) -> list[bytes]:
    """Raw bytes of every token id, in id order. Special tokens keep their literal text."""
    dec = _byte_decoder()
    vocab = tok.get_vocab()
    out: list[bytes] = [b""] * len(vocab)
    for piece, idx in vocab.items():
        if piece == EOS:
            out[idx] = piece.encode()
        else:
            out[idx] = bytes(dec[ch] for ch in piece)
    return out


def merges(tok: Tokenizer) -> list[tuple[int, int, int]]:
    """BPE merges in rank order as (left id, right id, merged id)."""
    import json

    model = json.loads(tok.to_str())["model"]
    vocab = model["vocab"]
    out = []
    for m in model["merges"]:
        a, b = m if isinstance(m, list) else m.split(" ")
        out.append((vocab[a], vocab[b], vocab[a + b]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--text", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--sample-mb", type=int, default=400)
    p.add_argument("--out", type=Path, default=Path("out/tokenizer.json"))
    args = p.parse_args()
    tok = train(args.text, args.vocab_size, args.sample_mb * 1_000_000)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(args.out))
    print(f"saved {args.out} ({tok.get_vocab_size()} tokens)")


if __name__ == "__main__":
    main()
