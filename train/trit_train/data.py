"""Tokenize TinyStoriesV2 into flat uint16 token files.

Layout: EOS story EOS story EOS ... so every story is preceded by EOS, which is
also how generation starts.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from .tokenizer import EOS, iter_stories


def tokenize_file(tok: Tokenizer, src: Path, dst: Path, chunk: int = 20_000) -> int:
    eos = tok.token_to_id(EOS)
    total = 0
    t0 = time.time()
    with dst.open("wb") as f:
        np.array([eos], dtype=np.uint16).tofile(f)
        total += 1
        batch: list[str] = []

        def flush() -> int:
            encs = tok.encode_batch(batch, add_special_tokens=False)
            ids = []
            for e in encs:
                ids.extend(e.ids)
                ids.append(eos)
            np.asarray(ids, dtype=np.uint16).tofile(f)
            batch.clear()
            return len(ids)

        for story in iter_stories(src):
            batch.append(story)
            if len(batch) >= chunk:
                total += flush()
                print(f"\r{dst.name}: {total / 1e6:.1f}M tokens, {time.time() - t0:.0f}s", end="")
        if batch:
            total += flush()
    print(f"\r{dst.name}: {total / 1e6:.2f}M tokens in {time.time() - t0:.0f}s")
    return total


class TokenStream:
    """Random fixed-length windows over a flat token file."""

    def __init__(self, path: Path, seq_len: int, device: str):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.device = device

    def __len__(self) -> int:
        return len(self.data)

    def batch(self, batch_size: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        hi = len(self.data) - self.seq_len - 1
        starts = torch.randint(0, hi, (batch_size,), generator=gen).tolist()
        rows = np.stack([self.data[s : s + self.seq_len + 1] for s in starts]).astype(np.int64)
        xy = torch.from_numpy(rows)
        if self.device.startswith("cuda"):
            xy = xy.pin_memory().to(self.device, non_blocking=True)
        return xy[:, :-1], xy[:, 1:]

    def fixed_batches(
        self, batch_size: int, n_batches: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Deterministic, non-overlapping windows from the start of the file (for validation)."""
        out = []
        step = self.seq_len + 1
        for b in range(n_batches):
            rows = []
            for i in range(batch_size):
                s = (b * batch_size + i) * step
                if s + step > len(self.data):
                    return out
                rows.append(self.data[s : s + step])
            xy = torch.from_numpy(np.stack(rows).astype(np.int64)).to(self.device)
            out.append((xy[:, :-1], xy[:, 1:]))
        return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tokenizer", type=Path, default=Path("out/tokenizer.json"))
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("out"))
    args = p.parse_args()
    tok = Tokenizer.from_file(str(args.tokenizer))
    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("valid", "train"):
        src = args.data / f"TinyStoriesV2-GPT4-{split}.txt"
        tokenize_file(tok, src, args.out / f"{split}.bin")


if __name__ == "__main__":
    main()
