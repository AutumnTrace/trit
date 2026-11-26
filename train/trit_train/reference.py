"""Reference forward pass over an exported `.trit` file, in float32 PyTorch.

This mirrors crates/trit-model/src/model.rs step for step. It is used to
measure the validation loss of the file as shipped, and to write the fixtures
the Rust tests compare against (logits and tokenizer output).
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from . import fileformat as ff
from .data import TokenStream
from .model import Config, activation_quant, apply_rope, rope_tables


class Reference:
    def __init__(self, f: ff.TritFile):
        self.f = f
        t = {k: torch.from_numpy(np.array(v.data)) for k, v in f.tensors.items()}
        self.scale = {k: v.scale for k, v in f.tensors.items()}
        self.embed = t["embed.weight"].float() * t["embed.scale"].float()[:, None]
        self.t = {k: v.float() for k, v in t.items()}
        cfg = Config(
            vocab_size=f.vocab_size,
            dim=f.dim,
            n_layers=f.n_layers,
            n_heads=f.n_heads,
            ffn_dim=f.ffn_dim,
            max_seq_len=f.max_seq_len,
            rope_theta=f.rope_theta,
            norm_eps=f.norm_eps,
        )
        self.cfg = cfg
        self.cos, self.sin = rope_tables(cfg)
        # Share of feed-forward hidden activations that are exactly zero after ReLU².
        self.ffn_zeros = 0
        self.ffn_total = 0

    def rmsnorm(self, x: torch.Tensor, name: str, n: int | None = None) -> torch.Tensor:
        """RMSNorm over the last axis. `n` is the count the mean divides by,
        when pruned units (always zero) are missing from `x`."""
        ms = x.pow(2).mean(-1, keepdim=True) if n is None else x.pow(2).sum(-1, keepdim=True) / n
        x = x * torch.rsqrt(ms + self.cfg.norm_eps)
        return x * self.t[name]

    def bitlinear(self, x: torch.Tensor, name: str) -> torch.Tensor:
        xq, s = activation_quant(x)
        return F.linear(xq, self.t[name]) * (self.scale[name] / s)

    @torch.no_grad()
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        b, n = idx.shape
        hd = cfg.head_dim
        x = self.embed[idx]
        for i in range(cfg.n_layers):
            p = f"layers.{i}."
            h = self.rmsnorm(x, p + "attn_norm")

            def heads(z: torch.Tensor) -> torch.Tensor:
                return z.view(b, n, cfg.n_heads, hd).transpose(1, 2)

            q = apply_rope(heads(self.bitlinear(h, p + "wq")), self.cos, self.sin)
            k = apply_rope(heads(self.bitlinear(h, p + "wk")), self.cos, self.sin)
            v = heads(self.bitlinear(h, p + "wv"))
            att = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
            att = att.masked_fill(torch.ones(n, n, dtype=torch.bool).triu(1), float("-inf"))
            a = (att.softmax(-1) @ v).transpose(1, 2).reshape(b, n, cfg.dim)
            x = x + self.bitlinear(self.rmsnorm(a, p + "attn_sub_norm"), p + "wo")
            u = F.relu(self.bitlinear(self.rmsnorm(x, p + "ffn_norm"), p + "w_up")).square()
            # Count over the trained width: pruned units are zeros that aren't stored.
            pruned = (cfg.ffn_dim - u.shape[-1]) * b * n
            self.ffn_zeros += (u == 0).sum().item() + pruned
            self.ffn_total += u.numel() + pruned
            x = x + self.bitlinear(self.rmsnorm(u, p + "ffn_sub_norm", cfg.ffn_dim), p + "w_down")
        x = self.rmsnorm(x, "norm")
        return x @ self.embed.T

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> float:
        logits = self.forward(x)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1)).item()


def write_logit_fixture(ref: Reference, tokens: list[int], path: Path) -> None:
    logits = ref.forward(torch.tensor([tokens]))[0].numpy().astype("<f4")
    with path.open("wb") as fh:
        fh.write(struct.pack("<2I", len(tokens), logits.shape[1]))
        fh.write(np.asarray(tokens, dtype="<u4").tobytes())
        fh.write(logits.tobytes())


TOKENIZER_CASES = [
    "Once upon a time, there was a little girl named Lily.",
    "She didn't want to go. \"We're here!\" they'll say, I'm sure you've seen it'd rain.",
    "Tom had 3 apples and 12 pears.\nOne day...\n\nThe end!",
    "  two spaces,\ttab, trailing space ",
    "Café naïve — “quoted” ’tis \U0001f600 日本",
    "'s 'S 'll 'x ''",
    "",
]


def write_tokenizer_fixture(tok: Tokenizer, path: Path) -> None:
    with path.open("wb") as fh:
        fh.write(struct.pack("<I", len(TOKENIZER_CASES)))
        for text in TOKENIZER_CASES:
            raw = text.encode()
            ids = tok.encode(text, add_special_tokens=False).ids
            fh.write(struct.pack("<I", len(raw)) + raw)
            fh.write(struct.pack("<I", len(ids)) + np.asarray(ids, dtype="<u4").tobytes())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, default=Path("out/tinystories-15m.trit"))
    p.add_argument("--tokenizer", type=Path, default=Path("out/tokenizer.json"))
    p.add_argument("--val", type=Path, default=Path("out/valid.bin"))
    p.add_argument("--val-batches", type=int, default=40)
    p.add_argument("--fixtures", type=Path, help="write Rust test fixtures into this directory")
    args = p.parse_args()

    ref = Reference(ff.read(args.model))
    tok = Tokenizer.from_file(str(args.tokenizer))
    result: dict = {"model": str(args.model)}
    if args.val_batches:
        batches = TokenStream(args.val, 512, "cpu").fixed_batches(32, args.val_batches)
        losses = [ref.loss(x, y) for x, y in batches]
        result["val_loss"] = round(sum(losses) / len(losses), 4)
        result["val_tokens"] = sum(x.numel() for x, _ in batches)
        result["ffn_zero_fraction"] = round(ref.ffn_zeros / ref.ffn_total, 4)
    if args.fixtures:
        args.fixtures.mkdir(parents=True, exist_ok=True)
        prompt = [ref.f.eos_id] + tok.encode("Once upon a time, there was a dragon").ids
        write_logit_fixture(ref, prompt, args.fixtures / "logits.bin")
        write_tokenizer_fixture(tok, args.fixtures / "tokenizer.bin")
        result["fixtures"] = str(args.fixtures)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
