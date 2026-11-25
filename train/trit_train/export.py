"""Export a trained checkpoint to a `.trit` file.

BitLinear weights become trits plus their absmean scale, exactly as the
training forward pass saw them. The tied embedding is quantized after
training to int8 with one absmax scale per row; `reference.py` measures
what that costs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from . import fileformat as ff
from .model import Config, Transformer, weight_quant
from .tokenizer import EOS, merges, token_bytes

LINEARS = ("wq", "wk", "wv", "wo", "w_up", "w_down")
NORMS = ("attn_norm", "attn_sub_norm", "ffn_norm", "ffn_sub_norm")


def quantize_embedding(w: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    q = (w / scale[:, None]).round().clamp(-127, 127)
    return q.to(torch.int8).numpy(), scale.numpy().astype(np.float32)


def build(model: Transformer, tok: Tokenizer) -> ff.TritFile:
    cfg = model.cfg
    if not cfg.ternary:
        raise ValueError("only ternary checkpoints can be exported")
    f = ff.TritFile(
        vocab_size=cfg.vocab_size,
        dim=cfg.dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        ffn_dim=cfg.ffn_dim,
        max_seq_len=cfg.max_seq_len,
        rope_theta=cfg.rope_theta,
        norm_eps=cfg.norm_eps,
        eos_id=tok.token_to_id(EOS),
        tokens=token_bytes(tok),
        merges=merges(tok),
    )
    sd = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}

    def add(name: str, dtype: int, data: np.ndarray, scale: float = 1.0) -> None:
        f.tensors[name] = ff.Tensor(name, dtype, tuple(data.shape), data, scale)

    q, s = quantize_embedding(sd["embed.weight"])
    add("embed.weight", ff.DT_I8, q)
    add("embed.scale", ff.DT_F32, s)
    for i in range(cfg.n_layers):
        p = f"blocks.{i}."
        for n in NORMS:
            add(f"layers.{i}.{n}", ff.DT_F32, sd[p + n + ".weight"].numpy())
        for n in LINEARS:
            trits, gamma = weight_quant(sd[p + n + ".weight"])
            add(f"layers.{i}.{n}", ff.DT_TERNARY, trits.to(torch.int8).numpy(), float(gamma))
    add("norm", ff.DT_F32, sd["norm.weight"].numpy())
    return f


def load_checkpoint(path: Path) -> Transformer:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = Transformer(Config(**ck["config"]))
    model.load_state_dict(ck["model"])
    return model.eval()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, default=Path("runs/ternary/model.pt"))
    p.add_argument("--tokenizer", type=Path, default=Path("out/tokenizer.json"))
    p.add_argument("--out", type=Path, default=Path("out/tinystories-15m.trit"))
    args = p.parse_args()
    model = load_checkpoint(args.checkpoint)
    f = build(model, Tokenizer.from_file(str(args.tokenizer)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = ff.write(args.out, f)
    trits = sum(t.data.size for t in f.tensors.values() if t.dtype == ff.DT_TERNARY)
    print(f"wrote {args.out}: {n:,} bytes, {trits:,} ternary weights")


if __name__ == "__main__":
    main()
