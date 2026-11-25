"""A small Llama-style decoder with BitNet b1.58 linear layers.

Every linear layer inside the blocks is a BitLinear: weights are quantized to
{-1, 0, +1} times one per-matrix scale (absmean), inputs to 8-bit integers per
token (absmax). Both use the straight-through estimator, so the latent weights
stay in full precision and receive ordinary gradients.

The token embedding is tied to the output head and stays in full precision
during training; export stores it as int8 with one scale per row.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

EPS_QUANT = 1e-5


@dataclass
class Config:
    vocab_size: int = 4096
    dim: int = 384
    n_layers: int = 8
    n_heads: int = 6
    ffn_dim: int = 1536
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    ternary: bool = True

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)


def weight_quant(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """absmean ternary quantization: returns (trits in {-1,0,1}, scale)."""
    scale = w.abs().mean().clamp(min=EPS_QUANT)
    return (w / scale).round().clamp(-1, 1), scale


def activation_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token absmax int8 quantization: returns (integers in [-127,127], 127/absmax)."""
    s = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=EPS_QUANT)
    return (x * s).round().clamp(-128, 127), s


class BitLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        xf = x.float()
        xq, s = activation_quant(xf)
        wq, scale = weight_quant(w.float())
        # Straight-through estimator: forward sees the quantized values,
        # backward treats quantization as identity.
        x_ste = xf + (xq / s - xf).detach()
        w_ste = w + (wq * scale - w).detach()
        return F.linear(x_ste.to(x.dtype), w_ste.to(x.dtype))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def rope_tables(
    cfg: Config, device: torch.device | str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    hd = cfg.head_dim
    inv = 1.0 / (
        cfg.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float32, device=device) / hd)
    )
    t = torch.arange(cfg.max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv)  # (T, hd/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE: pairs (i, i + hd/2). x is (B, H, T, hd)."""
    x1, x2 = x.float().chunk(2, dim=-1)
    c = cos[: x.shape[-2]]
    s = sin[: x.shape[-2]]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).to(x.dtype)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        lin = BitLinear if cfg.ternary else (lambda i, o: nn.Linear(i, o, bias=False))
        d, f = cfg.dim, cfg.ffn_dim
        self.n_heads = cfg.n_heads
        self.attn_norm = RMSNorm(d, cfg.norm_eps)
        self.wq, self.wk, self.wv = lin(d, d), lin(d, d), lin(d, d)
        self.attn_sub_norm = RMSNorm(d, cfg.norm_eps)
        self.wo = lin(d, d)
        self.ffn_norm = RMSNorm(d, cfg.norm_eps)
        self.w_up = lin(d, f)
        self.ffn_sub_norm = RMSNorm(f, cfg.norm_eps)
        self.w_down = lin(f, d)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.attn_norm(x)

        def heads(z: torch.Tensor) -> torch.Tensor:
            return z.view(b, t, self.n_heads, d // self.n_heads).transpose(1, 2)

        q = apply_rope(heads(self.wq(h)), cos, sin)
        k = apply_rope(heads(self.wk(h)), cos, sin)
        v = heads(self.wv(h))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(b, t, d)
        x = x + self.wo(self.attn_sub_norm(a))
        u = F.relu(self.w_up(self.ffn_norm(x))).square()
        return x + self.w_down(self.ffn_sub_norm(u))


class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        cos, sin = rope_tables(cfg)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        # Scale the residual-branch outputs down with depth (GPT-2 style).
        for blk in self.blocks:
            for lin in (blk.wo, blk.w_down):
                nn.init.normal_(lin.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x, self.rope_cos, self.rope_sin)
        x = self.norm(x)
        logits = F.linear(x, self.embed.weight.to(x.dtype))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def param_groups(self, weight_decay: float) -> list[dict]:
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            (no_decay if p.ndim < 2 or "norm" in name else decay).append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def count_params(self) -> dict[str, int]:
        linear = sum(
            p.numel() for n, p in self.named_parameters() if p.ndim == 2 and "embed" not in n
        )
        return {
            "linear": linear,
            "embedding": self.embed.weight.numel(),
            "norm": sum(p.numel() for p in self.parameters() if p.ndim == 1),
        }
