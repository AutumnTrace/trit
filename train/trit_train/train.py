"""Train a ternary (or, with --baseline, a full-precision) TinyStories model."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .data import TokenStream
from .model import Config, Transformer


def lr_at(step: int, total: int, warmup: int, peak: float, floor_frac: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def evaluate(model: Transformer, batches, amp: bool) -> float:
    model.eval()
    total = 0.0
    for x, y in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / len(batches)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("out"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--baseline", action="store_true", help="plain nn.Linear instead of BitLinear")
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=40)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--ffn", type=int, default=1536)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = Config(
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        ffn_dim=args.ffn,
        max_seq_len=args.seq,
        ternary=not args.baseline,
    )
    model = Transformer(cfg).to(device)
    counts = model.count_params()
    print(json.dumps({"config": cfg.to_dict(), "params": counts}))

    opt = torch.optim.AdamW(model.param_groups(args.wd), lr=args.lr, betas=(0.9, 0.95), fused=amp)
    fwd = torch.compile(model) if args.compile else model

    train = TokenStream(args.data / "train.bin", args.seq, device)
    val = TokenStream(args.data / "valid.bin", args.seq, device).fixed_batches(
        args.batch, args.eval_batches
    )
    gen = torch.Generator().manual_seed(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "log.jsonl").open("w")
    tokens_per_step = args.batch * args.seq
    t0 = time.time()
    x, y = train.batch(args.batch, gen)
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            vl = evaluate(model, val, amp)
            rec = {
                "step": step,
                "tokens": step * tokens_per_step,
                "val_loss": round(vl, 4),
                "elapsed_s": round(time.time() - t0, 1),
            }
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
        if step == args.steps:
            break
        if step and step % 5000 == 0:
            torch.save(
                {"config": cfg.to_dict(), "model": model.state_dict()}, args.out / "model.pt"
            )
        lr = lr_at(step, args.steps, args.warmup, args.lr, args.min_lr_frac)
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            _, loss = fwd(x, y)
        x, y = train.batch(args.batch, gen)  # overlap the next copy with backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 50 == 0:
            rec = {
                "step": step,
                "train_loss": round(loss.item(), 4),
                "lr": lr,
                "elapsed_s": round(time.time() - t0, 1),
            }
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step % 250 == 0:
                tps = (step + 1) * tokens_per_step / max(1e-9, time.time() - t0)
                print({**rec, "tok_per_s": round(tps)}, flush=True)

    torch.save({"config": cfg.to_dict(), "model": model.state_dict()}, args.out / "model.pt")
    summary = {
        "config": cfg.to_dict(),
        "params": counts,
        "steps": args.steps,
        "batch": args.batch,
        "seq": args.seq,
        "tokens": args.steps * tokens_per_step,
        "lr": args.lr,
        "wd": args.wd,
        "train_seconds": round(time.time() - t0, 1),
        "final_val_loss": round(vl, 4),
        "gpu": torch.cuda.get_device_name(0) if amp else None,
        "torch": torch.__version__,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
