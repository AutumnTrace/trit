#!/usr/bin/env bash
# The two v0.1.0 runs, back to back, same token budget (20k steps x 32 x 512).
set -euo pipefail
cd "$(dirname "$0")"
uv run python -m trit_train.train --out runs/ternary --steps 20000 --batch 32 --lr 2e-3 --eval-every 1000 --compile
uv run python -m trit_train.train --out runs/baseline --baseline --steps 20000 --batch 32 --lr 1e-3 --eval-every 1000 --compile
