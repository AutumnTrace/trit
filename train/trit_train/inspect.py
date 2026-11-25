"""Statistics of the trits in an exported `.trit` file.

    uv run python -m trit_train.inspect ../models/tinystories-15m.trit

Reports the share of -1 / 0 / +1 per matrix type, and the feed-forward hidden
units that training switched off entirely: a row of `w_up` that is all zeros
always outputs 0, so the matching column of `w_down` is never used either.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from . import fileformat as ff

NAMES = ("wq", "wk", "wv", "wo", "w_up", "w_down")


def shares(counts: np.ndarray) -> dict[str, float]:
    return {k: round(float(c / counts.sum()), 4) for k, c in zip("-0+", counts, strict=True)}


def stats(f: ff.TritFile) -> dict:
    out: dict = {"by_matrix": {}, "dead_ffn_units": [], "ffn_units_per_layer": f.ffn_dim}
    total = np.zeros(3, dtype=np.int64)
    for name in NAMES:
        counts = np.zeros(3, dtype=np.int64)
        for i in range(f.n_layers):
            a = np.asarray(f.tensors[f"layers.{i}.{name}"].data)
            counts += np.bincount((a + 1).ravel(), minlength=3)
        total += counts
        out["by_matrix"][name] = shares(counts)
    out["all"] = shares(total)
    for i in range(f.n_layers):
        up = np.asarray(f.tensors[f"layers.{i}.w_up"].data)
        down = np.asarray(f.tensors[f"layers.{i}.w_down"].data)
        dead = np.flatnonzero((up == 0).all(axis=1))
        out["dead_ffn_units"].append(
            {
                "layer": i,
                "w_up_rows_all_zero": int(dead.size),
                "w_down_columns_all_zero": int((down == 0).all(axis=0).sum()),
                "both": int((down[:, dead] == 0).all(axis=0).sum()),
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model")
    args = p.parse_args()
    print(json.dumps(stats(ff.read(args.model)), indent=2))


if __name__ == "__main__":
    main()
