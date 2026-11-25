"""Draw the README charts as static SVG, one light and one dark file each.

    uv run --no-project python bench/charts.py

Reads train/results/*.log.jsonl and bench/kernels-sweep.json; writes
docs/media/{loss,sweep}-{light,dark}.svg. No dependencies.

Colours are the first four slots of a categorical palette checked for
colour-vision deficiency separation in both modes; every line is also
labelled at its end, so identity never depends on colour alone.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / "docs" / "media"

THEMES = {
    "light": {
        "surface": "#ffffff",
        "text": "#1f2328",
        "dim": "#59636e",
        "grid": "#e7e7e7",
        "axis": "#c2c2c2",
        "series": ["#257de0", "#f47230", "#00b894", "#f2a600"],
    },
    "dark": {
        "surface": "#0d1117",
        "text": "#e6edf3",
        "dim": "#9198a1",
        "grid": "#262c36",
        "axis": "#3d444d",
        "series": ["#338ced", "#e76624", "#00a888", "#d28e05"],
    },
}

FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    start = math.ceil(lo / step) * step
    ticks = []
    v = start
    while v <= hi + 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


def fmt(v: float) -> str:
    if v >= 1000:
        return f"{v:,.0f}"
    if v == int(v):
        return str(int(v))
    return f"{v:g}"


def panel(
    out: list[str],
    theme: dict,
    x0: float,
    y0: float,
    w: float,
    h: float,
    series: list[dict],
    *,
    x_log: bool,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    x_ticks: list[float],
    y_ticks: list[float],
    x_label: str,
    y_label: str,
    title: str,
    x_fmt=fmt,
) -> None:
    """One plot area. series: {name, color, points[(x, y)], label}."""
    def sx(x: float) -> float:
        a, b = x_range
        t = (math.log10(x) - math.log10(a)) / (math.log10(b) - math.log10(a)) if x_log else (x - a) / (b - a)
        return x0 + t * w

    def sy(y: float) -> float:
        a, b = y_range
        return y0 + h - (y - a) / (b - a) * h

    out.append(
        f'<text x="{x0}" y="{y0 - 14}" font-size="13" font-weight="600" fill="{theme["text"]}">{title}</text>'
    )
    for t in y_ticks:
        y = sy(t)
        out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{theme["grid"]}" stroke-width="1"/>')
        out.append(
            f'<text x="{x0 - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end" fill="{theme["dim"]}" '
            f'style="font-variant-numeric: tabular-nums">{fmt(t)}</text>'
        )
    out.append(
        f'<line x1="{x0}" x2="{x0 + w}" y1="{y0 + h}" y2="{y0 + h}" stroke="{theme["axis"]}" stroke-width="1"/>'
    )
    for t in x_ticks:
        x = sx(t)
        out.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y0 + h}" y2="{y0 + h + 4}" stroke="{theme["axis"]}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{y0 + h + 18}" font-size="11" text-anchor="middle" fill="{theme["dim"]}">'
            f"{x_fmt(t)}</text>"
        )
    out.append(
        f'<text x="{x0 + w / 2}" y="{y0 + h + 38}" font-size="12" text-anchor="middle" fill="{theme["dim"]}">'
        f"{x_label}</text>"
    )
    out.append(
        f'<text transform="translate({x0 - 46} {y0 + h / 2}) rotate(-90)" font-size="12" text-anchor="middle" '
        f'fill="{theme["dim"]}">{y_label}</text>'
    )
    # Lines, then end markers with a surface ring, then end labels (nudged apart if close).
    ends = []
    for s in series:
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in s["points"])
        out.append(
            f'<polyline points="{pts}" fill="none" stroke="{s["color"]}" stroke-width="2" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        x, y = s["points"][-1]
        ends.append([sy(y), sx(x), s])
    for ey, ex, s in ends:
        out.append(
            f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4" fill="{s["color"]}" stroke="{theme["surface"]}" stroke-width="2"/>'
        )
    ends.sort(key=lambda e: e[0])
    placed: list[float] = []
    for ey, ex, s in ends:
        ly = ey + 4
        if placed and ly - placed[-1] < 14:
            ly = placed[-1] + 14
        placed.append(ly)
        if abs(ly - (ey + 4)) > 1:
            out.append(
                f'<line x1="{ex + 6:.1f}" y1="{ey:.1f}" x2="{ex + 12:.1f}" y2="{ly - 4:.1f}" '
                f'stroke="{theme["axis"]}" stroke-width="1"/>'
            )
        out.append(f'<text x="{ex + 14:.1f}" y="{ly:.1f}" font-size="11.5" fill="{theme["text"]}">{s["label"]}</text>')


def legend(out: list[str], theme: dict, x: float, y: float, series: list[dict]) -> None:
    for s in series:
        out.append(
            f'<line x1="{x}" x2="{x + 16}" y1="{y - 4}" y2="{y - 4}" stroke="{s["color"]}" stroke-width="2" '
            'stroke-linecap="round"/>'
        )
        out.append(f'<text x="{x + 22}" y="{y}" font-size="12" fill="{theme["text"]}">{s["name"]}</text>')
        x += 30 + 7.2 * len(s["name"])


def svg(width: int, height: int, theme: dict, body: list[str], desc: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="{FONT}" role="img" aria-label="{desc}">\n'
        f"<title>{desc}</title>\n"
        f'<rect width="100%" height="100%" rx="8" fill="{theme["surface"]}"/>\n' + "\n".join(body) + "\n</svg>\n"
    )


def read_log(name: str) -> list[tuple[float, float]]:
    pts = []
    for line in (ROOT / "train" / "results" / f"{name}.log.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "val_loss" in r and r["step"] > 0:
            pts.append((r["tokens"] / 1e6, r["val_loss"]))
    return pts


def loss_chart() -> None:
    tern, base = read_log("ternary"), read_log("baseline")
    ymax = max(y for _, y in tern + base)
    ymin = min(y for _, y in tern + base)
    for mode, theme in THEMES.items():
        c = theme["series"]
        series = [
            {"name": "ternary (1.58-bit)", "color": c[0], "points": tern, "label": f"ternary {tern[-1][1]:.3f}"},
            {"name": "bf16 baseline, same shape", "color": c[1], "points": base, "label": f"bf16 {base[-1][1]:.3f}"},
        ]
        body: list[str] = []
        legend(body, theme, 70, 26, series)
        xmax = max(x for x, _ in tern + base)
        panel(
            body, theme, 70, 64, 560, 250, series,
            x_log=False,
            x_range=(0, xmax),
            y_range=(math.floor(ymin * 10) / 10, math.ceil(ymax * 10) / 10),
            x_ticks=nice_ticks(0, xmax, 6),
            y_ticks=nice_ticks(math.floor(ymin * 10) / 10, math.ceil(ymax * 10) / 10, 5),
            x_label="training tokens (millions)",
            y_label="validation loss (nats/token)",
            title="Validation loss during training",
        )
        out = svg(720, 370, theme, body, "Validation loss of the ternary model and a bf16 baseline during training")
        (MEDIA / f"loss-{mode}.svg").write_text(out, encoding="utf-8")


def sweep_chart() -> None:
    rows = json.loads((ROOT / "bench" / "kernels-sweep.json").read_text())
    names = [("f32", "f32"), ("i8", "int8"), ("i2", "ternary I2 (2 bits)"), ("tq", "ternary TQ (1.6 bits)")]
    threads = sorted({r["threads"] for r in rows})
    for mode, theme in THEMES.items():
        body: list[str] = []
        series_for_legend = []
        for pi, t in enumerate(threads):
            series = []
            for i, (key, label) in enumerate(names):
                pts = sorted(
                    (r["bytes"] / 1e6, r["weights"] / r["seconds"] / 1e9)
                    for r in rows
                    if r["threads"] == t and r["kernel"].split("-")[0] == key
                )
                if pts:
                    series.append(
                        {"name": label, "color": theme["series"][i], "points": pts, "label": f"{pts[-1][1]:.0f}"}
                    )
            series_for_legend = series_for_legend or series
            ymax = max(y for s in series for _, y in s["points"])
            yt = nice_ticks(0, ymax * 1.05, 5)
            panel(
                body, theme, 70 + pi * 400, 70, 290, 240, series,
                x_log=True,
                x_range=(1, 1200),
                y_range=(0, yt[-1] if yt[-1] >= ymax else ymax * 1.1),
                x_ticks=[1, 10, 100, 1000],
                y_ticks=yt,
                x_label="weights in memory (MB, log scale)",
                y_label="billion weights per second",
                title=f"{t} thread{'s' if t > 1 else ''}",
                x_fmt=lambda v: f"{v:g}",
            )
        legend(body, theme, 70, 26, series_for_legend)
        out = svg(800, 370, theme, body, "GEMV throughput against the size of the weights, by weight format")
        (MEDIA / f"sweep-{mode}.svg").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    MEDIA.mkdir(parents=True, exist_ok=True)
    loss_chart()
    if (ROOT / "bench" / "kernels-sweep.json").exists():
        sweep_chart()
    print("wrote", sorted(p.name for p in MEDIA.glob("*.svg")))
