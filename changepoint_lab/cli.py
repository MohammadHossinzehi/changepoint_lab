"""Command line interface: python -m changepoint_lab <command> ...

  detect FILE   offline detection on a column of a CSV or whitespace file
  online FILE   streaming BOCPD, prints each changepoint as it is declared
  crops  FILE   every optimal segmentation across a penalty range
  demo          synthetic benchmark comparing all methods
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import metrics, offline, online, synth

BARS = " .:-=+*#%@"


def load_column(path: str, column: int) -> np.ndarray:
    rows = []
    with (sys.stdin if path == "-" else open(path, encoding="utf-8")) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            try:
                rows.append(float(parts[column]))
            except (ValueError, IndexError):
                if rows:
                    raise ValueError(f"cannot parse column {column} in line: {line!r}")
                continue  # header line
    if not rows:
        raise ValueError("no numeric data found")
    return np.array(rows)


def strip_plot(y, cps, width=72) -> str:
    """Two line terminal plot: a density strip of the signal (binned means)
    and a marker row with ^ under each changepoint."""
    y = np.asarray(y, float)
    n = y.size
    width = min(width, n)
    edges = np.linspace(0, n, width + 1).astype(int)
    means = np.array([y[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    lo, hi = means.min(), means.max()
    norm = (means - lo) / (hi - lo) if hi > lo else np.zeros_like(means)
    strip = "".join(BARS[int(round(v * (len(BARS) - 1)))] for v in norm)
    marks = [" "] * width
    for c in cps:
        marks[min(int(c * width / n), width - 1)] = "^"
    return f"|{strip}|\n {''.join(marks)}"


def cmd_detect(a) -> int:
    y = load_column(a.file, a.column)
    method = {"pelt": offline.pelt, "op": offline.optimal_partitioning,
              "binseg": offline.binary_segmentation}[a.method]
    pen = float(a.penalty) if _is_number(a.penalty) else a.penalty
    t0 = time.perf_counter()
    seg = method(y, cost=a.cost, penalty=pen, min_size=a.min_size)
    dt = time.perf_counter() - t0
    print(f"n={seg.n} method={seg.method} cost={a.cost} penalty={seg.penalty:.3f} "
          f"time={dt * 1000:.1f}ms")
    print("changepoints:", " ".join(map(str, seg.changepoints)) or "(none)")
    for s, e in seg.segments:
        print(f"  [{s:>6}, {e:>6})  len={e - s:<6} mean={y[s:e].mean():.4g} sd={y[s:e].std():.4g}")
    if not a.no_plot:
        print(strip_plot(y, seg.changepoints))
    return 0


def cmd_online(a) -> int:
    y = load_column(a.file, a.column)
    model = (online.NormalGamma.from_data(y[: a.warmup]) if a.model == "normal"
             else online.PoissonGamma.from_data(y[: a.warmup]))
    det = online.BOCPD(model, hazard=a.hazard, persistence=a.persistence)
    for x in y:
        info = det.update(x)
        if info["declared"] is not None:
            c = info["declared"]
            print(f"t={info['t']:>6}  changepoint at {c}  (delay {info['t'] - c})")
    print(f"done: {len(det.changepoints)} changepoints, max hypotheses kept {det.max_runs_kept}, "
          f"log evidence {det.log_evidence:.2f}")
    return 0


def cmd_crops(a) -> int:
    y = load_column(a.file, a.column)
    res = offline.crops(y, a.pen_min, a.pen_max, cost=a.cost, min_size=a.min_size)
    print(f"{len(res.solutions)} distinct optimal segmentations from {res.pelt_runs} PELT runs")
    print(f"{'penalty range':>24}  {'K':>4}  {'cost':>12}  changepoints")
    for lo, hi, seg in res.solutions:
        cps = " ".join(map(str, seg.changepoints))
        if len(cps) > 60:
            cps = cps[:57] + "..."
        print(f"{lo:>11.3f} .. {hi:<10.3f}  {len(seg.changepoints):>4}  {seg.cost:>12.3f}  {cps}")
    return 0


def cmd_demo(a) -> int:
    rows = []
    for kind, cost, model in (("mean", "l2", "normal"), ("meanvar", "normal", "normal"),
                              ("poisson", "poisson", "poisson")):
        y, true = synth.piecewise(n=a.n, k=a.k, kind=kind, seed=a.seed)
        print(f"\n== {kind} signal, n={a.n}, true changepoints {true}")
        print(strip_plot(y, true))
        runs = [
            ("optimal_partitioning", lambda: offline.optimal_partitioning(y, cost=cost).changepoints),
            ("pelt", lambda: offline.pelt(y, cost=cost).changepoints),
            ("binary_segmentation", lambda: offline.binary_segmentation(y, cost=cost).changepoints),
            ("bocpd", lambda: online.bocpd(y, model=model, hazard=a.n / (a.k + 1)).changepoints),
        ]
        for name, fn in runs:
            t0 = time.perf_counter()
            cps = fn()
            dt = time.perf_counter() - t0
            s = metrics.score(true, cps, a.n, margin=a.margin)
            rows.append((kind, name, s, dt))
            print(f"  {name:<22} f1={s['f1']:.2f} hausdorff={s['hausdorff']:<6g} "
                  f"rand={s['rand_index']:.3f} time={dt * 1000:7.1f}ms  {cps}")
    return 0


def _is_number(s) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="changepoint_lab", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("file", help="CSV or whitespace separated file, '-' for stdin")
        sp.add_argument("--column", type=int, default=0)

    d = sub.add_parser("detect", help="offline detection")
    common(d)
    d.add_argument("--method", choices=["pelt", "op", "binseg"], default="pelt")
    d.add_argument("--cost", choices=["l2", "normal", "poisson"], default="l2")
    d.add_argument("--penalty", default="bic", help="number, bic, aic or hq")
    d.add_argument("--min-size", type=int, default=None)
    d.add_argument("--no-plot", action="store_true")
    d.set_defaults(fn=cmd_detect)

    o = sub.add_parser("online", help="streaming Bayesian detection")
    common(o)
    o.add_argument("--model", choices=["normal", "poisson"], default="normal")
    o.add_argument("--hazard", type=float, default=250.0, help="expected segment length")
    o.add_argument("--persistence", type=int, default=5)
    o.add_argument("--warmup", type=int, default=50, help="points used to set the prior")
    o.set_defaults(fn=cmd_online)

    c = sub.add_parser("crops", help="all optimal segmentations over a penalty range")
    common(c)
    c.add_argument("--cost", choices=["l2", "normal", "poisson"], default="l2")
    c.add_argument("--pen-min", type=float, default=2.0)
    c.add_argument("--pen-max", type=float, default=200.0)
    c.add_argument("--min-size", type=int, default=None)
    c.set_defaults(fn=cmd_crops)

    m = sub.add_parser("demo", help="synthetic comparison of every method")
    m.add_argument("--n", type=int, default=600)
    m.add_argument("--k", type=int, default=5)
    m.add_argument("--seed", type=int, default=7)
    m.add_argument("--margin", type=int, default=5)
    m.set_defaults(fn=cmd_demo)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
