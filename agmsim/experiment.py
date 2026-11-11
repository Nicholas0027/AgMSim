"""Factorial experiments.

``run_matrix`` runs every combination of the given factors over a list of
seeds. Because demand, personas and the initial fleet depend only on the seed,
all arms that share a seed face identical conditions (common random numbers),
so differences between arms can be compared seed by seed.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

from .config import Config
from .metrics import flatten

# two-sided 95% Student-t critical values by degrees of freedom
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}

DEFAULT_METRICS = [
    "travelers.completion_rate", "travelers.trip_time_s_mean", "travelers.generalised_cost_usd_mean",
    "vehicles.travel_time_ratio", "vehicles.vkt_km", "llm.calls",
]

LLM_2X2 = {"human_agents.use_llm": [False, True], "machine_agents.use_llm": [False, True]}


def _t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    keys = sorted(_T95)
    return _T95[max((k for k in keys if k <= df), default=1)] if df <= 30 else 1.96


def mean_ci(values: list) -> tuple[float, float]:
    """Mean and half-width of the 95% confidence interval."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, _t95(n - 1) * math.sqrt(var / n)


def _one(job):
    from .engine import Simulation      # imported in the worker process
    from .recorder import Recorder
    source, overrides, label, seed = job
    # only the metrics are needed: no trace in memory or on disk (state fingerprints are still chained)
    result = Simulation(source, overrides=overrides, recorder=Recorder(keep=False)).run()
    return {"arm": label, "seed": seed, "trace_digest": result.trace_digest, **flatten(result.metrics)}


def run_matrix(source, factors: dict, seeds: list, overrides: Optional[list] = None,
               workers: int = 1, out_dir: Optional[str] = None, progress=None) -> dict:
    """Run all factor combinations x seeds. Returns ``{"rows": [...], "summary": {...}}``."""
    base = Config.load(source).to_dict() if not isinstance(source, dict) else source
    names = list(factors)
    jobs = []
    for combo in itertools.product(*(factors[n] for n in names)):
        label = ", ".join(f"{n.split('.', 1)[-1] if n.startswith('agents.') else n}={v}" for n, v in zip(names, combo))
        for seed in seeds:
            ov = {**{"record.dir": None, "simulation.seed": seed}, **dict(zip(names, combo))}
            jobs.append((base, [*(overrides or []), *[f"{k}={json.dumps(v)}" for k, v in ov.items()]], label or "base", seed))
    rows = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for row in pool.map(_one, jobs):
                rows.append(row)
                if progress:
                    progress(len(rows), len(jobs))
    else:
        for job in jobs:
            rows.append(_one(job))
            if progress:
                progress(len(rows), len(jobs))
    summary = summarize(rows)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for row in rows for k in row})
        with open(out / "runs.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["arm", "seed"] + [k for k in keys if k not in ("arm", "seed")])
            writer.writeheader()
            writer.writerows(rows)
        with open(out / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    return {"rows": rows, "summary": summary}


def summarize(rows: list) -> dict:
    """Per arm: mean and 95% CI half-width of every numeric metric across seeds."""
    arms: dict[str, list] = {}
    for row in rows:
        arms.setdefault(row["arm"], []).append(row)
    summary = {}
    for arm, items in arms.items():
        stats = {}
        for key in sorted({k for item in items for k in item}):
            values = [item[key] for item in items if isinstance(item.get(key), (int, float))]
            if key in ("seed",) or len(values) != len(items):
                continue
            mean, half = mean_ci(values)
            stats[key] = {"mean": round(mean, 4), "ci95": None if math.isnan(half) else round(half, 4)}
        summary[arm] = {"n": len(items), "metrics": stats}
    return summary


def format_table(summary: dict, metrics: Optional[list] = None) -> str:
    metrics = metrics or DEFAULT_METRICS
    present = [m for m in metrics if any(m in arm["metrics"] for arm in summary.values())]
    width = max(len(a) for a in summary) + 2
    header = "arm".ljust(width) + "".join(m.split(".", 1)[-1][:22].rjust(24) for m in present)
    lines = [header, "-" * len(header)]
    for arm, data in summary.items():
        cells = []
        for m in present:
            s = data["metrics"].get(m)
            if s is None:
                cells.append("-".rjust(24))
            else:
                ci = "" if s["ci95"] is None else f" ±{s['ci95']:.3g}"
                cells.append(f"{s['mean']:.4g}{ci}".rjust(24))
        lines.append(arm.ljust(width) + "".join(cells))
    lines.append(f"(mean ± 95% CI half-width over {next(iter(summary.values()))['n']} seed(s))")
    return "\n".join(lines)


# --------------------------------------------------------------------------- designed comparisons


def run_arms(source, arms: dict, seeds: list, workers: int = 1, progress=None) -> list:
    """Run named arms (``{label: {dotted.key: value}}``) over the same seeds.

    Unlike :func:`run_matrix` the arms need not form a full factorial design.
    """
    base = Config.load(source).to_dict() if not isinstance(source, dict) else source
    jobs = []
    for label, overrides in arms.items():
        for seed in seeds:
            ov = {"record.dir": None, "simulation.seed": seed, **overrides}
            jobs.append((base, [f"{k}={json.dumps(v)}" for k, v in ov.items()], label, seed))
    rows = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for row in pool.map(_one, jobs):
                rows.append(row)
                if progress:
                    progress(len(rows), len(jobs))
    else:
        for job in jobs:
            rows.append(_one(job))
            if progress:
                progress(len(rows), len(jobs))
    return rows


def paired(rows: list, baseline: str, metric: str) -> dict:
    """Per arm: mean of the metric, and its seed-by-seed difference from ``baseline`` with a 95% CI.

    Arms share seeds and therefore travellers, so differencing within a seed removes most of the
    between-seed variance (common random numbers).
    """
    by_arm: dict[str, dict] = {}
    for row in rows:
        if isinstance(row.get(metric), (int, float)):
            by_arm.setdefault(row["arm"], {})[row["seed"]] = row[metric]
    base = by_arm.get(baseline, {})
    out = {}
    for arm, values in by_arm.items():
        mean = sum(values.values()) / len(values)
        entry = {"mean": mean, "n": len(values)}
        if arm != baseline:
            diffs = [values[s] - base[s] for s in values if s in base]
            if diffs:
                d_mean, half = mean_ci(diffs)
                entry.update(diff=d_mean, ci95=half,
                             significant=(not math.isnan(half)) and abs(d_mean) > half,
                             same_sign=sum(1 for d in diffs if d * d_mean > 0), pairs=len(diffs))
        out[arm] = entry
    return out


def paired_markdown(rows: list, baseline: str, metrics: dict) -> str:
    """Markdown table: arms in rows, ``metrics`` (``{key: column title}``) in columns.

    Cells show the arm mean and, for non-baseline arms, the paired difference from the baseline
    as ``mean (Δ ± half-width)``; ``*`` marks differences whose 95% interval excludes zero.
    """
    tables = {key: paired(rows, baseline, key) for key in metrics}
    arms = list(dict.fromkeys(row["arm"] for row in rows))
    lines = ["| 实验组 | " + " | ".join(metrics.values()) + " |", "| - |" + " - |" * len(metrics)]
    for arm in arms:
        cells = []
        for key in metrics:
            e = tables[key].get(arm)
            if e is None:
                cells.append("–")
                continue
            text = f"{e['mean']:.4g}"
            if "diff" in e:
                half = "" if math.isnan(e["ci95"]) else f" ± {e['ci95']:.3g}"
                text += f" ({e['diff']:+.3g}{half}){'*' if e['significant'] else ''}"
            cells.append(text)
        lines.append(f"| {arm}{'（基准）' if arm == baseline else ''} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
