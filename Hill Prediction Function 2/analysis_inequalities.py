

from __future__ import annotations

import csv
import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


PROJECT_ROOT = (
    ROOT
    / "Hill-Ramp-Function-Prediction-Edition-20_04_30_2026"
    / "Hill-Ramp-Function-Prediction-Edition-20_04_30_2026"
)
RESULT_SUBDIRS = {
    "ex1": Path("results_two_node_cycle_ex1") / "results_ex1.csv",
    "ex2": Path("results_two_node_cycle_ex2") / "results_ex2.csv",
    "ex3": Path("results_three_node_cycle_ex3") / "results_ex3.csv",
}
LEAST_SQUARES_RESULTS = (
    ROOT
    / "least square prediction Edition 20"
    / "outputs"
    / "least_squares_parameter_estimates.csv"
)
LOW_DATA_LEAST_SQUARES_RESULTS: Dict[str, Path] = {}
OUTPUT_DIR = ROOT / "analysis_outputs"
STANDALONE_PLOT_DIR = OUTPUT_DIR / "standalone_parameter_boxplots"
FIGURE_DPI = 600

PINN_COLOR = "#0072B2"
PINN_FILL = "#DCEAF7"
LS_COLOR = "#D55E00"
LS_FILL = "#F6D7B8"
LS_METHOD_LABEL = "Vanilla least squares"
LS_METHOD_SHORT = "Vanilla LS"
TRUE_COLOR = "#202020"
GRID_COLOR = "#D9DEE7"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#2F3542",
        "axes.labelcolor": "#1F2933",
        "axes.labelsize": 11,
        "axes.linewidth": 0.9,
        "axes.titlecolor": "#111827",
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "font.family": "DejaVu Serif",
        "font.size": 10.5,
        "legend.fontsize": 9.5,
        "mathtext.fontset": "stix",
        "xtick.color": "#1F2933",
        "xtick.labelsize": 9.5,
        "ytick.color": "#1F2933",
        "ytick.labelsize": 9.5,
    }
)


@dataclass(frozen=True)
class SummaryRow:
    noise_ub: float
    total: int
    matches: int
    fails: int


@dataclass(frozen=True)
class ExampleConfig:
    key: str
    base: str
    label: str
    variant: str
    directory: Path
    result_path: Path


INEQUALITIES = {
    "ex1": {
        "inequalities": (
            "L[y->x] < T[x->y] && T[x->y] < U[y->x] && 0 < T[x->y] && "
            "0 < L[y->x] < U[y->x] && L[x->y] < T[y->x] && T[y->x] < U[x->y] && "
            "0 < T[y->x] && 0 < L[x->y] < U[x->y]"
        ),
        "variables": "{L[y->x], U[y->x], T[x->y], L[x->y], U[x->y], T[y->x]}",
    },
    "ex2": {
        "inequalities": (
            "(L[x->x] + L[y->x]) < T[x->y] && T[x->x] < (U[x->x] + L[y->x]) && "
            "T[x->y] < (L[x->x] + U[y->x]) < T[x->x] && "
            "T[x->x] < (U[x->x] + U[y->x]) && 0 < T[x->y] < T[x->x] && "
            "0 < L[x->x] < U[x->x] && 0 < L[y->x] < U[y->x] && "
            "L[x->y]*L[y->y] < T[y->x] && T[y->x] < U[x->y]*L[y->y] && "
            "U[x->y]*L[y->y] < T[y->y] && T[y->y] < L[x->y]*U[y->y] && "
            "T[y->y] < U[x->y]*U[y->y] && 0 < T[y->x] < T[y->y] && "
            "0 < L[x->y] < U[x->y] && 0 < L[y->y] < U[y->y]"
        ),
        "variables": (
            "{L[x->x], L[y->x], U[x->x], U[y->x], T[x->y], T[x->x], L[x->y], "
            "L[y->y], U[x->y], U[y->y], T[y->x], T[y->y]}"
        ),
    },
    "ex3": {
        "inequalities": (
            "L[z->x] < T[x->y] && T[x->y] < U[z->x] && 0 < T[x->y] && "
            "0 < L[z->x] < U[z->x] && L[x->y] < T[y->z] && "
            "T[y->z] < U[x->y] && 0 < T[y->z] && 0 < L[x->y] < U[x->y] && "
            "L[y->z] < T[z->x] && T[z->x] < U[y->z] && 0 < T[z->x] && "
            "0 < L[y->z] < U[y->z]"
        ),
        "variables": (
            "{L[z->x], U[z->x], T[x->y], L[x->y], U[x->y], T[y->z], L[y->z], "
            "U[y->z], T[z->x]}"
        ),
    },
}

EXERCISE_LABELS = {
    "ex1": "Exercise 1",
    "ex2": "Exercise 2",
    "ex3": "Exercise 3",
}
BASE_EXERCISE: Dict[str, str] = {}


def _save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    if output_path.suffix.lower() == ".png":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def _style_axis(ax, grid_axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.7, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#3B4351")
    ax.spines["bottom"].set_color("#3B4351")
    ax.tick_params(axis="both", which="major", length=3.5, width=0.8, direction="out")


def _exercise_label(exercise: str) -> str:
    return EXERCISE_LABELS.get(exercise, exercise.replace("ex", "Exercise "))


def _base_exercise(exercise: str) -> str:
    return BASE_EXERCISE.get(exercise, exercise)


def _example_base_from_dirname(dirname: str) -> Optional[Tuple[str, str]]:
    for example_num, base in (("1", "ex1"), ("2", "ex2"), ("3", "ex3")):
        prefix = f"Example_{example_num}"
        if dirname == prefix:
            return base, ""
        if dirname.startswith(f"{prefix}_"):
            return base, dirname[len(prefix) + 1 :]
    return None


def _variant_label(variant: str) -> str:
    if not variant:
        return ""
    labels = {
        "low_traj_low_data": "low trajectory / low data",
        "no_feature_map": "no feature map",
    }
    return labels.get(variant, variant.replace("_", " "))


def _exercise_key(base: str, variant: str) -> str:
    return base if not variant else f"{base}_{variant}"


def _exercise_label_from_parts(base: str, variant: str) -> str:
    base_label = EXERCISE_LABELS.get(base, base)
    variant_label = _variant_label(variant)
    return base_label if not variant_label else f"{base_label} ({variant_label})"


def _example_sort_key(config: ExampleConfig) -> Tuple[int, int, str]:
    base_order = {"ex1": 1, "ex2": 2, "ex3": 3}
    variant_order = {"": 0, "low_traj_low_data": 1, "no_feature_map": 2}
    return (
        base_order.get(config.base, 99),
        variant_order.get(config.variant, 50),
        config.variant,
    )


def _discover_example_configs(project_root: Path = PROJECT_ROOT) -> List[ExampleConfig]:
    configs: List[ExampleConfig] = []
    if not project_root.exists():
        return configs

    for directory in sorted(path for path in project_root.iterdir() if path.is_dir()):
        parsed = _example_base_from_dirname(directory.name)
        if parsed is None:
            continue
        base, variant = parsed
        result_relpath = RESULT_SUBDIRS.get(base)
        if result_relpath is None:
            continue
        result_path = directory / result_relpath
        if not result_path.exists():
            continue
        configs.append(
            ExampleConfig(
                key=_exercise_key(base, variant),
                base=base,
                label=_exercise_label_from_parts(base, variant),
                variant=variant,
                directory=directory,
                result_path=result_path,
            )
        )
    return sorted(configs, key=_example_sort_key)


def _register_example_configs(configs: Iterable[ExampleConfig]) -> None:
    for config in configs:
        BASE_EXERCISE[config.key] = config.base
        EXERCISE_LABELS[config.key] = config.label
        if config.key not in INEQUALITIES:
            INEQUALITIES[config.key] = {
                "inequalities": INEQUALITIES[config.base]["inequalities"],
                "variables": INEQUALITIES[config.base]["variables"],
            }
        if config.key not in EDGE_MAP:
            EDGE_MAP[config.key] = list(EDGE_MAP[config.base])


def _parameter_label(param: str, edge: str) -> str:
    symbol = {"L": "L", "U": "U", "Theta": r"\Theta"}.get(param, param)
    return rf"${symbol}_{{{edge}}}$"


def _parameter_filename(param: str, edge: str) -> str:
    return f"{param.lower().replace('theta', 'theta')}_{edge}"


def _noise_tick_label(noise: float) -> str:
    return f"{noise:g}"


def _to_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _load_rows(path: Path) -> List[Dict[str, Optional[float]]]:
    rows: List[Dict[str, Optional[float]]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: _to_float(value) for key, value in row.items()})
    return rows


def _load_least_squares_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: (value or "").strip() for key, value in row.items()})
    return rows


def _load_optional_least_squares_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    return _load_least_squares_rows(path)


def _load_tagged_least_squares_rows(path: Path, exercise: str) -> List[Dict[str, str]]:
    rows = _load_optional_least_squares_rows(path)
    return [{**row, "exercise": exercise} for row in rows]


def _parse_matrix(value: str) -> np.ndarray:
    if not value:
        return np.array([])
    try:
        return np.array(json.loads(value), dtype=float)
    except Exception:
        return np.array([])


def _check_ex1(row: Dict[str, Optional[float]]) -> bool:
    L_yx = row.get("L_learned_10")
    U_yx = row.get("U_learned_10")
    T_xy = row.get("Th_learned_01")
    L_xy = row.get("L_learned_01")
    U_xy = row.get("U_learned_01")
    T_yx = row.get("Th_learned_10")
    required = [L_yx, U_yx, T_xy, L_xy, U_xy, T_yx]
    if any(value is None for value in required):
        return False
    return (
        L_yx < T_xy < U_yx
        and T_xy > 0
        and 0 < L_yx < U_yx
        and L_xy < T_yx < U_xy
        and T_yx > 0
        and 0 < L_xy < U_xy
    )


def _check_ex2(row: Dict[str, Optional[float]]) -> bool:
    L_xx = row.get("L_learned_00")
    U_xx = row.get("U_learned_00")
    T_xx = row.get("Th_learned_00")
    L_yx = row.get("L_learned_10")
    U_yx = row.get("U_learned_10")
    L_xy = row.get("L_learned_01")
    U_xy = row.get("U_learned_01")
    L_yy = row.get("L_learned_11")
    U_yy = row.get("U_learned_11")
    T_xy = row.get("Th_learned_01")
    T_yx = row.get("Th_learned_10")
    T_yy = row.get("Th_learned_11")

    required = [
        L_xx,
        U_xx,
        T_xx,
        L_yx,
        U_yx,
        L_xy,
        U_xy,
        L_yy,
        U_yy,
        T_xy,
        T_yx,
        T_yy,
    ]
    if any(value is None for value in required):
        return False

    return (
        (L_xx + L_yx) < T_xy
        and T_xx < (U_xx + L_yx)
        and T_xy < (L_xx + U_yx) < T_xx
        and T_xx < (U_xx + U_yx)
        and 0 < T_xy < T_xx
        and 0 < L_xx < U_xx
        and 0 < L_yx < U_yx
        and (L_xy * L_yy) < T_yx
        and T_yx < (U_xy * L_yy)
        and (U_xy * L_yy) < T_yy
        and T_yy < (L_xy * U_yy)
        and T_yy < (U_xy * U_yy)
        and 0 < T_yx < T_yy
        and 0 < L_xy < U_xy
        and 0 < L_yy < U_yy
    )


def _check_ex3(row: Dict[str, Optional[float]]) -> bool:
    L_zx = row.get("L_20")
    U_zx = row.get("U_20")
    T_xy = row.get("Th_01")
    L_xy = row.get("L_01")
    U_xy = row.get("U_01")
    T_yz = row.get("Th_12")
    L_yz = row.get("L_12")
    U_yz = row.get("U_12")
    T_zx = row.get("Th_20")

    required = [L_zx, U_zx, T_xy, L_xy, U_xy, T_yz, L_yz, U_yz, T_zx]
    if any(value is None for value in required):
        return False

    return (
        L_zx < T_xy < U_zx
        and T_xy > 0
        and 0 < L_zx < U_zx
        and L_xy < T_yz < U_xy
        and T_yz > 0
        and 0 < L_xy < U_xy
        and L_yz < T_zx < U_yz
        and T_zx > 0
        and 0 < L_yz < U_yz
    )


def _check_ex1_ls(row: Dict[str, str]) -> bool:
    L = _parse_matrix(row.get("L_hat", ""))
    U = _parse_matrix(row.get("U_hat", ""))
    Th = _parse_matrix(row.get("Th_hat", ""))
    if L.size == 0 or U.size == 0 or Th.size == 0:
        return False
    L_yx, U_yx, T_xy = L[1, 0], U[1, 0], Th[0, 1]
    L_xy, U_xy, T_yx = L[0, 1], U[0, 1], Th[1, 0]
    return (
        L_yx < T_xy < U_yx
        and T_xy > 0
        and 0 < L_yx < U_yx
        and L_xy < T_yx < U_xy
        and T_yx > 0
        and 0 < L_xy < U_xy
    )


def _check_ex2_ls(row: Dict[str, str]) -> bool:
    L = _parse_matrix(row.get("L_hat", ""))
    U = _parse_matrix(row.get("U_hat", ""))
    Th = _parse_matrix(row.get("Th_hat", ""))
    if L.size == 0 or U.size == 0 or Th.size == 0:
        return False
    L_xx, U_xx, T_xx = L[0, 0], U[0, 0], Th[0, 0]
    L_yx, U_yx = L[1, 0], U[1, 0]
    L_xy, U_xy = L[0, 1], U[0, 1]
    L_yy, U_yy = L[1, 1], U[1, 1]
    T_xy, T_yx, T_yy = Th[0, 1], Th[1, 0], Th[1, 1]
    return (
        (L_xx + L_yx) < T_xy
        and T_xx < (U_xx + L_yx)
        and T_xy < (L_xx + U_yx) < T_xx
        and T_xx < (U_xx + U_yx)
        and 0 < T_xy < T_xx
        and 0 < L_xx < U_xx
        and 0 < L_yx < U_yx
        and (L_xy * L_yy) < T_yx
        and T_yx < (U_xy * L_yy)
        and (U_xy * L_yy) < T_yy
        and T_yy < (L_xy * U_yy)
        and T_yy < (U_xy * U_yy)
        and 0 < T_yx < T_yy
        and 0 < L_xy < U_xy
        and 0 < L_yy < U_yy
    )


def _check_ex3_ls(row: Dict[str, str]) -> bool:
    L = _parse_matrix(row.get("L_hat", ""))
    U = _parse_matrix(row.get("U_hat", ""))
    Th = _parse_matrix(row.get("Th_hat", ""))
    if L.size == 0 or U.size == 0 or Th.size == 0:
        return False
    L_zx, U_zx, T_xy = L[2, 0], U[2, 0], Th[0, 1]
    L_xy, U_xy, T_yz = L[0, 1], U[0, 1], Th[1, 2]
    L_yz, U_yz, T_zx = L[1, 2], U[1, 2], Th[2, 0]
    return (
        L_zx < T_xy < U_zx
        and T_xy > 0
        and 0 < L_zx < U_zx
        and L_xy < T_yz < U_xy
        and T_yz > 0
        and 0 < L_xy < U_xy
        and L_yz < T_zx < U_yz
        and T_zx > 0
        and 0 < L_yz < U_yz
    )


def _summarize(rows: Iterable[Dict[str, Optional[float]]], checker) -> Tuple[List[SummaryRow], List[Dict[str, Optional[float]]]]:
    grouped: Dict[float, List[bool]] = {}
    enriched_rows: List[Dict[str, Optional[float]]] = []

    for row in rows:
        noise = row.get("noise_ub")
        if noise is None:
            continue
        match = bool(checker(row))
        grouped.setdefault(noise, []).append(match)
        enriched = dict(row)
        enriched["inequality_match"] = float(match)
        enriched_rows.append(enriched)

    summary: List[SummaryRow] = []
    for noise, matches in sorted(grouped.items()):
        total = len(matches)
        match_count = sum(matches)
        summary.append(SummaryRow(noise, total, match_count, total - match_count))
    return summary, enriched_rows


def _summarize_ls(
    rows: Iterable[Dict[str, str]],
    checker,
    exercise: str,
) -> Tuple[List[SummaryRow], List[Dict[str, str]]]:
    grouped: Dict[float, List[bool]] = {}
    enriched_rows: List[Dict[str, str]] = []

    for row in rows:
        if row.get("exercise", "").strip().lower() != exercise:
            continue
        if row.get("data_type", "").strip().lower() != "hill":
            continue
        noise = _to_float(row.get("noise_ub", ""))
        if noise is None:
            continue
        match = bool(checker(row))
        grouped.setdefault(noise, []).append(match)
        enriched = dict(row)
        enriched["inequality_match"] = "1" if match else "0"
        enriched_rows.append(enriched)

    summary: List[SummaryRow] = []
    for noise, matches in sorted(grouped.items()):
        total = len(matches)
        match_count = sum(matches)
        summary.append(SummaryRow(noise, total, match_count, total - match_count))
    return summary, enriched_rows


def _summarize_from_matches(rows: Iterable[Dict[str, str]], match_key: str) -> List[SummaryRow]:
    grouped: Dict[float, List[bool]] = {}
    for row in rows:
        noise = _to_float(row.get("noise_ub", ""))
        if noise is None:
            continue
        match_raw = row.get(match_key, "")
        match = str(match_raw).strip() in {"1", "True", "true", "yes"}
        grouped.setdefault(noise, []).append(match)
    summary: List[SummaryRow] = []
    for noise, matches in sorted(grouped.items()):
        total = len(matches)
        match_count = sum(matches)
        summary.append(SummaryRow(noise, total, match_count, total - match_count))
    return summary


def _write_raw_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_csv(path: Path, rows: List[Dict[str, Optional[float]]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_float(value: Optional[float], digits: int = 6) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.{digits}g}"


def _success_by_noise_pinn(rows: List[Dict[str, Optional[float]]]) -> Dict[str, Tuple[int, int]]:
    grouped: Dict[str, List[bool]] = {"ALL": []}
    for row in rows:
        noise = row.get("noise_ub")
        if noise is None:
            continue
        match = bool(row.get("inequality_match"))
        grouped["ALL"].append(match)
        grouped.setdefault(f"{float(noise):g}", []).append(match)
    return {noise: (sum(matches), len(matches)) for noise, matches in grouped.items()}


def _success_by_noise_ls(rows: List[Dict[str, str]]) -> Dict[str, Tuple[int, int]]:
    grouped: Dict[str, List[bool]] = {"ALL": []}
    for row in rows:
        noise = _to_float(row.get("noise_ub", ""))
        if noise is None:
            continue
        match = row.get("inequality_match") == "1"
        grouped["ALL"].append(match)
        grouped.setdefault(f"{float(noise):g}", []).append(match)
    return {noise: (sum(matches), len(matches)) for noise, matches in grouped.items()}


def _summary_stat_rows(
    exercise: str,
    method: str,
    grouped_values: Dict[str, Dict[str, Dict[float, List[float]]]],
    true_values: Dict[str, Dict[str, float]],
    success_by_noise: Dict[str, Tuple[int, int]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    noise_values = sorted(
        {
            noise
            for by_edge in grouped_values.values()
            for grouped in by_edge.values()
            for noise in grouped
        }
    )
    noise_labels = ["ALL", *[f"{float(noise):g}" for noise in noise_values]]

    for param in ["L", "U", "Theta"]:
        for edge in sorted(grouped_values.get(param, {})):
            grouped = grouped_values[param][edge]
            for noise_label in noise_labels:
                if noise_label == "ALL":
                    values = [value for items in grouped.values() for value in items]
                else:
                    values = grouped.get(float(noise_label), [])
                finite = _finite_values(values)
                n = len(finite)
                mean = float(np.mean(finite)) if finite else None
                std = float(np.std(finite, ddof=1)) if n > 1 else 0.0 if n == 1 else None
                success, total = success_by_noise.get(noise_label, (0, 0))
                success_rate = success / total if total else None
                fail_rate = 1.0 - success_rate if success_rate is not None else None
                rows.append(
                    {
                        "exercise": exercise,
                        "method": method,
                        "noise_ub": noise_label,
                        "parameter": param,
                        "edge": edge,
                        "true_value": _format_float(true_values.get(param, {}).get(edge)),
                        "mean": _format_float(mean),
                        "std": _format_float(std),
                        "n": str(n),
                        "region_success_rate": _format_float(success_rate),
                        "region_fail_rate": _format_float(fail_rate),
                        "region_success_count": str(success),
                        "region_total": str(total),
                    }
                )
    return rows


def _write_parameter_summary_tables(
    path: Path,
    exercise_edges: Dict[str, List[str]],
    pinn_grouped_by_exercise: Dict[str, Dict[str, Dict[float, List[float]]]],
    ls_grouped_by_exercise: Dict[str, Dict[str, Dict[float, List[float]]]],
    true_values_by_exercise: Dict[str, Dict[str, Dict[str, float]]],
    pinn_enriched_by_exercise: Dict[str, List[Dict[str, Optional[float]]]],
    ls_enriched_by_exercise: Dict[str, List[Dict[str, str]]],
) -> None:
    header = [
        "exercise",
        "method",
        "noise_ub",
        "parameter",
        "edge",
        "true_value",
        "mean",
        "std",
        "n",
        "region_success_rate",
        "region_fail_rate",
        "region_success_count",
        "region_total",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        for index, exercise in enumerate(exercise_edges):
            if index:
                handle.write("\n")
            handle.write(f"{exercise} parameter summary\n")
            writer.writeheader()
            for row in _summary_stat_rows(
                exercise,
                "PINN",
                pinn_grouped_by_exercise.get(exercise, {}),
                true_values_by_exercise.get(exercise, {}),
                _success_by_noise_pinn(pinn_enriched_by_exercise.get(exercise, [])),
            ):
                writer.writerow(row)
            for row in _summary_stat_rows(
                exercise,
                LS_METHOD_LABEL,
                ls_grouped_by_exercise.get(exercise, {}),
                true_values_by_exercise.get(exercise, {}),
                _success_by_noise_ls(ls_enriched_by_exercise.get(exercise, [])),
            ):
                writer.writerow(row)


def _stats_for_group(
    grouped_values: Dict[str, Dict[str, Dict[float, List[float]]]],
    param: str,
    edge: str,
    noise_label: str,
) -> Tuple[str, str, str]:
    grouped = grouped_values.get(param, {}).get(edge, {})
    if noise_label == "ALL":
        values = [value for items in grouped.values() for value in items]
    else:
        values = grouped.get(float(noise_label), [])
    finite = _finite_values(values)
    if not finite:
        return "", "", "0"
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return _format_float(mean), _format_float(std), str(len(finite))


def _rate_text(success_by_noise: Dict[str, Tuple[int, int]], noise_label: str) -> Tuple[str, str]:
    success, total = success_by_noise.get(noise_label, (0, 0))
    if not total:
        return "", "0/0"
    return _format_float(success / total, digits=4), f"{success}/{total}"


def _rate_class(rate: str) -> str:
    if not rate:
        return ""
    value = float(rate)
    if value >= 0.95:
        return "good"
    if value >= 0.8:
        return "warn"
    return "bad"


def _noise_labels_for_exercise(
    exercise: str,
    pinn_grouped_by_exercise: Dict[str, Dict[str, Dict[float, List[float]]]],
    ls_grouped_by_exercise: Dict[str, Dict[str, Dict[float, List[float]]]],
) -> List[str]:
    noises = set()
    for source in (pinn_grouped_by_exercise, ls_grouped_by_exercise):
        for by_edge in source.get(exercise, {}).values():
            for grouped in by_edge.values():
                noises.update(grouped)
    return ["ALL", *[f"{float(noise):g}" for noise in sorted(noises)]]


def _mean_and_std(values: Iterable[float]) -> Tuple[Optional[float], Optional[float]]:
    finite = _finite_values(values)
    if not finite:
        return None, None
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return mean, std


def _comparison_label(
    pinn_value: Optional[float],
    ls_value: Optional[float],
    *,
    lower_is_better: bool,
) -> str:
    if pinn_value is None or ls_value is None:
        return ""
    if np.isclose(pinn_value, ls_value, rtol=1e-9, atol=1e-12):
        return "Tie"
    pinn_wins = pinn_value < ls_value if lower_is_better else pinn_value > ls_value
    return "PINN" if pinn_wins else LS_METHOD_LABEL


def _accuracy_std_snapshot_rows(
    exercise_edges: Dict[str, List[str]],
    pinn_grouped_by_exercise: Dict[str, Dict[str, Dict[float, List[float]]]],
    ls_grouped_by_exercise: Dict[str, Dict[str, Dict[float, List[float]]]],
    true_values_by_exercise: Dict[str, Dict[str, Dict[str, float]]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for exercise, edges in exercise_edges.items():
        pinn_noises = {
            noise
            for by_edge in pinn_grouped_by_exercise.get(exercise, {}).values()
            for grouped in by_edge.values()
            for noise, values in grouped.items()
            if _finite_values(values)
        }
        ls_noises = {
            noise
            for by_edge in ls_grouped_by_exercise.get(exercise, {}).values()
            for grouped in by_edge.values()
            for noise, values in grouped.items()
            if _finite_values(values)
        }
        available_noises = sorted(pinn_noises & ls_noises)
        snapshot_noises = [noise for noise in (0.0, 50.0) if noise in available_noises]
        if not snapshot_noises:
            snapshot_noises = available_noises

        for noise in snapshot_noises:
            pinn_errors: List[float] = []
            ls_errors: List[float] = []
            pinn_stds: List[float] = []
            ls_stds: List[float] = []

            for param in ["L", "U", "Theta"]:
                for edge in edges:
                    true_value = true_values_by_exercise.get(exercise, {}).get(param, {}).get(edge)
                    if true_value is None:
                        continue

                    pinn_values = (
                        pinn_grouped_by_exercise.get(exercise, {})
                        .get(param, {})
                        .get(edge, {})
                        .get(noise, [])
                    )
                    ls_values = (
                        ls_grouped_by_exercise.get(exercise, {})
                        .get(param, {})
                        .get(edge, {})
                        .get(noise, [])
                    )
                    pinn_mean, pinn_std = _mean_and_std(pinn_values)
                    ls_mean, ls_std = _mean_and_std(ls_values)
                    if pinn_mean is None or pinn_std is None or ls_mean is None or ls_std is None:
                        continue

                    pinn_errors.append(abs(pinn_mean - true_value))
                    ls_errors.append(abs(ls_mean - true_value))
                    pinn_stds.append(pinn_std)
                    ls_stds.append(ls_std)

            if not pinn_errors or not ls_errors:
                continue

            pinn_mae = float(np.mean(pinn_errors)) if pinn_errors else None
            ls_mae = float(np.mean(ls_errors)) if ls_errors else None
            pinn_avg_std = float(np.mean(pinn_stds)) if pinn_stds else None
            ls_avg_std = float(np.mean(ls_stds)) if ls_stds else None
            error_gap = (
                float(pinn_mae - ls_mae)
                if pinn_mae is not None and ls_mae is not None
                else None
            )
            std_gap = (
                float(pinn_avg_std - ls_avg_std)
                if pinn_avg_std is not None and ls_avg_std is not None
                else None
            )

            rows.append(
                {
                    "exercise": _exercise_label(exercise),
                    "noise": _noise_tick_label(noise),
                    "parameter_cells": str(len(pinn_errors)),
                    "pinn_mae": _format_float(pinn_mae),
                    "ls_mae": _format_float(ls_mae),
                    "error_gap": _format_float(error_gap),
                    "more_accurate": _comparison_label(pinn_mae, ls_mae, lower_is_better=True),
                    "pinn_avg_std": _format_float(pinn_avg_std),
                    "ls_avg_std": _format_float(ls_avg_std),
                    "std_gap": _format_float(std_gap),
                    "higher_std": _comparison_label(pinn_avg_std, ls_avg_std, lower_is_better=False),
                }
            )
    return rows


def _example_manifest_rows(configs: Iterable[ExampleConfig]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for config in configs:
        rows.append(
            {
                "exercise": config.key,
                "base_exercise": config.base,
                "variant": config.variant or "standard",
                "label": config.label,
                "directory": str(config.directory.relative_to(ROOT)),
                "results_csv": str(config.result_path.relative_to(ROOT)),
            }
        )
    return rows


def _mean_abs_error_for_noise(
    exercise: str,
    edges: List[str],
    noise_label: str,
    pinn_grouped_by_exercise: Dict[str, Dict[str, Dict[float, List[float]]]],
    true_values_by_exercise: Dict[str, Dict[str, Dict[str, float]]],
) -> Tuple[int, Optional[float], Optional[float]]:
    abs_errors: List[float] = []
    std_values: List[float] = []
    exercise_values = pinn_grouped_by_exercise.get(exercise, {})
    true_values = true_values_by_exercise.get(exercise, {})

    for param in ["L", "U", "Theta"]:
        for edge in edges:
            true_value = true_values.get(param, {}).get(edge)
            grouped = exercise_values.get(param, {}).get(edge, {})
            if noise_label == "ALL":
                values = [value for items in grouped.values() for value in items]
            else:
                values = grouped.get(float(noise_label), [])
            finite = _finite_values(values)
            if not finite or true_value is None:
                continue
            mean = float(np.mean(finite))
            std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
            abs_errors.append(abs(mean - true_value))
            std_values.append(std)

    if not abs_errors:
        return 0, None, None
    return len(abs_errors), float(np.mean(abs_errors)), float(np.mean(std_values))


def _pinn_example_comparison_rows(
    configs: Iterable[ExampleConfig],
    exercise_edges: Dict[str, List[str]],
    summaries_by_exercise: Dict[str, List[SummaryRow]],
    pinn_grouped_by_exercise: Dict[str, Dict[str, Dict[float, List[float]]]],
    true_values_by_exercise: Dict[str, Dict[str, Dict[str, float]]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for config in configs:
        summary = summaries_by_exercise.get(config.key, [])
        if not summary:
            continue

        summary_by_label = {
            _noise_tick_label(item.noise_ub): item
            for item in summary
        }
        total = sum(item.total for item in summary)
        matches = sum(item.matches for item in summary)
        summary_by_label["ALL"] = SummaryRow(
            noise_ub=float("nan"),
            total=total,
            matches=matches,
            fails=total - matches,
        )

        noise_labels = ["ALL", *[_noise_tick_label(item.noise_ub) for item in summary]]
        for noise_label in noise_labels:
            item = summary_by_label[noise_label]
            param_cells, mean_abs_error, avg_std = _mean_abs_error_for_noise(
                config.key,
                exercise_edges.get(config.key, []),
                noise_label,
                pinn_grouped_by_exercise,
                true_values_by_exercise,
            )
            match_rate = item.matches / item.total if item.total else None
            rows.append(
                {
                    "base_exercise": config.base,
                    "exercise": config.key,
                    "variant": config.variant or "standard",
                    "label": config.label,
                    "noise_ub": noise_label,
                    "total": str(item.total),
                    "matches": str(item.matches),
                    "fails": str(item.fails),
                    "match_rate": _format_float(match_rate, digits=4),
                    "parameter_cells": str(param_cells),
                    "mean_abs_parameter_error": _format_float(mean_abs_error),
                    "avg_parameter_std": _format_float(avg_std),
                    "results_csv": str(config.result_path.relative_to(ROOT)),
                }
            )
    return rows


def _plot_pinn_example_match_rates_by_base(
    configs: List[ExampleConfig],
    summaries_by_exercise: Dict[str, List[SummaryRow]],
    output_path: Path,
) -> None:
    grouped_configs: Dict[str, List[ExampleConfig]] = {}
    for config in configs:
        grouped_configs.setdefault(config.base, []).append(config)
    if not grouped_configs:
        return

    fig, axes = plt.subplots(
        len(grouped_configs),
        1,
        figsize=(8.2, 3.2 * len(grouped_configs)),
        sharex=True,
        sharey=True,
    )
    if len(grouped_configs) == 1:
        axes = [axes]

    colors = (PINN_COLOR, LS_COLOR, "#009E73", "#CC79A7", "#56B4E9", "#F0E442")
    for ax, (base, base_configs) in zip(axes, grouped_configs.items()):
        for index, config in enumerate(base_configs):
            summary = summaries_by_exercise.get(config.key, [])
            if not summary:
                continue
            xs = [item.noise_ub for item in summary]
            rates = [item.matches / item.total if item.total else 0 for item in summary]
            ax.plot(
                xs,
                rates,
                marker="o",
                markersize=5.4,
                markeredgecolor="white",
                markeredgewidth=0.8,
                linewidth=2.0,
                label=_variant_label(config.variant) or "standard",
                color=colors[index % len(colors)],
            )
        ax.set_title(f"{EXERCISE_LABELS.get(base, base)} variants", pad=10)
        ax.set_ylabel("Match rate")
        ax.set_ylim(0, 1.05)
        _style_axis(ax)
        ax.legend(frameon=False, loc="lower right")

    axes[-1].set_xlabel("Noise upper level")
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def _plot_pinn_example_error_by_base(
    comparison_rows: List[Dict[str, str]],
    output_path: Path,
) -> None:
    grouped: Dict[str, Dict[str, List[Tuple[float, float, str]]]] = {}
    for row in comparison_rows:
        if row["noise_ub"] == "ALL" or not row["mean_abs_parameter_error"]:
            continue
        grouped.setdefault(row["base_exercise"], {}).setdefault(row["exercise"], []).append(
            (
                float(row["noise_ub"]),
                float(row["mean_abs_parameter_error"]),
                row["variant"],
            )
        )
    if not grouped:
        return

    fig, axes = plt.subplots(
        len(grouped),
        1,
        figsize=(8.2, 3.2 * len(grouped)),
        sharex=True,
    )
    if len(grouped) == 1:
        axes = [axes]

    colors = (PINN_COLOR, LS_COLOR, "#009E73", "#CC79A7", "#56B4E9", "#F0E442")
    for ax, (base, by_exercise) in zip(axes, grouped.items()):
        for index, (exercise, values) in enumerate(by_exercise.items()):
            values = sorted(values)
            xs = [item[0] for item in values]
            errors = [item[1] for item in values]
            variant = values[0][2]
            ax.plot(
                xs,
                errors,
                marker="o",
                markersize=5.4,
                markeredgecolor="white",
                markeredgewidth=0.8,
                linewidth=2.0,
                label=_variant_label("" if variant == "standard" else variant) or "standard",
                color=colors[index % len(colors)],
            )
        ax.set_title(f"{EXERCISE_LABELS.get(base, base)} variants", pad=10)
        ax.set_ylabel("Mean abs. error")
        _style_axis(ax)
        ax.legend(frameon=False, loc="upper left")

    axes[-1].set_xlabel("Noise upper level")
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def _write_comparison_html(
    path: Path,
    exercise_edges: Dict[str, List[str]],
    pinn_grouped_by_exercise: Dict[str, Dict[str, Dict[float, List[float]]]],
    ls_grouped_by_exercise: Dict[str, Dict[str, Dict[float, List[float]]]],
    true_values_by_exercise: Dict[str, Dict[str, Dict[str, float]]],
    pinn_enriched_by_exercise: Dict[str, List[Dict[str, Optional[float]]]],
    ls_enriched_by_exercise: Dict[str, List[Dict[str, str]]],
) -> None:
    style = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:32px;color:#172033}
h1{margin-bottom:4px}h2{margin-top:34px;padding-top:20px;border-top:2px solid #d9e0ea}
h3{margin:22px 0 8px;color:#5b667a;text-transform:uppercase;font-size:13px;letter-spacing:.05em}
.sub{color:#5b667a;margin-bottom:20px}.cards{display:grid;grid-template-columns:repeat(3,minmax(170px,1fr));gap:12px;max-width:900px}
.card{background:#f5f7fb;border:1px solid #d9e0ea;border-radius:8px;padding:12px}.label{font-size:12px;color:#5b667a;text-transform:uppercase}
.value{font-size:24px;font-weight:750}.pinn{color:#1f77b4;font-weight:650}.ls{color:#ff7f0e;font-weight:650}
table{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:24px}th,td{border:1px solid #d9e0ea;padding:6px 8px;text-align:right;white-space:nowrap}
th{background:#eef3f8}td:first-child,td:nth-child(2),th:first-child,th:nth-child(2){text-align:left}.good{background:#e8f5e9}.warn{background:#fff2cc}.bad{background:#fce4e4}
"""
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>PINN vs {html.escape(LS_METHOD_LABEL)} Parameter Comparison</title>",
        f"<style>{style}</style></head><body>",
        f"<h1>PINN vs {html.escape(LS_METHOD_LABEL)} Parameter Comparison</h1>",
        "<div class='sub'>Side-by-side mean, standard deviation, true value, and right-region success rates.</div>",
        f"<div class='sub'>{html.escape(LS_METHOD_SHORT)} source CSV: <code>{html.escape(str(LEAST_SQUARES_RESULTS.relative_to(ROOT)))}</code></div>",
    ]

    snapshot_rows = _accuracy_std_snapshot_rows(
        exercise_edges,
        pinn_grouped_by_exercise,
        ls_grouped_by_exercise,
        true_values_by_exercise,
    )
    parts.extend(
        [
            "<h2>Accuracy and standard deviation snapshot</h2>",
            f"<div class='sub'>Mean absolute error and standard deviation are averaged over all parameter-edge cells. Lower error is more accurate. The gap columns are PINN minus {html.escape(LS_METHOD_SHORT)}.</div>",
            "<table><thead><tr><th>Exercise</th><th>Noise UB</th><th>Parameters</th>"
            f"<th>PINN Avg. Abs. Error</th><th>{html.escape(LS_METHOD_SHORT)} Avg. Abs. Error</th><th>Error Gap</th><th>More Accurate</th>"
            f"<th>PINN Avg. Std.</th><th>{html.escape(LS_METHOD_SHORT)} Avg. Std.</th><th>Std. Gap</th><th>Higher Std.</th></tr></thead><tbody>",
        ]
    )
    for row in snapshot_rows:
        more_accurate_class = "pinn" if row["more_accurate"] == "PINN" else "ls" if row["more_accurate"] == LS_METHOD_LABEL else ""
        higher_std_class = "pinn" if row["higher_std"] == "PINN" else "ls" if row["higher_std"] == LS_METHOD_LABEL else ""
        parts.append(
            "<tr>"
            f"<td>{html.escape(row['exercise'])}</td>"
            f"<td>{html.escape(row['noise'])}</td>"
            f"<td>{html.escape(row['parameter_cells'])}</td>"
            f"<td>{html.escape(row['pinn_mae'])}</td>"
            f"<td>{html.escape(row['ls_mae'])}</td>"
            f"<td>{html.escape(row['error_gap'])}</td>"
            f"<td class='{more_accurate_class}'>{html.escape(row['more_accurate'])}</td>"
            f"<td>{html.escape(row['pinn_avg_std'])}</td>"
            f"<td>{html.escape(row['ls_avg_std'])}</td>"
            f"<td>{html.escape(row['std_gap'])}</td>"
            f"<td class='{higher_std_class}'>{html.escape(row['higher_std'])}</td>"
            "</tr>"
        )
    parts.append("</tbody></table>")

    for exercise, edges in exercise_edges.items():
        pinn_success = _success_by_noise_pinn(pinn_enriched_by_exercise.get(exercise, []))
        ls_success = _success_by_noise_ls(ls_enriched_by_exercise.get(exercise, []))
        pinn_all, pinn_all_count = _rate_text(pinn_success, "ALL")
        ls_all, ls_all_count = _rate_text(ls_success, "ALL")
        gap = _format_float(float(pinn_all) - float(ls_all), digits=4) if pinn_all and ls_all else ""
        noise_labels = _noise_labels_for_exercise(exercise, pinn_grouped_by_exercise, ls_grouped_by_exercise)

        parts.extend(
            [
                f"<h2>{html.escape(_exercise_label(exercise))}</h2>",
                "<div class='cards'>",
                f"<div class='card'><div class='label'>PINN right-region</div><div class='value pinn'>{html.escape(pinn_all)}</div><div>{html.escape(pinn_all_count)}</div></div>",
                f"<div class='card'><div class='label'>{html.escape(LS_METHOD_SHORT)} right-region</div><div class='value ls'>{html.escape(ls_all)}</div><div>{html.escape(ls_all_count)}</div></div>",
                f"<div class='card'><div class='label'>PINN - {html.escape(LS_METHOD_SHORT)} gap</div><div class='value'>{html.escape(gap)}</div><div>positive favors PINN</div></div>",
                "</div>",
                "<h3>Right-region success by noise</h3>",
                f"<table><thead><tr><th>Noise UB</th><th>PINN Rate</th><th>PINN Count</th><th>{html.escape(LS_METHOD_SHORT)} Rate</th><th>{html.escape(LS_METHOD_SHORT)} Count</th><th>Gap</th></tr></thead><tbody>",
            ]
        )
        for noise_label in noise_labels:
            pinn_rate, pinn_count = _rate_text(pinn_success, noise_label)
            ls_rate, ls_count = _rate_text(ls_success, noise_label)
            rate_gap = _format_float(float(pinn_rate) - float(ls_rate), digits=4) if pinn_rate and ls_rate else ""
            parts.append(
                "<tr>"
                f"<td>{html.escape(noise_label)}</td>"
                f"<td class='{_rate_class(pinn_rate)}'>{html.escape(pinn_rate)}</td>"
                f"<td>{html.escape(pinn_count)}</td>"
                f"<td class='{_rate_class(ls_rate)}'>{html.escape(ls_rate)}</td>"
                f"<td>{html.escape(ls_count)}</td>"
                f"<td>{html.escape(rate_gap)}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")

        parts.append(
            "<h3>Parameter statistics</h3>"
            "<table><thead><tr><th>Noise UB</th><th>Parameter</th><th>Edge</th><th>True</th>"
            "<th>PINN Mean</th><th>PINN Std.</th><th>PINN n</th>"
            f"<th>{html.escape(LS_METHOD_SHORT)} Mean</th><th>{html.escape(LS_METHOD_SHORT)} Std.</th><th>{html.escape(LS_METHOD_SHORT)} n</th><th>Mean Gap</th></tr></thead><tbody>"
        )
        for noise_label in noise_labels:
            for param in ["L", "U", "Theta"]:
                for edge in edges:
                    true_value = true_values_by_exercise.get(exercise, {}).get(param, {}).get(edge)
                    pinn_mean, pinn_std, pinn_n = _stats_for_group(
                        pinn_grouped_by_exercise.get(exercise, {}), param, edge, noise_label
                    )
                    ls_mean, ls_std, ls_n = _stats_for_group(
                        ls_grouped_by_exercise.get(exercise, {}), param, edge, noise_label
                    )
                    mean_gap = _format_float(float(pinn_mean) - float(ls_mean)) if pinn_mean and ls_mean else ""
                    parts.append(
                        "<tr>"
                        f"<td>{html.escape(noise_label)}</td><td>{html.escape(param)}</td><td>{html.escape(edge)}</td>"
                        f"<td>{html.escape(_format_float(true_value))}</td>"
                        f"<td class='pinn'>{html.escape(pinn_mean)}</td><td>{html.escape(pinn_std)}</td><td>{html.escape(pinn_n)}</td>"
                        f"<td class='ls'>{html.escape(ls_mean)}</td><td>{html.escape(ls_std)}</td><td>{html.escape(ls_n)}</td>"
                        f"<td>{html.escape(mean_gap)}</td>"
                        "</tr>"
                    )
        parts.append("</tbody></table>")

    parts.append("</body></html>")
    path.write_text("\n".join(parts))


def _plot_summary(summary: List[SummaryRow], title: str, output_path: Path) -> None:
    if not summary:
        return
    labels = [_noise_tick_label(item.noise_ub) for item in summary]
    matches = [item.matches for item in summary]
    fails = [item.fails for item in summary]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar(labels, matches, label="Match", color="#009E73", edgecolor="white", linewidth=0.8)
    ax.bar(labels, fails, bottom=matches, label="Fail", color="#CC6677", edgecolor="white", linewidth=0.8)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Noise upper level")
    ax.set_ylabel("Count")
    _style_axis(ax)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def _plot_match_rate(
    summaries_by_exercise: Dict[str, List[SummaryRow]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    colors = (PINN_COLOR, LS_COLOR, "#009E73", "#CC79A7", "#56B4E9", "#F0E442")
    for index, (exercise, summary) in enumerate(summaries_by_exercise.items()):
        if not summary:
            continue
        color = colors[index % len(colors)]
        xs = [item.noise_ub for item in summary]
        rates = [item.matches / item.total if item.total else 0 for item in summary]
        ax.plot(
            xs,
            rates,
            marker="o",
            markersize=5.8,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=2.0,
            label=_exercise_label(exercise),
            color=color,
        )

    ax.set_xlabel("Noise upper level")
    ax.set_ylabel("Match rate")
    ax.set_title("Inequality match rate by noise upper level", pad=12)
    ax.set_ylim(0, 1.05)
    _style_axis(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def _plot_comparison_rate(
    pinn: List[SummaryRow],
    least_squares: List[SummaryRow],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    if pinn:
        xs = [item.noise_ub for item in pinn]
        rates = [item.matches / item.total if item.total else 0 for item in pinn]
        ax.plot(
            xs,
            rates,
            marker="o",
            markersize=5.8,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=2.0,
            label="PINN",
            color=PINN_COLOR,
        )
    if least_squares:
        xs = [item.noise_ub for item in least_squares]
        rates = [item.matches / item.total if item.total else 0 for item in least_squares]
        ax.plot(
            xs,
            rates,
            marker="s",
            markersize=5.4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=2.0,
            label=LS_METHOD_SHORT,
            color=LS_COLOR,
        )
    ax.set_xlabel("Noise upper level")
    ax.set_ylabel("Match rate")
    ax.set_title(title, pad=12)
    ax.set_ylim(0, 1.05)
    _style_axis(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


EDGE_MAP = {
    "ex1": [(0, 1), (1, 0)],
    "ex2": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "ex3": [(0, 1), (1, 2), (2, 0)],
}


def _extract_matrix_values(matrix: np.ndarray, edges: List[Tuple[int, int]]) -> List[float]:
    if matrix.ndim != 2:
        return []
    values: List[float] = []
    for i, j in edges:
        if i < matrix.shape[0] and j < matrix.shape[1]:
            values.append(float(matrix[i, j]))
    return values


def _collect_pinn_values(row: Dict[str, Optional[float]], keys: List[str]) -> List[float]:
    values: List[float] = []
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        values.append(float(value))
    return values


def _finite_values(values: Iterable[float]) -> List[float]:
    return [float(value) for value in values if np.isfinite(value)]


def _robust_ylim(values: Iterable[float]) -> Tuple[float, float]:
    finite = _finite_values(values)
    if not finite:
        return 0.0, 1.0

    arr = np.array(finite, dtype=float)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if lo == hi:
        center = float(lo)
        pad = max(abs(center) * 0.12, 0.25)
        return center - pad, center + pad

    span = hi - lo
    pad = max(0.18 * span, 0.05 * max(abs(lo), abs(hi), 1.0))
    y_min = lo - pad
    y_max = hi + pad
    if y_min > 0 and lo > 0:
        y_min = max(0.0, y_min)
    return float(y_min), float(y_max)


def _boxplot_visible_values(
    grouped: Dict[float, List[float]],
    noises: List[float],
) -> List[float]:
    visible: List[float] = []
    for noise in noises:
        finite = _finite_values(grouped.get(noise, []))
        if not finite:
            continue
        arr = np.array(finite, dtype=float)
        q1, median, q3 = np.percentile(arr, [25, 50, 75])
        iqr = q3 - q1
        if iqr > 0:
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            whisker_low = float(np.min(arr[arr >= lower_fence]))
            whisker_high = float(np.max(arr[arr <= upper_fence]))
        else:
            whisker_low = float(np.min(arr))
            whisker_high = float(np.max(arr))
        visible.extend([float(q1), float(median), float(q3), whisker_low, whisker_high])
    return visible


def _boxplot_ylim(
    pinn_grouped: Dict[float, List[float]],
    ls_grouped: Dict[float, List[float]],
    noises: List[float],
    true_value: Optional[float],
) -> Tuple[float, float]:
    visible = [
        *_boxplot_visible_values(pinn_grouped, noises),
        *_boxplot_visible_values(ls_grouped, noises),
    ]
    if true_value is not None:
        visible.append(true_value)
    finite = _finite_values(visible)
    if not finite:
        return 0.0, 1.0
    y_min = min(finite)
    y_max = max(finite)
    if y_min == y_max:
        pad = max(abs(y_min) * 0.12, 0.05)
    else:
        pad = max((y_max - y_min) * 0.18, 0.035 * max(abs(y_min), abs(y_max), 1.0))
    y_min -= pad
    y_max += pad
    if y_min > 0 and min(finite) > 0:
        y_min = max(0.0, y_min)
    return float(y_min), float(y_max)


def _values_for_param(
    param: str,
    edge: str,
    pinn_values: Dict[str, Dict[str, Dict[float, List[float]]]],
    ls_values: Dict[str, Dict[str, Dict[float, List[float]]]],
) -> List[float]:
    values: List[float] = []
    for grouped in (
        pinn_values.get(param, {}).get(edge, {}),
        ls_values.get(param, {}).get(edge, {}),
    ):
        for items in grouped.values():
            values.extend(items)
    return values


def _collect_true_values(
    rows: Iterable[Dict[str, str]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    field_map = {
        "L": "L_true",
        "U": "U_true",
        "Theta": "Th_true",
    }
    collected: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    for row in rows:
        exercise = row.get("exercise", "").strip().lower()
        if row.get("data_type", "").strip().lower() != "hill":
            continue
        edges = EDGE_MAP.get(exercise, [])
        if not edges:
            continue
        for param, field in field_map.items():
            matrix = _parse_matrix(row.get(field, ""))
            if matrix.size == 0:
                continue
            for i, j in edges:
                edge = f"{i}{j}"
                values = _extract_matrix_values(matrix, [(i, j)])
                if not values:
                    continue
                collected.setdefault(exercise, {}).setdefault(param, {}).setdefault(edge, []).extend(values)

    true_values: Dict[str, Dict[str, Dict[str, float]]] = {}
    for exercise, by_param in collected.items():
        for param, by_edge in by_param.items():
            for edge, values in by_edge.items():
                finite = _finite_values(values)
                if finite:
                    true_values.setdefault(exercise, {}).setdefault(param, {})[edge] = float(np.median(finite))
    return true_values


def _collect_true_values_from_pinn_paths(
    paths_by_exercise: Dict[str, Path],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    field_map = {
        "L": "L_np",
        "U": "U_np",
        "Theta": "Th_np",
    }
    collected: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    for exercise, path in paths_by_exercise.items():
        if not path.exists():
            continue
        edges = EDGE_MAP.get(exercise, [])
        if not edges:
            continue
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for param, field in field_map.items():
                    matrix = _parse_matrix(row.get(field, ""))
                    if matrix.size == 0:
                        continue
                    for i, j in edges:
                        edge = f"{i}{j}"
                        values = _extract_matrix_values(matrix, [(i, j)])
                        if not values:
                            continue
                        collected.setdefault(exercise, {}).setdefault(param, {}).setdefault(edge, []).extend(values)

    true_values: Dict[str, Dict[str, Dict[str, float]]] = {}
    for exercise, by_param in collected.items():
        for param, by_edge in by_param.items():
            for edge, values in by_edge.items():
                finite = _finite_values(values)
                if finite:
                    true_values.setdefault(exercise, {}).setdefault(param, {})[edge] = float(np.median(finite))
    return true_values


def _merge_true_values(
    primary: Dict[str, Dict[str, Dict[str, float]]],
    fallback: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    merged = {
        exercise: {
            param: dict(by_edge)
            for param, by_edge in by_param.items()
        }
        for exercise, by_param in primary.items()
    }
    for exercise, by_param in fallback.items():
        for param, by_edge in by_param.items():
            for edge, value in by_edge.items():
                merged.setdefault(exercise, {}).setdefault(param, {}).setdefault(edge, value)
    return merged


def _style_boxplot(parts, color: str, fill: str) -> None:
    for box in parts.get("boxes", []):
        box.set(color=color, linewidth=1.45)
        box.set_facecolor(fill)
        box.set_alpha(0.86)
    for median in parts.get("medians", []):
        median.set(color="#111827", linewidth=1.9)
    for whisker in parts.get("whiskers", []):
        whisker.set(color=color, linewidth=1.15)
    for cap in parts.get("caps", []):
        cap.set(color=color, linewidth=1.15)
    for mean in parts.get("means", []):
        mean.set(markeredgecolor=color, markerfacecolor="white", markersize=4.2, markeredgewidth=1.0)


def _scatter_boxplot_points(
    ax,
    grouped: Dict[float, List[float]],
    noises: List[float],
    positions: np.ndarray,
    offset: float,
    color: str,
) -> None:
    rng = np.random.default_rng(12345)
    for position, noise in zip(positions, noises):
        values = _finite_values(grouped.get(noise, []))
        if not values:
            continue
        arr = np.array(values, dtype=float)
        jitter = rng.normal(0.0, 0.035, size=len(arr))
        jitter = np.clip(jitter, -0.085, 0.085)
        ax.scatter(
            np.full(len(arr), position + offset) + jitter,
            arr,
            s=8,
            color=color,
            alpha=0.18,
            linewidths=0,
            zorder=1,
        )


def _draw_parameter_boxplot_panel(
    ax,
    param: str,
    edge: str,
    pinn_grouped: Dict[float, List[float]],
    ls_grouped: Dict[float, List[float]],
    true_value: Optional[float],
    show_xlabel: bool,
    show_ylabel: bool,
    title: Optional[str] = None,
    show_title: bool = True,
) -> None:
    noises = sorted(set(pinn_grouped) | set(ls_grouped))
    label = _parameter_label(param, edge)

    if not noises:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        if show_title:
            ax.set_title(title or label)
        _style_axis(ax)
        return

    positions = np.arange(len(noises), dtype=float) * 1.28
    box_width = 0.27
    offset = 0.20
    common_box_kwargs = {
        "widths": box_width,
        "patch_artist": True,
        "showfliers": False,
        "showmeans": True,
        "manage_ticks": False,
        "medianprops": {"linewidth": 1.9, "color": "#111827"},
        "meanprops": {
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgewidth": 1.0,
            "markersize": 4.2,
        },
    }

    if pinn_grouped:
        pinn_data = [pinn_grouped.get(n, []) or [np.nan] for n in noises]
        parts = ax.boxplot(pinn_data, positions=positions - offset, **common_box_kwargs)
        _style_boxplot(parts, PINN_COLOR, PINN_FILL)
        _scatter_boxplot_points(ax, pinn_grouped, noises, positions, -offset, PINN_COLOR)
    if ls_grouped:
        ls_data = [ls_grouped.get(n, []) or [np.nan] for n in noises]
        parts = ax.boxplot(ls_data, positions=positions + offset, **common_box_kwargs)
        _style_boxplot(parts, LS_COLOR, LS_FILL)
        _scatter_boxplot_points(ax, ls_grouped, noises, positions, offset, LS_COLOR)

    if true_value is not None:
        ax.axhline(
            true_value,
            color=TRUE_COLOR,
            linestyle=(0, (5, 3)),
            linewidth=1.35,
            alpha=0.95,
            zorder=0,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([_noise_tick_label(n) for n in noises], rotation=35, ha="right")
    ax.set_xlim(positions[0] - 0.72, positions[-1] + 0.72)
    ax.set_ylim(*_boxplot_ylim(pinn_grouped, ls_grouped, noises, true_value))
    if show_title:
        ax.set_title(title or label, pad=8)
    if show_xlabel:
        ax.set_xlabel("Noise upper level")
    if show_ylabel:
        ax.set_ylabel(f"Estimated {label}")
    _style_axis(ax)


def _boxplot_legend_handles() -> List[Line2D]:
    return [
        Line2D([0], [0], color=PINN_COLOR, linewidth=5, label="PINN"),
        Line2D([0], [0], color=LS_COLOR, linewidth=5, label=LS_METHOD_SHORT),
        Line2D([0], [0], color=TRUE_COLOR, linestyle=(0, (5, 3)), linewidth=1.5, label="True value"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="#4B5563",
            markerfacecolor="white",
            markeredgewidth=1.0,
            linewidth=0,
            markersize=4.8,
            label="Mean",
        ),
    ]


def _plot_standalone_parameter_boxplot(
    exercise: str,
    param: str,
    edge: str,
    pinn_grouped: Dict[float, List[float]],
    ls_grouped: Dict[float, List[float]],
    true_value: Optional[float],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    label = _parameter_label(param, edge)
    _draw_parameter_boxplot_panel(
        ax,
        param,
        edge,
        pinn_grouped,
        ls_grouped,
        true_value,
        show_xlabel=True,
        show_ylabel=True,
        title=f"{_exercise_label(exercise)}: {label}",
        show_title=False,
    )
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def _plot_exercise_edge_boxplots(
    exercise: str,
    edges: List[str],
    pinn_values: Dict[str, Dict[str, Dict[float, List[float]]]],
    ls_values: Dict[str, Dict[str, Dict[float, List[float]]]],
    true_values: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    parameters = ["L", "U", "Theta"]
    n_cols = max(len(edges), 1)
    fig, axes = plt.subplots(
        nrows=len(parameters),
        ncols=n_cols,
        figsize=(4.25 * n_cols, 9.1),
        sharex=True,
        constrained_layout=False,
    )
    if n_cols == 1:
        axes = np.array(axes).reshape(len(parameters), 1)

    for row_idx, param in enumerate(parameters):
        for col_idx, edge in enumerate(edges):
            ax = axes[row_idx][col_idx]
            pinn_grouped = pinn_values.get(param, {}).get(edge, {})
            ls_grouped = ls_values.get(param, {}).get(edge, {})
            true_value = true_values.get(param, {}).get(edge)
            _draw_parameter_boxplot_panel(
                ax,
                param,
                edge,
                pinn_grouped,
                ls_grouped,
                true_value,
                show_xlabel=row_idx == len(parameters) - 1,
                show_ylabel=col_idx == 0,
                title=_parameter_label(param, edge),
            )
            _plot_standalone_parameter_boxplot(
                exercise,
                param,
                edge,
                pinn_grouped,
                ls_grouped,
                true_value,
                STANDALONE_PLOT_DIR / f"{exercise}_{_parameter_filename(param, edge)}_boxplot.png",
            )

    fig.legend(
        handles=_boxplot_legend_handles(),
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.978),
    )
    fig.suptitle(f"{_exercise_label(exercise)}: parameter estimates by edge", fontsize=15, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.935], h_pad=1.25, w_pad=1.0)
    _save_figure(fig, output_path)
    plt.close(fig)


def _plot_l01_u01_theta01_examples_boxplots(
    pinn_grouped_by_exercise: Dict[str, Dict[str, Dict[str, Dict[float, List[float]]]]],
    ls_grouped_by_exercise: Dict[str, Dict[str, Dict[str, Dict[float, List[float]]]]],
    true_values_by_exercise: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path,
) -> None:
    exercises = ["ex1", "ex2", "ex3"]
    parameters = ["L", "U", "Theta"]
    edge = "01"
    fig, axes = plt.subplots(
        nrows=len(exercises),
        ncols=len(parameters),
        figsize=(13.2, 9.8),
        sharex=False,
        constrained_layout=False,
    )

    for row_idx, exercise in enumerate(exercises):
        for col_idx, param in enumerate(parameters):
            ax = axes[row_idx][col_idx]
            pinn_grouped = pinn_grouped_by_exercise.get(exercise, {}).get(param, {}).get(edge, {})
            ls_grouped = ls_grouped_by_exercise.get(exercise, {}).get(param, {}).get(edge, {})
            true_value = true_values_by_exercise.get(exercise, {}).get(param, {}).get(edge)
            _draw_parameter_boxplot_panel(
                ax,
                param,
                edge,
                pinn_grouped,
                ls_grouped,
                true_value,
                show_xlabel=False,
                show_ylabel=col_idx == 0,
                title=_parameter_label(param, edge),
                show_title=False,
            )

    fig.tight_layout(h_pad=2.4, w_pad=1.0)
    _save_figure(fig, output_path)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    example_configs = _discover_example_configs(PROJECT_ROOT)
    if not example_configs:
        raise FileNotFoundError(f"No Example_* results CSV files found under {PROJECT_ROOT}")
    _register_example_configs(example_configs)

    (OUTPUT_DIR / "inequalities.json").write_text(json.dumps(INEQUALITIES, indent=2))
    _write_raw_csv(OUTPUT_DIR / "pinn_example_manifest.csv", _example_manifest_rows(example_configs))

    pinn_result_paths = {
        config.key: config.result_path
        for config in example_configs
    }
    exercise_edges = {
        config.key: [f"{i}{j}" for i, j in EDGE_MAP[config.key]]
        for config in example_configs
    }
    base_pinn_checkers = {
        "ex1": _check_ex1,
        "ex2": _check_ex2,
        "ex3": _check_ex3,
    }
    pinn_checkers = {
        config.key: base_pinn_checkers[config.base]
        for config in example_configs
    }
    base_ls_checkers = {
        "ex1": _check_ex1_ls,
        "ex2": _check_ex2_ls,
        "ex3": _check_ex3_ls,
    }
    ls_checkers = {
        config.key: base_ls_checkers[config.base]
        for config in example_configs
    }
    base_pinn_columns = {
        "ex1": {
            "L": {"01": "L_learned_01", "10": "L_learned_10"},
            "U": {"01": "U_learned_01", "10": "U_learned_10"},
            "Theta": {"01": "Th_learned_01", "10": "Th_learned_10"},
        },
        "ex2": {
            "L": {
                "00": "L_learned_00",
                "01": "L_learned_01",
                "10": "L_learned_10",
                "11": "L_learned_11",
            },
            "U": {
                "00": "U_learned_00",
                "01": "U_learned_01",
                "10": "U_learned_10",
                "11": "U_learned_11",
            },
            "Theta": {
                "00": "Th_learned_00",
                "01": "Th_learned_01",
                "10": "Th_learned_10",
                "11": "Th_learned_11",
            },
        },
        "ex3": {
            "L": {"01": "L_01", "12": "L_12", "20": "L_20"},
            "U": {"01": "U_01", "12": "U_12", "20": "U_20"},
            "Theta": {"01": "Th_01", "12": "Th_12", "20": "Th_20"},
        },
    }
    pinn_columns = {
        config.key: base_pinn_columns[config.base]
        for config in example_configs
    }

    pinn_exercise_rows = {
        exercise: _load_rows(path)
        for exercise, path in pinn_result_paths.items()
        if path.exists()
    }

    summaries_by_exercise: Dict[str, List[SummaryRow]] = {}
    pinn_enriched_by_exercise: Dict[str, List[Dict[str, Optional[float]]]] = {}
    for exercise, rows in pinn_exercise_rows.items():
        summary, enriched = _summarize(rows, pinn_checkers[exercise])
        summaries_by_exercise[exercise] = summary
        pinn_enriched_by_exercise[exercise] = enriched
        _write_summary_csv(OUTPUT_DIR / f"results_{exercise}_with_inequality.csv", enriched)
        _plot_summary(
            summary,
            f"{_exercise_label(exercise)}: inequality matches by noise upper level",
            OUTPUT_DIR / f"{exercise}_match_counts.png",
        )

    ls_rows = _load_optional_least_squares_rows(LEAST_SQUARES_RESULTS)
    if not LEAST_SQUARES_RESULTS.exists():
        print(f"Warning: {LS_METHOD_LABEL} CSV not found at {LEAST_SQUARES_RESULTS}")
    elif not ls_rows:
        print(f"Warning: {LS_METHOD_LABEL} CSV is empty at {LEAST_SQUARES_RESULTS}")
    else:
        print(f"Loaded {len(ls_rows)} {LS_METHOD_LABEL} rows from {LEAST_SQUARES_RESULTS}")
    for exercise, path in LOW_DATA_LEAST_SQUARES_RESULTS.items():
        ls_rows.extend(_load_tagged_least_squares_rows(path, exercise))

    ls_summaries_by_exercise: Dict[str, List[SummaryRow]] = {}
    ls_exercise_rows: Dict[str, List[Dict[str, str]]] = {}
    for exercise in exercise_edges:
        base_exercise = _base_exercise(exercise)
        if base_exercise != exercise and base_exercise in ls_exercise_rows:
            source_rows = ls_exercise_rows.get(base_exercise, [])
            enriched = [{**row, "exercise": exercise} for row in source_rows]
            summary = _summarize_from_matches(enriched, "inequality_match")
        else:
            summary, enriched = _summarize_ls(ls_rows, ls_checkers[exercise], exercise)
        ls_summaries_by_exercise[exercise] = summary
        ls_exercise_rows[exercise] = enriched
        _write_raw_csv(OUTPUT_DIR / f"least_squares_{exercise}_with_inequality.csv", enriched)

    _plot_match_rate(summaries_by_exercise, OUTPUT_DIR / "match_rate.png")
    for exercise in exercise_edges:
        _plot_comparison_rate(
            summaries_by_exercise.get(exercise, []),
            ls_summaries_by_exercise.get(exercise, []),
            f"{_exercise_label(exercise)}: PINN vs vanilla least-squares match rate",
            OUTPUT_DIR / f"comparison_{exercise}_match_rate.png",
        )

    pinn_grouped_by_exercise: Dict[str, Dict[str, Dict[float, List[float]]]] = {}
    for exercise, rows in pinn_exercise_rows.items():
        column_map = pinn_columns.get(exercise, {})
        for label, edge_map in column_map.items():
            grouped_by_edge: Dict[str, Dict[float, List[float]]] = {}
            for edge, key in edge_map.items():
                grouped: Dict[float, List[float]] = {}
                for row in rows:
                    noise = row.get("noise_ub")
                    if noise is None:
                        continue
                    values = _collect_pinn_values(row, [key])
                    if not values:
                        continue
                    grouped.setdefault(float(noise), []).extend(values)
                grouped_by_edge[edge] = grouped
            pinn_grouped_by_exercise.setdefault(exercise, {})[label] = grouped_by_edge

    true_values_by_exercise = _merge_true_values(
        _collect_true_values_from_pinn_paths(pinn_result_paths),
        _collect_true_values(ls_rows),
    )
    for config in example_configs:
        if config.base in true_values_by_exercise:
            true_values_by_exercise.setdefault(config.key, true_values_by_exercise[config.base])

    pinn_example_comparison_rows = _pinn_example_comparison_rows(
        example_configs,
        exercise_edges,
        summaries_by_exercise,
        pinn_grouped_by_exercise,
        true_values_by_exercise,
    )
    _write_raw_csv(
        OUTPUT_DIR / "pinn_example_comparison.csv",
        pinn_example_comparison_rows,
    )
    _plot_pinn_example_match_rates_by_base(
        example_configs,
        summaries_by_exercise,
        OUTPUT_DIR / "pinn_example_match_rate_by_base.png",
    )
    _plot_pinn_example_error_by_base(
        pinn_example_comparison_rows,
        OUTPUT_DIR / "pinn_example_mean_abs_error_by_base.png",
    )

    ls_field_map = {
        "L": "L_hat",
        "U": "U_hat",
        "Theta": "Th_hat",
    }
    ls_grouped_by_exercise: Dict[str, Dict[str, Dict[float, List[float]]]] = {}
    for exercise, rows in ls_exercise_rows.items():
        edges = EDGE_MAP.get(exercise, [])
        edge_labels = [f"{i}{j}" for i, j in edges]
        for label, field in ls_field_map.items():
            grouped_by_edge: Dict[str, Dict[float, List[float]]] = {}
            for edge_label, (i, j) in zip(edge_labels, edges):
                grouped: Dict[float, List[float]] = {}
                for row in rows:
                    noise = _to_float(row.get("noise_ub", ""))
                    if noise is None:
                        continue
                    matrix = _parse_matrix(row.get(field, ""))
                    values = _extract_matrix_values(matrix, [(i, j)])
                    if not values:
                        continue
                    grouped.setdefault(float(noise), []).extend(values)
                grouped_by_edge[edge_label] = grouped
            ls_grouped_by_exercise.setdefault(exercise, {})[label] = grouped_by_edge

    for exercise, edges in exercise_edges.items():
        pinn_values = pinn_grouped_by_exercise.get(exercise, {})
        ls_values = ls_grouped_by_exercise.get(exercise, {})
        _plot_exercise_edge_boxplots(
            exercise,
            edges,
            pinn_values,
            ls_values,
            true_values_by_exercise.get(exercise, {}),
            OUTPUT_DIR / f"combined_{exercise}_boxplots.png",
        )
    _plot_l01_u01_theta01_examples_boxplots(
        pinn_grouped_by_exercise,
        ls_grouped_by_exercise,
        true_values_by_exercise,
        OUTPUT_DIR / "combined_examples_l01_u01_theta01_boxplots.png",
    )
    _write_parameter_summary_tables(
        OUTPUT_DIR / "parameter_summary_tables.csv",
        exercise_edges,
        pinn_grouped_by_exercise,
        ls_grouped_by_exercise,
        true_values_by_exercise,
        pinn_enriched_by_exercise,
        ls_exercise_rows,
    )
    _write_comparison_html(
        OUTPUT_DIR / "pinn_vs_least_squares_comparison.html",
        exercise_edges,
        pinn_grouped_by_exercise,
        ls_grouped_by_exercise,
        true_values_by_exercise,
        pinn_enriched_by_exercise,
        ls_exercise_rows,
    )

    comparison_rows: List[Dict[str, str]] = []
    pinn_rows = [
        {**row, "exercise": exercise}
        for exercise, rows in pinn_enriched_by_exercise.items()
        for row in rows
        if row.get("set_id") is not None
    ]

    def _key_from_pinn(row: Dict[str, Optional[float]]) -> Optional[Tuple[str, int, float]]:
        set_id = row.get("set_id")
        noise = row.get("noise_ub")
        if set_id is None or noise is None:
            return None
        return (row.get("exercise", ""), int(set_id), float(noise))

    pinn_map: Dict[Tuple[str, int, float], List[bool]] = {}
    for row in pinn_rows:
        key = _key_from_pinn(row)
        if key is None:
            continue
        pinn_map.setdefault(key, []).append(bool(row.get("inequality_match")))

    ls_map: Dict[Tuple[str, int, float], List[bool]] = {}
    for rows in ls_exercise_rows.values():
        for row in rows:
            exercise = row.get("exercise", "").strip().lower()
            set_raw = row.get("set", "")
            noise_raw = row.get("noise_ub", "")
            if not set_raw or not noise_raw:
                continue
            key = (exercise, int(float(set_raw)), float(noise_raw))
            ls_map.setdefault(key, []).append(row.get("inequality_match") == "1")

    for key in sorted(set(pinn_map) | set(ls_map)):
        exercise, set_id, noise = key
        pinn_matches = pinn_map.get(key, [])
        ls_matches = ls_map.get(key, [])
        pinn_rate = sum(pinn_matches) / len(pinn_matches) if pinn_matches else 0.0
        ls_rate = sum(ls_matches) / len(ls_matches) if ls_matches else 0.0
        comparison_rows.append(
            {
                "exercise": exercise,
                "set_id": str(set_id),
                "noise_ub": str(noise),
                "pinn_match_rate": f"{pinn_rate:.3f}",
                "least_squares_match_rate": f"{ls_rate:.3f}",
                "pinn_total": str(len(pinn_matches)),
                "least_squares_total": str(len(ls_matches)),
                "match_rate_gap": f"{(pinn_rate - ls_rate):.3f}",
            }
        )

    _write_raw_csv(OUTPUT_DIR / "comparison_pinn_least_squares.csv", comparison_rows)

    print("PINN summaries:")
    for exercise, summary in summaries_by_exercise.items():
        print(f"\n{_exercise_label(exercise)}:")
        for row in summary:
            print(row)

    print(f"\n{LS_METHOD_LABEL} summaries:")
    for exercise, summary in ls_summaries_by_exercise.items():
        for row in summary:
            print(exercise, row)


if __name__ == "__main__":
    main()
