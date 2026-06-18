









from __future__ import annotations

import argparse
import ast
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = (
    ROOT.parent
    / "Hill-Ramp-Function-Prediction-Edition-20_04_30_2026"
    / "Hill-Ramp-Function-Prediction-Edition-20_04_30_2026"
)


@dataclass(frozen=True)
class DatasetSpec:
    exercise: str
    data_type: str
    example_dir: str
    param_file: str
    data_dir: str
    pattern: str
    dimension: int


SPECS = [
    DatasetSpec("ex1", "hill", "Example_1", "param_ex1.csv", "generated_data/hill", "Data_Set*_HILL_ex1.csv", 2),
    DatasetSpec("ex2", "hill", "Example_2", "param_ex2.csv", "generated_data_ex2/hill", "Data_Set*_HILL_ex2.csv", 2),
    DatasetSpec("ex3", "hill", "Example_3", "param_ex3.csv", "generated_data/hill", "Data_Set*_HILL_ex3.csv", 3),
    DatasetSpec("ex1", "hill", "Example_1_May27", "param_ex1.csv", "generated_data/hill", "Data_Set*_HILL_ex1.csv", 2),
    DatasetSpec("ex2", "hill", "Example_2_May27", "param_ex2.csv", "generated_data_ex2/hill", "Data_Set*_HILL_ex2.csv", 2),
    DatasetSpec("ex3", "hill", "Example_3_May27", "param_ex3.csv", "generated_data/hill", "Data_Set*_HILL_ex3.csv", 3),
    DatasetSpec("ex3", "ramp", "Example_3", "param_ex3.csv", "generated_data/ramp", "Data_Set*_RAMP_ex3.csv", 3),
    DatasetSpec("ex4", "hill", "Example_4", "param_ex4.csv", "generated_data/hill", "Data_Set*_HILL_ex4.csv", 3),
]


def parse_matrix(value, shape: tuple[int, int], default: float = 0.0) -> np.ndarray:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.full(shape, default, dtype=float)
    try:
        arr = np.array(ast.literal_eval(str(value)), dtype=float)
    except Exception:
        arr = np.full(shape, default, dtype=float)
    if arr.shape != shape:
        out = np.full(shape, default, dtype=float)
        slices = tuple(slice(0, min(a, b)) for a, b in zip(out.shape, arr.shape))
        out[slices] = arr[slices]
        return out
    return arr


def hill(x, a, b, theta, d):
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    theta = max(float(theta), 1e-8)
    d = max(float(d), 1e-8)
    xd = np.power(x, d)
    td = theta**d
    return a + (b - a) * xd / (td + xd + 1e-12)


def ramp(x, a, b, theta, d):
    d = max(float(d), 1e-8)
    x = np.asarray(x, dtype=float)
    width = 1.0 / d
    left = theta - width
    right = theta + width
    y = np.empty_like(x, dtype=float)
    y[x <= left] = a
    y[x >= right] = b
    mid = (x > left) & (x < right)
    y[mid] = a + (x[mid] - left) * (b - a) / (2.0 * width)
    return y


def derivative_table(df: pd.DataFrame, dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_cols = [f"x{i}" for i in range(dim)]
    xs = []
    dxs = []
    times = []
    for _, group in df.groupby("Trajectory", sort=False):
        group = group.sort_values("Time")
        if len(group) < 3:
            continue
        t = group["Time"].to_numpy(dtype=float)
        x = group[x_cols].to_numpy(dtype=float)
        dx = np.column_stack([np.gradient(x[:, i], t, edge_order=2) for i in range(dim)])
        xs.append(x)
        dxs.append(dx)
        times.append(t)
    if not xs:
        raise ValueError("Need at least one trajectory with three or more time points.")
    return np.vstack(xs), np.vstack(dxs), np.concatenate(times)


def derivative_for_trajectory(df: pd.DataFrame, dim: int, trajectory_id) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_cols = [f"x{i}" for i in range(dim)]
    group = df[df["Trajectory"] == trajectory_id].sort_values("Time")
    if len(group) < 3:
        raise ValueError(f"Trajectory {trajectory_id} has fewer than three time points.")
    t = group["Time"].to_numpy(dtype=float)
    x = group[x_cols].to_numpy(dtype=float)
    dx = np.column_stack([np.gradient(x[:, i], t, edge_order=2) for i in range(dim)])
    return x, dx, t


def active_edges(exercise: str, data_type: str) -> list[tuple[int, int]]:
    if exercise == "ex1":
        return [(1, 0), (0, 1)]
    if exercise == "ex2":
        return [(0, 0), (0, 1), (1, 0), (1, 1)]
    if exercise == "ex3":
        return [(2, 0), (0, 1), (1, 2)]
    if exercise == "ex4":
        return [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
    raise ValueError(f"Unsupported exercise: {exercise}")


def unpack_params(p: np.ndarray, edges: list[tuple[int, int]], dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    L = np.zeros((dim, dim), dtype=float)
    U = np.zeros((dim, dim), dtype=float)
    Th = np.zeros((dim, dim), dtype=float)
    D = np.zeros((dim, dim), dtype=float)
    for k, (src, dst) in enumerate(edges):
        L[src, dst], U[src, dst], Th[src, dst], D[src, dst] = p[4 * k : 4 * k + 4]
    return L, U, Th, D


def rhs(exercise: str, data_type: str, x: np.ndarray, p: np.ndarray, dim: int) -> np.ndarray:
    edges = active_edges(exercise, data_type)
    L, U, Th, D = unpack_params(p, edges, dim)
    out = -x.copy()

    if data_type == "ramp":
        fn = ramp
    else:
        fn = hill

    if exercise == "ex1":
        out[:, 0] += fn(x[:, 1], U[1, 0], L[1, 0], Th[1, 0], D[1, 0])
        out[:, 1] += fn(x[:, 0], U[0, 1], L[0, 1], Th[0, 1], D[0, 1])
    elif exercise == "ex2":
        h00 = hill(x[:, 0], L[0, 0], U[0, 0], Th[0, 0], D[0, 0])
        h01 = hill(x[:, 0], U[0, 1], L[0, 1], Th[0, 1], D[0, 1])
        h10 = hill(x[:, 1], L[1, 0], U[1, 0], Th[1, 0], D[1, 0])
        h11 = hill(x[:, 1], L[1, 1], U[1, 1], Th[1, 1], D[1, 1])
        out[:, 0] += h00 + h10
        out[:, 1] += h01 * h11
    elif exercise == "ex3":
        out[:, 0] += fn(x[:, 2], U[2, 0], L[2, 0], Th[2, 0], D[2, 0])
        out[:, 1] += fn(x[:, 0], U[0, 1], L[0, 1], Th[0, 1], D[0, 1])
        out[:, 2] += fn(x[:, 1], U[1, 2], L[1, 2], Th[1, 2], D[1, 2])
    elif exercise == "ex4":
        f00 = fn(x[:, 0], L[0, 0], U[0, 0], Th[0, 0], D[0, 0])
        f10 = fn(x[:, 1], U[1, 0], L[1, 0], Th[1, 0], D[1, 0])
        f20 = fn(x[:, 2], U[2, 0], L[2, 0], Th[2, 0], D[2, 0])
        f01 = fn(x[:, 0], U[0, 1], L[0, 1], Th[0, 1], D[0, 1])
        f11 = fn(x[:, 1], L[1, 1], U[1, 1], Th[1, 1], D[1, 1])
        f12 = fn(x[:, 1], U[1, 2], L[1, 2], Th[1, 2], D[1, 2])
        f22 = fn(x[:, 2], L[2, 2], U[2, 2], Th[2, 2], D[2, 2])
        out[:, 0] += f00 * f10 * f20
        out[:, 1] += f01 * f11
        out[:, 2] += f12 * f22
    return out


def set_number(path: Path) -> int:
    match = re.search(r"Data_Set(\d+)_", path.name)
    if not match:
        raise ValueError(f"Cannot parse set number from {path.name}")
    return int(match.group(1))


def true_params(row: pd.Series, spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (spec.dimension, spec.dimension)
    L = parse_matrix(row.get("L_np"), shape)
    U = parse_matrix(row.get("U_np"), shape)
    Th = parse_matrix(row.get("Th_np"), shape)
    if spec.exercise in {"ex1", "ex2"}:
        D = np.zeros(shape, dtype=float)
        for i in range(spec.dimension):
            for j in range(spec.dimension):
                col = f"D{i + 1}{j + 1}_input"
                if col in row and pd.notna(row[col]):
                    D[i, j] = float(row[col])
    else:
        D = parse_matrix(row.get("D_np"), shape, default=12.0)
    return L, U, Th, D


def initial_guess(x: np.ndarray, dx: np.ndarray, edges: list[tuple[int, int]], dim: int, rng: np.random.Generator) -> np.ndarray:
    scale = max(float(np.nanmax(x)), float(np.nanmax(dx + x)), 1.0)
    guess = []
    for src, dst in edges:
        response = np.clip(dx[:, dst] + x[:, dst], 0.0, None)
        low = max(float(np.percentile(response, 10)), 1e-4)
        high = max(float(np.percentile(response, 90)), low + 1e-3)
        theta = max(float(np.percentile(x[:, src], 50)), 1e-3)
        guess.extend(
            [
                low * rng.uniform(0.8, 1.2),
                high * rng.uniform(0.8, 1.2),
                theta * rng.uniform(0.75, 1.25),
                min(20.0, max(2.0, 8.0 * rng.uniform(0.75, 1.25))),
            ]
        )
    return np.array(guess, dtype=float)


def bounds(x: np.ndarray, dx: np.ndarray, n_edges: int, data_type: str) -> tuple[np.ndarray, np.ndarray]:
    y_scale = max(float(np.nanmax(np.abs(dx + x))), float(np.nanmax(x)), 1.0)
    theta_scale = max(float(np.nanmax(x)), 1.0)
    lower = []
    upper = []
    for _ in range(n_edges):
        lower.extend([0.0, 0.0, 1e-6, 0.25])
        upper.extend([10.0 * y_scale, 10.0 * y_scale, 3.0 * theta_scale, 80.0])
    return np.array(lower, dtype=float), np.array(upper, dtype=float)


def fit_arrays(
    x: np.ndarray,
    dx: np.ndarray,
    path: Path,
    row: pd.Series,
    spec: DatasetSpec,
    args,
    trajectory_id=None,
) -> dict:
    if args.max_points and len(x) > args.max_points:
        idx = np.linspace(0, len(x) - 1, args.max_points).astype(int)
        x = x[idx]
        dx = dx[idx]

    edges = active_edges(spec.exercise, spec.data_type)
    lb, ub = bounds(x, dx, len(edges), spec.data_type)
    scale = np.maximum(np.std(dx, axis=0), 1e-6)

    best = None
    traj_seed_offset = 0 if trajectory_id is None else int(float(trajectory_id)) * 1009
    rng = np.random.default_rng(args.seed + set_number(path) * 10007 + traj_seed_offset)
    for start_idx in range(1, args.starts + 1):
        print(
            f"    Start {start_idx}/{args.starts} for {spec.exercise} {spec.data_type} "
            f"set {set_number(path)}"
        )
        p0 = np.clip(initial_guess(x, dx, edges, spec.dimension, rng), lb + 1e-9, ub - 1e-9)

        def residual(p):
            return ((rhs(spec.exercise, spec.data_type, x, p, spec.dimension) - dx) / scale).ravel()

        start_time = time.perf_counter()
        result = least_squares(
            residual,
            p0,
            bounds=(lb, ub),
            loss=args.loss,
            max_nfev=args.max_nfev,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )
        elapsed = time.perf_counter() - start_time
        print(
            f"    Completed start {start_idx}/{args.starts} "
            f"(cost={result.cost:.6g}, nfev={result.nfev}, {elapsed:.1f}s)"
        )
        if best is None or result.cost < best.cost:
            best = result

    assert best is not None
    L_hat, U_hat, Th_hat, D_hat = unpack_params(best.x, edges, spec.dimension)
    L_true, U_true, Th_true, D_true = true_params(row, spec)
    pred = rhs(spec.exercise, spec.data_type, x, best.x, spec.dimension)
    rmse_dx = float(np.sqrt(np.mean((pred - dx) ** 2)))
    add_noise = str(row.get("add_noise", "")).strip()
    noise_ub = row.get("noise_ub", "")
    ic_seed = row.get("ic_seed", "")
    nn_seed = row.get("NN_seed", "")

    return {
        "exercise": spec.exercise,
        "data_type": spec.data_type,
        "set": set_number(path),
        "trajectory": "" if trajectory_id is None else trajectory_id,
        "ic_seed": ic_seed,
        "NN_seed": nn_seed,
        "add_noise": add_noise,
        "noise_ub": noise_ub,
        "source_csv": str(path),
        "n_points": int(len(x)),
        "success": bool(best.success),
        "cost": float(best.cost),
        "rmse_derivative": rmse_dx,
        "L_hat": json.dumps(L_hat.tolist()),
        "U_hat": json.dumps(U_hat.tolist()),
        "Th_hat": json.dumps(Th_hat.tolist()),
        "D_hat": json.dumps(D_hat.tolist()),
        "L_true": json.dumps(L_true.tolist()),
        "U_true": json.dumps(U_true.tolist()),
        "Th_true": json.dumps(Th_true.tolist()),
        "D_true": json.dumps(D_true.tolist()),
        "L_abs_error": json.dumps(np.abs(L_hat - L_true).tolist()),
        "U_abs_error": json.dumps(np.abs(U_hat - U_true).tolist()),
        "Th_abs_error": json.dumps(np.abs(Th_hat - Th_true).tolist()),
        "D_abs_error": json.dumps(np.abs(D_hat - D_true).tolist()),
    }


def fit_dataset(path: Path, row: pd.Series, spec: DatasetSpec, args) -> list[dict]:
    df = pd.read_csv(path)
    x, dx, _ = derivative_table(df, spec.dimension)
    return [fit_arrays(x, dx, path, row, spec, args)]


def scalar_summary(value: str) -> float:
    try:
        arr = np.array(json.loads(value), dtype=float)
    except Exception:
        return np.nan
    return float(np.nanmean(np.abs(arr)))


def write_grouped_outputs(result: pd.DataFrame, out_dir: Path) -> None:
    return


def run(args) -> pd.DataFrame:
    source = Path(args.source).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / "least_squares_parameter_estimates.csv"
    selected_sets = parse_set_filter(args.sets)
    header_needed = not output_file.exists()

    for spec in SPECS:
        if "all" not in args.exercises and spec.exercise not in args.exercises:
            continue
        if args.data_type != "all" and spec.data_type != args.data_type:
            continue

        example = source / spec.example_dir
        param_path = example / spec.param_file
        data_dir = example / spec.data_dir
        if not param_path.exists() or not data_dir.exists():
            continue

        params = pd.read_csv(param_path)
        files = sorted(data_dir.glob(spec.pattern), key=set_number)
        if selected_sets is not None:
            files = [path for path in files if set_number(path) in selected_sets]
        if args.max_sets:
            files = files[: args.max_sets]

        total_files = len(files)
        if total_files == 0:
            continue
        print(
            f"Processing {total_files} set(s) for {spec.exercise} {spec.data_type} "
            f"from {data_dir}"
        )

        for idx, file_path in enumerate(files, start=1):
            set_id = set_number(file_path)
            row_idx = set_id - 1
            if row_idx < 0 or row_idx >= len(params):
                print(f"Skipping {file_path.name}: no parameter row {row_idx}.")
                continue
            row = params.iloc[row_idx]
            if str(row.get("data_type", "")).strip().lower() != spec.data_type:
                continue
            percent_complete = 100.0 * idx / total_files
            print(
                f"[{idx}/{total_files} | {percent_complete:.1f}%] Fitting {spec.exercise} {spec.data_type} set {set_id} "
                f"(pooled trajectories) from {file_path.name}"
            )
            set_start = time.perf_counter()
            try:
                new_rows = fit_dataset(file_path, row, spec, args)
            except Exception as exc:
                new_rows = [
                    {
                        "exercise": spec.exercise,
                        "data_type": spec.data_type,
                        "set": set_id,
                        "source_csv": str(file_path),
                        "success": False,
                        "error": str(exc),
                    }
                ]
            status = "ok" if new_rows and new_rows[0].get("success") else "failed"
            set_elapsed = time.perf_counter() - set_start
            print(f"Completed set {set_id}: {status} ({set_elapsed:.1f}s)")
            if new_rows:
                pd.DataFrame(new_rows).to_csv(output_file, mode="a", header=header_needed, index=False)
                header_needed = False

    if output_file.exists():
        result = pd.read_csv(output_file)
    else:
        result = pd.DataFrame([])
    print(f"Wrote {output_file}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Read-only Edition 20 source folder.")
    parser.add_argument("--output", default=str(ROOT / "outputs"), help="Folder for estimate CSVs.")
    parser.add_argument(
        "--exercise",
        choices=["prompt", "all", "ex1", "ex2", "ex3", "ex4"],
        default="prompt",
        help="Exercise to run; use 'prompt' to choose interactively.",
    )
    parser.add_argument("--data-type", choices=["all", "hill", "ramp"], default="all")
    parser.add_argument("--max-sets", type=int, default=0, help="Limit sets per exercise/data type; 0 means all.")
    parser.add_argument("--sets", default="", help="Comma-separated set numbers/ranges to process, e.g. 1,3,10-15.")
    parser.add_argument("--merge-existing", action="store_true", help="Merge processed rows into an existing output CSV.")
    parser.add_argument("--max-points", type=int, default=2500, help="Subsample derivative points per fit; 0 means all.")
    parser.add_argument("--starts", type=int, default=8, help="Random multistart fits per dataset.")
    parser.add_argument("--max-nfev", type=int, default=4000, help="Least-squares function evaluations per start.")
    parser.add_argument("--loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="soft_l1")
    parser.add_argument("--seed", type=int, default=20)
    args = parser.parse_args()
    if args.max_sets == 0:
        args.max_sets = None
    if args.max_points == 0:
        args.max_points = None
    args.exercises = normalize_exercise_selection(args.exercise)
    return args


def normalize_exercise_selection(selection: str) -> list[str]:
    selection = str(selection or "").strip().lower()
    if selection == "prompt":
        selection = prompt_exercise_selection()
    if not selection or selection == "all":
        return ["all"]
    raw = [part.strip().lower() for part in selection.split(",") if part.strip()]
    valid = {"ex1", "ex2", "ex3", "ex4", "all"}
    for item in raw:
        if item not in valid:
            raise ValueError(f"Unsupported exercise selection '{item}'. Use ex1, ex2, ex3, ex4, or all.")
    return raw


def prompt_exercise_selection() -> str:
    prompt = (
        "Which exercises should the least-squares run? "
        "Enter comma-separated values (ex1, ex2, ex3, ex4) or 'all': "
    )
    while True:
        response = input(prompt).strip().lower()
        if not response:
            print("Please enter at least one exercise or 'all'.")
            continue
        try:
            normalize_exercise_selection(response)
        except ValueError as exc:
            print(exc)
            continue
        return response


def parse_set_filter(text: str) -> set[int] | None:
    text = str(text or "").strip()
    if not text:
        return None
    selected = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            selected.update(range(int(start), int(end) + 1))
        else:
            selected.add(int(part))
    return selected


def result_key(row: dict) -> tuple[str, str, int, str]:
    trajectory = row.get("trajectory", "")
    if pd.isna(trajectory) or str(trajectory).strip().lower() == "nan":
        trajectory = ""
    return (
        str(row.get("exercise", "")),
        str(row.get("data_type", "")),
        int(float(row.get("set", 0))),
        str(trajectory),
    )


if __name__ == "__main__":
    run(parse_args())
