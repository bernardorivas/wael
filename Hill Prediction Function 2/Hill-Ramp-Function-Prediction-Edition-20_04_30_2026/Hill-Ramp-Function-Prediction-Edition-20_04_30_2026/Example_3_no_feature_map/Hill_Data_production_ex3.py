import ast
import json
import sys
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.stats import qmc


BASE_DIR = Path(__file__).resolve().parent
PARAM_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "param_ex3.csv"
if not PARAM_PATH.is_absolute():
    PARAM_PATH = BASE_DIR / PARAM_PATH
DATA_DIR = BASE_DIR / "generated_data" / "hill"
PLOT_DIR = BASE_DIR / "generated_plots" / "hill"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def compute_noise_reference_scales(L, U, csv_row):
    L_nonzero = L[L != 0]
    U_nonzero = U[U != 0]
    if L_nonzero.size == 0 or U_nonzero.size == 0:
        raise ValueError(f"Cannot compute noise scale for CSV row {csv_row}: L_np and U_np must have nonzero entries.")

    old_scale = float(np.min(np.abs(L_nonzero[:, None] - U_nonzero[None, :])))
    same_index_mask = (L != 0) & (U != 0)
    if not np.any(same_index_mask):
        raise ValueError(f"Cannot compute noise scale for CSV row {csv_row}: no matching nonzero Lij and Uij entries.")

    same_index_scale = float(np.min(np.abs(L[same_index_mask] - U[same_index_mask])))
    min_nonzero_l = float(np.min(L_nonzero))
    new_scale = float(min(same_index_scale, min_nonzero_l))
    return old_scale, new_scale, same_index_scale, min_nonzero_l


def resolve_ic_seed(row_like, set_id):
    s = None
    try:
        s = row_like.get("ic_seed", None)
    except Exception:
        try:
            s = row_like["ic_seed"]
        except Exception:
            s = None
    if s is not None and str(s).strip() not in ("", "nan", "None"):
        try:
            return int(float(s))
        except Exception:
            pass
    try:
        d = row_like.to_dict() if hasattr(row_like, "to_dict") else dict(row_like)
    except Exception:
        d = {}
    blob = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    h = zlib.crc32(blob) ^ int(set_id)
    seed = int(h & 0xFFFFFFFF) or 1
    return seed


def hill(x, a, b, theta, d):
    return a + (b - a) * (x**d) / (theta**d + x**d)


ensure_dir(DATA_DIR)
ensure_dir(PLOT_DIR)

if not PARAM_PATH.exists():
    raise FileNotFoundError(f"Params CSV not found: {PARAM_PATH}")

param_df = pd.read_csv(PARAM_PATH)
hill_rows = param_df[param_df["data_type"].str.lower() == "hill"].copy()
hill_rows["__csv_row"] = hill_rows.index
hill_rows = hill_rows.reset_index(drop=True)

if hill_rows.empty:
    print("No rows with data_type == 'Hill' were found in the CSV.")
    raise SystemExit(0)

print(f"Found {len(hill_rows)} Hill set(s) in {PARAM_PATH}.")

for _, row in hill_rows.iterrows():
    csv_row = int(row["__csv_row"])
    set_num = csv_row + 1
    print(f"\n=== Simulating set {set_num} (CSV row {csv_row}) ===")
    ic_seed = resolve_ic_seed(row, set_num)
    rng = np.random.default_rng(ic_seed)

    xMax = float(row.get("xMax", 1.3))
    yMax = float(row.get("yMax", 1.3))
    Tfinal = float(row.get("Tfinal", 20))
    n_t = int(row.get("n_timepoints", 8000))
    n_ic = int(row.get("Hill_IC", 1))

    try:
        L = np.array(ast.literal_eval(row["L_np"]), dtype=float)
        U = np.array(ast.literal_eval(row["U_np"]), dtype=float)
        Th = np.array(ast.literal_eval(row["Th_np"]), dtype=float)
    except Exception as e:
        raise ValueError(f"Failed to parse L_np/U_np/Th_np for CSV row {csv_row}: {e}")

    if L.shape != (3, 3) or U.shape != (3, 3) or Th.shape != (3, 3):
        raise ValueError(f"L, U, Th must be 3x3 for CSV row {csv_row}")

    try:
        scale_max = float(max(np.max(L), np.max(U)))
    except ValueError:
        scale_max = 1.0
    xMax = yMax = 1.5 * scale_max
    print(f"  Auto-set xMax=yMax={xMax:.4g} (1.5 * max(L,U) with max={scale_max:.4g})")

    try:
        D = np.array(ast.literal_eval(row["D_np"]), dtype=float)
    except Exception:
        D = np.full((3, 3), 12.0)

    def odefun(t, x):
        x0, x1, x2 = x
        dx0 = -x0 + hill(x2, U[2, 0], L[2, 0], Th[2, 0], D[2, 0])
        dx1 = -x1 + hill(x0, U[0, 1], L[0, 1], Th[0, 1], D[0, 1])
        dx2 = -x2 + hill(x1, U[1, 2], L[1, 2], Th[1, 2], D[1, 2])
        return [dx0, dx1, dx2]

    t_eval = np.linspace(0.0, Tfinal, n_t)

    xMin = 0.0
    yMin = 0.0
    zMin = 0.0
    zMax = xMax

    sampler = qmc.LatinHypercube(d=3, seed=ic_seed)
    initial_conditions = qmc.scale(
        sampler.random(n_ic),
        l_bounds=[xMin, yMin, zMin],
        u_bounds=[xMax, yMax, zMax],
    )
    print(f"  ICs (Latin hypercube): Hill_IC={n_ic} in [0,{xMax}]^3")

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_zlabel("x2")
    ax.set_title(f"3-Node Hill Cycle (Set {set_num}), t in [0, {Tfinal}]")
    ax.grid(True)

    all_data = []

    for traj_id, x0 in enumerate(initial_conditions, start=1):
        sol = solve_ivp(
            odefun,
            [0.0, Tfinal],
            x0,
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10,
        )
        if not sol.success:
            print(f"  Warning: solver failed on traj {traj_id} with message: {sol.message}")

        Y = sol.y.T
        ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], linewidth=1.2)

        for t_val, (y0, y1, y2) in zip(sol.t, Y):
            all_data.append([traj_id, t_val, y0, y1, y2])

    ax.scatter(initial_conditions[:, 0], initial_conditions[:, 1], initial_conditions[:, 2], c="r", marker="o", s=10)

    plot_path = PLOT_DIR / f"Set{set_num}_TrajectoryHILL_ex3.png"
    csv_path = DATA_DIR / f"Data_Set{set_num}_HILL_ex3.csv"

    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    df_out = pd.DataFrame(all_data, columns=["Trajectory", "Time", "x0", "x1", "x2"])
    df_out["ic_seed"] = ic_seed

    add_noise_flag = str(row.get("add_noise", "no")).strip().lower()
    if add_noise_flag in ("yes", "y", "true", "1"):
        try:
            noise_ub_pct = float(row.get("noise_ub"))
        except Exception:
            raise ValueError(f"add_noise requested for CSV row {csv_row} but noise_ub is missing or invalid.")
        if noise_ub_pct < 0:
            raise ValueError(f"noise_ub must be non-negative for CSV row {csv_row}; got {noise_ub_pct}")

        old_noise_scale, new_noise_scale, same_index_scale, min_nonzero_l = compute_noise_reference_scales(L, U, csv_row)
        noise_ub_abs = (noise_ub_pct / 100.0) * new_noise_scale

        n_rows = len(df_out)
        noise_x0 = rng.uniform(-noise_ub_abs, noise_ub_abs, size=n_rows)
        noise_x1 = rng.uniform(-noise_ub_abs, noise_ub_abs, size=n_rows)
        noise_x2 = rng.uniform(-noise_ub_abs, noise_ub_abs, size=n_rows)

        df_out["x0"] = df_out["x0"] + noise_x0
        df_out["x1"] = df_out["x1"] + noise_x1
        df_out["x2"] = df_out["x2"] + noise_x2
        df_out[["x0", "x1", "x2"]] = df_out[["x0", "x1", "x2"]].clip(lower=0.0)
        df_out["add_noise"] = "yes"
        df_out["noise_ub"] = noise_ub_pct
        df_out["noise_ref_old_min_abs_lij_minus_ukl"] = old_noise_scale
        df_out["noise_ref_new_min_same_lu_or_l"] = new_noise_scale
        df_out["noise_ref_same_index_min_abs_lij_minus_uij"] = same_index_scale
        df_out["noise_ref_min_nonzero_l"] = min_nonzero_l
        print(f"  Noise scale old min(|Lij - Ukl|) = {old_noise_scale:.6g}")
        print(f"  Noise scale new min(min(|Lij - Uij|), Lkl) = {new_noise_scale:.6g} (same-index={same_index_scale:.6g}, min nonzero L={min_nonzero_l:.6g})")
        print(f"  Added bounded uniform observation noise with ub={noise_ub_pct}% (abs ub={noise_ub_abs:.6g}) to {n_rows} rows")
    else:
        df_out["add_noise"] = "no"

    df_out.to_csv(csv_path, index=False)

    print(f"  Saved data  -> {csv_path.relative_to(BASE_DIR)}")
    print(f"  Saved plot  -> {plot_path.relative_to(BASE_DIR)}")

    if add_noise_flag in ("yes", "y", "true", "1"):
        noisy_plot_path = PLOT_DIR / f"Set{set_num}_TrajectoryHILL_ex3_noisy.png"
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(projection="3d")
        ax2.set_xlabel("x0")
        ax2.set_ylabel("x1")
        ax2.set_zlabel("x2")
        ax2.set_title(f"3-Node Hill Cycle (Set {set_num}) - noisy observations, t in [0, {Tfinal}]")
        ax2.grid(True)

        for traj_id in sorted(df_out["Trajectory"].unique()):
            sub = df_out[df_out["Trajectory"] == traj_id].sort_values("Time")
            ax2.plot(sub["x0"].values, sub["x1"].values, sub["x2"].values, linewidth=1.2)

        try:
            ax2.scatter(initial_conditions[:, 0], initial_conditions[:, 1], initial_conditions[:, 2], c="r", marker="o", s=10)
        except Exception:
            pass

        fig2.savefig(noisy_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved noisy plot -> {noisy_plot_path.relative_to(BASE_DIR)}")

    print(f"   Trajectories: {n_ic}, points total: {len(all_data)}")

print("\nAll Hill sets processed for exercise 3.")
