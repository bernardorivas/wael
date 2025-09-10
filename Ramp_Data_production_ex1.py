#!/usr/bin/env python3
# two_node_cycle_save_ex1.py  -> Hill replaced by Ramp
# This script processes ONLY rows whose data_type is "Ramp".
# If there is no such row (or no data_type column), it exits cleanly.

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path

# -------------------------------
# Global ramp steepness used in ODE
# -------------------------------
D_12 = float(13.0)
D_21 = float(14.0)

# -------------------------------
# Ramp function
# -------------------------------
def ramp(x, a, b, theta, d):
    """
    Piecewise-linear ramp:
      if x <= theta - 1/d: returns a
      if x >= theta + 1/d: returns b
      else: linear from a to b across width 2*(1/d)
    """
    d = float(d)
    if d == 0:
        return 0.5 * (a + b)

    w = 1.0 / d
    left = theta - w
    right = theta + w

    x = np.asarray(x, dtype=float)
    y = np.empty_like(x)

    mask_left = x <= left
    mask_right = x >= right
    mask_mid = ~(mask_left | mask_right)

    y[mask_left] = a
    y[mask_right] = b
    if np.any(mask_mid):
        y[mask_mid] = a + (x[mask_mid] - left) * (b - a) / (2.0 * w)

    return y.item() if y.shape == () else y

# -------------------------------
# Load parameter rows
# -------------------------------
param_path = "param_ex1.csv"
if not Path(param_path).exists():
    raise FileNotFoundError(f"Params CSV not found: {param_path}")

param_df = pd.read_csv(param_path)

# Require data_type column
if "data_type" not in param_df.columns:
    print("Column 'data_type' not found in CSV. Nothing to do.")
    raise SystemExit(0)

# Keep ONLY rows where data_type == Ramp (case-insensitive, trimmed)
mask_ramp = (
    param_df["data_type"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("ramp")
)
ramp_rows = param_df.loc[mask_ramp].copy()

# If no Ramp rows, exit cleanly
if ramp_rows.empty:
    print("No rows with data_type == 'Ramp' were found in the CSV.")
    raise SystemExit(0)

# Preserve original CSV row index for set numbering like the Hill code
ramp_rows["__csv_row"] = ramp_rows.index
ramp_rows = ramp_rows.reset_index(drop=True)

print(f"Found {len(ramp_rows)} Ramp set(s) in {param_path}.")

# -------------------------------
# Process each Ramp row
# -------------------------------
for _, row in ramp_rows.iterrows():
    csv_row = int(row["__csv_row"])
    set_num = csv_row + 1
    print(f"\n=== Simulating set {set_num} (CSV row {csv_row}) ===")

    # Core controls
    Tfinal = float(row.get("Tfinal", 20))
    n_t    = int(row.get("n_timepoints", 8000))
    n_ic   = int(row.get("Ramp_IC", 1))

    # Parse matrices
    try:
        L  = np.array(ast.literal_eval(row["L_np"]),  dtype=float)
        U  = np.array(ast.literal_eval(row["U_np"]),  dtype=float)
        Th = np.array(ast.literal_eval(row["Th_np"]), dtype=float)
    except Exception as e:
        raise ValueError(f"Failed to parse L_np/U_np/Th_np for CSV row {csv_row}: {e}")

    if L.shape != (2, 2) or U.shape != (2, 2) or Th.shape != (2, 2):
        raise ValueError(f"L, U, Th must be 2x2 for CSV row {csv_row}")

    # Auto scale like Hill code: figure limits = 1.5 * max(L, U)
    try:
        scale_max = float(max(np.max(L), np.max(U)))
    except ValueError:
        scale_max = 1.0
    xMax = yMax = 1.5 * scale_max
    print(f"  Auto-set xMax=yMax={xMax:.4g} (1.5 * max(L,U) with max={scale_max:.4g})")

    # RHS with ramp
    def odefun(t, x):
        dx0 = -x[0] + float(ramp(x[1], U[1, 0], L[1, 0], Th[1, 0], D_21))
        dx1 = -x[1] + float(ramp(x[0], U[0, 1], L[0, 1], Th[0, 1], D_12))
        return [dx0, dx1]

    t_eval = np.linspace(0.0, Tfinal, n_t)

    # Equally spaced initial conditions on a grid (take first n_ic)
    s = int(np.ceil(np.sqrt(n_ic)))
    xs = np.linspace(0.0, xMax, s)
    ys = np.linspace(0.0, yMax, s)
    grid = np.array([(xi, yi) for yi in ys for xi in xs], dtype=float)
    initial_conditions = grid[:n_ic]

    fig, ax = plt.subplots()
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title(f"2-Node Ramp Cycle (Set {set_num}), t in [0, {Tfinal}]")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, xMax)
    ax.set_ylim(0.0, yMax)

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
        ax.plot(Y[:, 0], Y[:, 1], linewidth=1.2)

        for t_val, (y0, y1) in zip(sol.t, Y):
            all_data.append([traj_id, t_val, y0, y1])

    ax.plot(
        initial_conditions[:, 0],
        initial_conditions[:, 1],
        "ro",
        markersize=4,
        markerfacecolor="r",
    )

    plot_path = f"Set{set_num}_TrajectoryRAMP_ex1.png"
    csv_path  = f"Data_Set{set_num}_RAMP_ex1.csv"

    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    df_out = pd.DataFrame(all_data, columns=["Trajectory", "Time", "x0", "x1"])
    df_out.to_csv(csv_path, index=False)

    print(f" • Saved data  -> {csv_path}")
    print(f" • Saved plot  -> {plot_path}")
    print(f"   Trajectories: {n_ic}, points total: {len(all_data)}")

print("\nAll Ramp sets processed for exercise 1.")
