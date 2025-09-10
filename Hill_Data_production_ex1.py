#!/usr/bin/env python3
# two_node_cycle_save_ex1.py

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path

D_12 = 13
D_21 = 14

# -------------------------------
# Hill function
# -------------------------------
def hill(x, a, b, theta, d):
    # same form as MATLAB: a + (b-a) * x^d / (theta^d + x^d)
    return a + (b - a) * (x**d) / (theta**d + x**d)

# -------------------------------
# Load parameter rows for "Hill"
# -------------------------------
param_path = "param_ex1.csv"
if not Path(param_path).exists():
    raise FileNotFoundError(f"Params CSV not found: {param_path}")

param_df  = pd.read_csv(param_path)
# Filter to hill rows and keep a copy of original CSV row indices for logging
hill_rows = param_df[param_df["data_type"].str.lower() == "hill"].copy()
hill_rows["__csv_row"] = hill_rows.index
hill_rows = hill_rows.reset_index(drop=True)

if hill_rows.empty:
    print("No rows with data_type == 'Hill' were found in the CSV.")
    exit(0)

print(f"Found {len(hill_rows)} Hill set(s) in {param_path}.")

# -------------------------------
# Process each Hill row
# -------------------------------
for k, row in hill_rows.iterrows():
    # set number is the absolute CSV row number (1-based)
    csv_row = int(row["__csv_row"])  # original CSV row index (0-based)
    set_num = csv_row + 1
    print(f"\n=== Simulating set {set_num} (CSV row {csv_row}) ===")

    # core simulation controls from CSV
    # xMax / yMax will be auto-set later from L,U; initial CSV values (if any) ignored.
    xMax   = float(row.get("xMax", 1.3))  # placeholder (will be overridden)
    yMax   = float(row.get("yMax", 1.3))  # placeholder (will be overridden)
    Tfinal = float(row.get("Tfinal", 20))
    n_t    = int(row.get("n_timepoints", 8000))
    n_ic   = int(row.get("Hill_IC", 1))

    # matrices L, U, Theta from CSV (as stringified 2x2 lists)
    try:
        L  = np.array(ast.literal_eval(row["L_np"]),  dtype=float)
        U  = np.array(ast.literal_eval(row["U_np"]),  dtype=float)
        Th = np.array(ast.literal_eval(row["Th_np"]), dtype=float)
    except Exception as e:
        raise ValueError(f"Failed to parse L_np/U_np/Th_np for CSV row {csv_row}: {e}")

    if L.shape != (2,2) or U.shape != (2,2) or Th.shape != (2,2):
        raise ValueError(f"L, U, Th must be 2x2 for CSV row {csv_row}")

    # After loading matrices, choose xMax,yMax = 1.5 * max(L,U) as requested
    try:
        scale_max = float(max(np.max(L), np.max(U)))
    except ValueError:
        scale_max = 1.0
    xMax = yMax = 1.5 * scale_max
    print(f"  Auto-set xMax=yMax={xMax:.4g} (1.5 * max(L,U) with max={scale_max:.4g})")

    # Build RHS that matches the MATLAB code exactly
    # dx0/dt = -x0 + h(x1; U(2,1), L(2,1), T(2,1), D(2,1))
    # dx1/dt = -x1 + h(x0; U(1,2), L(1,2), T(1,2), D(1,2))
    def odefun(t, x):
        dx0 = -x[0] + hill(x[1], U[1,0], L[1,0], Th[1,0], D_21)
        dx1 = -x[1] + hill(x[0], U[0,1], L[0,1], Th[0,1], D_12)
        return [dx0, dx1]

    # time grid
    t_eval = np.linspace(0.0, Tfinal, n_t)

    # Equally spaced initial conditions on a grid (take first n_ic, like Ramp)
    s = int(np.ceil(np.sqrt(n_ic)))
    xs = np.linspace(0.0, xMax, s)
    ys = np.linspace(0.0, yMax, s)
    grid = np.array([(xi, yi) for yi in ys for xi in xs], dtype=float)
    initial_conditions = grid[:n_ic]

    # figure and axis for phase portrait
    fig, ax = plt.subplots()
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title(f"2-Node Hill Cycle (Set {set_num}), t ∈ [0, {Tfinal}]")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, xMax)
    ax.set_ylim(0.0, yMax)

    # collect all trajectories for this set
    all_data = []  # columns: Trajectory, Time, x0, x1

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

        Y = sol.y.T  # shape (n_t, 2)
        ax.plot(Y[:,0], Y[:,1], linewidth=1.2)

        # append rows to all_data
        for t_val, (y0, y1) in zip(sol.t, Y):
            all_data.append([traj_id, t_val, y0, y1])

    # mark ICs
    ax.plot(initial_conditions[:,0], initial_conditions[:,1],
            "ro", markersize=4, markerfacecolor="r")

    # output names aligned with exercise 1
    plot_path = f"Set{set_num}_TrajectoryHILL_ex1.png"
    csv_path  = f"Data_Set{set_num}_HILL_ex1.csv"

    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # per-set trajectory CSV
    df_out = pd.DataFrame(all_data, columns=["Trajectory", "Time", "x0", "x1"])
    df_out.to_csv(csv_path, index=False)

    print(f" • Saved data  -> {csv_path}")
    print(f" • Saved plot  -> {plot_path}")
    print(f"   Trajectories: {n_ic}, points total: {len(all_data)}")

print("\nAll Hill sets processed for exercise 1.")
