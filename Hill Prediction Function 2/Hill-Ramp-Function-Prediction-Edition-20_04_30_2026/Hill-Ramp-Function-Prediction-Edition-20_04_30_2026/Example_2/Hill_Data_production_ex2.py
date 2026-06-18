


import ast
import json
import zlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc




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

from scipy.integrate import solve_ivp
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "generated_data_ex2" / "hill"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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




def hill(x, a, b, theta, d):

    return a + (b - a) * (x**d) / (theta**d + x**d)




param_path = BASE_DIR / "param_ex2.csv"
if not param_path.exists():
    raise FileNotFoundError(f"Params CSV not found: {param_path}")

param_df  = pd.read_csv(param_path)

hill_rows = param_df[param_df["data_type"].str.lower() == "hill"].copy()
hill_rows["__csv_row"] = hill_rows.index
hill_rows = hill_rows.reset_index(drop=True)

if hill_rows.empty:
    print("No rows with data_type == 'Hill' were found in the CSV.")
    exit(0)

print(f"Found {len(hill_rows)} Hill set(s) in {param_path}.")




for k, row in hill_rows.iterrows():

    csv_row = int(row["__csv_row"])
    set_num = csv_row + 1
    print(f"\n=== Simulating set {set_num} (CSV row {csv_row}) ===")
    ic_seed = resolve_ic_seed(row, set_num)
    rng = np.random.default_rng(ic_seed)



    xMax   = float(row.get("xMax", 1.3))
    yMax   = float(row.get("yMax", 1.3))
    Tfinal = float(row.get("Tfinal", 20))
    n_t    = int(row.get("n_timepoints", 8000))
    n_ic   = int(row.get("Hill_IC", 1))


    try:
        L  = np.array(ast.literal_eval(row["L_np"]),  dtype=float)
        U  = np.array(ast.literal_eval(row["U_np"]),  dtype=float)
        Th = np.array(ast.literal_eval(row["Th_np"]), dtype=float)
    except Exception as e:
        raise ValueError(f"Failed to parse L_np/U_np/Th_np for CSV row {csv_row}: {e}")

    if L.shape != (2,2) or U.shape != (2,2) or Th.shape != (2,2):
        raise ValueError(f"L, U, Th must be 2x2 for CSV row {csv_row}")


    try:
        scale_max = float(max(np.max(L), np.max(U)))
    except ValueError:
        scale_max = 1.0
    xMax = yMax = 1.5 * scale_max
    print(f"  Auto-set xMax=yMax={xMax:.4g} (1.5 * max(L,U) with max={scale_max:.4g})")



    D = np.zeros((2, 2), dtype=float)
    try:
        D[0, 0] = float(row["D11_input"])
    except Exception:
        D[0, 0] = 12.0
    try:
        D[0, 1] = float(row["D12_input"])
    except Exception:
        D[0, 1] = 12.0
    try:
        D[1, 0] = float(row["D21_input"])
    except Exception:
        D[1, 0] = 12.0
    try:
        D[1, 1] = float(row["D22_input"])
    except Exception:
        D[1, 1] = 12.0


    def odefun(t, x):
        T = Th
        hill00 = lambda xx: hill(xx, L[0,0], U[0,0], T[0,0], D[0,0])
        hill01 = lambda xx: hill(xx, U[0,1], L[0,1], T[0,1], D[0,1])
        hill10 = lambda xx: hill(xx, L[1,0], U[1,0], T[1,0], D[1,0])
        hill11 = lambda xx: hill(xx, L[1,1], U[1,1], T[1,1], D[1,1])

        x0, x1 = x
        dx0 = -x0 + hill00(x0) + hill10(x1)
        dx1 = -x1 + hill01(x0) * hill11(x1)
        return [dx0, dx1]


    t_eval = np.linspace(0.0, Tfinal, n_t)




    xMin = 0.0
    yMin = 0.0
    xMax = 6.0
    yMax = 3.0

    sampler = qmc.LatinHypercube(d=2, seed=ic_seed)
    initial_conditions = qmc.scale(sampler.random(n_ic), l_bounds=[xMin, yMin], u_bounds=[xMax, yMax])
    print(f"  ICs (Latin hypercube): Hill_IC={n_ic} in [0,{xMax}]×[0,{yMax}]")


    fig, ax = plt.subplots()
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title(f"2-Node Hill Cycle (Set {set_num}), t ∈ [0, {Tfinal}]")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")


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
        ax.plot(Y[:,0], Y[:,1], linewidth=1.2)


        for t_val, (y0, y1) in zip(sol.t, Y):
            all_data.append([traj_id, t_val, y0, y1])


    ax.plot(initial_conditions[:,0], initial_conditions[:,1],
            "ro", markersize=4, markerfacecolor="r")


    plot_path = OUTPUT_DIR / f"Set{set_num}_TrajectoryHILL_ex2.png"
    csv_path  = OUTPUT_DIR / f"Data_Set{set_num}_HILL_ex2.csv"

    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


    df_out = pd.DataFrame(all_data, columns=["Trajectory", "Time", "x0", "x1"])
    df_out['ic_seed'] = ic_seed







    add_noise_flag = str(row.get("add_noise", "no")).strip().lower()
    if add_noise_flag in ("yes", "y", "true", "1"):
        try:
            noise_ub = float(row.get("noise_ub"))
        except Exception:
            raise ValueError(f"add_noise requested for CSV row {csv_row} but noise_ub is missing or invalid.")
        if noise_ub < 0:
            raise ValueError(f"noise_ub must be non-negative for CSV row {csv_row}; got {noise_ub}")

        n_rows = len(df_out)
        old_noise_scale, new_noise_scale, same_index_scale, min_nonzero_l = compute_noise_reference_scales(L, U, csv_row)
        noise_scale = noise_ub / 100.0
        noise_amp = noise_scale * new_noise_scale
        noise_x0 = rng.uniform(-1.0, 1.0, size=n_rows) * noise_amp
        noise_x1 = rng.uniform(-1.0, 1.0, size=n_rows) * noise_amp

        df_out['x0'] = np.clip(df_out['x0'] + noise_x0, a_min=0.0, a_max=None)
        df_out['x1'] = np.clip(df_out['x1'] + noise_x1, a_min=0.0, a_max=None)

        df_out['add_noise'] = 'yes'
        df_out['noise_ub'] = noise_ub
        df_out['noise_ref_old_min_abs_lij_minus_ukl'] = old_noise_scale
        df_out['noise_ref_new_min_same_lu_or_l'] = new_noise_scale
        df_out['noise_ref_same_index_min_abs_lij_minus_uij'] = same_index_scale
        df_out['noise_ref_min_nonzero_l'] = min_nonzero_l
        print(f"  Noise scale old min(|Lij - Ukl|) = {old_noise_scale:.6g}")
        print(f"  Noise scale new min(min(|Lij - Uij|), Lkl) = {new_noise_scale:.6g} (same-index={same_index_scale:.6g}, min nonzero L={min_nonzero_l:.6g})")
        print(f"  Added bounded uniform observation noise with ub={noise_ub}% to {n_rows} rows (clipped at 0)")
    else:
        df_out['add_noise'] = 'no'

    df_out.to_csv(csv_path, index=False)

    print(f" • Saved data  -> {csv_path.relative_to(BASE_DIR)}")
    print(f" • Saved plot  -> {plot_path.relative_to(BASE_DIR)}")



    if add_noise_flag in ("yes", "y", "true", "1"):
        noisy_plot_path = OUTPUT_DIR / f"Set{set_num}_TrajectoryHILL_ex2_noisy.png"
        fig2, ax2 = plt.subplots()
        ax2.set_xlabel("x0")
        ax2.set_ylabel("x1")
        ax2.set_title(f"2-Node Hill Cycle (Set {set_num}) — noisy observations, t ∈ [0, {Tfinal}]")
        ax2.grid(True)
        ax2.set_aspect("equal", adjustable="box")


        for traj_id in sorted(df_out['Trajectory'].unique()):
            sub = df_out[df_out['Trajectory'] == traj_id].sort_values('Time')
            ax2.plot(sub['x0'].values, sub['x1'].values, linewidth=1.2)


        try:
            ax2.plot(initial_conditions[:,0], initial_conditions[:,1],
                     "ro", markersize=4, markerfacecolor="r")
        except Exception:

            pass

        fig2.savefig(noisy_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f" • Saved noisy plot -> {noisy_plot_path.relative_to(BASE_DIR)}")
    print(f"   Trajectories: {n_ic}, points total: {len(all_data)}")

print("\nAll Hill sets processed for exercise 2.")
