import pandas as pd
import numpy as np
import zlib
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os
import json
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

def logmsg(s):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {s}")


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

def get_or_default(df, index, col, default, dtype=None):
    val = default
    if col in df.columns and index in df.index:
        raw_val = df.loc[index, col]
        if pd.notna(raw_val) and raw_val != '':
            val = raw_val
    if dtype:
        try:
            float_val = float(val)
            return dtype(float_val)
        except (ValueError, TypeError):
            return dtype(default)
    return val

def parse_mat2x2(cellstr, name, row_idx):
    s = str(cellstr).strip()
    try:
        A = np.array(json.loads(s), dtype=float)
    except (json.JSONDecodeError, TypeError):
        s = s.replace('],[', ';').replace('], [', ';')
        s = s.replace('[', '').replace(']', '')
        s = s.replace(',', ' ')
        rows = s.split(';')
        A = np.array([list(map(float, row.split())) for row in rows if row.strip()])
    if A.shape != (3, 3):
        raise ValueError(f"'{name}' must be a 3x3 matrix for CSV row {row_idx}, but got shape {A.shape}")
    return A

def get_required(df, r, name):
    if name not in df.columns:
        raise ValueError(f"CSV missing required column: {name}")
    val = df.loc[r, name]
    if pd.isna(val) or (isinstance(val, str) and not val.strip()):
        raise ValueError(f"CSV column {name} is empty at row {r}")
    return val





def piecewise_ode(t, y, xBound, yBound, zBound, U, L, D):
    x, v, z = y

















    term_from_z = U[2,0] if z <= zBound else L[2,0]










    fx2 = U[2,0] if z <= zBound else L[2,0]
    fx0 = U[0,1] if x <= xBound else L[0,1]
    fx1 = U[1,2] if v <= yBound else L[1,2]


    return [-x + fx2, -v + fx0, -z + fx1]

def boundary_event(t, y, xMin, xMax, yMin, yMax, zMin, zMax):
    tol = 1e-9
    return min([y[0]-xMin, xMax-y[0],
                y[1]-yMin, yMax-y[1],
                y[2]-zMin, zMax-y[2]]) - tol
boundary_event.terminal = True
boundary_event.direction = -1


def x_thresh(t, y, xb): return y[0] - xb
x_thresh.terminal = True
x_thresh.direction = 0


def y_thresh(t, y, yb): return y[1] - yb
y_thresh.terminal = True
y_thresh.direction = 0


def z_thresh(t, y, zb): return y[2] - zb
z_thresh.terminal = True
z_thresh.direction = 0

def piecewise_data_production_ex3():



    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path(os.getcwd())

    data_dir = base_dir / "generated_data" / "piecewise"
    plot_dir = base_dir / "generated_plots" / "piecewise"
    ensure_dir(data_dir)
    ensure_dir(plot_dir)

    param_path = base_dir / "param_ex3.csv"
    logmsg("Starting piecewise data production (ex3)...")
    t_global = time.time()

    if not param_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {param_path}")

    logmsg("Loading parameters from 'param_ex3.csv'")
    Tcsv = pd.read_csv(param_path, dtype='string', keep_default_na=False)

    if 'data_type' not in Tcsv.columns:
        raise ValueError("CSV must contain a 'data_type' column.")

    piecewise_rows = Tcsv[Tcsv['data_type'].str.strip().str.lower() == 'piecewise']
    print(f"Found {len(piecewise_rows)} piecewise set(s) in 'param_ex3.csv'.")

    for r, row_data in piecewise_rows.iterrows():
        t_set = time.time()




        set_id = int(r) + 1
        print(f"=== piecewise set {set_id} (CSV row {set_id - 1}) ===")
        row_dict = Tcsv.iloc[int(r)].to_dict()
        ic_seed = resolve_ic_seed(row_dict, set_id)
        rng = np.random.default_rng(ic_seed)
        xMin = get_or_default(Tcsv, r, "xMin", 0.0, dtype=float)

        _csv_xMax_ignored = get_or_default(Tcsv, r, "xMax", 1.5, dtype=float)
        yMin = get_or_default(Tcsv, r, "yMin", 0.0, dtype=float)
        _csv_yMax_ignored = get_or_default(Tcsv, r, "yMax", 1.5, dtype=float)

        L = parse_mat2x2(get_required(Tcsv, r, "L_np"), "L_np", r)
        U = parse_mat2x2(get_required(Tcsv, r, "U_np"), "U_np", r)
        Th = parse_mat2x2(get_required(Tcsv, r, "Th_np"), "Th_np", r)

        xBound = Th[0, 1]
        yBound = Th[1, 2]
        zBound = Th[2, 0]


        try:
            _scale_max = float(max(np.max(L), np.max(U)))
        except ValueError:
            _scale_max = 1.0
        xMax = yMax = zMax = 1.5 * _scale_max
        logmsg(f"Auto-set xMax=yMax=zMax={xMax:.4g} (1.5 * max(L,U) with max={_scale_max:.4g}; CSV xMax/yMax ignored)")

        Tfinal = get_or_default(Tcsv, r, "Tfinal", 20.0, dtype=float)
        n_t = get_or_default(Tcsv, r, "n_timepoints", 2000, dtype=int)
        tspan_full = np.linspace(0, Tfinal, max(2, n_t))

        logmsg(f"Set params: box=([0,{xMax}]^3), "
               f"bounds=(x:{xBound}, y:{yBound}, z:{zBound}), Tfinal={Tfinal}, n_timepoints={n_t}")

        num_ic = get_or_default(Tcsv, r, "piecewise_IC", 0, dtype=int)
        logmsg(f"Sampling {num_ic} ICs via Latin hypercube on [0,6]x[0,6] (ic_seed={ic_seed})")

        if num_ic <= 0:
            print("  No initial conditions requested in CSV. Skipping this set.")
            continue

        sampler = qmc.LatinHypercube(d=3, seed=rng)
        lhs_unit = sampler.random(num_ic)
        domain_lower = np.array([0.0, 0.0, 0.0])
        zMin = 0.0
        domain_upper = np.array([6.0, 6.0, 6.0])






        domain_upper = np.array([xMax, yMax, zMax])

        ICs = qmc.scale(lhs_unit, domain_lower, domain_upper)




        labels = np.zeros(num_ic, dtype=int)
        logmsg(f"IC distribution: {num_ic} points sampled.")

        numTotal = ICs.shape[0]
        print(f"  Total trajectories: {numTotal}")
        logmsg("Integrating trajectories...")

        relTol = get_or_default(Tcsv, r, "relTol", 1e-10, dtype=float)
        absTol = get_or_default(Tcsv, r, "absTol", 1e-12, dtype=float)

        trajectories, all_traj_data_list = [], []
        step = max(1, numTotal // 10)

        for i in range(numTotal):
            y0, reg_i = ICs[i, :], labels[i]
            t_current, y_current = 0.0, y0
            t_full_segs, y_full_segs = [], []

            while t_current < Tfinal:
                seg_t_mask = tspan_full >= t_current
                if not np.any(seg_t_mask):
                    break

                seg_t = tspan_full[seg_t_mask]
                if seg_t[0] > t_current:
                    seg_t = np.insert(seg_t, 0, t_current)

                sol = solve_ivp(
                    piecewise_ode, (seg_t[0], seg_t[-1]), y_current, t_eval=seg_t,
                    args=(xBound, yBound, zBound, U, L, None),
                    events=[
                        lambda t, y, *args: boundary_event(t, y, xMin, xMax, yMin, yMax, zMin, zMax),
                        lambda t, y, *args: x_thresh(t, y, xBound),
                        lambda t, y, *args: y_thresh(t, y, yBound),
                        lambda t, y, *args: z_thresh(t, y, zBound),
                    ],
                    rtol=relTol, atol=absTol
                )

                if not t_full_segs:
                    t_full_segs.append(sol.t)
                    y_full_segs.append(sol.y.T)
                else:
                    t_full_segs.append(sol.t[1:])
                    y_full_segs.append(sol.y.T[1:, :])

                if sol.status == 1:
                    event_times = [ev[0] for ev in sol.t_events if ev.size > 0]
                    if not event_times:
                        break
                    first_event_time = min(event_times)
                    event_idx = next((j for j, ev in enumerate(sol.t_events)
                                      if ev.size > 0 and ev[0] == first_event_time), -1)

                    if event_idx == 0:
                        break
                    elif event_idx > 0:
                        t_current = sol.t_events[event_idx][0]
                        y_current = sol.y_events[event_idx][0, :]
                        continue
                else:
                    break

            t_full = np.concatenate(t_full_segs) if t_full_segs else np.array([])
            y_full = np.vstack(y_full_segs) if y_full_segs else np.empty((0, 3))
            y_full = np.clip(y_full, 0.0, None)
            trajectories.append(np.hstack([t_full[:, np.newaxis], y_full]))

            nP = len(t_full)

            traj_data = np.column_stack([np.full(nP, i + 1), np.zeros(nP), t_full, y_full])
            all_traj_data_list.append(traj_data)

            if (i + 1) % step == 0 or (i + 1) == numTotal:
                done, elapsed = i + 1, time.time() - t_set
                rate = elapsed / done
                eta = rate * (numTotal - done)
                logmsg(f"Set {set_id}: {done}/{numTotal} ({100.0 * done / numTotal:5.1f}%) "
                       f"elapsed={elapsed:5.1f}s ETA={eta:5.1f}s")

        logmsg(f"Set {set_id}: integration complete in {time.time() - t_set:5.2f}s")

        logmsg(f"Set {set_id}: plotting trajectories")
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection='3d')





        for i in range(numTotal):
            if trajectories[i].shape[0] > 0:
                ax.plot(trajectories[i][:, 1], trajectories[i][:, 2], trajectories[i][:, 3], linewidth=1.5)
            ax.scatter(ICs[i, 0], ICs[i, 1], ICs[i, 2], c='k', marker='o', s=10)

        ax.set_xlabel('x0'), ax.set_ylabel('x1'), ax.set_zlabel('x2')
        ax.set_title(f'Piecewise ODE Trajectories (set {set_id})')

        ax.grid(True)

        plot_path = plot_dir / f"piecewise_Set{set_id}_ex3.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  • Saved plot -> {os.path.relpath(plot_path)}")

        logmsg(f"Set {set_id}: writing per-set CSV")
        per_set_csv_path = data_dir / f"piecewise_trajectories_ex3_set{set_id}.csv"
        if all_traj_data_list:
            all_traj_data = np.vstack(all_traj_data_list)
            df_out = pd.DataFrame(all_traj_data, columns=['Trajectory', 'Region', 'Time', 'x0', 'x1', 'x2'])
            df_out[["x0", "x1", "x2"]] = df_out[["x0", "x1", "x2"]].clip(lower=0.0)
            df_out['ic_seed'] = ic_seed


            add_noise_flag = str(get_or_default(Tcsv, r, "add_noise", "")).strip().lower()
            add_noise = add_noise_flag in ("yes", "y", "true", "1")
            noise_ub_pct = get_or_default(Tcsv, r, "noise_ub", 0.0, dtype=float)

            if noise_ub_pct < 0:
                raise ValueError(f"noise_ub must be non-negative for CSV row {r}; got {noise_ub_pct}")

            old_noise_scale, new_noise_scale, same_index_scale, min_nonzero_l = compute_noise_reference_scales(L, U, r)
            noise_ub_abs = (noise_ub_pct / 100.0) * new_noise_scale

            if add_noise and noise_ub_pct > 0:
                noise = rng.uniform(-noise_ub_abs, noise_ub_abs, size=(len(df_out), 3))
                df_out[["x0", "x1", "x2"]] = df_out[["x0", "x1", "x2"]] + noise
                df_out[["x0", "x1", "x2"]] = df_out[["x0", "x1", "x2"]].clip(lower=0.0)
                df_out['add_noise'] = 'yes'
                df_out['noise_ub'] = float(noise_ub_pct)
                df_out['noise_ref_old_min_abs_lij_minus_ukl'] = old_noise_scale
                df_out['noise_ref_new_min_same_lu_or_l'] = new_noise_scale
                df_out['noise_ref_same_index_min_abs_lij_minus_uij'] = same_index_scale
                df_out['noise_ref_min_nonzero_l'] = min_nonzero_l
                print(f"  Noise scale old min(|Lij - Ukl|) = {old_noise_scale:.6g}")
                print(f"  Noise scale new min(min(|Lij - Uij|), Lkl) = {new_noise_scale:.6g} (same-index={same_index_scale:.6g}, min nonzero L={min_nonzero_l:.6g})")
                print(f"  Added bounded uniform observation noise with ub={noise_ub_pct}% (abs ub={noise_ub_abs:.6g}) to {len(df_out)} rows")
            else:
                df_out['add_noise'] = 'no'
                df_out['noise_ub'] = float(noise_ub_pct)
                df_out['noise_ref_old_min_abs_lij_minus_ukl'] = old_noise_scale
                df_out['noise_ref_new_min_same_lu_or_l'] = new_noise_scale
                df_out['noise_ref_same_index_min_abs_lij_minus_uij'] = same_index_scale
                df_out['noise_ref_min_nonzero_l'] = min_nonzero_l

            df_out.to_csv(per_set_csv_path, index=False)
            print(f"  • Saved data -> {os.path.relpath(per_set_csv_path)}")



            if add_noise and noise_ub_pct > 0:
                noisy_plot_path = plot_dir / f"piecewise_Set{set_id}_ex3_noisy.png"
                try:
                    fig2 = plt.figure(figsize=(6, 6))
                    ax2 = fig2.add_subplot(projection='3d')

                    for tr in sorted(df_out['Trajectory'].unique()):
                        sub = df_out[df_out['Trajectory'] == tr].sort_values('Time')
                        ax2.plot(sub['x0'].values, sub['x1'].values, sub['x2'].values, linewidth=0.8)


                    ax2.set_xlabel('x0'); ax2.set_ylabel('x1'); ax2.set_zlabel('x2')
                    ax2.set_title(f'Piecewise Trajectories (noisy) for Set {set_id}')
                    ax2.grid(True)

                    fig2.savefig(noisy_plot_path, dpi=150)
                    plt.close(fig2)
                    print(f"  • Saved noisy plot -> {os.path.relpath(noisy_plot_path)}")
                except Exception as e:
                    print(f"  • Failed to save noisy plot: {e}")
        else:
            print(f"  • No data to save for set {set_id}.")

        logmsg(f"Finished set {set_id} in {time.time() - t_set:5.2f}s")

if __name__ == '__main__':
    piecewise_data_production_ex3()
