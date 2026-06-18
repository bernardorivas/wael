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

def logmsg(s):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {s}")

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
    if A.shape != (2, 2):
        raise ValueError(f"'{name}' must be a 2x2 matrix for CSV row {row_idx}, but got shape {A.shape}")
    return A

def get_required(df, r, name):
    if name not in df.columns:
        raise ValueError(f"CSV missing required column: {name}")
    val = df.loc[r, name]
    if pd.isna(val) or (isinstance(val, str) and not val.strip()):
        raise ValueError(f"CSV column {name} is empty at row {r}")
    return val



def piecewise_ode(t, y, xBound, yBound, U, L):
    x, v = y
    cx = U[1, 0] if v <= yBound else L[1, 0]
    cy = U[0, 1] if x <= xBound else L[0, 1]
    return [-x + cx, -v + cy]

def outer_boundary_event(t, y, xMin, xMax, yMin, yMax):
    tol = 1e-9
    return min([y[0] - xMin, xMax - y[0], y[1] - yMin, yMax - y[1]]) - tol
outer_boundary_event.terminal = True
outer_boundary_event.direction = -1

def x_boundary_event(t, y, xBound):
    return y[0] - xBound
x_boundary_event.terminal = True
x_boundary_event.direction = 0

def y_boundary_event(t, y, yBound):
    return y[1] - yBound
y_boundary_event.terminal = True
y_boundary_event.direction = 0

def piecewise_data_production_ex1():




    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()


    data_dir = os.path.join(base_dir, 'generated_data', 'piecewise')
    plot_dir = os.path.join(base_dir, 'generated_plots', 'piecewise')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    param_path = os.path.join(base_dir, 'param_ex1.csv')
    logmsg("Starting piecewise data production (ex1)...")
    t_global = time.time()

    if not os.path.isfile(param_path):
        raise FileNotFoundError(f"CSV file not found: {param_path}")

    logmsg("Loading parameters from 'param_ex1.csv'")
    Tcsv = pd.read_csv(param_path, dtype='string', keep_default_na=False)

    if 'data_type' not in Tcsv.columns:
        raise ValueError("CSV must contain a 'data_type' column.")

    piecewise_rows = Tcsv[Tcsv['data_type'].str.strip().str.lower() == 'piecewise']
    print(f"Found {len(piecewise_rows)} piecewise set(s) in 'param_ex1.csv'.")

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
        yBound = Th[1, 0]


        try:
            _scale_max = float(max(np.max(L), np.max(U)))
        except ValueError:
            _scale_max = 1.0
        xMax = yMax = 1.5 * _scale_max
        logmsg(f"Auto-set xMax=yMax={xMax:.4g} (1.5 * max(L,U) with max={_scale_max:.4g}; CSV xMax/yMax ignored)")

        Tfinal = get_or_default(Tcsv, r, "Tfinal", 20.0, dtype=float)
        n_t = get_or_default(Tcsv, r, "n_timepoints", 2000, dtype=int)
        tspan_full = np.linspace(0, Tfinal, max(2, n_t))

        logmsg(f"Set params: box=([x:{xMin},{xMax}], [y:{yMin},{yMax}]), "
               f"bounds=(xBound:{xBound}, yBound:{yBound}), Tfinal={Tfinal}, n_timepoints={n_t}")

        num_ic = get_or_default(Tcsv, r, "piecewise_IC", 0, dtype=int)
        logmsg(f"Sampling {num_ic} ICs via Latin hypercube on [0,6]x[0,6] (ic_seed={ic_seed})")

        if num_ic <= 0:
            print("  No initial conditions requested in CSV. Skipping this set.")
            continue

        sampler = qmc.LatinHypercube(d=2, seed=rng)
        lhs_unit = sampler.random(num_ic)
        domain_lower = np.array([0.0, 0.0])
        domain_upper = np.array([6.0, 6.0])
        ICs = qmc.scale(lhs_unit, domain_lower, domain_upper)

        labels = np.ones(num_ic, dtype=int)
        labels[(ICs[:, 0] > xBound) & (ICs[:, 1] <= yBound)] = 2
        labels[(ICs[:, 0] <= xBound) & (ICs[:, 1] > yBound)] = 3
        labels[(ICs[:, 0] > xBound) & (ICs[:, 1] > yBound)] = 4

        region_counts = {reg: int(np.sum(labels == reg)) for reg in range(1, 5)}
        logmsg("IC distribution by region after LHS: "
               f"n1={region_counts[1]}, n2={region_counts[2]}, "
               f"n3={region_counts[3]}, n4={region_counts[4]}")

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
                    args=(xBound, yBound, U, L),
                    events=[
                        lambda t, y, *args: outer_boundary_event(t, y, xMin, xMax, yMin, yMax),
                        lambda t, y, *args: x_boundary_event(t, y, xBound),
                        lambda t, y, *args: y_boundary_event(t, y, yBound)
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
            y_full = np.vstack(y_full_segs) if y_full_segs else np.empty((0, 2))
            trajectories.append(np.hstack([t_full[:, np.newaxis], y_full]))

            nP = len(t_full)
            traj_data = np.column_stack([np.full(nP, i + 1), np.full(nP, reg_i), t_full, y_full])
            all_traj_data_list.append(traj_data)

            if (i + 1) % step == 0 or (i + 1) == numTotal:
                done, elapsed = i + 1, time.time() - t_set
                rate = elapsed / done
                eta = rate * (numTotal - done)
                logmsg(f"Set {set_id}: {done}/{numTotal} ({100.0 * done / numTotal:5.1f}%) "
                       f"elapsed={elapsed:5.1f}s ETA={eta:5.1f}s")

        logmsg(f"Set {set_id}: integration complete in {time.time() - t_set:5.2f}s")

        logmsg(f"Set {set_id}: plotting trajectories")
        fig, ax = plt.subplots(figsize=(6, 6))

        ax.add_patch(patches.Rectangle((xMin, yMin), xBound - xMin, yBound - yMin, color=[0.8, 0.8, 1], alpha=0.5, ec='none'))
        ax.add_patch(patches.Rectangle((xBound, yMin), xMax - xBound, yBound - yMin, color=[0.8, 1, 0.8], alpha=0.5, ec='none'))
        ax.add_patch(patches.Rectangle((xMin, yBound), xBound - xMin, yMax - yBound, color=[1, 0.8, 0.8], alpha=0.5, ec='none'))
        ax.add_patch(patches.Rectangle((xBound, yBound), xMax - xBound, yMax - yBound, color=[1, 1, 0.8], alpha=0.5, ec='none'))

        ax.plot([xMin, xMax], [yBound, yBound], 'k', linewidth=2)
        ax.plot([xBound, xBound], [yMin, yMax], 'k', linewidth=2)

        colors = np.array([[0, 0, 1], [0, 0.5, 0], [1, 0, 0], [0.85, 0.65, 0]])
        for i in range(numTotal):
            if trajectories[i].shape[0] > 0:
                ax.plot(trajectories[i][:, 1], trajectories[i][:, 2], color=colors[labels[i] - 1], linewidth=1.5)
            ax.plot(ICs[i, 0], ICs[i, 1], 'ko', markerfacecolor='k', markersize=4)

        ax.set_xlabel('x'), ax.set_ylabel('y')
        ax.set_title(f'Piecewise ODE Trajectories (set {set_id})')
        ax.set_xlim(xMin, xMax), ax.set_ylim(yMin, yMax)
        ax.set_aspect('equal', adjustable='box'), ax.grid(True)

        plot_path = os.path.join(plot_dir, f'piecewise_Set{set_id}_ex1.png')
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  • Saved plot -> {os.path.relpath(plot_path, base_dir)}")

        logmsg(f"Set {set_id}: writing per-set CSV")
        per_set_csv_path = os.path.join(data_dir, f'piecewise_trajectories_ex1_set{set_id}.csv')
        if all_traj_data_list:
            all_traj_data = np.vstack(all_traj_data_list)
            df_out = pd.DataFrame(all_traj_data, columns=['Trajectory', 'Region', 'Time', 'x0', 'x1'])
            df_out['ic_seed'] = ic_seed




            add_noise_flag = str(get_or_default(Tcsv, r, "add_noise", "")).strip().lower()
            add_noise = add_noise_flag in ("yes", "y", "true", "1")
            noise_ub_percent = get_or_default(Tcsv, r, "noise_ub", 0.0, dtype=float)
            old_noise_scale, new_noise_scale, same_index_scale, min_nonzero_l = compute_noise_reference_scales(L, U, r)
            noise_ub_abs = (noise_ub_percent / 100.0) * new_noise_scale

            if add_noise and noise_ub_abs > 0:
                noise = rng.uniform(-noise_ub_abs, noise_ub_abs, size=(len(df_out), 2))
                df_out[["x0", "x1"]] = df_out[["x0", "x1"]] + noise

                df_out['x0'] = np.maximum(df_out['x0'].to_numpy(dtype=float), 0.0)
                df_out['x1'] = np.maximum(df_out['x1'].to_numpy(dtype=float), 0.0)
                df_out['add_noise'] = 'yes'
                df_out['noise_ub_percent'] = float(noise_ub_percent)
                df_out['noise_ub_abs'] = float(noise_ub_abs)
                df_out['noise_ref_old_min_abs_lij_minus_ukl'] = old_noise_scale
                df_out['noise_ref_new_min_same_lu_or_l'] = new_noise_scale
                df_out['noise_ref_same_index_min_abs_lij_minus_uij'] = same_index_scale
                df_out['noise_ref_min_nonzero_l'] = min_nonzero_l
                print(
                    f"  Added bounded uniform observation noise with ub={noise_ub_percent}% "
                    f"of new scale min(min(|Lij-Uij|), Lkl)={new_noise_scale:.6g} "
                    f"(old scale min(|Lij-Ukl|)={old_noise_scale:.6g}; "
                    f"same-index={same_index_scale:.6g}; min nonzero L={min_nonzero_l:.6g}) "
                    f"(abs ub={noise_ub_abs:.6g}) to {len(df_out)} rows; "
                    f"applied post-noise clipping at 0"
                )
            else:
                df_out['add_noise'] = 'no'
                df_out['noise_ub_percent'] = float(noise_ub_percent)
                df_out['noise_ub_abs'] = float(noise_ub_abs)
                df_out['noise_ref_old_min_abs_lij_minus_ukl'] = old_noise_scale
                df_out['noise_ref_new_min_same_lu_or_l'] = new_noise_scale
                df_out['noise_ref_same_index_min_abs_lij_minus_uij'] = same_index_scale
                df_out['noise_ref_min_nonzero_l'] = min_nonzero_l

            df_out.to_csv(per_set_csv_path, index=False)
            print(f"  • Saved data -> {os.path.relpath(per_set_csv_path, base_dir)}")



            if add_noise and noise_ub_abs > 0:
                noisy_plot_path = os.path.join(plot_dir, f'piecewise_Set{set_id}_ex1_noisy.png')
                try:
                    fig2, ax2 = plt.subplots(figsize=(6, 6))
                    ax2.add_patch(patches.Rectangle((xMin, yMin), xBound - xMin, yBound - yMin, color=[0.8, 0.8, 1], alpha=0.5, ec='none'))
                    ax2.add_patch(patches.Rectangle((xBound, yMin), xMax - xBound, yBound - yMin, color=[0.8, 1, 0.8], alpha=0.5, ec='none'))
                    ax2.add_patch(patches.Rectangle((xMin, yBound), xBound - xMin, yMax - yBound, color=[1, 0.8, 0.8], alpha=0.5, ec='none'))
                    ax2.add_patch(patches.Rectangle((xBound, yBound), xMax - xBound, yMax - yBound, color=[1, 1, 0.8], alpha=0.5, ec='none'))
                    ax2.plot([xMin, xMax], [yBound, yBound], 'k', linewidth=2)
                    ax2.plot([xBound, xBound], [yMin, yMax], 'k', linewidth=2)

                    for tr in sorted(df_out['Trajectory'].unique()):
                        sub = df_out[df_out['Trajectory'] == tr].sort_values('Time')
                        ax2.plot(sub['x0'].values, sub['x1'].values, linewidth=0.8, color='tab:blue')
                        ax2.plot(sub['x0'].values[0], sub['x1'].values[0], 'ko', markerfacecolor='k', markersize=3)

                    ax2.set_xlabel('x'); ax2.set_ylabel('y')
                    ax2.set_title(f'Piecewise Trajectories (noisy) for Set {set_id}')
                    ax2.set_xlim(xMin, xMax); ax2.set_ylim(yMin, yMax)
                    ax2.set_aspect('equal', adjustable='box'); ax2.grid(True)

                    fig2.savefig(noisy_plot_path, dpi=150)
                    plt.close(fig2)
                    print(f"  • Saved noisy plot -> {os.path.relpath(noisy_plot_path, base_dir)}")
                except Exception as e:
                    print(f"  • Failed to save noisy plot: {e}")
        else:
            print(f"  • No data to save for set {set_id}.")

        logmsg(f"Finished set {set_id} in {time.time() - t_set:5.2f}s")

    print("\nAll piecewise sets processed for exercise 1.")
    logmsg(f"Total wall time: {time.time() - t_global:5.2f}s")

if __name__ == '__main__':
    piecewise_data_production_ex1()
