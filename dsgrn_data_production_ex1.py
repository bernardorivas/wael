import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os
import json
from datetime import datetime

def logmsg(s):
    """Prints a log message with a timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {s}")

def get_or_default(df, index, col, default, dtype=None):
    """Gets a value from a DataFrame row, with a default, and optional type conversion."""
    val = default
    if col in df.columns and index in df.index:
        raw_val = df.loc[index, col]
        # In pandas, empty strings from CSV might be read as such, or as NaN
        if pd.notna(raw_val) and raw_val != '':
            val = raw_val
    if dtype:
        try:
            # First, attempt to convert to float, as it's more general
            float_val = float(val)
            # Then, convert to the desired dtype
            return dtype(float_val)
        except (ValueError, TypeError):
            return dtype(default)
    return val

def parse_mat2x2(cellstr, name, row_idx):
    """Parses a 2x2 matrix from a string (JSON or MATLAB-like)."""
    s = str(cellstr).strip()
    try:
        # Try JSON-like format, e.g., "[[1, 2], [3, 4]]"
        A = np.array(json.loads(s), dtype=float)
    except (json.JSONDecodeError, TypeError):
        # Fallback to MATLAB-like format, e.g., "[1 2; 3 4]"
        s = s.replace('],[', ';').replace('], [', ';')
        s = s.replace('[', '').replace(']', '')
        s = s.replace(',', ' ')
        rows = s.split(';')
        A = np.array([list(map(float, row.split())) for row in rows if row.strip()])

    if A.shape != (2, 2):
        raise ValueError(f"'{name}' must be a 2x2 matrix for CSV row {row_idx}, but got shape {A.shape}")
    return A

def get_required(df, r, name):
    """Gets a required value from the DataFrame."""
    if name not in df.columns:
        raise ValueError(f"CSV missing required column: {name}")
    val = df.loc[r, name]
    if pd.isna(val) or (isinstance(val, str) and not val.strip()):
        raise ValueError(f"CSV column {name} is empty at row {r}")
    return val

# --- ODE and Event Functions ---

def piecewise_ode(t, y, xBound, yBound, U, L):
    """Piecewise ODE system definition."""
    x, v = y
    # Note: MATLAB is 1-based, Python is 0-based.
    # cx depends on y vs yBound: cx = U(2,1) if v<=yBound else L(2,1) -> U[1,0], L[1,0]
    # cy depends on x vs xBound: cy = U(1,2) if x<=xBound else L(1,2) -> U[0,1], L[0,1]
    cx = U[1, 0] if v <= yBound else L[1, 0]
    cy = U[0, 1] if x <= xBound else L[0, 1]
    return [-x + cx, -v + cy]

# Event function to detect leaving the overall domain
def outer_boundary_event(t, y, xMin, xMax, yMin, yMax):
    tol = 1e-9
    # Value is positive inside the domain, becomes zero at the boundary
    return min([y[0] - xMin, xMax - y[0], y[1] - yMin, yMax - y[1]]) - tol
outer_boundary_event.terminal = True
outer_boundary_event.direction = -1  # Event when value goes from positive to negative

# Event function for the vertical switching boundary
def x_boundary_event(t, y, xBound):
    return y[0] - xBound
x_boundary_event.terminal = True
x_boundary_event.direction = 0  # Trigger on any crossing

# Event function for the horizontal switching boundary
def y_boundary_event(t, y, yBound):
    return y[1] - yBound
y_boundary_event.terminal = True
y_boundary_event.direction = 0  # Trigger on any crossing


def dsgrn_data_production_ex1():
    """
    Python translation of DSGRN_data_production_ex1.m.
    Reads parameter sets from param_ex1.csv and, for each row with data_type='dsgrn',
    samples ICs per region, integrates with solve_ivp using boundary events,
    records trajectories, plots, and writes a per-set CSV.
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
        
    param_path = os.path.join(base_dir, 'param_ex1.csv')
    logmsg("Starting DSGRN data production (ex1)...")
    t_global = time.time()

    if not os.path.isfile(param_path):
        raise FileNotFoundError(f"CSV file not found: {param_path}")

    logmsg(f"Loading parameters from 'param_ex1.csv'")
    Tcsv = pd.read_csv(param_path, dtype='string', keep_default_na=False)

    if 'data_type' not in Tcsv.columns:
        raise ValueError("CSV must contain a 'data_type' column.")

    dsgrn_rows = Tcsv[Tcsv['data_type'].str.strip().str.lower() == 'dsgrn']
    print(f"Found {len(dsgrn_rows)} DSGRN set(s) in 'param_ex1.csv'.")

    for r, row_data in dsgrn_rows.iterrows():
        t_set = time.time()
        set_id = r + 1
        print(f"=== DSGRN set {set_id} (CSV row {r}) ===")

        xMin = get_or_default(Tcsv, r, "xMin", 0.0, dtype=float)
        xMax = get_or_default(Tcsv, r, "xMax", 1.5, dtype=float)
        yMin = get_or_default(Tcsv, r, "yMin", 0.0, dtype=float)
        yMax = get_or_default(Tcsv, r, "yMax", 1.5, dtype=float)

        L = parse_mat2x2(get_required(Tcsv, r, "L_np"), "L_np", r)
        U = parse_mat2x2(get_required(Tcsv, r, "U_np"), "U_np", r)
        Th = parse_mat2x2(get_required(Tcsv, r, "Th_np"), "Th_np", r)

        xBound = Th[0, 1]
        yBound = Th[1, 0]

        Tfinal = get_or_default(Tcsv, r, "Tfinal", 20.0, dtype=float)
        n_t = get_or_default(Tcsv, r, "n_timepoints", 2000, dtype=int)
        tspan_full = np.linspace(0, Tfinal, max(2, n_t))

        logmsg(f"Set params: box=([x:{xMin},{xMax}], [y:{yMin},{yMax}]), "
               f"bounds=(xBound:{xBound}, yBound:{yBound}), Tfinal={Tfinal}, n_timepoints={n_t}")

        n1 = get_or_default(Tcsv, r, "numPoints1", 0, dtype=int)
        n2 = get_or_default(Tcsv, r, "numPoints2", 0, dtype=int)
        n3 = get_or_default(Tcsv, r, "numPoints3", 0, dtype=int)
        n4 = get_or_default(Tcsv, r, "numPoints4", 0, dtype=int)

        logmsg(f"Sampling ICs per region: n1={n1}, n2={n2}, n3={n3}, n4={n4}")

        ICs_list, labels_list = [], []
        if n1 > 0:
            ICs_list.append(np.random.rand(n1, 2) * [xBound - xMin, yBound - yMin] + [xMin, yMin])
            labels_list.extend([1] * n1)
        if n2 > 0:
            ICs_list.append(np.random.rand(n2, 2) * [xMax - xBound, yBound - yMin] + [xBound, yMin])
            labels_list.extend([2] * n2)
        if n3 > 0:
            ICs_list.append(np.random.rand(n3, 2) * [xBound - xMin, yMax - yBound] + [xMin, yBound])
            labels_list.extend([3] * n3)
        if n4 > 0:
            ICs_list.append(np.random.rand(n4, 2) * [xMax - xBound, yMax - yBound] + [xBound, yBound])
            labels_list.extend([4] * n4)

        if not ICs_list:
            print("  No initial conditions requested in CSV. Skipping this set.")
            continue

        ICs = np.vstack(ICs_list)
        labels = np.array(labels_list)
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
                if not np.any(seg_t_mask): break
                
                seg_t = tspan_full[seg_t_mask]
                if seg_t[0] > t_current:
                    seg_t = np.insert(seg_t, 0, t_current)

                sol = solve_ivp(
                    piecewise_ode, (seg_t[0], seg_t[-1]), y_current, t_eval=seg_t,
                    args=(xBound, yBound, U, L),
                    events=[
                        lambda t,y, *args: outer_boundary_event(t,y,xMin,xMax,yMin,yMax),
                        lambda t,y, *args: x_boundary_event(t,y,xBound),
                        lambda t,y, *args: y_boundary_event(t,y,yBound)
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
                    if not event_times: break
                    
                    first_event_time = min(event_times)
                    event_idx = next((j for j, ev in enumerate(sol.t_events) if ev.size > 0 and ev[0] == first_event_time), -1)

                    if event_idx == 0: break
                    elif event_idx > 0:
                        t_current = sol.t_events[event_idx][0]
                        y_current = sol.y_events[event_idx][0, :]
                        continue
                else:
                    break
            
            t_full = np.concatenate(t_full_segs) if t_full_segs else np.array([])
            y_full = np.vstack(y_full_segs) if y_full_segs else np.empty((0,2))
            trajectories.append(np.hstack([t_full[:, np.newaxis], y_full]))

            nP = len(t_full)
            traj_data = np.column_stack([np.full(nP, i + 1), np.full(nP, reg_i), t_full, y_full])
            all_traj_data_list.append(traj_data)

            if (i + 1) % step == 0 or (i + 1) == numTotal:
                done, elapsed = i + 1, time.time() - t_set
                rate = elapsed / done
                eta = rate * (numTotal - done)
                logmsg(f"Set {set_id}: {done}/{numTotal} ({100.0 * done / numTotal:5.1f}%) elapsed={elapsed:5.1f}s ETA={eta:5.1f}s")

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
                ax.plot(trajectories[i][:, 1], trajectories[i][:, 2], color=colors[labels[i]-1], linewidth=1.5)
            ax.plot(ICs[i, 0], ICs[i, 1], 'ko', markerfacecolor='k', markersize=4)

        ax.set_xlabel('x'), ax.set_ylabel('y')
        ax.set_title(f'Piecewise ODE Trajectories (set {set_id})')
        ax.set_xlim(xMin, xMax), ax.set_ylim(yMin, yMax)
        ax.set_aspect('equal', adjustable='box'), ax.grid(True)
        
        plot_path = os.path.join(output_dir, f'DSGRN_Set{set_id}_ex1.png')
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  • Saved plot -> {os.path.relpath(plot_path)}")

        logmsg(f"Set {set_id}: writing per-set CSV")
        per_set_csv_path = os.path.join(output_dir, f'DSGRN_trajectories_ex1_set{set_id}.csv')
        if all_traj_data_list:
            all_traj_data = np.vstack(all_traj_data_list)
            df_out = pd.DataFrame(all_traj_data, columns=['Trajectory', 'Region', 'Time', 'x0', 'x1'])
            df_out.to_csv(per_set_csv_path, index=False)
            print(f"  • Saved data -> {os.path.relpath(per_set_csv_path)}")
        else:
            print(f"  • No data to save for set {set_id}.")


        logmsg(f"Finished set {set_id} in {time.time() - t_set:5.2f}s")

    print("\nAll DSGRN sets processed for exercise 1.")
    logmsg(f"Total wall time: {time.time() - t_global:5.2f}s")


if __name__ == '__main__':
    dsgrn_data_production_ex1()
