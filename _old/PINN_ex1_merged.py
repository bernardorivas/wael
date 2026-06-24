#!/usr/bin/env python3
# PINN_ex1_merged.py
#
# Unified runner that merges PINN_ex1.py and PINN_ex1_learnable_feats.py.
# - Reads param_ex1.csv and a column "Learned_Feats" (yes/no).
# - If Learned_Feats == yes, enables learnable feature parameters for the time feature map
#   (Fourier, RBF, or Spline). Otherwise, feature map is fixed (original behavior).
# - Results file includes ALL columns from param_ex1.csv plus training metrics and,
#   when learned features are enabled, the learned feature parameters.
# - Results file now also includes final losses after Adam and after LBFGS.
#
# Fixes in this version:
# - Treat "On"/"Off" as valid values for the "LBFGS ON/OFF" CSV column.
# - Only run LBFGS when LBFGS ON/OFF is On; otherwise do not call the optimizer.
# - Respect the device passed to train_once (no silent overrides).
#
# Usage:
#   python PINN_ex1_merged.py param_ex1.csv
#
# Author: ChatGPT
# ---------------------------------------------

import os, sys, math, subprocess, random, time, ast, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

def log(msg): 
    print(msg, flush=True)

def set_seed(seed: int | None):
    if seed is None: 
        return
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path): 
    p.mkdir(parents=True, exist_ok=True)

def try_run_generator(script_path: str, args=None):
    sp = Path(script_path)
    if not sp.is_absolute():
        sp = BASE_DIR / sp
    log(f"[generator] requested: {sp} {args or ''}")
    if sp.exists():
        cmd = [sys.executable, "-u", str(sp)] + (list(map(str, args)) if args else [])
        log(f"[generator] running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, cwd=str(BASE_DIR))
            log("[generator] finished ok")
        except subprocess.CalledProcessError as e:
            log(f"[generator][warn] nonzero exit: {e}")
    else:
        log(f"[generator][warn] script NOT FOUND at: {sp}")

def parse_mat2x2(row, name, row_idx=0):
    if name not in row or pd.isna(row[name]):
        raise ValueError(f"Missing required CSV column: {name}")
    try:
        A = np.array(ast.literal_eval(row[name]), dtype=float)
    except Exception as e:
        raise ValueError(f"Failed to parse {name} on row {row_idx}: {e}")
    if A.shape != (2, 2):
        raise ValueError(f"{name} must be a 2x2 matrix on row {row_idx}")
    return A

def pick_data_csv(cfg_row, set_id: int, data_type: str):
    # Explicit override
    if "data_csv" in cfg_row and isinstance(cfg_row["data_csv"], str) and cfg_row["data_csv"].strip():
        p = Path(cfg_row["data_csv"])
        if not p.is_absolute():
            p = BASE_DIR / p
        return p
    # Heuristics
    data_type = (data_type or "").strip().lower()
    candidates = []
    if data_type == "hill":
        candidates.append(BASE_DIR / f"Data_Set{set_id}_HILL_ex1.csv")
        candidates.append(BASE_DIR / "Hill_trajectories_ex1.csv")
    elif data_type == "piecewise":
        candidates.append(BASE_DIR / f"piecewise_trajectories_ex1_set{set_id}.csv")
    elif data_type == "ramp":
        candidates.append(BASE_DIR / f"Ramp_trajectories_ex1_set{set_id}.csv")
        candidates.append(BASE_DIR / f"Data_Set{set_id}_RAMP_ex1.csv")
        candidates.append(BASE_DIR / "Ramp_trajectories_ex1.csv")
    else:
        candidates.append(BASE_DIR / "Hill_trajectories_ex1.csv")
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

def parse_yes_no(val, default=False):
    """Robust yes/no parser. Accepts: yes/no, y/n, true/false, 1/0, on/off (case-insensitive)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return bool(default)
    s = str(val).strip().lower()
    if s in ("y", "yes", "true", "1", "on"):
        return True
    if s in ("n", "no", "false", "0", "off"):
        return False
    try:
        return bool(int(s))
    except Exception:
        return bool(default)

# ------------------------------
# Feature map with learnable options for fourier, rbf, and spline
# ------------------------------
class FeatureMap(nn.Module):
    """
    Time feature map on normalized t in [0,1].
    kind: 'fourier' | 'rbf' | 'spline' | 'none'
    - learn_feature_params=False reproduces original fixed features.
    - learn_feature_params=True enables learning of Fourier cycles/phase/amp,
      RBF centers/sigmas, or Spline knot centers and width.
    """
    def __init__(self, kind="fourier", n_features=16, a_max=10.0, rbf_width=0.1, spline_knots=32,
                 learn_feature_params=False,
                 fourier_learn_phase=True, fourier_learn_amp=False,
                 rbf_learn_sigma=True, rbf_share_sigma=False,
                 spline_learn_width=True):
        super().__init__()
        self.kind = (kind or "fourier").lower()
        self.n_features = int(n_features) if n_features else 0
        self.a_max = float(a_max); self.spline_knots = int(spline_knots)

        self.learn_feature_params = bool(learn_feature_params)
        self.fourier_learn_phase = bool(fourier_learn_phase)
        self.fourier_learn_amp = bool(fourier_learn_amp)
        self.rbf_learn_sigma = bool(rbf_learn_sigma)
        self.rbf_share_sigma = bool(rbf_share_sigma)
        self.spline_learn_width = bool(spline_learn_width)

        if self.kind == "fourier":
            n = max(self.n_features, 8)
            a_init = torch.linspace(1.0, max(self.a_max, 1.0), n)  # cycles
            if self.learn_feature_params:
                self.log_a = nn.Parameter(torch.log(a_init))
                self.phi   = nn.Parameter(torch.zeros(n)) if self.fourier_learn_phase else nn.Parameter(torch.zeros(n), requires_grad=False)
                self.A     = nn.Parameter(torch.ones(n)) if self.fourier_learn_amp else nn.Parameter(torch.ones(n), requires_grad=False)
            else:
                self.register_buffer("_w", 2.0 * math.pi * a_init)
                self.register_buffer("phi", torch.zeros(n))
                self.register_buffer("A", torch.ones(n))

        elif self.kind == "rbf":
            n = max(self.n_features, 16)
            c_init = torch.linspace(0.0, 1.0, n).view(-1, 1)
            if self.learn_feature_params:
                self.c_raw = nn.Parameter(torch.logit(torch.clamp(c_init, 1e-4, 1-1e-4)))
                if self.rbf_learn_sigma:
                    if self.rbf_share_sigma:
                        self.log_sigma = nn.Parameter(torch.tensor(math.log(max(rbf_width, 1e-6)), dtype=torch.float32))
                    else:
                        self.log_sigma = nn.Parameter(torch.full((n, 1), math.log(max(rbf_width, 1e-6)), dtype=torch.float32))
                else:
                    self.register_buffer("log_sigma", torch.tensor(math.log(max(rbf_width, 1e-6)), dtype=torch.float32))
            else:
                self.register_buffer("c_fix", c_init)
                self.register_buffer("log_sigma", torch.tensor(math.log(max(rbf_width, 1e-6)), dtype=torch.float32))

        elif self.kind == "spline":
            k = max(self.spline_knots, 16)
            centers = torch.linspace(0.0, 1.0, k).view(-1, 1)
            h0 = 1.0 / max(k - 1, 1)
            if self.learn_feature_params:
                # Learn knot centers in (0,1) and a positive width h
                self.c_raw = nn.Parameter(torch.logit(torch.clamp(centers, 1e-4, 1-1e-4)))
                self.log_h = nn.Parameter(torch.tensor(math.log(h0), dtype=torch.float32)) if self.spline_learn_width else nn.Parameter(torch.tensor(math.log(h0), dtype=torch.float32), requires_grad=False)
            else:
                self.register_buffer("_knots", centers)
                self.register_buffer("_hbuf", torch.tensor(h0, dtype=torch.float32))

    def fourier_params(self):
        assert self.kind == "fourier"
        if self.learn_feature_params:
            a = torch.exp(self.log_a)
            w = 2.0 * math.pi * a
            return a, w, self.phi, self.A
        else:
            return self._w / (2.0 * math.pi), self._w, self.phi, self.A

    def rbf_params(self):
        assert self.kind == "rbf"
        if self.learn_feature_params:
            c = torch.sigmoid(self.c_raw)
            if isinstance(self.log_sigma, torch.Tensor):
                sigma = torch.exp(self.log_sigma)
            else:
                sigma = torch.exp(torch.tensor(self.log_sigma))
            return c, sigma
        else:
            c = self.c_fix
            sigma = torch.exp(self.log_sigma)
            return c, sigma

    def spline_params(self):
        assert self.kind == "spline"
        if self.learn_feature_params:
            c = torch.sigmoid(self.c_raw)  # centers in (0,1)
            h = torch.exp(self.log_h)      # positive width
            return c, h
        else:
            return self._knots, self._hbuf

    def forward(self, t):
        t = t.view(-1, 1)
        if self.kind == "none": 
            return t
        if self.kind == "fourier":
            a, w, phi, A = self.fourier_params()
            wt = t * w.view(1, -1) + phi.view(1, -1)
            s = torch.sin(wt) * A.view(1, -1)
            c = torch.cos(wt) * A.view(1, -1)
            return torch.cat([s, c], dim=1)
        if self.kind == "rbf":
            c, sigma = self.rbf_params()
            if sigma.ndim == 1:
                sigma = sigma.view(-1, 1) if sigma.numel() > 1 else sigma
            sigma = torch.clamp(sigma, min=1e-6)
            dif = t - c.T
            z = dif / sigma.T
            return torch.exp(-0.5 * (z ** 2))
        if self.kind == "spline":
            c, h = self.spline_params()
            h = torch.clamp(h, min=1e-6)
            dif = (t - c.T).abs() / h
            return torch.clamp(1.0 - dif, min=0.0)
        return t

    def short_summary(self, k=6):
        if self.kind == "fourier":
            a, _, phi, A = self.fourier_params()
            return {
                "kind":"fourier",
                "cycles": a.detach().cpu().numpy().tolist(),
                "phi": phi.detach().cpu().numpy().tolist(),
                "A": A.detach().cpu().numpy().tolist()
            }
        if self.kind == "rbf":
            c, sigma = self.rbf_params()
            return {
                "kind":"rbf",
                "centers": c.detach().cpu().numpy().ravel().tolist(),
                "sigma": sigma.detach().cpu().numpy().ravel().tolist()
            }
        if self.kind == "spline":
            c, h = self.spline_params()
            return {"kind":"spline","centers": c.detach().cpu().numpy().ravel().tolist(),"h": float(h.detach().cpu().item())}
        return {"kind":"none"}

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_width=20, hidden_depth=8, act=nn.Tanh):
        super().__init__()
        layers = []; last = in_dim
        for _ in range(hidden_depth):
            layers += [nn.Linear(last, hidden_width), act()]
            last = hidden_width
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): 
        return self.net(x)

class Sine(nn.Module):
    def __init__(self, w0=30.0): 
        super().__init__(); self.w0=float(w0)
    def forward(self, x): 
        return torch.sin(self.w0 * x)

def siren_init(m, w0=30.0):
    if isinstance(m, nn.Linear):
        with torch.no_grad():
            in_dim = m.weight.size(1)
            bound = math.sqrt(6 / in_dim) / w0
            nn.init.uniform_(m.weight, -bound, bound)
            if m.bias is not None: 
                nn.init.zeros_(m.bias)

class SIREN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_width=32, hidden_depth=6, omega0=30.0):
        super().__init__()
        layers, last = [], in_dim
        for _ in range(hidden_depth):
            layers += [nn.Linear(last, hidden_width), Sine(omega0)]
            last = hidden_width
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
        self.net.apply(lambda m: siren_init(m, w0=omega0))
    def forward(self, x): 
        return self.net(x)

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim); self.fc2 = nn.Linear(dim, dim)
        self.act = nn.Tanh()
    def forward(self, x):
        y = self.act(self.fc1(x)); y = self.fc2(y)
        return self.act(x + y)

class ResNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_width=64, hidden_depth=6):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_width)
        self.blocks = nn.Sequential(*[ResBlock(hidden_width) for _ in range(hidden_depth)])
        self.outp = nn.Linear(hidden_width, out_dim)
        self.act = nn.Tanh()
    def forward(self, x):
        h = self.act(self.inp(x)); h = self.blocks(h); return self.outp(h)

def build_model(model_type, in_dim, out_dim, hidden_width, hidden_depth, omega0):
    mt = (model_type or "fnn").lower()
    if mt == "fnn":   return MLP(in_dim, out_dim, hidden_width, hidden_depth, nn.Tanh)
    if mt == "siren": return SIREN(in_dim, out_dim, hidden_width, hidden_depth, omega0)
    if mt == "resnet":return ResNet(in_dim, out_dim, hidden_width, hidden_depth)
    return MLP(in_dim, out_dim, hidden_width, hidden_depth, nn.Tanh)

class TwoNodeCyclePINN(nn.Module):
    def __init__(self, feature_map: FeatureMap, backbone: nn.Module,
                 U_01, U_10, L_01, L_10, Th_01, Th_10,
                 ic_embed_dim: int = 0,
                 ode_type: str = "hill"):
        super().__init__()
        self.feature_map = feature_map
        self.backbone = backbone
        self.ode_type = (ode_type or "hill").lower()

        U_np = np.array([[0.0, U_01],[U_10, 0.0]], dtype=np.float32)
        L_np = np.array([[0.0, L_01],[L_10, 0.0]], dtype=np.float32)
        Th_np = np.array([[0.0, Th_01],[Th_10, 0.0]], dtype=np.float32)
        self.register_buffer("U_t",  torch.tensor(U_np))
        self.register_buffer("L_t",  torch.tensor(L_np))
        self.register_buffer("Th_t", torch.tensor(Th_np))

        self.ic_embed = None
        if ic_embed_dim > 0:
            self.ic_embed = nn.Sequential(
                nn.Linear(2, ic_embed_dim), nn.Tanh(),
                nn.Linear(ic_embed_dim, 2)
            )

        eps_small = 1e-3
        init_d = torch.full((2, 2), eps_small)
        init_d[0, 1] = 5.0; init_d[1, 0] = 5.0
        self.logd = nn.Parameter(torch.log(init_d))

    @staticmethod
    def hill(x, a, b, th, d):
        eps = 1e-12
        x_c = torch.clamp(x, min=eps)
        num = x_c ** d; den = th ** d + num
        return a + (b - a) * num / den

    @staticmethod
    def ramp(x, a, b, th, d):
        # Piecewise-linear ramp: a for x <= th-1/d, linear to b over width 2/d, then b for x >= th+1/d
        eps = 1e-12
        dpos = torch.clamp(d, min=eps)
        left  = th - 1.0 / dpos
        right = th + 1.0 / dpos
        width = right - left  # = 2/d
        m = (b - a) / torch.clamp(width, min=eps)  # slope
        lin = a + m * (x - left)
        y = torch.where(x <= left, a, torch.where(x >= right, b, lin))
        return y

    def forward(self, t_norm, IC, inv_t_scale: float = 1.0):
        phi_t = self.feature_map(t_norm)
        ic_feat = self.ic_embed(IC) if self.ic_embed is not None else IC
        X = self.backbone(torch.cat([phi_t, ic_feat], dim=1))
        x0, x1 = X[:, 0:1], X[:, 1:2]

        dx0_dt_norm = torch.autograd.grad(x0.sum(), t_norm, create_graph=True)[0]
        dx1_dt_norm = torch.autograd.grad(x1.sum(), t_norm, create_graph=True)[0]
        x0_t = inv_t_scale * dx0_dt_norm
        x1_t = inv_t_scale * dx1_dt_norm

        d = torch.exp(self.logd)
        if self.ode_type == "ramp":
            r0 = -x0 + self.ramp(x1, self.U_t[1, 0], self.L_t[1, 0], self.Th_t[1, 0], d[1, 0])
            r1 = -x1 + self.ramp(x0, self.U_t[0, 1], self.L_t[0, 1], self.Th_t[0, 1], d[0, 1])
        else:
            r0 = -x0 + self.hill(x1, self.U_t[1, 0], self.L_t[1, 0], self.Th_t[1, 0], d[1, 0])
            r1 = -x1 + self.hill(x0, self.U_t[0, 1], self.L_t[0, 1], self.Th_t[0, 1], d[0, 1])

        f0 = x0_t - r0
        f1 = x1_t - r1
        return x0, x1, f0, f1

def train_once(cfg_row, results_dir: Path, row_index_for_errors=0, device=None, use_amp=False, amp_dtype_str="fp16"):
    # Choose device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # AMP dtype
    amp_dtype = torch.float16 if amp_dtype_str == "fp16" else (torch.bfloat16 if amp_dtype_str == "bf16" else torch.float32)

    data_type     = str(cfg_row.get("data_type", "hill")).lower()
    n_runs        = int(cfg_row.get("n_runs", 1))
    seed          = int(cfg_row.get("seed", 1234)) if "seed" in cfg_row else 1234

    feature_map_kind = str(cfg_row.get("feature_map", "fourier")).lower()
    adam_steps    = int(cfg_row.get("adam_steps", 10000))
    lr_adam       = float(cfg_row.get("lr_adam", 1e-3))
    lbfgs_max_it  = int(cfg_row.get("lbfgs_max_iter", 5000))
    hidden_width  = int(cfg_row.get("hidden_width", 20))
    hidden_depth  = int(cfg_row.get("hidden_depth", 8))
    omega0        = float(cfg_row.get("omega0", 30.0))
    n_features    = int(cfg_row.get("n_features", 16))
    a_max         = float(cfg_row.get("a_max", 10.0))
    rbf_width     = float(cfg_row.get("rbf_width", 0.1))
    spline_knots  = int(cfg_row.get("spline_knots", 32))
    ic_embed_dim  = int(cfg_row.get("ic_embed_dim", 0)) if "ic_embed_dim" in cfg_row else 0
    ic_loss_wt    = float(cfg_row.get("ic_loss_wt", 1e-3)) if "ic_loss_wt" in cfg_row else 1e-3

    learned_feats = parse_yes_no(cfg_row.get("Learned_Feats", "no"), default=False)
    lbfgs_on = parse_yes_no(cfg_row.get("LBFGS ON/OFF", "yes"), default=True)
    ode_type = str(cfg_row.get("ODE_type", "Hill")).strip().lower()
    learn_feature_params = learned_feats and (feature_map_kind in ("fourier","rbf","spline"))

    U = parse_mat2x2(cfg_row, "U_np", row_index_for_errors)
    L = parse_mat2x2(cfg_row, "L_np", row_index_for_errors)
    Th = parse_mat2x2(cfg_row, "Th_np", row_index_for_errors)
    U_01, U_10 = float(U[0,1]), float(U[1,0])
    L_01, L_10 = float(L[0,1]), float(L[1,0])
    Th_01, Th_10 = float(Th[0,1]), float(Th[1,0])

    data_type = (data_type or "hill").strip().lower()
    set_id = int(row_index_for_errors) + 1

    data_path = pick_data_csv(cfg_row, set_id=set_id, data_type=data_type)
    if not data_path.exists():
        here = BASE_DIR
        msg = [
            f"Data file not found: {data_path}",
            f"Generator cwd: {here}",
            "Looked for:",
            f"  • Hill:   {here / f'Data_Set{set_id}_HILL_ex1.csv'}",
            f"  • piecewise:  {here / f'piecewise_trajectories_ex1_set{set_id}.csv'}",
            "Or set a custom path via the 'data_csv' column in param_ex1.csv."
        ]
        raise FileNotFoundError("\\n".join(msg))

    log(f"[data] using CSV: {data_path}")
    df = pd.read_csv(data_path)

    if data_type == "piecewise":
        if "x0" not in df.columns and {"x", "y"}.issubset(df.columns):
            df = df.rename(columns={"x": "x0", "y": "x1"})

    for col in ["Time", "x0", "x1"]:
        if col not in df.columns:
            raise ValueError(f"Data CSV missing column: {col} (after harmonization)")

    if "Trajectory" in df.columns:
        t0_df = df.groupby("Trajectory", as_index=False)["Time"].min().rename(columns={"Time": "t0"})
        ic_df = df.merge(t0_df, on="Trajectory")
        ic_df = ic_df[np.isclose(ic_df["Time"], ic_df["t0"])]
        ic_pick = ic_df.groupby("Trajectory", as_index=False).first()[["Trajectory", "x0", "x1"]]
        ic_pick = ic_pick.rename(columns={"x0": "ic0", "x1": "ic1"})
        df = df.merge(ic_pick, on="Trajectory", how="left")
    else:
        first_idx = df["Time"].idxmin()
        ic0, ic1 = float(df.loc[first_idx, "x0"]), float(df.loc[first_idx, "x1"])
        df["Trajectory"] = 1; df["ic0"] = ic0; df["ic1"] = ic1

    t_min = float(df["Time"].min())
    t_max = float(df["Time"].max())
    t_scale = t_max - t_min
    if not np.isfinite(t_scale) or t_scale <= 0.0:
        raise ValueError(f"Invalid time scale: t_min={t_min}, t_max={t_max}")
    inv_t_scale = 1.0 / t_scale

    t_norm_np = (df["Time"].values - t_min) / t_scale
    df["t_norm"] = t_norm_np

    # place tensors on the chosen device
    t_all_norm  = torch.tensor(df["t_norm"].values[:, None], dtype=torch.float32, device=device)
    ic_all      = torch.tensor(df[["ic0", "ic1"]].values,   dtype=torch.float32, device=device)
    x0_all      = torch.tensor(df["x0"].values[:, None],    dtype=torch.float32, device=device)
    x1_all      = torch.tensor(df["x1"].values[:, None],    dtype=torch.float32, device=device)
    N_total = t_all_norm.size(0)

    ensure_dir(results_dir)
    plots_dir = results_dir / "plots_ex1"
    ensure_dir(plots_dir)

    all_run_rows = []
    for run_idx in range(1, n_runs + 1):
        set_seed(seed + run_idx - 1)

        idx = np.arange(N_total)

        t_train_base_norm = t_all_norm[idx].detach()
        ic_train = ic_all[idx]
        x0_train = x0_all[idx]
        x1_train = x1_all[idx]

        fmap = FeatureMap(
            kind=feature_map_kind,
            n_features=n_features,
            a_max=a_max,
            rbf_width=rbf_width,
            spline_knots=spline_knots,
            learn_feature_params=learn_feature_params,
            fourier_learn_phase=True,
            fourier_learn_amp=False,
            rbf_learn_sigma=True,
            rbf_share_sigma=False,
            spline_learn_width=True
        ).to(device)
        with torch.no_grad():
            Dt = fmap(t_train_base_norm[:1]).shape[1]
        in_dim = Dt + 2
        backbone = build_model("fnn", in_dim, 2, hidden_width, hidden_depth, omega0).to(device)

        model = TwoNodeCyclePINN(
            feature_map=fmap, backbone=backbone,
            U_01=U_01, U_10=U_10, L_01=L_01, L_10=L_10, Th_01=Th_01, Th_10=Th_10,
            ic_embed_dim=ic_embed_dim,
            ode_type=ode_type
        ).to(device)
        model.train()

        opt_adam = optim.Adam(model.parameters(), lr=lr_adam)
        opt_lbfgs = optim.LBFGS(
            model.parameters(), lr=1.0,
            max_iter=lbfgs_max_it, max_eval=lbfgs_max_it,
            history_size=50, tolerance_grad=1e-12, tolerance_change=1e-12,
            line_search_fn='strong_wolfe'
        )

        with torch.no_grad():
            if "Trajectory" in df.columns:
                t0_map = df.groupby("Trajectory", as_index=False)["Time"].min().rename(columns={"Time":"t0"})
                t0_norm = ((t0_map["t0"].values - t_min) / t_scale).astype(np.float32)
                t0_per_row_norm = (
                    df.loc[idx, ["Trajectory"]]
                    .merge(pd.DataFrame({"Trajectory": t0_map["Trajectory"].values,
                                         "t0_norm": t0_norm}), on="Trajectory", how="left")["t0_norm"].values
                )
                t0_per_row_norm = torch.tensor(t0_per_row_norm[:, None], dtype=torch.float32, device=device)
                ic_mask = (torch.abs(t_train_base_norm - t0_per_row_norm) <= 1e-8).squeeze(1)
            else:
                ic_mask = (torch.abs(t_train_base_norm - t_train_base_norm.min()) <= 1e-8).squeeze(1)

        # Adam warm-up
        t0_time = time.time()
        last_adam_losses = {"total": None, "data": None, "res": None, "ic": None}
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        for it in range(adam_steps):
            opt_adam.zero_grad(set_to_none=True)
            t_in_norm = t_train_base_norm.clone().requires_grad_(True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                x0_p, x1_p, f0, f1 = model(t_in_norm, ic_train, inv_t_scale=inv_t_scale)
                loss_data = ((x0_train - x0_p)**2 + (x1_train - x1_p)**2).mean()
                loss_res  = (f0**2 + f1**2).mean()
                if ic_mask.any():
                    ic_err = (x0_p[ic_mask] - ic_train[ic_mask, 0:1]).pow(2).mean() + \
                             (x1_p[ic_mask] - ic_train[ic_mask, 1:2]).pow(2).mean()
                    loss_ic = ic_err
                else:
                    loss_ic = torch.tensor(0.0, device=device)
                loss = loss_data + loss_res + ic_loss_wt * loss_ic
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt_adam)
                scaler.update()
            else:
                loss.backward(); opt_adam.step()

            if it % 500 == 0:
                d_now = torch.exp(model.logd).detach().cpu().numpy()
                msg = (f"[ex1 set {set_id} run {run_idx}] Adam {it:05d} | Total {loss.item():.3e} | "
                       f"Data {loss_data.item():.3e} | Res {loss_res.item():.3e} | IC {loss_ic.item():.3e} | "
                       f"d_01={d_now[0,1]:.4f}, d_10={d_now[1,0]:.4f}")
                if learn_feature_params:
                    msg += f" | fmap_head={str(fmap.short_summary())[:120]}"
                log(msg)
            last_adam_losses = {
                "total": float(loss.item()),
                "data": float(loss_data.item()),
                "res":  float(loss_res.item()),
                "ic":   float(loss_ic.item())
            }

        # L-BFGS fine-tune (only if lbfgs_on is True)
        last_lbfgs_losses = {"total": None, "data": None, "res": None, "ic": None}
        if lbfgs_on:
            log("[info] Running LBFGS fine-tuning...")
            def closure():
                # Use full precision for LBFGS for numerical stability
                opt_lbfgs.zero_grad()
                t_in_norm = t_train_base_norm.clone().requires_grad_(True)
                x0_p, x1_p, f0, f1 = model(t_in_norm, ic_train, inv_t_scale=inv_t_scale)
                loss_data = ((x0_train - x0_p)**2 + (x1_train - x1_p)**2).mean()
                loss_res  = (f0**2 + f1**2).mean()
                if ic_mask.any():
                    ic_err = (x0_p[ic_mask] - ic_train[ic_mask, 0:1]).pow(2).mean() + \
                             (x1_p[ic_mask] - ic_train[ic_mask, 1:2]).pow(2).mean()
                    loss_ic = ic_err
                else:
                    loss_ic = torch.tensor(0.0, device=device)
                loss = loss_data + loss_res + ic_loss_wt * loss_ic
                loss.backward()
                d_now = torch.exp(model.logd).detach().cpu().numpy()
                msg = (f"[ex1 set {set_id} run {run_idx}] LBFGS loss: {loss.item():.3e} | "
                       f"d_01={d_now[0,1]:.4f}, d_10={d_now[1,0]:.4f}")
                if learn_feature_params:
                    msg += f" | fmap_head={str(fmap.short_summary())[:120]}"
                log(msg)
                last_lbfgs_losses["total"] = float(loss.item())
                last_lbfgs_losses["data"]  = float(loss_data.item())
                last_lbfgs_losses["res"]   = float(loss_res.item())
                last_lbfgs_losses["ic"]    = float(loss_ic.item())
                return loss

            opt_lbfgs.step(closure)
        else:
            log("[info] Skipping LBFGS (LBFGS ON/OFF=Off)")

        t1_time = time.time()

        # Evaluate on all points
        model.eval()
        t_all_req_norm = t_all_norm.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            x0_pred, x1_pred, _, _ = model(t_all_req_norm, ic_all, inv_t_scale=inv_t_scale)
        err0 = torch.norm(x0_all - x0_pred.detach()) / torch.norm(x0_all)
        err1 = torch.norm(x1_all - x1_pred.detach()) / torch.norm(x1_all)

        d_learned = torch.exp(model.logd).detach().cpu().numpy()

        # Plots
        plt.figure()
        plt.plot(df["Time"].values, df["x0"].values, '.', label='x0 true', alpha=0.6)
        plt.plot(df["Time"].values, x0_pred.detach().cpu().numpy(), '-', label='x0 pred', linewidth=1)
        plt.legend(); plt.xlabel('Time'); plt.ylabel('x0'); plt.tight_layout()
        plt.savefig(plots_dir / f"x0_ex1_run{run_idx}.png", dpi=150); plt.close()

        plt.figure()
        plt.plot(df["Time"].values, df["x1"].values, '.', label='x1 true', alpha=0.6)
        plt.plot(df["Time"].values, x1_pred.detach().cpu().numpy(), '-', label='x1 pred', linewidth=1)
        plt.legend(); plt.xlabel('Time'); plt.ylabel('x1'); plt.tight_layout()
        plt.savefig(plots_dir / f"x1_ex1_run{run_idx}.png", dpi=150); plt.close()

        # Compose results row
        row_copy = {}
        for k, v in dict(cfg_row).items():
            try:
                if isinstance(v, (np.floating,)):
                    v = float(v)
                elif isinstance(v, (np.integer,)):
                    v = int(v)
            except Exception:
                pass
            row_copy[k] = v

        fmap_info = fmap.short_summary() if learn_feature_params else {"kind": feature_map_kind}
        row_dict = dict(row_copy) | dict(
            run=run_idx,
            set_id=set_id,
            data_csv=str(data_path),
            feature_map=feature_map_kind,
            ODE_type=ode_type,
            Learned_Feats="yes" if learned_feats else "no",
            learned_feature_params=int(learn_feature_params),
            fmap_summary=str(fmap_info) if learn_feature_params else "",
            adam_steps=adam_steps, lr_adam=lr_adam,
            lbfgs_max_iter=lbfgs_max_it, hidden_width=hidden_width, hidden_depth=hidden_depth,
            omega0=omega0, n_features=n_features, a_max=a_max, rbf_width=rbf_width, spline_knots=spline_knots,
            ic_embed_dim=ic_embed_dim, ic_loss_wt=ic_loss_wt,
            # learned Hill exponents
            d_01=float(d_learned[0,1]), d_10=float(d_learned[1,0]), d_00=float(d_learned[0,0]), d_11=float(d_learned[1,1]),
            # relative errors
            rel_err_x0=float(err0.item()), rel_err_x1=float(err1.item()),
            # timing
            train_time_sec=float(t1_time - t0_time),
            # normalization time bounds
            t_min=t_min, t_max=t_max,
            # losses at the end of Adam and LBFGS
            adam_loss_total=last_adam_losses["total"],
            adam_loss_data=last_adam_losses["data"],
            adam_loss_res=last_adam_losses["res"],
            adam_loss_ic=last_adam_losses["ic"],
            lbfgs_loss_total=last_lbfgs_losses["total"],
            lbfgs_loss_data=last_lbfgs_losses["data"],
            lbfgs_loss_res=last_lbfgs_losses["res"],
            lbfgs_loss_ic=last_lbfgs_losses["ic"],
            LBFGS_ONOFF="On" if lbfgs_on else "Off"
        )
        all_run_rows.append(row_dict)

    return all_run_rows

def main():
    parser = argparse.ArgumentParser(description="Unified PINN ex1 runner (merged) with GPU/AMP options")
    parser.add_argument("params_csv", nargs="?", default="param_ex1.csv", help="Parameter CSV path")
    parser.add_argument("--device", default=os.environ.get("PINN_DEVICE", "auto"), help="cpu|cuda|cuda:0|auto")
    parser.add_argument("--amp", action="store_true", default=bool(int(os.environ.get("PINN_USE_AMP", "0"))), help="Enable mixed precision (Adam phase)")
    parser.add_argument("--amp-dtype", default=os.environ.get("PINN_AMP_PRECISION", "fp16"), choices=["fp16","bf16"], help="AMP dtype")
    args = parser.parse_args()

    log("=== PINN ex1 merged start ===")
    log(f"[cwd] os.getcwd() = {os.getcwd()}")
    log(f"[base] BASE_DIR     = {BASE_DIR}")
    params_csv = args.params_csv
    # Device selection
    if args.device.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            device = torch.device(args.device)
        except Exception:
            log(f"[warn] Could not parse device '{args.device}', falling back to auto")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log(f"[gpu] Using CUDA device: {torch.cuda.get_device_name(device.index or 0)}")
        torch.backends.cudnn.benchmark = True
    else:
        log("[gpu] CUDA not available or CPU forced.")
    use_amp = args.amp and (device.type == "cuda")
    log(f"[amp] mixed precision: {'ON' if use_amp else 'OFF'} (dtype={args.amp_dtype if use_amp else 'fp32'})")
    if not Path(params_csv).exists():
        raise FileNotFoundError(f"Params CSV not found: {Path(params_csv).resolve()}")

    cfg_df = pd.read_csv(params_csv)
    results_dir = BASE_DIR / "results_two_node_cycle_ex1"; ensure_dir(results_dir)
    results_path = results_dir / "results_ex1.csv"
    old_results_path = results_dir / "runs_ex1.csv"
    if (not results_path.exists()) and old_results_path.exists():
        try:
            prev = pd.read_csv(old_results_path)
            prev.to_csv(results_path, index=False)
            log("[results] Migrated existing runs_ex1.csv to results_ex1.csv")
        except Exception as e:
            log(f"[results][warn] failed to migrate old results: {e}")

    # Optional dataset generation if CSV mixes types
    if "data_type" in cfg_df.columns:
        present_types = set(cfg_df["data_type"].astype(str).str.lower().unique())
    else:
        present_types = set()
    if "hill" in present_types:
        log("[prep] generating all Hill datasets upfront...")
        try_run_generator("Hill_Data_production_ex1.py")
    if "piecewise" in present_types:
        log("[prep] generating all piecewise datasets upfront (python)...")
        try_run_generator("piecewise_Data_production_ex1.py")
    if "ramp" in present_types:
        log("[prep] generating all Ramp datasets upfront...")
        try_run_generator("Ramp_Data_production_ex1.py")

    all_rows = []
    for i, row in cfg_df.iterrows():
        log(f"\\n=== Running ex1 row {i+1}/{len(cfg_df)} ===")
        rows = train_once(row, results_dir, row_index_for_errors=i, device=device, use_amp=use_amp, amp_dtype_str=args.amp_dtype)
        all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows)
    append = results_path.exists() and os.path.getsize(results_path) > 0
    results_df.to_csv(results_path, mode=('a' if append else 'w'), header=not append, index=False)
    log(f"\\nAppended {len(results_df)} rows to {results_path.resolve()}")

if __name__ == "__main__":
    main()
