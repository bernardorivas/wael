




















import os, sys, math, subprocess, random, time, ast, argparse
import atexit
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Optional
import copy

BASE_DIR = Path(__file__).resolve().parent




_CONSOLE_STDOUT = sys.stdout
_CONSOLE_STDERR = sys.stderr
_LOG_FH = None
_LOG_PATH = None


def console(msg: str):

    print(msg, file=_CONSOLE_STDOUT, flush=True)
    if _LOG_FH is not None:
        print(msg, file=_LOG_FH, flush=True)


def setup_run_logging(log_dir: Path = None):

    global _LOG_FH, _LOG_PATH
    if _LOG_FH is not None:
        return _LOG_PATH
    if log_dir is None:
        log_dir = BASE_DIR / "logs"
    ensure_dir(log_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    _LOG_PATH = log_dir / f"PINN_ex2_{ts}.txt"
    _LOG_FH = open(_LOG_PATH, mode="w", encoding="utf-8")
    sys.stdout = _LOG_FH
    sys.stderr = _LOG_FH

    def _restore_and_close():
        try:
            sys.stdout = _CONSOLE_STDOUT
            sys.stderr = _CONSOLE_STDERR
        finally:
            try:
                if _LOG_FH is not None:
                    _LOG_FH.flush()
                    _LOG_FH.close()
            except Exception:
                pass

    atexit.register(_restore_and_close)
    return _LOG_PATH

def log(msg):
    print(msg, flush=True)

def set_seed(seed: Optional[int]):
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
            subprocess.run(cmd, check=True, cwd=str(BASE_DIR), stdout=sys.stdout, stderr=sys.stderr)
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

    if "data_csv" in cfg_row and isinstance(cfg_row["data_csv"], str) and cfg_row["data_csv"].strip():
        p = Path(cfg_row["data_csv"])
        if not p.is_absolute():
            p = BASE_DIR / p
        return p

    data_type = (data_type or "").strip().lower()
    candidates = []
    if data_type == "hill":
        candidates.append(BASE_DIR / "generated_data_ex2" / "hill" / f"Data_Set{set_id}_HILL_ex2.csv")
        candidates.append(BASE_DIR / f"Data_Set{set_id}_HILL_ex2.csv")
        candidates.append(BASE_DIR / "Hill_trajectories_ex2.csv")
    elif data_type == "piecewise":
        candidates.append(BASE_DIR / "generated_data_ex2" / "piecewise" / f"Data_Set{set_id}_PIECEWISE_ex2.csv")
        candidates.append(BASE_DIR / f"piecewise_trajectories_ex2_set{set_id}.csv")
    elif data_type == "ramp":
        candidates.append(BASE_DIR / "generated_data_ex2" / "ramp" / f"Data_Set{set_id}_RAMP_ex2.csv")
        candidates.append(BASE_DIR / f"Ramp_trajectories_ex2_set{set_id}.csv")
        candidates.append(BASE_DIR / f"Data_Set{set_id}_RAMP_ex2.csv")
        candidates.append(BASE_DIR / "Ramp_trajectories_ex2.csv")
    else:
        candidates.append(BASE_DIR / "Hill_trajectories_ex2.csv")
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

def parse_yes_no(val, default=False):

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




class FeatureMap(nn.Module):







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
            a_init = torch.linspace(1.0, max(self.a_max, 1.0), n)
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
            c = torch.sigmoid(self.c_raw)
            h = torch.exp(self.log_h)
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
                 U_init, L_init, Th_init,
                 ic_embed_dim: int = 0,
                 ode_type: str = "hill"):
        super().__init__()
        self.feature_map = feature_map
        self.backbone = backbone
        self.ode_type = (ode_type or "hill").lower()


        U_np = np.array(U_init, dtype=np.float32)
        L_np = np.array(L_init, dtype=np.float32)
        Th_np = np.array(Th_init, dtype=np.float32)
        if U_np.shape != (2, 2) or L_np.shape != (2, 2) or Th_np.shape != (2, 2):
            raise ValueError("U_init, L_init, and Th_init must be 2x2 arrays")


        self._param_eps = 1e-6

        def to_raw(val_mat):


            arr = np.array(val_mat, dtype=np.float32)

            arr = np.maximum(arr, self._param_eps + 1e-6)
            val = arr - self._param_eps
            return torch.sqrt(torch.tensor(val, dtype=torch.float32))

        self.U_t  = nn.Parameter(to_raw(U_init))
        self.L_t  = nn.Parameter(to_raw(L_init))
        self.Th_t = nn.Parameter(to_raw(Th_init))

        self.ic_embed = None
        if ic_embed_dim > 0:
            self.ic_embed = nn.Sequential(
                nn.Linear(2, ic_embed_dim), nn.Tanh(),
                nn.Linear(ic_embed_dim, 2)
            )

        init_d = torch.full((2, 2), 10.0)
        self.d_t = nn.Parameter(to_raw(init_d))

    def _pos(self, raw: torch.Tensor) -> torch.Tensor:
        return raw**2 + self._param_eps

    def pos_matrices(self):

        return self._pos(self.U_t), self._pos(self.L_t), self._pos(self.Th_t)

    @staticmethod
    def hill(x, a, b, th, d):
        eps = 1e-12
        x_c = torch.clamp(x, min=eps)
        th_pos = torch.clamp(th, min=1e-6)
        num = x_c ** d; den = th_pos ** d + num
        return a + (b - a) * num / den

    @staticmethod
    def ramp(x, a, b, th, d):

        eps = 1e-12
        dpos = torch.clamp(d, min=eps)
        th_pos = torch.clamp(th, min=1e-6)
        left  = th_pos - 1.0 / dpos
        right = th_pos + 1.0 / dpos
        width = right - left
        m = (b - a) / torch.clamp(width, min=eps)
        lin = a + m * (x - left)
        y = torch.where(x <= left, a, torch.where(x >= right, b, lin))
        return y

    def forward(self, t_norm, IC, inv_t_scale: float = 1.0):

        U_pos, L_pos, Th_pos = self.pos_matrices()

        phi_t = self.feature_map(t_norm)
        ic_feat = self.ic_embed(IC) if self.ic_embed is not None else IC
        X = self.backbone(torch.cat([phi_t, ic_feat], dim=1))
        x0, x1 = X[:, 0:1], X[:, 1:2]

        dx0_dt_norm = torch.autograd.grad(x0.sum(), t_norm, create_graph=True)[0]
        dx1_dt_norm = torch.autograd.grad(x1.sum(), t_norm, create_graph=True)[0]
        x0_t = inv_t_scale * dx0_dt_norm
        x1_t = inv_t_scale * dx1_dt_norm

        d = self._pos(self.d_t)
        fn = self.ramp if self.ode_type == "ramp" else self.hill

        fn00 = fn(x0, L_pos[0, 0], U_pos[0, 0], Th_pos[0, 0], d[0, 0])
        fn01 = fn(x0, U_pos[0, 1], L_pos[0, 1], Th_pos[0, 1], d[0, 1])
        fn10 = fn(x1, L_pos[1, 0], U_pos[1, 0], Th_pos[1, 0], d[1, 0])
        fn11 = fn(x1, L_pos[1, 1], U_pos[1, 1], Th_pos[1, 1], d[1, 1])

        dx0dt = -x0 + fn00 + fn10
        dx1dt = -x1 + fn01 * fn11

        f0 = x0_t - dx0dt
        f1 = x1_t - dx1dt
        return x0, x1, f0, f1

def train_once(cfg_row, results_dir: Path, row_index_for_errors=0, device=None, use_amp=False, amp_dtype_str="fp16"):

    if device is None:

        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    amp_dtype = torch.float16 if amp_dtype_str == "fp16" else (torch.bfloat16 if amp_dtype_str == "bf16" else torch.float32)

    data_type     = str(cfg_row.get("data_type", "hill")).lower()
    n_runs        = int(cfg_row.get("n_runs", 1))


    if "NN_seed" in cfg_row and not pd.isna(cfg_row["NN_seed"]):
        seed = int(cfg_row.get("NN_seed"))
    elif "seed" in cfg_row and not pd.isna(cfg_row["seed"]):
        seed = int(cfg_row.get("seed"))
    else:
        seed = 1234

    feature_map_kind = str(cfg_row.get("feature_map", "fourier")).lower()

    model_type   = str(cfg_row.get("model_type", "fnn")).lower()
    if model_type not in {"fnn", "siren", "resnet"}:
        log(f"[warn] Unknown model_type '{model_type}' in CSV; falling back to 'fnn'. Allowed: fnn|siren|resnet")
        model_type = "fnn"
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
    data_loss_wt  = float(cfg_row.get("data_loss_wt", 1.0)) if "data_loss_wt" in cfg_row else 1.0
    physics_loss_wt = float(cfg_row.get("physics_loss_wt", 1.0)) if "physics_loss_wt" in cfg_row else 1.0

    learned_feats = parse_yes_no(cfg_row.get("Learned_Feats", "no"), default=False)
    lbfgs_on = parse_yes_no(cfg_row.get("LBFGS ON/OFF", "yes"), default=True)
    corners_only = parse_yes_no(cfg_row.get("corners only", "no"), default=False)
    ode_type = str(cfg_row.get("ODE_type", "Hill")).strip().lower()
    learn_feature_params = learned_feats and (feature_map_kind in ("fourier","rbf","spline"))

    U = parse_mat2x2(cfg_row, "U_np", row_index_for_errors)
    L = parse_mat2x2(cfg_row, "L_np", row_index_for_errors)
    Th = parse_mat2x2(cfg_row, "Th_np", row_index_for_errors)


    U_init = np.ones((2, 2), dtype=np.float32)
    L_init = np.ones((2, 2), dtype=np.float32)
    Th_init = np.ones((2, 2), dtype=np.float32)

    data_type = (data_type or "hill").strip().lower()
    set_id = int(row_index_for_errors) + 1

    data_path = pick_data_csv(cfg_row, set_id=set_id, data_type=data_type)
    if not data_path.exists():
        here = BASE_DIR
        msg = [
            f"Data file not found: {data_path}",
            f"Generator cwd: {here}",
            "Looked for:",
            f"  - Hill (new): {here / 'generated_data_ex2' / 'hill' / f'Data_Set{set_id}_HILL_ex2.csv'}",
            f"  â€¢ Hill:   {here / f'Data_Set{set_id}_HILL_ex2.csv'}",
            f"  - Piecewise (new): {here / 'generated_data_ex2' / 'piecewise' / f'Data_Set{set_id}_PIECEWISE_ex2.csv'}",
            f"  â€¢ piecewise:  {here / f'piecewise_trajectories_ex2_set{set_id}.csv'}",
            f"  - Ramp (new): {here / 'generated_data_ex2' / 'ramp' / f'Data_Set{set_id}_RAMP_ex2.csv'}",
            "Or set a custom path via the 'data_csv' column in param_ex2.csv."
        ]
        raise FileNotFoundError("\\n".join(msg))

    log(f"[data] using CSV: {data_path}")
    df = pd.read_csv(data_path)

    ic_seed_in_data = None
    if "ic_seed" in df.columns and len(df) > 0:
        try:
            ic_seed_in_data = int(df["ic_seed"].iloc[0])
        except Exception:
            ic_seed_in_data = None
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

    if corners_only and data_type in {"ramp", "piecewise"}:
        def _keep_turn_regions(df_local: pd.DataFrame,
                                   time_col="Time", x_cols=("x0","x1"),
                                   angle_deg_thresh=8.0,
                                   pad_points=2,
                                   min_speed=1e-9):








                df_pos = df_local.reset_index(drop=True)
                keep = np.zeros(len(df_pos), dtype=bool)

                def mark_keeps(idx_array, t, x0, x1):
                    n = len(idx_array)
                    if n < 3:
                        keep[idx_array] = True
                        return



                    stride = max(1, int(round(n / 50.0)))

                    if 2 * stride >= n:
                        stride = max(1, (n - 1) // 2)


                    vx = np.zeros(n); vy = np.zeros(n)
                    if 2 * stride < n:
                        vx[stride:-stride] = (x0[2*stride:] - x0[:-2*stride]) / np.maximum(t[2*stride:] - t[:-2*stride], min_speed)
                        vy[stride:-stride] = (x1[2*stride:] - x1[:-2*stride]) / np.maximum(t[2*stride:] - t[:-2*stride], min_speed)
                    else:

                        vx[1:-1] = (x0[2:] - x0[:-2]) / np.maximum(t[2:] - t[:-2], min_speed)
                        vy[1:-1] = (x1[2:] - x1[:-2]) / np.maximum(t[2:] - t[:-2], min_speed)


                    speed = np.sqrt(vx**2 + vy**2)
                    ux = np.divide(vx, np.maximum(speed, min_speed))
                    uy = np.divide(vy, np.maximum(speed, min_speed))


                    for k in range(stride, n - stride):
                        i_before = k - stride
                        i_after = k + stride

                        if speed[i_before] < min_speed or speed[i_after] < min_speed:
                            continue
                        u1 = np.array([ux[i_before], uy[i_before]])
                        u2 = np.array([ux[i_after], uy[i_after]])
                        dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
                        ang = np.degrees(np.arccos(dot))
                        if ang >= angle_deg_thresh:
                            a = max(0, k - pad_points)
                            b = min(n, k + pad_points + 1)
                            keep[idx_array[a:b]] = True

                if "Trajectory" in df_pos.columns:
                    for _, g in df_pos.groupby("Trajectory", sort=False):
                        idx = g.index.values
                        t  = g[time_col].to_numpy()
                        x0 = g[x_cols[0]].to_numpy()
                        x1 = g[x_cols[1]].to_numpy()
                        mark_keeps(idx, t, x0, x1)
                else:
                    idx = np.arange(len(df_pos))
                    t  = df_pos[time_col].to_numpy()
                    x0 = df_pos[x_cols[0]].to_numpy()
                    x1 = df_pos[x_cols[1]].to_numpy()
                    mark_keeps(idx, t, x0, x1)


                if "Trajectory" in df_pos.columns:
                    for _, g in df_pos.groupby("Trajectory", sort=False):
                        i0 = g.index.min(); i1 = g.index.max()
                        keep[i0] = True; keep[i1] = True
                else:
                    if len(df_pos) > 0:
                        keep[0] = True; keep[-1] = True

                kept_df = df_pos.loc[keep].reset_index(drop=True)
                return kept_df, keep
        ensure_dir(results_dir)
        preproc_plots = results_dir / "plots_preproc"
        ensure_dir(preproc_plots)

        angle_deg_thresh = 8.0
        pad_points = 2
        log(f"[preproc] turn-keeper for {data_type}... (angle_deg_thresh={angle_deg_thresh}Â°, pad={pad_points})")
        before = len(df)
        df_filtered, keep_mask = _keep_turn_regions(
            df, time_col="Time", x_cols=("x0","x1"),
            angle_deg_thresh=angle_deg_thresh, pad_points=pad_points
        )
        after = len(df_filtered)
        log(f"[preproc] removed {before - after} near-linear points; kept {after} turn points")


        try:
            plt.figure(figsize=(8,4))
            plt.subplot(1,2,1)
            plt.plot(df["Time"].values, df["x0"].values, '.', color='gray', alpha=0.25, label='orig x0')
            plt.plot(df_filtered["Time"].values, df_filtered["x0"].values, 'o', ms=4, label='kept x0')
            plt.xlabel('Time'); plt.ylabel('x0'); plt.legend(); plt.title('x0 kept')
            plt.subplot(1,2,2)
            plt.plot(df["Time"].values, df["x1"].values, '.', color='gray', alpha=0.25, label='orig x1')
            plt.plot(df_filtered["Time"].values, df_filtered["x1"].values, 'o', ms=4, label='kept x1')
            plt.xlabel('Time'); plt.ylabel('x1'); plt.legend(); plt.title('x1 kept')
            plt.tight_layout()
            plt.savefig(preproc_plots / f"preproc_kept_set{set_id}.png", dpi=150)
            plt.close()

            plt.figure(figsize=(5,5))
            plt.plot(df["x0"].values, df["x1"].values, '.', color='gray', alpha=0.25, label='original')
            plt.plot(df_filtered["x0"].values, df_filtered["x1"].values, 'o', ms=4, label='kept')
            plt.xlabel('x0'); plt.ylabel('x1'); plt.legend(); plt.title('Phase plane: kept turns')
            plt.tight_layout()
            plt.savefig(preproc_plots / f"preproc_kept_phaseplane_set{set_id}.png", dpi=150)
            plt.close()


            try:
                plt.figure(figsize=(8,4))
                plt.plot(df_filtered["Time"].values, df_filtered["x0"].values, 'o', ms=4, label='kept x0')
                plt.plot(df_filtered["Time"].values, df_filtered["x1"].values, 'o', ms=4, label='kept x1')
                plt.xlabel('Time'); plt.ylabel('x'); plt.legend(); plt.title('Kept points only (time series)')
                plt.tight_layout()
                plt.savefig(preproc_plots / f"preproc_kept_only_set{set_id}.png", dpi=150)
                plt.close()

                plt.figure(figsize=(5,5))
                plt.plot(df_filtered["x0"].values, df_filtered["x1"].values, 'o', ms=4)
                plt.xlabel('x0'); plt.ylabel('x1'); plt.title('Kept points only (phase plane)')
                plt.tight_layout()
                plt.savefig(preproc_plots / f"preproc_kept_only_phaseplane_set{set_id}.png", dpi=150)
                plt.close()
            except Exception as e:
                log(f"[preproc][warn] kept-only plot fail: {e}")
        except Exception as e:
            log(f"[preproc][warn] plot fail: {e}")


        df = df_filtered
        t_min = float(df["Time"].min())
        t_max = float(df["Time"].max())
        t_scale = t_max - t_min
        if not np.isfinite(t_scale) or t_scale <= 0.0:
            raise ValueError(f"Invalid time scale after preprocessing: t_min={t_min}, t_max={t_max}")
        inv_t_scale = 1.0 / t_scale
        df["t_norm"] = (df["Time"].values - t_min) / t_scale


    t_all_norm  = torch.tensor(df["t_norm"].values[:, None], dtype=torch.float32, device=device)
    ic_all      = torch.tensor(df[["ic0", "ic1"]].values,   dtype=torch.float32, device=device)
    x0_all      = torch.tensor(df["x0"].values[:, None],    dtype=torch.float32, device=device)
    x1_all      = torch.tensor(df["x1"].values[:, None],    dtype=torch.float32, device=device)
    N_total = t_all_norm.size(0)

    ensure_dir(results_dir)
    plots_dir = results_dir / "plots_ex2"
    ensure_dir(plots_dir)

    all_run_rows = []
    for run_idx in range(1, n_runs + 1):

        console(f"Set {set_id} | Run {run_idx}/{n_runs}")
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


        if str(feature_map_kind).lower() == "none" and Dt != 1:
            raise RuntimeError(f"feature_map='none' should output Dt==1, got Dt={Dt}.")



        if str(feature_map_kind).lower() != "none":
            try:
                t_probe = torch.tensor([[0.123], [0.456]], dtype=torch.float32, device=device)
                phi_probe = fmap(t_probe)
                if phi_probe.shape[1] <= 1:
                    log(f"[feature_map][warn] kind={feature_map_kind} produced Dt={phi_probe.shape[1]} (expected > 1).")

                if phi_probe.shape[1] == 1 and torch.allclose(phi_probe, t_probe, atol=1e-7, rtol=0.0):
                    log(f"[feature_map][warn] kind={feature_map_kind} appears to be an identity mapping.")
            except Exception as _e:
                log(f"[feature_map][warn] probe check failed: {_e}")




        fmap_used = (str(feature_map_kind).lower() != "none")
        fmap_msg = (
            f"[feature_map] used={fmap_used} kind={feature_map_kind} "
            f"learn_params={learn_feature_params} Dt={Dt} n_features={n_features}"
        )
        log(fmap_msg)
        if not fmap_used:
            log("[feature_map][warn] kind='none' provides very limited time representation; inverse parameter learning may stall. Consider feature_map='fourier' or 'rbf'.")




        in_dim = Dt + 2

        log(f"[model] backbone={model_type} in_dim={in_dim} out_dim=2 width={hidden_width} depth={hidden_depth}")
        backbone = build_model(model_type, in_dim, 2, hidden_width, hidden_depth, omega0).to(device)

        model = TwoNodeCyclePINN(
            feature_map=fmap, backbone=backbone,
            U_init=U_init, L_init=L_init, Th_init=Th_init,
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





        t0_time = time.time()
        last_adam_losses = {"total": None, "data": None, "res": None, "ic": None}

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None

        window_size = 3000
        window_sum = 0.0
        window_count = 0
        prev_window_avg = None
        adam_early_stopped = False

        window_best_loss = float("inf")
        window_best_state = None

        last_good_d_t = model.d_t.detach().clone()

        last_good_losses = {"total": None, "data": None, "res": None, "ic": None}



        last_good_adam_state = copy.deepcopy(model.state_dict())
        nan_detected = False


        loss_history = []
        lbfgs_eval_counter = 0

        for it in range(adam_steps):
            opt_adam.zero_grad(set_to_none=True)
            t_in_norm = t_train_base_norm.clone().requires_grad_(True)

            if device.type == "cuda" and use_amp:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    x0_p, x1_p, f0, f1 = model(t_in_norm, ic_train, inv_t_scale=inv_t_scale)
                    loss_data = ((x0_train - x0_p)**2 + (x1_train - x1_p)**2).mean()
                    loss_res  = (f0**2 + f1**2).mean()
                    if ic_mask.any():
                        ic_err = (x0_p[ic_mask] - ic_train[ic_mask, 0:1]).pow(2).mean() + \
                                 (x1_p[ic_mask] - ic_train[ic_mask, 1:2]).pow(2).mean()
                        loss_ic = ic_err
                    else:
                        loss_ic = torch.tensor(0.0, device=device)
                    loss = data_loss_wt * loss_data + physics_loss_wt * loss_res + ic_loss_wt * loss_ic
            else:

                x0_p, x1_p, f0, f1 = model(t_in_norm, ic_train, inv_t_scale=inv_t_scale)
                loss_data = ((x0_train - x0_p)**2 + (x1_train - x1_p)**2).mean()
                loss_res  = (f0**2 + f1**2).mean()
                if ic_mask.any():
                    ic_err = (x0_p[ic_mask] - ic_train[ic_mask, 0:1]).pow(2).mean() + \
                             (x1_p[ic_mask] - ic_train[ic_mask, 1:2]).pow(2).mean()
                    loss_ic = ic_err
                else:
                    loss_ic = torch.tensor(0.0, device=device)
                loss = data_loss_wt * loss_data + physics_loss_wt * loss_res + ic_loss_wt * loss_ic


            try:
                loss_finite = bool(torch.isfinite(loss).all().item())
                data_finite = bool(torch.isfinite(loss_data).all().item())
                res_finite = bool(torch.isfinite(loss_res).all().item())
                ic_finite = bool(torch.isfinite(loss_ic).all().item())
                if not (loss_finite and data_finite and res_finite and ic_finite):
                    log("[error] Detected NaN/Inf in loss components during Adam. Reverting to last good values and stopping Adam.")
                    with torch.no_grad():
                        model.d_t.copy_(last_good_d_t)
                    nan_detected = True

                    last_adam_losses = dict(last_good_losses)
                    break
                else:

                    try:
                        last_good_losses["total"] = float(loss.item())
                        last_good_losses["data"] = float(loss_data.item())
                        last_good_losses["res"] = float(loss_res.item())
                        last_good_losses["ic"] = float(loss_ic.item())

                        try:
                            last_good_adam_state = copy.deepcopy(model.state_dict())
                        except Exception:
                            pass
                    except Exception:

                        pass


                    try:
                        curr_total = float(loss.item())
                        if math.isfinite(curr_total) and curr_total < window_best_loss:
                            window_best_loss = curr_total
                            window_best_state = copy.deepcopy(model.state_dict())
                    except Exception:
                        pass

            except Exception:
                log("[error] Exception while checking loss finiteness; reverting to last good values and stopping Adam.")
                with torch.no_grad():
                    model.d_t.copy_(last_good_d_t)
                nan_detected = True
                last_adam_losses = dict(last_good_losses)
                break

            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt_adam)
                scaler.update()
            else:
                loss.backward(); opt_adam.step()


            try:

                try:
                    d_now = model._pos(model.d_t).detach().cpu().numpy()
                    d01 = float(d_now[0, 1])
                    d10 = float(d_now[1, 0])
                except Exception:
                    d01 = float('nan'); d10 = float('nan')

                loss_history.append({
                    "step": int(it + 1),
                    "phase": "adam",
                    "total": float(loss.item()),
                    "data": float(loss_data.item()),
                    "res": float(loss_res.item()),
                    "ic": float(loss_ic.item()),
                    "d_01": d01,
                    "d_10": d10
                })
            except Exception:

                pass


            try:
                d_tensor = model._pos(model.d_t)

                if not torch.isfinite(d_tensor).all():
                    log("[error] Detected NaN/Inf in learned d after Adam step. Reverting to last good values and stopping Adam.")
                    with torch.no_grad():
                        model.d_t.copy_(last_good_d_t)
                    nan_detected = True
                    break
                else:

                    last_good_d_t = model.d_t.detach().clone()
            except Exception:

                log("[error] Exception while checking learned d; reverting to last good values and stopping Adam.")
                with torch.no_grad():
                    model.d_t.copy_(last_good_d_t)
                nan_detected = True
                break


            try:
                loss_val = float(loss.item())
            except Exception:

                loss_val = float(loss.detach().cpu().numpy())
            window_sum += loss_val
            window_count += 1


            if (it + 1) % window_size == 0:
                curr_avg = window_sum / max(1, window_count)
                if prev_window_avg is None:
                    prev_window_avg = curr_avg
                    log(f"[info] Adam window {it+1-window_size+1}-{it+1} avg loss: {curr_avg:.6e} (no previous window)")
                else:
                    log(f"[info] Adam window {it+1-window_size+1}-{it+1} avg loss: {curr_avg:.6e} prev avg: {prev_window_avg:.6e}")

                    if curr_avg < prev_window_avg:
                        prev_window_avg = curr_avg
                        log("[info] Window average decreased; continuing Adam.")
                    else:
                        log("[info] Window average did NOT decrease; stopping Adam early.")
                        adam_early_stopped = True

                        window_sum = 0.0
                        window_count = 0
                        break

                window_sum = 0.0
                window_count = 0
                window_best_loss = float("inf")
                window_best_state = None

            if it % 500 == 0:
                d_now = model._pos(model.d_t).detach().cpu().numpy()
                U_now_t, L_now_t, Th_now_t = model.pos_matrices()
                U_now = U_now_t.detach().cpu().numpy()
                L_now = L_now_t.detach().cpu().numpy()
                Th_now = Th_now_t.detach().cpu().numpy()
                msg = (f"[ex2 set {set_id} run {run_idx}] Adam {it:05d} | Total {loss.item():.3e} | "
                       f"Data {loss_data.item():.3e}(w={data_loss_wt}) | Res {loss_res.item():.3e}(w={physics_loss_wt}) | IC {loss_ic.item():.3e}(w={ic_loss_wt}) | "
                       f"U=[[{U_now[0,0]:.4f},{U_now[0,1]:.4f}],[{U_now[1,0]:.4f},{U_now[1,1]:.4f}]] | "
                      f"L=[[{L_now[0,0]:.6f},{L_now[0,1]:.6f}],[{L_now[1,0]:.6f},{L_now[1,1]:.6f}]] | "
                       f"Th=[[{Th_now[0,0]:.4f},{Th_now[0,1]:.4f}],[{Th_now[1,0]:.4f},{Th_now[1,1]:.4f}]] | "
                       f"d=[[{d_now[0,0]:.4f},{d_now[0,1]:.4f}],[{d_now[1,0]:.4f},{d_now[1,1]:.4f}]]")
                if learn_feature_params:
                    msg += f" | fmap_head={str(fmap.short_summary())[:120]}"
                log(msg)
            last_adam_losses = {
                "total": float(loss.item()),
                "data": float(loss_data.item()),
                "res":  float(loss_res.item()),
                "ic":   float(loss_ic.item())
            }





        if nan_detected:
            if last_good_adam_state is not None:
                try:
                    model.load_state_dict(last_good_adam_state)
                    log("[info] Restored last-good Adam weights (pre-NaN) prior to LBFGS.")
                except Exception:
                    log("[warn] Failed to restore last-good Adam state; attempting best-of-window if available.")
                    if (window_best_state is not None):
                        try:
                            model.load_state_dict(window_best_state)
                            log(f"[info] Restored best-of-window weights (loss={window_best_loss:.6e}) prior to LBFGS.")
                        except Exception:
                            log("[warn] Failed to restore best-of-window state; proceeding with current weights.")
            elif (window_best_state is not None):
                try:
                    model.load_state_dict(window_best_state)
                    log(f"[info] Restored best-of-window weights (loss={window_best_loss:.6e}) prior to LBFGS.")
                except Exception:
                    log("[warn] Failed to restore best-of-window state; proceeding with current weights.")
        elif (adam_early_stopped and (window_best_state is not None)):
            try:
                model.load_state_dict(window_best_state)
                log(f"[info] Restored best-of-window weights (loss={window_best_loss:.6e}) prior to LBFGS.")
            except Exception:
                log("[warn] Failed to restore best-of-window state; proceeding with current weights.")



        last_lbfgs_losses = {"total": None, "data": None, "res": None, "ic": None}



        last_lbfgs_state = None

        last_lbfgs_d = None
        if lbfgs_on:
            log("[info] Running LBFGS fine-tuning...")
            def closure():
                nonlocal last_lbfgs_state, last_lbfgs_d
                nonlocal lbfgs_eval_counter

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
                loss = data_loss_wt * loss_data + physics_loss_wt * loss_res + ic_loss_wt * loss_ic

                try:
                    loss_finite = bool(torch.isfinite(loss).all().item())
                    data_finite = bool(torch.isfinite(loss_data).all().item())
                    res_finite = bool(torch.isfinite(loss_res).all().item())
                    ic_finite = bool(torch.isfinite(loss_ic).all().item())
                    if not (loss_finite and data_finite and res_finite and ic_finite):
                        log("[error] Detected NaN/Inf in loss components during LBFGS.")




                        if last_lbfgs_losses["total"] is None:
                            last_lbfgs_losses.update({k: v for k, v in last_good_losses.items()})
                        if last_lbfgs_d is None:
                            try:
                                last_lbfgs_d = model._pos(last_good_d_t).detach().cpu().numpy()
                            except Exception:
                                last_lbfgs_d = None
                        raise RuntimeError("NaN in loss during LBFGS")
                    else:

                        try:
                            last_good_losses["total"] = float(loss.item())
                            last_good_losses["data"] = float(loss_data.item())
                            last_good_losses["res"] = float(loss_res.item())
                            last_good_losses["ic"] = float(loss_ic.item())
                        except Exception:
                            pass



                    try:



                        last_lbfgs_state = copy.deepcopy(model.state_dict())
                        last_lbfgs_losses["total"] = float(loss.item())
                        last_lbfgs_losses["data"]  = float(loss_data.item())
                        last_lbfgs_losses["res"]   = float(loss_res.item())
                        last_lbfgs_losses["ic"]    = float(loss_ic.item())


                        try:

                            try:
                                d_tmp = model._pos(model.d_t).detach().cpu().numpy()
                                d01_l = float(d_tmp[0, 1])
                                d10_l = float(d_tmp[1, 0])
                            except Exception:
                                d01_l = float('nan'); d10_l = float('nan')

                            loss_history.append({
                                "step": int(adam_steps + lbfgs_eval_counter + 1),
                                "phase": "lbfgs",
                                "total": float(loss.item()),
                                "data": float(loss_data.item()),
                                "res": float(loss_res.item()),
                                "ic": float(loss_ic.item()),
                                "d_01": d01_l,
                                "d_10": d10_l
                            })
                            lbfgs_eval_counter += 1
                        except Exception:
                            pass

                        try:
                            d_tmp = model._pos(model.d_t).detach().cpu().numpy()
                            last_lbfgs_d = d_tmp
                        except Exception:
                            pass
                    except Exception:


                        pass
                except Exception:
                    log("[error] Exception while checking loss finiteness during LBFGS; reverting and aborting LBFGS.")
                    with torch.no_grad():
                        model.d_t.copy_(last_good_d_t)
                    last_lbfgs_losses.update({k: v for k, v in last_good_losses.items()})
                    raise RuntimeError("NaN or error while checking loss during LBFGS")
                loss.backward()

                d_tensor = model._pos(model.d_t)
                if not torch.isfinite(d_tensor).all():
                    log("[error] Detected NaN/Inf in learned d during LBFGS. Reverting to last good values and aborting LBFGS.")
                    with torch.no_grad():
                        model.d_t.copy_(last_good_d_t)

                    raise RuntimeError("NaN in learned d during LBFGS")
                d_now = d_tensor.detach().cpu().numpy()
                U_now_t, L_now_t, Th_now_t = model.pos_matrices()
                U_now = U_now_t.detach().cpu().numpy()
                L_now = L_now_t.detach().cpu().numpy()
                Th_now = Th_now_t.detach().cpu().numpy()
                msg = (f"[ex2 set {set_id} run {run_idx}] LBFGS loss: {loss.item():.3e} | "
                       f"Data {loss_data.item():.3e}(w={data_loss_wt}) | Res {loss_res.item():.3e}(w={physics_loss_wt}) | IC {loss_ic.item():.3e}(w={ic_loss_wt}) | "
                       f"U=[[{U_now[0,0]:.4f},{U_now[0,1]:.4f}],[{U_now[1,0]:.4f},{U_now[1,1]:.4f}]] | "
                       f"L=[[{L_now[0,0]:.6f},{L_now[0,1]:.6f}],[{L_now[1,0]:.6f},{L_now[1,1]:.6f}]] | "
                       f"Th=[[{Th_now[0,0]:.4f},{Th_now[0,1]:.4f}],[{Th_now[1,0]:.4f},{Th_now[1,1]:.4f}]] | "
                       f"d=[[{d_now[0,0]:.4f},{d_now[0,1]:.4f}],[{d_now[1,0]:.4f},{d_now[1,1]:.4f}]]")
                if learn_feature_params:
                    msg += f" | fmap_head={str(fmap.short_summary())[:120]}"
                log(msg)
                last_lbfgs_losses["total"] = float(loss.item())
                last_lbfgs_losses["data"]  = float(loss_data.item())
                last_lbfgs_losses["res"]   = float(loss_res.item())
                last_lbfgs_losses["ic"]    = float(loss_ic.item())
                return loss

            try:
                opt_lbfgs.step(closure)
            except RuntimeError as e:
                log(f"[warn] LBFGS aborted: {e}")



                if last_lbfgs_losses["total"] is None:
                    last_lbfgs_losses.update({k: v for k, v in last_good_losses.items()})
                if last_lbfgs_d is None:
                    try:
                        last_lbfgs_d = model._pos(last_good_d_t).detach().cpu().numpy()
                    except Exception:
                        last_lbfgs_d = None
                nan_detected = True
        else:
            log("[info] Skipping LBFGS (LBFGS ON/OFF=Off)")

        t1_time = time.time()


        model.eval()
        t_all_req_norm = t_all_norm.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            x0_pred, x1_pred, _, _ = model(t_all_req_norm, ic_all, inv_t_scale=inv_t_scale)
        err0 = torch.norm(x0_all - x0_pred.detach()) / torch.norm(x0_all)
        err1 = torch.norm(x1_all - x1_pred.detach()) / torch.norm(x1_all)




        try:
            if last_lbfgs_d is not None:
                d_learned = last_lbfgs_d
            else:
                d_learned = model._pos(model.d_t).detach().cpu().numpy()
        except Exception:
            try:
                d_learned = model._pos(last_good_d_t).detach().cpu().numpy()
            except Exception:

                d_learned = np.array([[float('nan'), float('nan')],[float('nan'), float('nan')]])


        def _get_param_matrix(name: str, clamp_positive: bool = False):
            try:
                if last_lbfgs_state is not None and name in last_lbfgs_state:
                    tensor_val = last_lbfgs_state[name]
                elif hasattr(model, name):
                    tensor_val = getattr(model, name)
                elif last_good_adam_state is not None and isinstance(last_good_adam_state, dict) and name in last_good_adam_state:
                    tensor_val = last_good_adam_state[name]
                else:
                    return np.full((2, 2), float('nan'), dtype=float)
                if clamp_positive:

                    if name in ("U_t", "L_t", "Th_t", "d_t"):
                        tensor_val = tensor_val**2 + 1e-6
                    else:
                        tensor_val = torch.clamp(tensor_val, min=1e-6)
                return tensor_val.detach().cpu().numpy()
            except Exception:
                return np.full((2, 2), float('nan'), dtype=float)

        U_learned = _get_param_matrix("U_t", clamp_positive=True)
        L_learned = _get_param_matrix("L_t", clamp_positive=True)
        Th_learned = _get_param_matrix("Th_t", clamp_positive=True)

        try:
            log(f"[learned params] "
                f"U=[[{U_learned[0,0]:.4f},{U_learned[0,1]:.4f}],[{U_learned[1,0]:.4f},{U_learned[1,1]:.4f}]] | "
                f"L=[[{L_learned[0,0]:.6f},{L_learned[0,1]:.6f}],[{L_learned[1,0]:.6f},{L_learned[1,1]:.6f}]] | "
                f"Th=[[{Th_learned[0,0]:.4f},{Th_learned[0,1]:.4f}],[{Th_learned[1,0]:.4f},{Th_learned[1,1]:.4f}]] | "
                f"d=[[{d_learned[0,0]:.4f},{d_learned[0,1]:.4f}],[{d_learned[1,0]:.4f},{d_learned[1,1]:.4f}]]")
        except Exception:
            pass




        dense_preds = {}
        if "Trajectory" in df.columns:
            for tr in sorted(df['Trajectory'].unique()):
                sub = df[df['Trajectory'] == tr]
                t_start, t_end = float(sub['Time'].min()), float(sub['Time'].max())
                n_dense = max(300, len(sub) * 4)
                t_dense = np.linspace(t_start, t_end, n_dense)
                t_dense_norm = (t_dense - t_min) / t_scale
                ic_pair = sub[['ic0','ic1']].iloc[0].values.astype(np.float32)
                t_dense_t = torch.tensor(t_dense_norm[:, None], dtype=torch.float32, device=device).requires_grad_(True)
                ic_dense_t = torch.tensor(np.repeat(ic_pair[None, :], n_dense, axis=0), dtype=torch.float32, device=device)
                with torch.enable_grad():
                    x0_d, x1_d, _, _ = model(t_dense_t, ic_dense_t, inv_t_scale=inv_t_scale)
                dense_preds[tr] = {
                    "t": t_dense,
                    "x0": x0_d.detach().cpu().numpy().ravel(),
                    "x1": x1_d.detach().cpu().numpy().ravel(),
                }
        else:
            t_start, t_end = float(df['Time'].min()), float(df['Time'].max())
            n_dense = max(300, len(df) * 4)
            t_dense = np.linspace(t_start, t_end, n_dense)
            t_dense_norm = (t_dense - t_min) / t_scale
            ic_pair = df[['ic0','ic1']].iloc[0].values.astype(np.float32)
            t_dense_t = torch.tensor(t_dense_norm[:, None], dtype=torch.float32, device=device).requires_grad_(True)
            ic_dense_t = torch.tensor(np.repeat(ic_pair[None, :], n_dense, axis=0), dtype=torch.float32, device=device)
            with torch.enable_grad():
                x0_d, x1_d, _, _ = model(t_dense_t, ic_dense_t, inv_t_scale=inv_t_scale)
            dense_preds[1] = {
                "t": t_dense,
                "x0": x0_d.detach().cpu().numpy().ravel(),
                "x1": x1_d.detach().cpu().numpy().ravel(),
            }


        times = df["Time"].values
        x0p = x0_pred.detach().cpu().numpy().ravel()
        x1p = x1_pred.detach().cpu().numpy().ravel()

        plt.figure()
        plt.plot(times, df["x0"].values, '.', label='x0 true', alpha=0.6)
        if "Trajectory" in df.columns:
            for tr in sorted(df['Trajectory'].unique()):
                mask = (df['Trajectory'].values == tr)
                dp = dense_preds.get(tr)
                if dp is not None:
                    plt.plot(dp['t'], dp['x0'], '-', linewidth=1.2, label=f'pred tr{tr}')
                else:
                    plt.plot(df.loc[mask, 'Time'].values, x0p[mask], '-', linewidth=1)
        else:
            dp = dense_preds.get(1)
            if dp is not None:
                plt.plot(dp['t'], dp['x0'], '-', linewidth=1.2, label='pred')
            else:
                plt.plot(times, x0p, '-', linewidth=1)
        plt.legend(); plt.xlabel('Time'); plt.ylabel('x0'); plt.tight_layout()
        plt.savefig(plots_dir / f"x0_ex2_set{set_id}_run{run_idx}.png", dpi=150); plt.close()

        plt.figure()
        plt.plot(times, df["x1"].values, '.', label='x1 true', alpha=0.6)
        if "Trajectory" in df.columns:
            for tr in sorted(df['Trajectory'].unique()):
                mask = (df['Trajectory'].values == tr)
                dp = dense_preds.get(tr)
                if dp is not None:
                    plt.plot(dp['t'], dp['x1'], '-', linewidth=1.2, label=f'pred tr{tr}')
                else:
                    plt.plot(df.loc[mask, 'Time'].values, x1p[mask], '-', linewidth=1)
        else:
            dp = dense_preds.get(1)
            if dp is not None:
                plt.plot(dp['t'], dp['x1'], '-', linewidth=1.2, label='pred')
            else:
                plt.plot(times, x1p, '-', linewidth=1)
        plt.legend(); plt.xlabel('Time'); plt.ylabel('x1'); plt.tight_layout()
        plt.savefig(plots_dir / f"x1_ex2_set{set_id}_run{run_idx}.png", dpi=150); plt.close()


        try:
            if len(loss_history) > 0:
                loss_hist_df = pd.DataFrame(loss_history)
                loss_csv_path = results_dir / f"loss_history_set{set_id}_run{run_idx}.csv"
                loss_hist_df.to_csv(loss_csv_path, index=False)


                plt.figure(figsize=(8,4))

                loss_hist_df = loss_hist_df.sort_values(by=["step"])
                steps = loss_hist_df["step"].values
                plt.plot(steps, loss_hist_df["total"].values, label="total", linewidth=1.5)
                plt.plot(steps, loss_hist_df["data"].values, label="data", linewidth=1.0)
                plt.plot(steps, loss_hist_df["res"].values, label="physics", linewidth=1.0)
                plt.plot(steps, loss_hist_df["ic"].values, label="IC", linewidth=1.0)
                plt.yscale('log')
                plt.xlabel('training step (Adam steps then LBFGS evals)')
                plt.ylabel('loss (log scale)')
                plt.legend()
                plt.tight_layout()
                loss_png_path = plots_dir / f"loss_history_set{set_id}_run{run_idx}.png"
                plt.savefig(loss_png_path, dpi=150)
                plt.close()
                log(f"[loss] Saved loss history CSV: {loss_csv_path} and PNG: {loss_png_path}")

                try:
                    if ("d_01" in loss_hist_df.columns) and ("d_10" in loss_hist_df.columns):
                        plt.figure(figsize=(8,4))
                        plt.plot(steps, loss_hist_df["d_01"].values, label="d_01", linewidth=1.5)
                        plt.plot(steps, loss_hist_df["d_10"].values, label="d_10", linewidth=1.5)
                        plt.xlabel('training step (Adam steps then LBFGS evals)')
                        plt.ylabel('learned d values')
                        plt.legend()
                        plt.tight_layout()
                        d_png_path = plots_dir / f"d_history_set{set_id}_run{run_idx}.png"
                        plt.savefig(d_png_path, dpi=150)
                        plt.close()
                        log(f"[loss] Saved d-history PNG: {d_png_path}")
                except Exception as e:
                    log(f"[loss][warn] Failed to save/plot d-history: {e}")
        except Exception as e:
            log(f"[loss][warn] Failed to save/plot loss history: {e}")


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
        row_dict = dict(row_copy) | dict(ic_seed=ic_seed_in_data,
            run=run_idx,
            set_id=set_id,
            data_csv=str(data_path),
            model_type=model_type,
            feature_map=feature_map_kind,
            ODE_type=ode_type,
            Learned_Feats="yes" if learned_feats else "no",
            learned_feature_params=int(learn_feature_params),
            fmap_summary=str(fmap_info) if learn_feature_params else "",
            adam_steps=adam_steps, lr_adam=lr_adam,
            lbfgs_max_iter=lbfgs_max_it, hidden_width=hidden_width, hidden_depth=hidden_depth,
            omega0=omega0, n_features=n_features, a_max=a_max, rbf_width=rbf_width, spline_knots=spline_knots,
            ic_embed_dim=ic_embed_dim, ic_loss_wt=ic_loss_wt, data_loss_wt=data_loss_wt, physics_loss_wt=physics_loss_wt,

            U_learned_01=float(U_learned[0,1]), U_learned_10=float(U_learned[1,0]), U_learned_00=float(U_learned[0,0]), U_learned_11=float(U_learned[1,1]),
            L_learned_01=float(L_learned[0,1]), L_learned_10=float(L_learned[1,0]), L_learned_00=float(L_learned[0,0]), L_learned_11=float(L_learned[1,1]),
            Th_learned_01=float(Th_learned[0,1]), Th_learned_10=float(Th_learned[1,0]), Th_learned_00=float(Th_learned[0,0]), Th_learned_11=float(Th_learned[1,1]),
            d_01=float(d_learned[0,1]), d_10=float(d_learned[1,0]), d_00=float(d_learned[0,0]), d_11=float(d_learned[1,1]),

            rel_err_x0=float(err0.item()), rel_err_x1=float(err1.item()),

            train_time_sec=float(t1_time - t0_time),

            t_min=t_min, t_max=t_max,

            adam_loss_total=last_adam_losses["total"],
            adam_loss_data=last_adam_losses["data"],
            adam_loss_res=last_adam_losses["res"],
            adam_loss_ic=last_adam_losses["ic"],
            lbfgs_loss_total=last_lbfgs_losses["total"],
            lbfgs_loss_data=last_lbfgs_losses["data"],
            lbfgs_loss_res=last_lbfgs_losses["res"],
            lbfgs_loss_ic=last_lbfgs_losses["ic"],
            LBFGS_ONOFF="On" if lbfgs_on else "Off",
            nan_d_detected=bool(nan_detected)
        )
        all_run_rows.append(row_dict)

    return all_run_rows

def main():
    parser = argparse.ArgumentParser(description="Unified PINN ex2 runner (merged) with GPU/AMP options")
    parser.add_argument("params_csv", nargs="?", default="param_ex2.csv", help="Parameter CSV path")
    parser.add_argument("--device", default=os.environ.get("PINN_DEVICE", "auto"), help="cpu|cuda|cuda:0|auto")
    parser.add_argument("--amp", action="store_true", default=bool(int(os.environ.get("PINN_USE_AMP", "0"))), help="Enable mixed precision (Adam phase)")
    parser.add_argument("--amp-dtype", default=os.environ.get("PINN_AMP_PRECISION", "fp16"), choices=["fp16","bf16"], help="AMP dtype")
    args = parser.parse_args()


    setup_run_logging()

    log("=== PINN ex2 merged start ===")
    log(f"[cwd] os.getcwd() = {os.getcwd()}")
    log(f"[base] BASE_DIR     = {BASE_DIR}")
    params_csv = args.params_csv

    dev_str = args.device.lower()
    if dev_str == "auto":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            if dev_str == "mps":
                device = torch.device("mps")
            else:
                device = torch.device(args.device)
        except Exception:
            log(f"[warn] Could not parse device '{args.device}', falling back to auto")
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            dev_name = torch.cuda.get_device_name(device.index or 0)
        except Exception:
            dev_name = str(device)
        log(f"[gpu] Using CUDA device: {dev_name}")
        torch.backends.cudnn.benchmark = True
    elif device.type == "mps":
        log("[gpu] Using Apple MPS device")
    else:
        log("[gpu] CUDA/MPS not available or CPU forced.")
    use_amp = args.amp and (device.type == "cuda")
    log(f"[amp] mixed precision: {'ON' if use_amp else 'OFF'} (dtype={args.amp_dtype if use_amp and device.type=='cuda' else 'fp32'})")
    if not Path(params_csv).exists():
        raise FileNotFoundError(f"Params CSV not found: {Path(params_csv).resolve()}")

    cfg_df = pd.read_csv(params_csv)
    results_dir = BASE_DIR / "results_two_node_cycle_ex2"; ensure_dir(results_dir)
    results_path = results_dir / "results_ex2.csv"
    old_results_path = results_dir / "runs_ex2.csv"
    if (not results_path.exists()) and old_results_path.exists():
        try:
            prev = pd.read_csv(old_results_path)
            prev.to_csv(results_path, index=False)
            log("[results] Migrated existing runs_ex2.csv to results_ex2.csv")
        except Exception as e:
            log(f"[results][warn] failed to migrate old results: {e}")


    if "data_type" in cfg_df.columns:
        present_types = set(cfg_df["data_type"].astype(str).str.lower().unique())
    else:
        present_types = set()
    if "hill" in present_types:
        log("[prep] generating all Hill datasets upfront...")
        try_run_generator("Hill_Data_production_ex2.py")
    if "piecewise" in present_types:
        log("[prep] generating all piecewise datasets upfront (python)...")
        try_run_generator("piecewise_Data_production_ex2.py")
    if "ramp" in present_types:
        log("[prep] generating all Ramp datasets upfront...")
        try_run_generator("Ramp_Data_production_ex2.py")

    for i, row in cfg_df.iterrows():
        log(f"\\n=== Running ex2 row {i+1}/{len(cfg_df)} ===")
        rows = train_once(row, results_dir, row_index_for_errors=i, device=device, use_amp=use_amp, amp_dtype_str=args.amp_dtype)
        results_df = pd.DataFrame(rows)
        append = results_path.exists() and os.path.getsize(results_path) > 0
        if append:
            try:
                prev_df = pd.read_csv(results_path)
                combined = pd.concat([prev_df, results_df], ignore_index=True, sort=False)
                combined.to_csv(results_path, index=False)
            except Exception as e:
                log(f"[results][warn] Failed to merge with existing results; appending without header ({e})")
                results_df.to_csv(results_path, mode='a', header=False, index=False)
        else:
            results_df.to_csv(results_path, index=False)
        log(f"\\nAppended {len(results_df)} rows to {results_path.resolve()}")


if __name__ == "__main__":
    main()
