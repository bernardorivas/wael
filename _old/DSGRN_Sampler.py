#!/usr/bin/env python3
# dsgrn_csv_from_fixed_region.py
#
# For each row in param_ex1.csv:
#   - Sample fresh (L, U, Theta) INSIDE THE SAME parameter region (default par_index=4)
#   - Overwrite CSV columns with new values (JSON arrays)
#
# Columns written:
#   - If CSV already has any of ["L_np","U_np","Th_np","Theta_np","L","U","Theta","Th"],
#     we overwrite those three consistently. Otherwise we create L_np, U_np, Th_np.
#
# Usage:
#   python dsgrn_csv_from_fixed_region.py
#   python dsgrn_csv_from_fixed_region.py --csv param_ex1.csv --par-index 4 --seed 123 --backup param_ex1.bak.csv

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import DSGRN


NET_SPEC = """x : (~y)
              y : (~x)"""


# -------------------------------
# Robust sampler output handling
# -------------------------------
def _ensure_dict(maybe_json: Any) -> Dict[str, Any]:
    return json.loads(maybe_json) if isinstance(maybe_json, str) else maybe_json


# Accept BOTH parentheses and square brackets; allow optional trailing index
# Examples matched: L(x->y), U[x->y], T(x,y), T[x,y,0]
_PAT_ARROW = re.compile(r"^([LUT])[[(]\s*([^\]\)>\s,]+)\s*->\s*([^\]\)>\s,]+)(?:\s*,\s*\d+)?\s*[\])]\s*$")
_PAT_COMMA = re.compile(r"^([LUT])[[(]\s*([^\]\)\s,]+)\s*,\s*([^\]\)\s,]+)(?:\s*,\s*\d+)?\s*[\])]\s*$")


def parse_sample_to_mats(sample_obj: Any, network: DSGRN.Network) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a DSGRN Parameter sample into dense L, U, Theta matrices."""
    data = _ensure_dict(sample_obj)
    p = data.get("Parameter", {})

    n = network.size()
    L = np.zeros((n, n), dtype=float)
    U = np.zeros((n, n), dtype=float)
    T = np.zeros((n, n), dtype=float)

    def node_idx(name: str) -> int:
        try:
            return network.index(name)
        except Exception:
            # fallback: allow "0","1"
            k = int(name)
            if 0 <= k < n:
                return k
            raise

    matched = 0
    for key, val in p.items():
        ks = key.strip() if isinstance(key, str) else ""
        m = _PAT_ARROW.match(ks) or _PAT_COMMA.match(ks)
        if not m:
            continue
        kind, a_name, b_name = m.groups()
        i, j = node_idx(a_name), node_idx(b_name)
        v = float(val)
        if kind == "L":
            L[i, j] = v
        elif kind == "U":
            U[i, j] = v
        elif kind == "T":
            T[i, j] = v
        matched += 1

    if matched == 0:
        # Helpful debug
        print("No keys matched. First 10 keys:", list(p.keys())[:10])
        raise RuntimeError("Could not parse any L/U/T entries. Check key format.")

    return L, U, T


# -------------------------------
# CSV I/O and main workflow
# -------------------------------
def choose_output_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """Pick which 3 columns to overwrite or create for L, U, Theta."""
    # Preference order for an existing schema
    candidates = [
        ("L_np", "U_np", "Th_np"),
        ("L_np", "U_np", "Theta_np"),
        ("L", "U", "Th"),
        ("L", "U", "Theta"),
    ]
    for Lc, Uc, Tc in candidates:
        if Lc in df.columns and Uc in df.columns and Tc in df.columns:
            return Lc, Uc, Tc
    # Default if nothing exists
    return "L_np", "U_np", "Th_np"


def main():
    ap = argparse.ArgumentParser(description="Overwrite CSV with fresh L,U,Theta from the same DSGRN parameter region.")
    ap.add_argument("--csv", type=str, default="param_ex1.csv", help="Path to CSV to modify.")
    ap.add_argument("--par-index", type=int, default=4, help="Fixed parameter region index.")
    ap.add_argument("--seed", type=int, default=None, help="Base RNG seed; row r uses seed+r for variation.")
    ap.add_argument("--backup", type=str, default=None, help="Optional backup file path.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if args.backup:
        Path(args.backup).write_bytes(csv_path.read_bytes())
        print(f"Backup written to: {args.backup}")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Ensure we know where to write
    L_col, U_col, T_col = choose_output_columns(df)
    for col in (L_col, U_col, T_col):
        if col not in df.columns:
            df[col] = ""

    # Build network and fixed region
    network = DSGRN.Network(NET_SPEC)
    pg = DSGRN.ParameterGraph(network)
    if args.par_index < 0 or args.par_index >= pg.size():
        raise IndexError(f"--par-index {args.par_index} out of range [0, {pg.size()-1}]")
    region = pg.parameter(args.par_index)

    print(f"Network nodes: {network.size()}, parameter regions: {pg.size()}")
    print(f"Using FIXED region index: {args.par_index}")
    print(f"Rows in CSV: {len(df)}")
    sampler = DSGRN.ParameterSampler(network)

    # For each row: draw a new sample from the SAME region
    for r in range(len(df)):
        # Optional reproducibility knob; ensures different draws per row if a base seed is provided
        if args.seed is not None:
            np.random.seed(args.seed + r)

        sample_obj = sampler.sample(region)  # fresh random draw inside region
        L, U, T = parse_sample_to_mats(sample_obj, network)

        # Store as JSON arrays
        df.at[r, L_col] = json.dumps(L.tolist())
        df.at[r, U_col] = json.dumps(U.tolist())
        df.at[r, T_col] = json.dumps(T.tolist())

        print(f"Row {r}: wrote new sample from region {args.par_index} into columns [{L_col}, {U_col}, {T_col}]")

    # Save
    df.to_csv(csv_path, index=False)
    print(f"Updated CSV saved: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
