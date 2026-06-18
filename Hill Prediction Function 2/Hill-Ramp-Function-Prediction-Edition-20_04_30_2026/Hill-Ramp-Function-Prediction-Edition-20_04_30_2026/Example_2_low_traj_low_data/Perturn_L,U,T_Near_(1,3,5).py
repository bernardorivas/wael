import os
import argparse
import pandas as pd
import numpy as np
import ast

def jitter_off_diag(matrix_str, base_value, noise_max, rng=None):




    mat = np.array(ast.literal_eval(matrix_str), dtype=float)
    n = mat.shape[0]


    off_diag_mask = ~np.eye(n, dtype=bool)


    mask = (mat != 0) & off_diag_mask



    if rng is None:
        rng = np.random.default_rng()


    noise = rng.uniform(-noise_max, noise_max, size=mat.shape)

    mat[mask] = base_value + noise[mask]

    return repr(mat.tolist())


def add_noise_inplace(path="param_ex1.csv", noise_max=0.4, seed=None):











    rng = np.random.default_rng(seed)

    df = pd.read_csv(path)

    for idx in df.index:
        df.at[idx, "L_np"]  = jitter_off_diag(df.at[idx, "L_np"],  base_value=1, noise_max=noise_max, rng=rng)
        df.at[idx, "Th_np"] = jitter_off_diag(df.at[idx, "Th_np"], base_value=3, noise_max=noise_max, rng=rng)
        df.at[idx, "U_np"]  = jitter_off_diag(df.at[idx, "U_np"],  base_value=5, noise_max=noise_max, rng=rng)


    df.to_csv(path, index=False)


def _default_param_path():


    return os.path.join(os.path.dirname(__file__), "param_ex1.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jitter off-diagonal entries in param_ex1.csv")
    parser.add_argument("--path", default=_default_param_path(),
                        help="Path to param CSV (default: param_ex1.csv next to this script)")
    parser.add_argument("--noise_max", type=float, default=0.4, help="Maximum absolute noise to add")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without modifying file")

    args = parser.parse_args()

    if args.dry_run:
        print(f"Dry run: would update {args.path} with noise_max={args.noise_max} seed={args.seed}")
    else:

        try:
            if os.path.exists(args.path):
                bak = args.path + ".bak"
                print(f"Backing up existing file to {bak}")
                with open(args.path, "r", encoding="utf-8") as fr, open(bak, "w", encoding="utf-8") as fw:
                    fw.write(fr.read())
        except Exception as e:
            print(f"Warning: could not create backup: {e}")

        add_noise_inplace(args.path, noise_max=args.noise_max, seed=args.seed)
        print(f"Updated {args.path}")
