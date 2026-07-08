# Companion notebooks

Colab notebooks that walk through the paper's pipeline on each test network.

The pipeline is

1. start from a fixed parameter set and identify the DSGRN region it lies in (the target),
2. simulate a steep Hill system,
3. add bounded noise,
4. recover `(L, U, theta, d)` with PINN and least-squares methods, scoring only `(L, U, theta)` against the target region (since DSGRN does not depend on `d`),
5. check whether the learned parameters land back in the target region, and if not, check if the Morse graph is isomorphic.

Each notebook installs DSGRN and DSGRN_utils uses them for the topological validation.

| Notebook | Network | Open in Colab |
|---|---|---|
| `example1_toggle_switch.ipynb` | toggle switch, region 4 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bernardorivas/wael/blob/main/notebooks/example1_toggle_switch.ipynb) |
| `example2_mixed_2network.ipynb` | mixed-feedback 2-network, region 712 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bernardorivas/wael/blob/main/notebooks/example2_mixed_2network.ipynb) |
| `example3_repressilator.ipynb` | repressilator, region 13 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bernardorivas/wael/blob/main/notebooks/example3_repressilator.ipynb) |
| `example4_mixed_3network.ipynb` | 3-gene regulatory network, region 2472287 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bernardorivas/wael/blob/main/notebooks/example4_mixed_3network.ipynb) |

## Running

- **Colab (recommended):** click the link above. `torch`, `numpy`, `scipy`, and `matplotlib` are preinstalled in Google Colab; the first cell installs DSGRN and DSGRN_utils. Select a GPU runtime and the PINN runs on CUDA automatically.
- **Local:** Python 3.10+ with `pip install numpy scipy matplotlib torch DSGRN tqdm` and `pip install git+https://github.com/marciogameiro/DSGRN_utils.git`.

## Reproducing the recovery sweep

Each notebook has an `HParams` config cell right after the imports — the single source of truth for every hyperparameter (model width/depth, optimizer settings, loss weights, and the sweep controls), read everywhere as `HP.<FIELD>`.

- Set `HP.N_TRIALS` (default `15`) to control how many independent trials run per noise level. Each trial reuses the fixed ground-truth parameters and clean trajectories, varying only the noise-realization seed and the network-initialization seed.
- The six noise levels are fixed to `(0, 1, 5, 10, 25, 50)%`, matching the paper.
- The sweep runs both the PINN and the least-squares baseline and reports a region-recovery table plus a recovery-rate-vs-noise plot.
- `HP.MAX_EPOCHS` is the main wall-clock knob (the sweep runs `N_TRIALS × 6` PINN fits); use a GPU runtime for Examples 2–4. The full-scale paper numbers use 100 trials per level (600 aggregated for Example 3); `N_TRIALS=15` is a smaller-scale reproduction.
- Example 3 exposes `HP.USE_FEATURE_MAP` (default `True`): setting it to `False` reproduces the no-Fourier-feature-map ablation.

## Notes

- Each notebook selects its compute device at the top: CUDA on a Colab GPU, MPS on Apple silicon, otherwise CPU. MPS is single precision, so the PINN uses float32 on GPU/MPS and float64 on CPU; the numpy simulation and least-squares baseline always run in double precision.
- The notebooks use a neutral initial guess for the physical parameters (`L ~ 0, U ~ 1, theta ~ 1`), not the data-generating values.
- Parameters are considered normalized with `gamma = 1` since a change of variables $y=\Gamma x$ is possible.
- Trajectory initial conditions are Latin-hypercube samples over the state box.
- Region membership is tested with DSGRN's `par_index_from_sample`: `to_matrices` packs the learned `(L, U, theta)` into DSGRN's parameter matrices and the returned region index is compared against the target node. The final section builds the Conley-Morse graph with `DSGRN_utils.ConleyMorseGraph` and compares it to the target via `DSGRN.isomorphic_morse_graphs` in case region membership differs.
- The Colab links point to [`bernardorivas/wael`](https://github.com/bernardorivas/wael) (branch `main`, under `notebooks/`).
