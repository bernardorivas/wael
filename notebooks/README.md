# Companion notebooks - PINN parameter learning with DSGRN topological validation

Standalone, Colab-ready notebooks that walk through the paper's pipeline on each test network.
The pipeline is

1. start from a fixed parameter set and identify the DSGRN region it lies in (the target),
2. simulate a steep Hill system,
3. add bounded noise,
4. recover `(L, U, theta, d)` with PINN and least-squares methods, scoring only `(L, U, theta)` against the target region since DSGRN does not depend on `d`,
5. check whether the learned parameters land back in the target region.

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

## Notes

- Each notebook selects its compute device at the top: CUDA on a Colab GPU, MPS on Apple silicon, otherwise CPU. MPS is single precision, so the PINN uses float32 on GPU/MPS and float64 on CPU; the numpy simulation and least-squares baseline always run in double precision.
- The notebooks use a neutral initial guess for the physical parameters (`L ~ 0, U ~ 1, theta ~ 1`), not the data-generating values.
- Parameters and degradation are normalized with `gamma = 1` since a change of variables $y=\Gamma x$ is possible.
- Trajectory initial conditions are Latin-hypercube samples over the state box.
- Region membership is tested with DSGRN's `par_index_from_sample`: `to_matrices` packs the learned `(L, U, theta)` into DSGRN's parameter matrices and the returned region index is compared against the target node. The final section builds the Conley-Morse graph with `DSGRN_utils.ConleyMorseGraph` and compares it to the target via `DSGRN.isomorphic_morse_graphs` in case region membership differs.
- The Colab links point to [`bernardorivas/wael`](https://github.com/bernardorivas/wael) (branch `main`, under `notebooks/`).
