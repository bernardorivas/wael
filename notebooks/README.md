# Companion notebooks - PINN parameter learning with DSGRN topological validation

Standalone, Colab-ready notebooks that walk through the paper's pipeline on each test network,
in a teaching style: sample parameters from a target DSGRN region -> simulate a steep Hill
system -> add bounded noise -> recover `(L, U, theta, d)` with a PINN and a least-squares
baseline -> check whether the learned parameters land back in the target region. The region
test is the explicit DSGRN inequality system, so **no DSGRN / pychomp install is required**.

| Notebook | Network | What it teaches | Open in Colab |
|---|---|---|---|
| `example1_toggle_switch.ipynb` | toggle switch (bistable) | the question, and the easy baseline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OWNER/REPO/blob/main/notebooks/example1_toggle_switch.ipynb) |
| `example2_mixed_feedback.ipynb` | mixed-feedback circuit | a good fit landing in the wrong region | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OWNER/REPO/blob/main/notebooks/example2_mixed_feedback.ipynb) |
| `example3_repressilator.ipynb` | repressilator (oscillatory) | oscillation + Fourier feature mapping | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OWNER/REPO/blob/main/notebooks/example3_repressilator.ipynb) |
| `example4_conley_morse.ipynb` | repressilator, region node 13 (periodic orbit / FC) | region recovery is not topology recovery, and two tweaks that close the gap | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OWNER/REPO/blob/main/notebooks/example4_conley_morse.ipynb) |

## Running

- **Colab (recommended):** click a badge above. `torch`, `numpy`, `scipy`, and `matplotlib`
  are preinstalled, and a free GPU speeds up the PINN cells.
- **Local:** Python 3.10-3.12 with `pip install numpy scipy matplotlib torch`.

## Notes

- The notebooks use a **neutral initial guess** for the physical parameters
  (`L ~ 0, U ~ 1, theta ~ 1`), not the data-generating values, so the recovery is an honest
  test rather than an initialization at the answer.
- Parameters and degradation are normalized with `gamma = 1`, matching the paper's examples.
- `example4` adds a second success criterion beyond region membership: whether the learned
  smooth model still reproduces the periodic orbit (the FC Morse set). Its parameters are a
  real `DSGRN.ParameterSampler` draw from node 13, chosen so each threshold sits mid-interval;
  its optional last section installs DSGRN to confirm the Morse graph directly.
- These are clean re-implementations for teaching; the numpy/scipy parts (ODE simulation, the
  region-inequality check, the least-squares baseline) are validated, and the PyTorch PINN
  cells are written to run in a torch-enabled environment (Colab).
- **Before sharing:** replace `OWNER/REPO` in this README and in each notebook's top badge
  with the public repository path so the Colab links resolve.
