# neural-operators

Two self-contained case studies applying neural operator architectures to physics problems with exact analytical solutions:

1. **2D Heat Equation** — FNO vs DeepONet2D on transient heat diffusion over a unit square. A Fourier-series solver provides the ground truth.
2. **Lamé Hollow Sphere** — DeepONet3D on 3D linear elasticity. Given inner/outer pressures, Young's modulus, and Poisson's ratio, the model predicts the von Mises stress field anywhere in the shell. The exact Lamé (1852) solution serves as ground truth.

Both problems have known closed-form solutions, making them rigorous benchmarks: every prediction can be checked against exact physics.

---

## Case 1: 2D Heat Equation

**Operator:** G(T\_L, T\_R, T\_B, T\_T, T\_init, α, t) → u(x, y, t) ∈ ℝ^{32×32}

Given four boundary temperatures, initial temperature, thermal diffusivity α, and time t, each model predicts the full 32×32 temperature field without solving the PDE at inference time.

| Model | Architecture | Params | Val MSE (ep 200) | Test Rel L2 |
|---|---|---|---|---|
| FNO | modes=(12,12), ch=32, 4 layers | ~700 K | 8.3 × 10⁻⁴ | ~4% |
| DeepONet2D | branch+trunk MLP, w=256, p=128, d=3 | ~330 K | 4.0 × 10⁻³ | ~5% |
| Analytical | Fourier series (40 steady + 20×20 transient modes) | — | exact | — |

---

## Case 2: Lamé Hollow Sphere

**Operator:** G(p\_i, p\_e, E, ν) → σ\_vm(x, y, z) ∈ ℝ^{N\_pts}

A thick-walled hollow sphere (inner radius a = 0.4 m, outer radius b = 1.0 m) under uniform internal and external pressure. The output is the von Mises equivalent stress at arbitrary query points in the shell.

| Model | Architecture | Params | Training |
|---|---|---|---|
| DeepONet3D | branch MLP(4→256³→128), trunk MLP(1→256³→128) | ~397 K | 100 epochs |

Trunk input dimension is 1 (radial distance r) — the field is radially symmetric, so (x, y, z) carries no additional information beyond r = ‖·‖. The model remains mesh-free: it can evaluate at any point in the shell at inference time.

![Lamé sphere — von Mises stress cross-section](outputs/lame_sphere_3d_case375_crosssection.png)

*Von Mises stress field, cross-section view. Highest stress at the inner wall (r = 0.4 m), decaying toward the outer surface — consistent with exact Lamé theory.*

Full figures and training curves: [RESULTS.md](RESULTS.md)

---

## Reproduce from scratch

**Requirements:** Python 3.10+, GPU optional.

```bash
git clone <repo-url>
cd neural-operators

python -m venv .venv && source .venv/bin/activate

# CPU (swap cu121 for GPU):
pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

### Case 1: 2D Heat Equation

```bash
# 5-minute demo (50 cases):
python scripts/generate_heat2d_dataset.py --config configs/heat2d_smoke.yaml
jupyter notebook notebooks/heat2d_train_compare.ipynb

# Full run (5 000 cases, ~15 min on 8 CPU cores):
python scripts/generate_heat2d_dataset.py
jupyter notebook notebooks/heat2d_train_compare.ipynb
python scripts/generate_carousel.py     # outputs/heat2d_carousel.pdf
python scripts/generate_animation.py    # outputs/heat2d_animation.mp4
```

### Case 2: Lamé Hollow Sphere

The sphere mesh (`data/sphere-FEMMeshGmsh.vtk`) is generated from the FreeCAD model in `data/` using Gmsh. Once the mesh is in place:

```bash
python scripts/generate_lame_sphere_fields.py   # compute analytical fields → dataset/lame_sphere_cases/
python scripts/generate_lame_sphere_dataset.py  # assemble Parquet splits → dataset/lame_sphere_*.parquet
jupyter notebook notebooks/lame_sphere_train.ipynb
python scripts/render_lame_sphere_3d.py         # outputs/lame_sphere_3d_*.png
```

Note: the full Lamé sphere dataset is ~30 GB. The notebook's `laptop:` config block supports subsampling query points for memory-constrained machines.

---

## Run tests and lint

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/ tests/ scripts/
```

CI runs on every push via GitHub Actions: ruff lint, 6 pytest smoke tests, and a 50-case end-to-end heat2d dataset generation.

---

## Project structure

```
src/neural_operators/
├── models/deeponet.py        # mlp(), DeepONet, DeepONet2D, DeepONet3D
├── data/heat2d.py            # Fourier solver, solve_case(), analytical_field()
├── data/lame_sphere.py       # Lamé analytical solver, solve_case(), QUERY_XYZ
├── data/anti_derivative.py   # loader for the NGC anti-derivative dataset
└── utils/metrics.py          # mse(), relative_l2(), metrics_summary()
configs/
├── heat2d.yaml               # full run (5 000 cases, 200 epochs)
├── heat2d_smoke.yaml         # quick demo (50 cases)
└── lame_sphere.yaml          # Lamé sphere (5 000 cases, 100 epochs, laptop subsampling)
notebooks/
├── heat2d_train_compare.ipynb   # FNO vs DeepONet2D on heat equation
├── lame_sphere_train.ipynb      # DeepONet3D on 3D elasticity
└── fno_anti_derivative.ipynb    # FNO on 1D anti-derivative (bonus)
scripts/
├── generate_heat2d_dataset.py      # heat2d Parquet dataset (parallel, --config flag)
├── generate_lame_sphere_fields.py  # per-case analytical field computation
├── generate_lame_sphere_dataset.py # assemble Lamé sphere Parquet splits
├── generate_carousel.py            # outputs/heat2d_carousel.pdf
├── generate_animation.py           # outputs/heat2d_animation.mp4
└── render_lame_sphere_3d.py        # outputs/lame_sphere_3d_*.png
tests/
├── test_heat2d.py               # solve_case, analytical_field smoke tests
└── test_models.py               # DeepONet, DeepONet2D, FNO forward-pass tests
docs/
└── lame_sphere_problem.md       # full problem statement, derivation, FEM setup
```

---

## What I learned

FNO reaches roughly 1.5× lower error than DeepONet on the heat equation because spectral convolution directly parameterises the Fourier modes the PDE lives in — it is, in a sense, the right inductive bias for this problem. By contrast, the Lamé sphere is a natural fit for DeepONet: the operator maps four scalar parameters to a field, and the branch-trunk decomposition separates "which physics" from "where to query" cleanly. For the Lamé problem the trunk input is just r = ‖xyz‖ (one scalar, not three coordinates) because the von Mises field is radially symmetric — discovering and exploiting this reduces the trunk's learning problem dramatically. FNO is grid-tied to the 32×32 training discretisation while DeepONet is inherently mesh-free, which proved important for the Lamé sphere where the query mesh is irregular and contains ~400 K points. On both problems, normalisation choices mattered more than architecture: log-scale for α and E, min-max for pressures and temperatures. Finally, having an exact analytical baseline for both problems made debugging unambiguous — when predictions diverged from the Lamé solution near the inner wall, it pointed immediately to insufficient radial resolution in the query grid rather than a model bug.

---

## Dependencies

- [PyTorch](https://pytorch.org/) 2.5
- [neuraloperator](https://neuraloperator.github.io/) >= 2.0 (FNO)
- NumPy, SciPy, matplotlib, tqdm
- PyArrow + pandas (Parquet datasets)
- PyVista (Lamé sphere mesh loading and 3D rendering)
- ruff, pytest (dev)

---

## License

MIT — see [LICENSE](LICENSE).
