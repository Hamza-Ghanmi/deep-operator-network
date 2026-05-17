# neural-operators

Two self-contained case studies applying neural operator architectures to physics problems with exact analytical solutions:

1. **2D Heat Equation** — FNO vs DeepONet2D on transient heat diffusion over a unit square. A Fourier-series solver provides the ground truth.
2. **Lamé Hollow Sphere** — DeepONet3D on 3D linear elasticity. Given the net pressure differential, the model predicts the von Mises stress field anywhere in the shell. The exact Lamé (1852) solution serves as ground truth.

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

**Operator:** G(Δp) → σ\_vm(r) ∈ ℝ^{N\_pts}

A thick-walled hollow sphere (inner radius a = 0.2 m, outer radius b = 0.5 m) under combined internal and external pressure. The model predicts the von Mises equivalent stress at arbitrary query points in the shell.

**EDA finding:** the Lamé solution shows σ\_vm(r) = |C₂| · 3/(2r³) where C₂ ∝ Δp = p\_i − p\_e. Young's modulus E and Poisson's ratio ν do not appear in the von Mises expression — confirmed by R² = 1.000 between Δp and max(σ\_vm) across 5 000 cases. The branch therefore takes only Δp as input (param\_dim = 1).

| Model | Architecture | Params | Test Rel L2 (mean ± std) | Max abs error |
|---|---|---|---|---|
| DeepONet3D | branch MLP(1→256→256→256→128), trunk MLP(1→256→256→256→128) | ~330 K | 0.008 ± 0.010 | 0.78 MPa |

Trunk input dimension is 1 (radial distance r = ‖xyz‖) — the field is radially symmetric, so (x, y, z) carries no additional information beyond r. The model is mesh-free and can evaluate at any point in the shell at inference time.

![Lamé sphere — von Mises stress cross-section](outputs/lame_sphere_3d_case375_crosssection.png)

*Von Mises stress field, cross-section view. Highest stress at the inner wall (r = 0.2 m), decaying as 1/r³ toward the outer surface — consistent with exact Lamé theory.*

Full figures and training curves: [RESULTS.md](RESULTS.md)

---

## Reproduce from scratch

**Requirements:** Python 3.12+, GPU optional. Uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Hamza-Ghanmi/deep-operator-network.git
cd deep-operator-network

# CPU (swap cu121 → cpu in pyproject.toml if no CUDA):
uv sync
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
python scripts/generate_lame_sphere_fields.py   # analytical fields → dataset/lame_sphere_cases/
python scripts/generate_lame_sphere_dataset.py  # assemble Parquet splits → dataset/lame_sphere_*.parquet
jupyter notebook notebooks/lame_sphere_train.ipynb
python scripts/render_lame_sphere_3d.py         # outputs/lame_sphere_3d_*.png
```

The full Lamé sphere dataset is ~30 GB. On the first notebook run the data-loading cell streams all three Parquet splits and saves a 69 MB cache (`dataset/lame_sphere_training_data.npz`); subsequent runs load from the cache in under a second. The `laptop:` config block in `configs/lame_sphere.yaml` controls query-point subsampling for memory-constrained machines.

---

## Serving with inference-hub

Trained models are exported as TorchScript files that [`inference-hub`](../inference-hub) can load and serve over HTTP. No model architecture code is needed in the serving layer — the `.pt` file is self-contained.

```bash
# After training (run from repo root):
python scripts/export_heat2d.py       # → outputs/don_heat2d_scripted.pt
python scripts/export_lame_sphere.py  # → outputs/don_lame_sphere_scripted.pt
```

Load and query via inference-hub:

```bash
# Load the heat2d model
curl -X POST http://localhost:8000/api/v1/models/heat2d/load \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/don_heat2d_scripted.pt", "framework": "pytorch"}'

# Predict: [T_L, T_R, T_B, T_T, T_init, alpha, t] in raw SI units
curl -X POST http://localhost:8000/api/v1/models/heat2d/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[0.0, 100.0, 0.0, 0.0, 20.0, 0.02, 5.0]]}'
# → (1, 1024) temperature field in °C on a fixed 32×32 grid

# Load and query the Lamé sphere model
curl -X POST http://localhost:8000/api/v1/models/lame_sphere/load \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/don_lame_sphere_scripted.pt", "framework": "pytorch"}'

curl -X POST http://localhost:8000/api/v1/models/lame_sphere/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[10e6, 2e6]]}'
# → (1, 200) σ_vm in Pa on a fixed radial grid [0.2 m, 0.5 m]
```

The inference wrappers (`models/inference.py`) embed normalisation constants as TorchScript buffers — callers pass raw SI units, no external config needed.

---

## Run tests and lint

```bash
uv run pytest -v -m unit      # fast unit tests (no data files or checkpoints needed)
uv run pytest -v               # all tests
uv run ruff check src/ tests/ scripts/
uv run mypy src/
```

CI runs on every push via GitHub Actions: ruff lint, 25 pytest unit tests, and a 50-case end-to-end heat2d dataset generation.

---

## Project structure

```
src/neural_operators/           # installable package (hexagonal architecture)
├── domain/
│   ├── entities/               # pure @dataclass types — no framework imports
│   │   ├── heat2d.py           #   Heat2DParams, Heat2DField
│   │   └── lame_sphere.py      #   LameSphereParams, LameSphereField
│   └── ports/
│       ├── physics_solver.py   #   IPhysicsSolver[P,R] ABC
│       └── model_exporter.py   #   IModelExporter ABC
├── models/
│   ├── deeponet.py             # mlp(), DeepONet, DeepONet2D, DeepONet3D
│   └── inference.py            # Heat2DPredictor, LameSpherePredictor (TorchScript wrappers)
├── use_cases/
│   ├── generate_dataset.py     # GenerateDatasetUseCase
│   └── export_model.py         # ExportModelUseCase
├── adapters/
│   ├── physics/
│   │   ├── heat2d_solver.py    #   FourierHeat2DSolver + module-level solve_case / sample_params
│   │   └── lame_sphere_solver.py #   LameSolver + module-level solve_case / sample_params
│   └── exporters/
│       └── torchscript.py      #   TorchScriptExporter (script → trace fallback)
├── data/anti_derivative.py     # loader for the NGC anti-derivative dataset
└── utils/metrics.py            # mse(), relative_l2(), metrics_summary()
configs/
├── heat2d.yaml                 # full run (5 000 cases, 200 epochs)
├── heat2d_smoke.yaml           # quick demo (50 cases)
└── lame_sphere.yaml            # Lamé sphere (5 000 cases, 100 epochs, laptop subsampling)
notebooks/
├── heat2d_train_compare.ipynb  # FNO vs DeepONet2D on heat equation
├── lame_sphere_train.ipynb     # DeepONet3D on Lamé sphere (EDA-driven improvements)
├── eda_lame_sphere.ipynb       # exploratory data analysis — Lamé sphere dataset
├── eda_anti_derivative.ipynb   # exploratory data analysis — anti-derivative dataset
└── fno_anti_derivative.ipynb   # FNO on 1D anti-derivative (bonus)
scripts/
├── generate_heat2d_dataset.py      # heat2d Parquet dataset (parallel, --config flag)
├── generate_lame_sphere_fields.py  # per-case analytical field computation
├── generate_lame_sphere_dataset.py # assemble Lamé sphere Parquet splits
├── export_heat2d.py                # → outputs/don_heat2d_scripted.pt (inference-hub)
├── export_lame_sphere.py           # → outputs/don_lame_sphere_scripted.pt (inference-hub)
├── generate_carousel.py            # outputs/heat2d_carousel.pdf
├── generate_animation.py           # outputs/heat2d_animation.mp4
└── render_lame_sphere_3d.py        # outputs/lame_sphere_3d_*.png
tests/
├── test_heat2d.py                  # solve_case, analytical_field smoke tests
├── test_models.py                  # DeepONet, DeepONet2D, FNO forward-pass tests
└── unit/
    ├── test_domain_entities.py     # Heat2DParams, LameSphereParams entity tests
    ├── test_export_use_case.py     # ExportModelUseCase with FakeExporter stub
    └── test_inference_wrappers.py  # predictor shapes + TorchScript scriptability
docs/
└── lame_sphere_problem.md          # full problem statement, derivation, FEM setup
```

---

## What I learned

**FNO vs DeepONet on the heat equation.** FNO reaches roughly 1.5× lower error than DeepONet2D because spectral convolution directly parameterises the Fourier modes the PDE lives in — it is, in a sense, the right inductive bias for this problem.

**DeepONet is the natural fit for the Lamé sphere.** The operator maps scalar parameters to a field, and the branch-trunk decomposition separates "which physics" from "where to query" cleanly. Because σ\_vm is radially symmetric the trunk takes r = ‖xyz‖ (one scalar, not three coordinates), reducing its learning problem to fitting 1/r³. FNO would require a regular grid; DeepONet is inherently mesh-free, which matters when query points live on an irregular 409 230-node FEM mesh.

**EDA before training pays off.** Exploratory analysis of the Lamé dataset (`eda_lame_sphere.ipynb`) revealed three actionable findings that each translated directly into a code change:

1. *R² = 1.000 for Δp → σ\_vm* — E and ν are algebraically absent from the Lamé von Mises formula. Passing them to the branch added two noise dimensions. Dropping them (param\_dim 4 → 1) reduced the worst-case test rel-L2 from 0.55 to 0.16.
2. *1590× dynamic range in σ\_vm* — the fixed-divisor normalisation made low-Δp cases nearly invisible in MSE loss. Switching to log-space targets made every case contribute equally without a custom ε heuristic.
3. *6:1 outer/inner vertex imbalance* — the inner wall (peak stress, safety-critical) had 6× fewer mesh points than the outer surface. Stratified radial sampling across 8 equal shells corrected the coverage.

Combined, these changes improved mean rel-L2 from 0.026 to 0.008 and max absolute error from 1.90 MPa to 0.78 MPa with identical architecture and training time.

**Normalisation choices matter more than architecture tuning.** On both problems, getting the input and target scales right (log-uniform for E and α, min-max for pressures and temperatures, log-space for σ\_vm targets) had larger impact than hyperparameter search. Having exact analytical baselines made debugging unambiguous — when predictions diverged near the inner wall it pointed immediately to the mesh imbalance rather than a model bug.

---

## Dependencies

- [PyTorch](https://pytorch.org/) 2.5
- [neuraloperator](https://neuraloperator.github.io/) >= 2.0 (FNO)
- NumPy, SciPy, matplotlib, tqdm
- PyArrow + pandas (Parquet datasets)
- PyVista (Lamé sphere mesh loading and 3D rendering)
- ruff, mypy, pytest (dev)

---

## License

MIT — see [LICENSE](LICENSE).
