"""Generate a LinkedIn carousel PDF (dark theme, 9 slides).
Output: outputs/heat2d_carousel.pdf

Slides:
  1  Title
  2  The Problem
  3  Two Architectures
  4  Training Curves
  5  Analytical Baseline
  6  Test Set Metrics
  7  Field Comparison
  8  Error vs Time
  9  Key Takeaways

Usage (from repo root):
    python scripts/generate_carousel.py           # full run (inference + PNGs + PDF)
    python scripts/generate_carousel.py --cached  # fast: skip inference, reuse existing PNGs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT_DIR  = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "dataset"
NX = NY  = 32
T_SCALE  = 200.0
ALPHA_MIN, ALPHA_MAX = 0.005, 0.05
T_MAX    = 30.0
EVAL_BATCH = 64

# ── Known training stats (from notebook run) ──────────────────────────────────
FNO_PARAMS    = 357_473
DON_PARAMS    = 331_777
FNO_TRAIN_S   = 15_095.2
DON_TRAIN_S   =    319.3
N_TRAIN_CASES = 3_500
N_VAL_CASES   =   750
N_TEST_CASES  =   750

# ── Cached test metrics (from notebook run) — used with --cached ───────────────
CACHED_METRICS = {
    "fno_mse": None, "fno_rel": None, "fno_std": None, "fno_max": None,
    "don_mse": None, "don_rel": None, "don_std": None, "don_max": None,
}

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--cached", action="store_true",
                    help="Skip model loading and inference; reuse existing output PNGs "
                         "and cached metrics. Fast — no GPU needed.")
args = parser.parse_args()

CACHED = args.cached

if CACHED:
    # Verify the PNGs we'll embed actually exist
    missing = [p for p in [
        OUT_DIR / "heat2d_compare_curves.png",
        OUT_DIR / "heat2d_field_comparison.png",
        OUT_DIR / "heat2d_error_vs_time.png",
    ] if not p.exists()]
    if missing:
        print("ERROR: --cached requires these files to exist (run without --cached first):")
        for p in missing: print(f"  {p}")
        sys.exit(1)

    # Check if metrics cache file exists
    metrics_path = OUT_DIR / "heat2d_metrics.npz"
    if metrics_path.exists():
        m = np.load(metrics_path)
        fno_mse, fno_rel, fno_std, fno_max = float(m["fno_mse"]), float(m["fno_rel"]), float(m["fno_std"]), float(m["fno_max"])
        don_mse, don_rel, don_std, don_max = float(m["don_mse"]), float(m["don_rel"]), float(m["don_std"]), float(m["don_max"])
        print(f"Loaded cached metrics from {metrics_path}")
    elif all(v is not None for v in CACHED_METRICS.values()):
        fno_mse = CACHED_METRICS["fno_mse"]; fno_rel = CACHED_METRICS["fno_rel"]
        fno_std = CACHED_METRICS["fno_std"]; fno_max = CACHED_METRICS["fno_max"]
        don_mse = CACHED_METRICS["don_mse"]; don_rel = CACHED_METRICS["don_rel"]
        don_std = CACHED_METRICS["don_std"]; don_max = CACHED_METRICS["don_max"]
    else:
        print("ERROR: --cached requires either outputs/heat2d_metrics.npz or CACHED_METRICS "
              "populated in the script. Run once without --cached to generate the cache.")
        sys.exit(1)
    print("Skipping inference (--cached mode).")

else:
    import pyarrow.parquet as pq
    import torch
    from neuralop.models import FNO

    from neural_operators.models import DeepONet2D
    from neural_operators.utils import metrics_summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load models ───────────────────────────────────────────────────────────────
    print("Loading FNO …")
    fno_model = FNO(n_modes=(12, 12), in_channels=5, out_channels=1,
                    hidden_channels=32, n_layers=4).to(device)
    fno_model.load_state_dict(
        torch.load(OUT_DIR / "fno_heat2d.pt", map_location=device)["state_dict"])
    fno_model.eval()

    print("Loading DeepONet …")
    don_model = DeepONet2D().to(device)
    don_model.load_state_dict(
        torch.load(OUT_DIR / "don_heat2d.pt", map_location=device)["state_dict"])
    don_model.eval()

    # ── Load test data ─────────────────────────────────────────────────────────────
    print("Loading test data …")
    d        = np.load(DATA_DIR / "heat2d_training_data.npz")
    fno_Xte  = torch.from_numpy(d["fno_X_test"])
    fno_Yte  = torch.from_numpy(d["fno_Y_test"])
    don_Pte  = torch.from_numpy(d["don_params_test"])
    don_Ute  = torch.from_numpy(d["don_u_test"])
    trunk_xy = torch.from_numpy(d["trunk_xy"]).to(device)
    IS_CUDA  = device.type == "cuda"

    test_df = pq.read_table(DATA_DIR / "heat2d_test.parquet").to_pandas()
    test_t  = test_df["t"].values

    # ── Batched inference ──────────────────────────────────────────────────────────
    print("Running inference …")
    fno_preds, don_preds = [], []
    with torch.no_grad():
        for i in range(0, len(fno_Xte), EVAL_BATCH):
            xb = fno_Xte[i:i+EVAL_BATCH].to(device, non_blocking=IS_CUDA)
            fno_preds.append(fno_model(xb).squeeze(1).cpu())
        for i in range(0, len(don_Pte), EVAL_BATCH):
            pb = don_Pte[i:i+EVAL_BATCH].to(device, non_blocking=IS_CUDA)
            don_preds.append(don_model(pb, trunk_xy).cpu())

    fno_pred_C = torch.cat(fno_preds).numpy() * T_SCALE
    don_pred_C = torch.cat(don_preds).numpy().reshape(-1, NX, NY) * T_SCALE
    true_C     = fno_Yte.squeeze(1).numpy() * T_SCALE

    fno_mse, fno_rel, fno_std, fno_max, fno_rel_all = metrics_summary(fno_pred_C, true_C)
    don_mse, don_rel, don_std, don_max, don_rel_all = metrics_summary(don_pred_C, true_C)

    print(f"FNO  test MSE={fno_mse:.4f}  Rel-L2={fno_rel:.4f}  MaxErr={fno_max:.2f}")
    print(f"DON  test MSE={don_mse:.4f}  Rel-L2={don_rel:.4f}  MaxErr={don_max:.2f}")

    # ── Save metrics cache so future --cached runs don't need inference ────────────
    np.savez(OUT_DIR / "heat2d_metrics.npz",
             fno_mse=fno_mse, fno_rel=fno_rel, fno_std=fno_std, fno_max=fno_max,
             don_mse=don_mse, don_rel=don_rel, don_std=don_std, don_max=don_max)
    print(f"Metrics cached → {OUT_DIR}/heat2d_metrics.npz")

pct_better  = 100 * (don_rel - fno_rel) / don_rel
speed_ratio = FNO_TRAIN_S / DON_TRAIN_S

# ── Regenerate comparison PNGs (skipped in --cached mode) ─────────────────────
if not CACHED:
    print("Generating comparison PNGs …")
    sample_idx = [0, 20, 80, 150]
    xv = np.linspace(0, 1, NX)
    yv = np.linspace(0, 1, NY)

    fig_fc, axes_fc = plt.subplots(len(sample_idx), 3, figsize=(10, 13), layout="constrained")
    for ax, lbl in zip(axes_fc[0], ["True field [°C]", "FNO |error| [°C]", "DeepONet |error| [°C]"]):
        ax.set_title(lbl, fontsize=9, fontweight="bold")
    for row, si in enumerate(sample_idx):
        true  = true_C[si]
        f_err = np.abs(fno_pred_C[si] - true)
        d_err = np.abs(don_pred_C[si] - true)
        emax  = max(f_err.max(), d_err.max())
        im_t = axes_fc[row, 0].pcolormesh(xv, yv, true.T, cmap="hot", shading="auto")
        fig_fc.colorbar(im_t, ax=axes_fc[row, 0], shrink=0.9)
        for col, (err, rl) in enumerate([(f_err, fno_rel_all[si]), (d_err, don_rel_all[si])], start=1):
            im = axes_fc[row, col].pcolormesh(xv, yv, err.T, cmap="Reds", vmin=0, vmax=emax, shading="auto")
            fig_fc.colorbar(im, ax=axes_fc[row, col], shrink=0.9)
            axes_fc[row, col].set_xlabel(f"rel-L2 = {rl:.4f}", fontsize=7)
        axes_fc[row, 0].set_ylabel(f"Sample {si}", fontsize=8)
        for ax in axes_fc[row]:
            ax.set_aspect("equal"); ax.tick_params(labelsize=6)
    fig_fc.suptitle("True field and absolute error — FNO vs DeepONet (test set)", fontsize=11)
    fig_fc.savefig(OUT_DIR / "heat2d_field_comparison.png", dpi=120)
    plt.close(fig_fc)

    fig_et, ax_et = plt.subplots(figsize=(9, 4))
    for label, rel_all, color, marker in [("FNO", fno_rel_all, "C0", "o"), ("DeepONet", don_rel_all, "C1", "s")]:
        unique_t = np.unique(test_t)
        mean_err = [rel_all[test_t == t].mean() for t in unique_t]
        std_err  = [rel_all[test_t == t].std()  for t in unique_t]
        ax_et.errorbar(unique_t, mean_err, yerr=std_err, marker=marker, ms=5,
                       lw=1.5, capsize=3, label=label, color=color)
    ax_et.set_xscale("log")
    ax_et.set_xlabel("Time t [s]"); ax_et.set_ylabel("Relative L2 error")
    ax_et.set_title("Error vs Time — FNO vs DeepONet (test set, mean ± std)")
    ax_et.legend(); ax_et.grid(True, which="both", ls="--", alpha=0.4)
    fig_et.tight_layout()
    fig_et.savefig(OUT_DIR / "heat2d_error_vs_time.png", dpi=120)
    plt.close(fig_et)
else:
    print("Reusing existing PNGs (--cached mode).")

# ── Slide helpers ─────────────────────────────────────────────────────────────
BG     = "#0f0f17"
PANEL  = "#1a1a2e"
ACCENT = "#e94560"
BLUE   = "#4fc3f7"
ORANGE = "#ffb74d"
GREEN  = "#81c784"
WHITE  = "#f0f0f0"
GREY   = "#888899"
SLIDE_W = SLIDE_H = 8.0

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": WHITE, "axes.labelcolor": WHITE,
    "xtick.color": GREY, "ytick.color": GREY,
    "axes.edgecolor": "#33334a", "axes.facecolor": PANEL,
    "figure.facecolor": BG, "grid.color": "#2a2a40", "grid.linestyle": "--",
})


def new_slide(title: str | None = None, subtitle: str | None = None) -> plt.Figure:
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H))
    fig.patch.set_facecolor(BG)
    if title:
        y = 0.91 if subtitle else 0.93
        fig.text(0.5, y, title, ha="center", va="top", fontsize=20, fontweight="bold", color=WHITE)
    if subtitle:
        fig.text(0.5, 0.86, subtitle, ha="center", va="top", fontsize=11, color=GREY)
    ax_rule = fig.add_axes([0.07, 0.83 if subtitle else 0.87, 0.86, 0.004])
    ax_rule.set_facecolor(ACCENT)
    ax_rule.axis("off")
    return fig


def save(pdf: PdfPages, fig: plt.Figure) -> None:
    pdf.savefig(fig, dpi=135, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


fno_train_str = f"{FNO_TRAIN_S/3600:.1f} h"
don_train_str = f"{DON_TRAIN_S:.0f} s"

print("Building carousel PDF …")
with PdfPages(OUT_DIR / "heat2d_carousel.pdf") as pdf:

    # Slide 1 — Title
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H))
    fig.patch.set_facecolor(BG)
    ax_bar = fig.add_axes([0, 0.62, 1, 0.06])
    ax_bar.imshow(np.linspace(0, 1, 256).reshape(1, -1), aspect="auto", cmap="cool", extent=[0, 1, 0, 1])
    ax_bar.axis("off")
    fig.text(0.5, 0.80, "FNO  vs  DeepONet", ha="center", fontsize=26, fontweight="bold", color=WHITE)
    fig.text(0.5, 0.70, "on the 2D Transient Heat Equation", ha="center", fontsize=15, color=GREY)
    fig.text(0.5, 0.52,
             "Two operator-learning architectures.\nOne physics problem.\nHead-to-head benchmark.",
             ha="center", fontsize=13, color=WHITE, linespacing=1.7)
    fig.text(0.5, 0.10, "Swipe to see results →", ha="center", fontsize=11, color=ACCENT, style="italic")
    save(pdf, fig)

    # Slide 2 — The Problem
    fig = new_slide("The Problem", "Learn the parameter-to-field operator")
    ax = fig.add_axes([0.07, 0.10, 0.86, 0.70])
    ax.set_facecolor(PANEL); ax.axis("off")
    dom_x, dom_y, dom_w, dom_h = 0.25, 0.20, 0.50, 0.55
    ax.add_patch(FancyBboxPatch((dom_x, dom_y), dom_w, dom_h,
                                boxstyle="round,pad=0,rounding_size=0.02",
                                facecolor="#16213e", edgecolor=BLUE, lw=2,
                                transform=ax.transAxes))
    ax.text(dom_x - 0.06, dom_y + dom_h/2, "T_L", ha="center", va="center",
            fontsize=11, color=BLUE, transform=ax.transAxes, fontweight="bold")
    ax.text(dom_x + dom_w + 0.06, dom_y + dom_h/2, "T_R", ha="center", va="center",
            fontsize=11, color=ORANGE, transform=ax.transAxes, fontweight="bold")
    ax.text(dom_x + dom_w/2, dom_y - 0.07, "T_B", ha="center", va="center",
            fontsize=11, color=GREEN, transform=ax.transAxes, fontweight="bold")
    ax.text(dom_x + dom_w/2, dom_y + dom_h + 0.07, "T_T", ha="center", va="center",
            fontsize=11, color=ACCENT, transform=ax.transAxes, fontweight="bold")
    ax.text(dom_x + dom_w/2, dom_y + dom_h/2 + 0.07, "T(x,y,t) = ?",
            ha="center", va="center", fontsize=13, color=WHITE, transform=ax.transAxes, fontweight="bold")
    ax.text(dom_x + dom_w/2, dom_y + dom_h/2 - 0.07, "32 × 32 grid",
            ha="center", va="center", fontsize=10, color=GREY, transform=ax.transAxes)
    ax.annotate("", xy=(0.83, 0.50), xytext=(0.17, 0.50),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.0),
                xycoords="axes fraction", textcoords="axes fraction")
    for i, p in enumerate(["T_L, T_R, T_B, T_T  (0–200 °C)", "T_init  (0–100 °C)",
                            "α  (0.005–0.05 m²/s)", "t  (query time)"]):
        ax.text(0.02, 0.80 - i*0.17, f"  {p}", ha="left", va="center",
                fontsize=9, color=WHITE, transform=ax.transAxes)
    ax.text(0.50, 0.93, "G : params  →  T(x,y,t)", ha="center", va="center",
            fontsize=13, color=ACCENT, transform=ax.transAxes, fontweight="bold")
    fig.text(0.5, 0.06,
             f"{N_TRAIN_CASES + N_VAL_CASES + N_TEST_CASES:,} cases × 10 time steps"
             f"  |  Train {N_TRAIN_CASES:,} / Val {N_VAL_CASES} / Test {N_TEST_CASES} (split by case_id)",
             ha="center", fontsize=9, color=GREY)
    save(pdf, fig)

    # Slide 3 — Two Architectures
    fig = new_slide("Two Approaches", "Same problem, different inductive biases")
    rows = [
        ("Input",    "5-ch 32×32 spatial field",     "7 scalars + (x,y) query"),
        ("Output",   "32×32 temperature field",        "scalar T at query point"),
        ("Key idea", "Spectral convolution (Fourier)", "Branch·Trunk dot-product"),
        ("Params",   f"{FNO_PARAMS//1000} K",          f"{DON_PARAMS//1000} K"),
        ("Training", fno_train_str,                    don_train_str),
    ]
    col_left, col_mid, col_right = 0.07, 0.43, 0.70
    row_start, row_step = 0.77, 0.11
    fig.text(col_mid,        row_start + 0.04, "FNO",       ha="center",
             fontsize=14, fontweight="bold", color=BLUE)
    fig.text(col_right+0.13, row_start + 0.04, "DeepONet",  ha="center",
             fontsize=14, fontweight="bold", color=ORANGE)
    for i, (label, fno_val, don_val) in enumerate(rows):
        y = row_start - i * row_step
        if i % 2 == 0:
            ax_bg = fig.add_axes([0.07, y - 0.06, 0.86, row_step], zorder=0)
            ax_bg.set_facecolor("#1f1f35"); ax_bg.axis("off")
        fig.text(col_left,        y, label,   ha="left",   fontsize=10, color=GREY,   fontweight="bold")
        fig.text(col_mid+0.10,    y, fno_val, ha="center", fontsize=10, color=BLUE)
        fig.text(col_right+0.13,  y, don_val, ha="center", fontsize=10, color=ORANGE)
    fig.text(0.50, 0.12,
             "FNO applies global Fourier convolutions — naturally captures\n"
             "long-range spatial correlations in the temperature field.",
             ha="center", fontsize=9, color=GREY, linespacing=1.5)
    fig.text(0.50, 0.04,
             f"DeepONet separates parameter encoding from spatial query —\n"
             f"{speed_ratio:.0f}× faster training, mesh-free inference at arbitrary points.",
             ha="center", fontsize=9, color=GREY, linespacing=1.5)
    save(pdf, fig)

    # Slide 4 — Training Curves
    fig = new_slide("Training Curves", "Validation MSE (normalised) over 200 epochs")
    img = np.array(Image.open(OUT_DIR / "heat2d_compare_curves.png"))
    ax = fig.add_axes([0.07, 0.12, 0.86, 0.68])
    ax.imshow(img[:, : img.shape[1]//2])
    ax.axis("off")
    fig.text(0.26, 0.11, "FNO converges slower\nbut reaches lower error",
             ha="center", fontsize=9, color=BLUE)
    fig.text(0.72, 0.11, f"DON trains {speed_ratio:.0f}× faster\nwith higher error floor",
             ha="center", fontsize=9, color=ORANGE)
    save(pdf, fig)

    # Slide 5 — Analytical Baseline
    fig = new_slide("Analytical Solution", "Ground truth source and its inherent limitations")
    ax = fig.add_axes([0.07, 0.09, 0.86, 0.72])
    ax.set_facecolor(PANEL); ax.axis("off")
    ax.text(0.04, 0.95, "Source of error in the analytical solution",
            ha="left", va="top", fontsize=11, fontweight="bold", color=GREEN, transform=ax.transAxes)
    y_pos = 0.83
    for title, body in [
        ("1  Fourier-series truncation",
         "The exact solution is an infinite double sine series.\n"
         "We truncate at M = N = 20 transient modes and K = 40\n"
         "steady-state modes — higher spatial frequencies are dropped."),
        ("2  Gibbs oscillations at corners",
         "Where adjacent boundary conditions conflict (e.g. T_L ≠ T_B),\n"
         "the truncated series overshoots near those corners.\n"
         "These are small but visible artefacts in the ground-truth data."),
    ]:
        ax.text(0.04, y_pos, title, ha="left", va="top", fontsize=10,
                fontweight="bold", color=WHITE, transform=ax.transAxes)
        ax.text(0.06, y_pos - 0.07, body, ha="left", va="top", fontsize=8.5,
                color=GREY, transform=ax.transAxes, linespacing=1.5)
        y_pos -= 0.28
    save(pdf, fig)

    # Slide 6 — Test Set Metrics
    fig = new_slide("Test Set Results",
                    f"{N_TEST_CASES} held-out cases × 10 time steps ({N_TEST_CASES*10:,} samples)")
    metrics_rows = [
        ("Test MSE (°C²)",       f"{fno_mse:.1f}",       f"{don_mse:.1f}",       "↓ better"),
        ("Rel. L₂ error (mean)", f"{fno_rel*100:.1f} %", f"{don_rel*100:.1f} %", "↓ better"),
        ("Rel. L₂ error (std)",  f"{fno_std*100:.1f} %", f"{don_std*100:.1f} %", "↓ better"),
        ("Max abs. error (°C)",  f"{fno_max:.1f}",        f"{don_max:.1f}",       "↓ better"),
        ("Training time",        fno_train_str,           don_train_str,           "↓ better"),
        ("Parameters",           f"{FNO_PARAMS//1000} K", f"{DON_PARAMS//1000} K", "—"),
    ]
    col_label, col_fno, col_don, col_hint = 0.07, 0.52, 0.72, 0.90
    row0, rstep = 0.76, 0.095
    fig.text(col_fno, row0+0.045, "FNO",      ha="center", fontsize=13, fontweight="bold", color=BLUE)
    fig.text(col_don, row0+0.045, "DeepONet", ha="center", fontsize=13, fontweight="bold", color=ORANGE)
    for i, (label, fno_v, don_v, hint) in enumerate(metrics_rows):
        y = row0 - i * rstep
        if i % 2 == 0:
            ax_bg = fig.add_axes([0.07, y - rstep*0.55, 0.86, rstep], zorder=0)
            ax_bg.set_facecolor("#1f1f35"); ax_bg.axis("off")
        try:
            fno_num = float(fno_v.replace(" %","").replace(" K","").replace(" h",""))
            don_num = float(don_v.replace(" %","").replace(" K","").replace(" s","").replace(" h",""))
            fno_win = fno_num < don_num if hint != "—" else True
        except ValueError:
            fno_win = True
        fig.text(col_label, y, label, ha="left",   fontsize=10, color=WHITE)
        fig.text(col_fno,   y, fno_v, ha="center", fontsize=10,
                 color=GREEN if fno_win else BLUE, fontweight="bold" if fno_win else "normal")
        fig.text(col_don,   y, don_v, ha="center", fontsize=10,
                 color=GREEN if not fno_win else ORANGE, fontweight="bold" if not fno_win else "normal")
        fig.text(col_hint,  y, hint,  ha="right",  fontsize=8, color=GREY)
    fig.text(0.5, 0.04,
             "Analytical solution is the ground truth — its error is not comparable (see previous slide)",
             ha="center", fontsize=8, color=GREY, style="italic")
    save(pdf, fig)

    # Slide 7 — Field Comparison
    fig = new_slide("Field Comparison", "True field  |  FNO error  |  DON error  (4 test samples)")
    ax = fig.add_axes([0.02, 0.09, 0.96, 0.73])
    ax.imshow(np.array(Image.open(OUT_DIR / "heat2d_field_comparison.png"))); ax.axis("off")
    fig.text(0.5, 0.04, "FNO errors are visibly smaller and more evenly distributed across the domain",
             ha="center", fontsize=9, color=GREY)
    save(pdf, fig)

    # Slide 8 — Error vs Time
    fig = new_slide("Error vs Time", "Relative L₂ error per time step (mean ± std)")
    ax = fig.add_axes([0.04, 0.11, 0.92, 0.70])
    ax.imshow(np.array(Image.open(OUT_DIR / "heat2d_error_vs_time.png"))); ax.axis("off")
    fig.text(0.5, 0.04, "Both models struggle at early times (sharp gradients) — FNO recovers faster",
             ha="center", fontsize=9, color=GREY)
    save(pdf, fig)

    # Slide 9 — Key Takeaways
    fig = new_slide("Key Takeaways")
    for i, (color, label, text) in enumerate([
        (BLUE,   "FNO",       f"{pct_better:.0f}% lower relative L₂ error\nthanks to spectral inductive bias"),
        (ORANGE, "DeepONet",  f"{speed_ratio:.0f}× faster training\nand mesh-free inference"),
        (ACCENT, "Both",      "run inference in milliseconds\nafter a one-time training cost"),
        (GREEN,  "Verdict",   "FNO for accuracy-critical tasks;\nDeepONet when speed or flexibility matters"),
    ]):
        y = 0.74 - i * 0.16
        ax_bar = fig.add_axes([0.07, y - 0.04, 0.008, 0.10])
        ax_bar.set_facecolor(color); ax_bar.axis("off")
        fig.text(0.10, y + 0.03, label, ha="left", fontsize=13, fontweight="bold", color=color)
        fig.text(0.10, y - 0.02, text,  ha="left", fontsize=10, color=WHITE, linespacing=1.5)
    fig.text(0.5, 0.07,
             "Operator learning is a compelling surrogate for PDE-constrained design\n"
             "where thousands of forward solves are required.",
             ha="center", fontsize=9.5, color=GREY, linespacing=1.5)
    save(pdf, fig)

print(f"\nSaved → {OUT_DIR}/heat2d_carousel.pdf  (9 slides)")