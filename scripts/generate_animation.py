"""Generate a 4-panel animation for the 2D heat equation results.

  [ Ground Truth ] [ FNO ] [ DeepONet ] [ |Error| FNO vs DON ]

Picks the test case with the highest temperature range (most visually striking).
Saves: outputs/heat2d_animation.mp4  (GIF fallback if ffmpeg unavailable)

Usage (from repo root):
    python scripts/generate_animation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pyarrow.parquet as pq
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neural_operators.models import DeepONet2D
from neuralop.models import FNO

DATA_DIR = PROJECT_ROOT / "dataset"
OUT_DIR  = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

NX = NY         = 32
T_SCALE         = 200.0
ALPHA_MIN, ALPHA_MAX = 0.005, 0.05
T_MAX           = 30.0
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load data ─────────────────────────────────────────────────────────────────
df = pq.read_table(DATA_DIR / "heat2d_dataset.parquet").to_pandas()

rng        = np.random.default_rng(0)
all_cases  = df["case_id"].unique()
rng.shuffle(all_cases)
n_train    = int(0.8 * len(all_cases))
test_cases = set(all_cases[n_train:])
test_df    = df[df["case_id"].isin(test_cases)]

# ── Pick the most visually interesting case ───────────────────────────────────
case_stats = (
    test_df.groupby("case_id")
    .first()[["T_L", "T_R", "T_B", "T_T", "T_init", "alpha"]]
    .assign(contrast=lambda d: d[["T_L", "T_R", "T_B", "T_T"]].max(axis=1) - d["T_init"])
)
best_case_id = case_stats["contrast"].idxmax()
chosen = case_stats.loc[best_case_id]
print(f"\nChosen case_id: {best_case_id}")
print(
    f"  T_L={chosen.T_L:.0f}°C  T_R={chosen.T_R:.0f}°C  "
    f"T_B={chosen.T_B:.0f}°C  T_T={chosen.T_T:.0f}°C"
)
print(f"  T_init={chosen.T_init:.0f}°C  α={chosen.alpha:.4f} m²/s")

case_rows = test_df[test_df["case_id"] == best_case_id].sort_values("t").reset_index(drop=True)
times     = case_rows["t"].values
T_fields  = np.stack([np.array(r["T_field"]).reshape(NX, NY) for _, r in case_rows.iterrows()])

x_grid = np.linspace(0, 1, NX, dtype=np.float32)
y_grid = np.linspace(0, 1, NY, dtype=np.float32)
X_GRID, Y_GRID = np.meshgrid(x_grid, y_grid, indexing="ij")
trunk_xy = np.stack([X_GRID.ravel(), Y_GRID.ravel()], axis=1).astype(np.float32)

T_L, T_R   = float(chosen.T_L),   float(chosen.T_R)
T_B, T_T   = float(chosen.T_B),   float(chosen.T_T)
T_init     = float(chosen.T_init)
alpha      = float(chosen.alpha)
alpha_norm = (alpha - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN)

fno_inputs = np.zeros((len(times), 5, NX, NY), dtype=np.float32)
for i, t in enumerate(times):
    bc = np.full((NX, NY), T_init / T_SCALE, dtype=np.float32)
    bc[0,  :]  = T_L / T_SCALE
    bc[-1, :]  = T_R / T_SCALE
    bc[:,  0]  = T_B / T_SCALE
    bc[:, -1]  = T_T / T_SCALE
    bc[0,  0]  = (T_L + T_B) / 2 / T_SCALE
    bc[0, -1]  = (T_L + T_T) / 2 / T_SCALE
    bc[-1, 0]  = (T_R + T_B) / 2 / T_SCALE
    bc[-1, -1] = (T_R + T_T) / 2 / T_SCALE
    fno_inputs[i, 0] = bc
    fno_inputs[i, 1] = alpha_norm
    fno_inputs[i, 2] = t / T_MAX
    fno_inputs[i, 3] = X_GRID
    fno_inputs[i, 4] = Y_GRID

don_params_arr = np.array(
    [[T_L/T_SCALE, T_R/T_SCALE, T_B/T_SCALE,
      T_T/T_SCALE, T_init/T_SCALE, alpha_norm, t/T_MAX]
     for t in times],
    dtype=np.float32,
)

# ── Load models ───────────────────────────────────────────────────────────────
fno_model = FNO(n_modes=(12, 12), in_channels=5, out_channels=1,
                hidden_channels=32, n_layers=4).to(device)
fno_model.load_state_dict(torch.load(OUT_DIR / "fno_heat2d.pt", map_location=device)["state_dict"])
fno_model.eval()

don_model = DeepONet2D().to(device)
don_model.load_state_dict(torch.load(OUT_DIR / "don_heat2d.pt", map_location=device)["state_dict"])
don_model.eval()

# ── Run inference ─────────────────────────────────────────────────────────────
with torch.no_grad():
    fno_preds = fno_model(
        torch.from_numpy(fno_inputs).to(device)
    ).squeeze(1).cpu().numpy() * T_SCALE

    don_preds = don_model(
        torch.from_numpy(don_params_arr).to(device),
        torch.from_numpy(trunk_xy).to(device),
    ).cpu().numpy().reshape(len(times), NX, NY) * T_SCALE

# ── Build animation ───────────────────────────────────────────────────────────
vmin = min(T_fields.min(), fno_preds.min(), don_preds.min())
vmax = max(T_fields.max(), fno_preds.max(), don_preds.max())
emax = max(np.abs(fno_preds - T_fields).max(), np.abs(don_preds - T_fields).max())

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
fig.patch.set_facecolor("#0d0d0d")
for ax in axes:
    ax.set_facecolor("#0d0d0d")
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

for ax, title in zip(axes, ["Ground Truth", "FNO", "DeepONet", "|Error| FNO vs DON"]):
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)

xv = np.linspace(0, 1, NX)
yv = np.linspace(0, 1, NY)

ims = []
for ax, data, cmap, vm in [
    (axes[0], T_fields[0],                       "inferno", (vmin, vmax)),
    (axes[1], fno_preds[0],                      "inferno", (vmin, vmax)),
    (axes[2], don_preds[0],                      "inferno", (vmin, vmax)),
    (axes[3], np.abs(fno_preds[0]-don_preds[0]), "hot",     (0, emax)),
]:
    im = ax.pcolormesh(xv, yv, data.T, cmap=cmap, vmin=vm[0], vmax=vm[1], shading="auto")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    cb.outline.set_edgecolor("#444")
    ims.append(im)

for ax in axes[:3]:
    ax.set_xlabel("x", color="white", fontsize=8)
    ax.set_ylabel("y", color="white", fontsize=8)
axes[3].set_xlabel("x", color="white", fontsize=8)

bc_text = (
    f"T_L={T_L:.0f}°C  T_R={T_R:.0f}°C  T_B={T_B:.0f}°C  T_T={T_T:.0f}°C  "
    f"T₀={T_init:.0f}°C  α={alpha:.4f} m²/s"
)
sup = fig.suptitle("", color="white", fontsize=9, y=0.995)


def update(frame: int) -> list:
    arrays = [
        T_fields[frame],
        fno_preds[frame],
        don_preds[frame],
        np.abs(fno_preds[frame] - don_preds[frame]),
    ]
    for im, arr in zip(ims, arrays):
        im.set_array(arr.T.ravel())
    fno_rel = np.linalg.norm(fno_preds[frame] - T_fields[frame]) / (np.linalg.norm(T_fields[frame]) + 1e-8)
    don_rel = np.linalg.norm(don_preds[frame] - T_fields[frame]) / (np.linalg.norm(T_fields[frame]) + 1e-8)
    sup.set_text(
        f"t = {times[frame]:.3f} s    FNO rel-L2 = {fno_rel:.4f}   DON rel-L2 = {don_rel:.4f}\n{bc_text}"
    )
    return ims + [sup]


ani = animation.FuncAnimation(fig, update, frames=len(times), interval=700, blit=False)
plt.tight_layout(rect=[0, 0, 1, 0.93])

# ── Save ──────────────────────────────────────────────────────────────────────
mp4_path = OUT_DIR / "heat2d_animation.mp4"
gif_path = OUT_DIR / "heat2d_animation.gif"
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    writer = animation.FFMpegWriter(fps=2, bitrate=1800,
                                    extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    ani.save(str(mp4_path), writer=writer, dpi=150, savefig_kwargs={"facecolor": "#0d0d0d"})
    print(f"\nSaved MP4 → {mp4_path}  ({mp4_path.stat().st_size/1e6:.1f} MB)")
except Exception as e:
    print(f"MP4 failed ({e}), falling back to GIF …")
    ani.save(str(gif_path), writer="pillow", fps=2, dpi=120, savefig_kwargs={"facecolor": "#0d0d0d"})
    print(f"Saved GIF → {gif_path}  ({gif_path.stat().st_size/1e6:.1f} MB)")

plt.close()
print("Done.")