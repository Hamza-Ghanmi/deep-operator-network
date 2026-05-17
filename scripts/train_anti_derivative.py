"""Train DeepONet on the 1-D anti-derivative operator.

Reads hyperparameters from configs/anti_derivative.yaml.
Saves checkpoint to outputs/deeponet.pt and training history to outputs/history.npz.

Usage (from repo root):
    python scripts/train_anti_derivative.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neural_operators.data.anti_derivative import load_anti_derivative
from neural_operators.models import DeepONet, mlp


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "anti_derivative.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = (
        PROJECT_ROOT
        / "physicsnemo_sym_examples_supplemental_materials_v0.0.1"
        / "examples"
        / "anti_derivative"
        / "data"
        / "anti_derivative.npy"
    )
    if not data_path.exists():
        warnings.warn(
            f"Data file not found at {data_path}. "
            "Download from NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/"
            "physicsnemo/resources/physicsnemo_sym_examples_supplemental_materials"
        )
        sys.exit(1)

    raw = load_anti_derivative(data_path)

    def to_gpu(key: str) -> torch.Tensor:
        return torch.tensor(raw[key], dtype=torch.float32, device=device)

    a_train, x_train, u_train = to_gpu("a_train"), to_gpu("x_train"), to_gpu("u_train")
    a_test,  x_test,  u_test  = to_gpu("a_test"),  to_gpu("x_test"),  to_gpu("u_test")

    n_train    = a_train.shape[0]
    batch_size = cfg["batch_size"]["train"]

    branch_cfg = cfg["arch"]["branch"]
    trunk_cfg  = cfg["arch"]["trunk"]
    dropout    = cfg["arch"].get("dropout", 0.0)

    branch_net = mlp(branch_cfg["n_sensors"], branch_cfg["layer_size"], branch_cfg["nr_layers"], dropout=dropout)
    trunk_net  = mlp(trunk_cfg["n_dim"],      trunk_cfg["layer_size"],  trunk_cfg["nr_layers"], dropout=dropout)
    model      = DeepONet(branch_net, trunk_net).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 0.0),
    )
    gamma     = cfg["scheduler"]["decay_rate"] ** (1.0 / cfg["scheduler"]["decay_steps"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    loss_fn   = nn.MSELoss()

    max_steps    = cfg["training"]["max_steps"]
    rec_val_freq = cfg["training"]["rec_validation_freq"]

    step = 0
    model.train()
    train_history: list[tuple[int, float]] = []
    val_history:   list[tuple[int, float]] = []

    while step < max_steps:
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, batch_size):
            if step >= max_steps:
                break
            idx = perm[i : i + batch_size]
            a_b, x_b, u_b = a_train[idx], x_train[idx], u_train[idx]

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(a_b, x_b), u_b)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            train_history.append((step, loss.item()))

            if step % rec_val_freq == 0 or step == 1:
                model.eval()
                with torch.no_grad():
                    val_losses = [
                        loss_fn(
                            model(a_test[k * 100 : (k + 1) * 100],
                                  x_test[k * 100 : (k + 1) * 100]),
                            u_test[k * 100 : (k + 1) * 100],
                        ).item()
                        for k in range(10)
                    ]
                val_mean = float(np.mean(val_losses))
                val_history.append((step, val_mean))
                print(f"step {step:>6}/{max_steps}  train={loss.item():.4e}  val={val_mean:.4e}")
                model.train()

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    checkpoint_path = output_dir / "deeponet.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg": cfg,
        },
        checkpoint_path,
    )

    model.eval()
    with torch.no_grad():
        u_pred = model(a_test, x_test).cpu().numpy().ravel()
    u_true = u_test.cpu().numpy().ravel()

    train_steps, train_losses = zip(*train_history)
    val_steps,   val_losses   = zip(*val_history)
    np.savez(
        output_dir / "history.npz",
        train_steps=train_steps,
        train_losses=train_losses,
        val_steps=val_steps,
        val_losses=val_losses,
        u_pred=u_pred,
        u_true=u_true,
    )
    print(f"Training complete. Model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()