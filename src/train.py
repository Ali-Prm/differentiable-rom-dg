"""
Training utilities for the nonlinear-manifold Burgers ROM.

Latent-state definition
-----------------------
The learned latent state is obtained from the solution snapshot itself:

    q(t) = E_phi(U(t))

The physical parameters used to generate a trajectory are NOT q and are
never concatenated to q or passed to the encoder/decoder.

Decoder:

    U_theta(x,t) = psi_theta(x, q(t))

Physics operator:

    F(U_theta)

For the periodic Burgers problem, the Julia/Trixi DG operator depends on the
current state U_theta and discretization (K, p), not on the initial-condition
parameters. Therefore Tesseract B is called as

    F_dg = tesseract_session.flux(U_pred)

and its discrete analytical VJP is used during backpropagation when
use_tesseract_grad=True.


Data splitting
--------------
The dataset is flattened over time, but train/validation splitting is done
BY TRAJECTORY / PARAMETER REALIZATION using dataset.sample_sim_ids.

This is important: all snapshots from one FOM simulation stay in the same
split. Randomly splitting individual snapshots would leak trajectories into
validation and overstate parameter generalization.

Two training stages
-------------------
1. train_manifold_only
       reconstruction loss only.

2. train_all_losses
       reconstruction + tangent physics + manifold regularization.

No x-coordinate normalization is performed. The decoder receives the
physical GLL coordinates stored in dataset.x_coords.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from model import ManifoldROM
from losses import (
    manifold_loss,
    tangent_loss,
    reg_orthonormal,
    orth_trace_loss,
    orth_cond_loss,
    temporal_loss,
)
from dataset import BurgersSnapshotDataset


# ============================================================================
# Basic helpers
# ============================================================================

def _batch_to(
    batch: dict,
    device: str | torch.device,
) -> dict:
    """Move tensor-valued batch entries to the requested device."""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _validate_model_dataset(
    model: ManifoldROM,
    dataset: BurgersSnapshotDataset,
) -> None:
    """
    Check that the current model and FOM mesh are compatible.

    The current CNN encoder is fixed-grid, so its expected node count must
    equal dataset.Nx.
    """
    if not hasattr(model, "n_nodes"):
        raise AttributeError(
            "The current ManifoldROM must expose `n_nodes` because the "
            "location-aware CNN encoder is fixed to the FOM node count."
        )

    if int(model.n_nodes) != int(dataset.Nx):
        raise ValueError(
            f"Model expects n_nodes={model.n_nodes}, but dataset has "
            f"Nx={dataset.Nx}. Recreate the model for this FOM mesh."
        )

    if not hasattr(model, "q_dim"):
        raise AttributeError("Model must expose `q_dim`.")


def _get_simulation_ids(
    dataset: BurgersSnapshotDataset,
) -> np.ndarray:
    """
    Return one trajectory ID for every flattened snapshot.

    The revised dataset exposes this as `sample_sim_ids`.
    """
    if not hasattr(dataset, "sample_sim_ids"):
        raise AttributeError(
            "Dataset must provide `sample_sim_ids`. "
            "The training split must be performed by trajectory, not by "
            "individual snapshots."
        )

    sim_ids = np.asarray(dataset.sample_sim_ids)

    if sim_ids.ndim != 1:
        raise ValueError(
            f"sample_sim_ids must be 1-D, got shape {sim_ids.shape}."
        )

    if len(sim_ids) != len(dataset):
        raise ValueError(
            f"sample_sim_ids has length {len(sim_ids)}, but dataset has "
            f"{len(dataset)} samples."
        )

    return sim_ids


def _make_grouped_loaders(
    dataset: BurgersSnapshotDataset,
    train_frac: float,
    batch_size: int,
    seed: int = 42,
):
    """
    Split the dataset by trajectory.

    Every snapshot from a given parameter realization remains entirely in
    train or validation.
    """
    if not (0.0 < train_frac < 1.0):
        raise ValueError(
            f"train_frac must be in (0,1), got {train_frac}."
        )

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    sim_ids = _get_simulation_ids(dataset)
    unique_sim_ids = np.unique(sim_ids)

    if len(unique_sim_ids) < 2:
        raise ValueError(
            "Need at least two distinct trajectories for a train/validation "
            "split."
        )

    rng = np.random.default_rng(seed)
    shuffled = unique_sim_ids.copy()
    rng.shuffle(shuffled)

    n_train_sims = int(round(train_frac * len(shuffled)))
    n_train_sims = min(
        max(n_train_sims, 1),
        len(shuffled) - 1,
    )

    train_sim_ids = set(
        shuffled[:n_train_sims].tolist()
    )

    train_indices = [
        i
        for i, sim_id in enumerate(sim_ids)
        if sim_id in train_sim_ids
    ]

    val_indices = [
        i
        for i, sim_id in enumerate(sim_ids)
        if sim_id not in train_sim_ids
    ]

    if not train_indices or not val_indices:
        raise RuntimeError(
            "Grouped split produced an empty train or validation set."
        )

    train_ds = Subset(
        dataset,
        train_indices,
    )

    val_ds = Subset(
        dataset,
        val_indices,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return (
        train_loader,
        val_loader,
        len(train_ds),
        len(val_ds),
        len(train_sim_ids),
        len(unique_sim_ids) - len(train_sim_ids),
    )


def _call_dg_operator(
    tesseract_session,
    U: torch.Tensor,
    params: torch.Tensor
) -> torch.Tensor:
    """
    Evaluate the Julia/Trixi DG spatial operator.
    """
    # Extract the mu1 and mu2 parameters from the batch metadata
    mu1 = params[:, 0].contiguous()
    mu2 = params[:, 1].contiguous()
    
    F = tesseract_session.flux(U.cpu(), mu1.cpu(), mu2.cpu())

    if not isinstance(F, torch.Tensor):
        F = torch.as_tensor(F)

    return F


def _save_checkpoint(
    save_dir,
    epoch,
    model,
    optimizer,
    history,
    dataset,
    tag="checkpoint.pt",
):
    """Save model, optimizer, training history, and mesh metadata."""
    os.makedirs(save_dir, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "meta": {
                "q_dim": int(model.q_dim),
                "n_nodes": int(dataset.Nx),
                "K": int(dataset.K),
                "polydeg": int(dataset.polydeg),
                "n_parameters": int(
                    getattr(dataset, "n_parameters", 0)
                ),
            },
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        os.path.join(save_dir, tag),
    )


def load_checkpoint(
    path: str,
    model: ManifoldROM,
    optimizer: Optional[optim.Optimizer] = None,
):
    """Load a saved checkpoint into model and optionally optimizer."""
    ckpt = torch.load(
        path,
        map_location="cpu",
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            ckpt["optimizer_state_dict"]
        )

    return (
        ckpt["epoch"],
        ckpt.get("history", {}),
    )


# ============================================================================
# Gram diagnostics
# ============================================================================

def gram_diagnostics(
    G: torch.Tensor,
    eps: float = 1e-8,
) -> dict:
    """
    Diagnostics for

        G = J^T W J.

    Parameters
    ----------
    G:
        Tensor of shape (B,q_dim,q_dim).
    """
    if G.ndim != 3:
        raise ValueError(
            f"G must have shape (B,q,q), got {tuple(G.shape)}."
        )

    with torch.no_grad():

        ev = torch.linalg.eigvalsh(G)

        min_eig = (
            ev[:, 0]
            .clamp(min=0)
            .mean()
            .item()
        )

        max_eig = (
            ev[:, -1]
            .mean()
            .item()
        )

        cond = (
            ev[:, -1]
            /
            ev[:, 0].clamp(min=eps)
        ).mean().item()

        ev_pos = ev.clamp(min=0)

        ev_norm = (
            ev_pos
            /
            ev_pos.sum(
                dim=1,
                keepdim=True,
            ).clamp(min=eps)
        )

        eff_rank = torch.exp(
            -(
                ev_norm
                *
                torch.log(
                    ev_norm.clamp(min=1e-10)
                )
            ).sum(dim=1)
        ).mean().item()

    return {
        "min_eig": min_eig,
        "max_eig": max_eig,
        "cond": cond,
        "eff_rank": eff_rank,
    }


def _format_rank_status(
    diag: dict,
    q_dim: int,
) -> str:
    er = diag["eff_rank"]

    frac = er / max(q_dim, 1)

    n_filled = min(
        max(int(frac * 10), 0),
        10,
    )

    bar = (
        "█" * n_filled
        +
        "░" * (10 - n_filled)
    )

    return (
        f"eff_rank={er:.1f}/{q_dim} "
        f"[{bar}] cond={diag['cond']:.1e}"
    )


# ============================================================================
# Stage 1 — reconstruction pretraining
# ============================================================================

def train_manifold_only(
    model: ManifoldROM,
    dataset: BurgersSnapshotDataset,
    tesseract_session=None,
    n_epochs: int = 500,
    lr_model: float = 1e-3,
    batch_size: int = 256,
    train_frac: float = 0.85,
    normalize: bool = False,
    grad_clip: float = 1.0,
    use_scheduler: bool = True,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 20,
    scheduler_min_lr: float = 1e-6,
    seed: int = 42,
    device: str = "cpu",
    log_every: int = 10,
    save_dir: str | None = "checkpoints",
    save_every: int = 100,
) -> tuple[ManifoldROM, dict]:
    """
    Stage 1: train encoder + decoder using reconstruction loss only.

        U -> q = Encoder(U) -> U_pred = Decoder(x,q)

    The parameter vector is never supplied to the model.
    """

    _validate_model_dataset(
        model,
        dataset,
    )

    model.to(device)

    # Physical GLL coordinates are used directly.
    x_grid = dataset.x_coords.to(device)
    quad_weights = dataset.quad_weights.to(device)

    (
        train_loader,
        val_loader,
        n_train,
        n_val,
        n_train_sims,
        n_val_sims,
    ) = _make_grouped_loaders(
        dataset,
        train_frac,
        batch_size,
        seed,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr_model,
    )

    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr,
        )
        if use_scheduler
        else None
    )

    history: dict[str, list] = {
        "train_man": [],
        "val_man": [],
        "val_tan": [],
        "lr": [],
    }

    print("=" * 75)
    print("Stage 1: Manifold-only training")
    print(
        f"  snapshots   : {len(dataset):,} "
        f"(train {n_train:,}, val {n_val:,})"
    )
    print(
        f"  trajectories: "
        f"(train {n_train_sims}, val {n_val_sims})"
    )
    print(f"  q_dim       : {model.q_dim}")
    print(
        f"  Nx          : {dataset.Nx} "
        f"(K={dataset.K}, p={dataset.polydeg})"
    )
    print("  q           : Encoder(U)")
    print("  parameters  : metadata only")
    print("-" * 75)

    for epoch in range(1, n_epochs + 1):

        model.train()

        train_sum = 0.0

        for batch in train_loader:

            batch = _batch_to(
                batch,
                device,
            )

            B = batch["U"].shape[0]

            q = model.encode(
                batch["U"]
            )

            U_pred = model.decode(
                x_grid,
                q,
            )

            L_man = manifold_loss(
                U_pred,
                batch["U"],
                quad_weights,
                normalize=normalize,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            L_man.backward()

            if grad_clip:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

            optimizer.step()

            train_sum += L_man.item() * B

        train_man = (
            train_sum /
            max(n_train, 1)
        )

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------
        model.eval()

        val_sum_man = 0.0
        val_sum_tan = 0.0

        val_g_sum = dict(
            min_eig=0.0,
            max_eig=0.0,
            cond=0.0,
            eff_rank=0.0,
        )

        val_batches = 0

        for batch in val_loader:

            batch = _batch_to(
                batch,
                device,
            )

            B = batch["U"].shape[0]

            # Jacobian requires autograd.
            with torch.enable_grad():

                q_v = model.encode(
                    batch["U"]
                )

                U_v, J_v = model.decoder_jacobian(
                    x_grid,
                    q_v,
                )

            U_v = U_v.detach()
            J_v = J_v.detach()

            # No parameter vector enters the decoder or DG operator.
            with torch.no_grad():

                L_m = manifold_loss(
                    U_v,
                    batch["U"],
                    quad_weights,
                    normalize=normalize,
                )

                val_sum_man += L_m.item() * B

                if tesseract_session is not None:

                    F_dg = _call_dg_operator(
                        tesseract_session,
                        U_v,
                        batch["params"]
                    ).to(device)

                    L_t, G_v, _ = tangent_loss(
                        J_v,
                        F_dg,
                        quad_weights,
                    )

                    val_sum_tan += L_t.item() * B

                    diag = gram_diagnostics(
                        G_v
                    )

                    for key in val_g_sum:
                        val_g_sum[key] += diag[key]

                    val_batches += 1

        val_man = (
            val_sum_man / n_val
            if n_val > 0
            else float("nan")
        )

        val_tan = (
            val_sum_tan / n_val
            if tesseract_session is not None and n_val > 0
            else float("nan")
        )

        for key in val_g_sum:
            val_g_sum[key] /= max(
                val_batches,
                1,
            )

        if (
            scheduler is not None
            and np.isfinite(val_man)
        ):
            scheduler.step(val_man)

        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_man"].append(
            train_man
        )
        history["val_man"].append(
            val_man
        )
        history["val_tan"].append(
            val_tan
        )
        history["lr"].append(
            cur_lr
        )

        if (
            epoch % log_every == 0
            or epoch == 1
        ):
            tan_str = (
                f"{val_tan:.4e}"
                if np.isfinite(val_tan)
                else "n/a"
            )

            print(
                f"Ep {epoch:5d}/{n_epochs}  "
                f"man: {train_man:.4e}/{val_man:.4e}  "
                f"val_tan: {tan_str}  "
                f"lr: {cur_lr:.2e}"
            )

            if val_batches > 0:
                print(
                    f"{'':>12}  "
                    f"val-Gram: "
                    f"{_format_rank_status(val_g_sum, model.q_dim)}"
                )

        if save_dir and (
            epoch % save_every == 0
            or epoch == n_epochs
        ):
            _save_checkpoint(
                save_dir,
                epoch,
                model,
                optimizer,
                history,
                dataset,
                tag=f"stage1_ep{epoch:05d}.pt",
            )

    # --------------------------------------------------------------
    # Training curve
    # --------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ep_ax = range(
        1,
        n_epochs + 1,
    )

    ax.semilogy(
        ep_ax,
        history["train_man"],
        label="train",
    )

    ax.semilogy(
        ep_ax,
        history["val_man"],
        label="val",
        linestyle="--",
    )

    if not all(
        np.isnan(history["val_tan"])
    ):
        ax.semilogy(
            ep_ax,
            history["val_tan"],
            label="val tangent",
            linestyle=":",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Stage 1: Manifold-only")
    ax.legend()
    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )

    plt.tight_layout()
    plt.show()

    return model, history


# ============================================================================
# Stage 2 — physics-informed manifold training
# ============================================================================

def train_all_losses(
    model: ManifoldROM,
    dataset: BurgersSnapshotDataset,
    tesseract_session,
    lambda_schedulers: dict | None = None,
    tangent_param_sch: Callable | None = None,
    use_tesseract_grad: bool = True,
    n_epochs: int = 1000,
    lr_model: float = 1e-3,
    orth_reg_type: str | None = "trace",
    tangent_solver: str = "tikhonov",
    eps_reg: float = 1e-8,
    normalize_man: bool = False,
    normalize_tan: bool = True,
    grad_clip: float = 1.0,
    use_scheduler: bool = True,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 20,
    scheduler_min_lr: float = 1e-6,
    batch_size: int = 256,
    train_frac: float = 0.85,
    seed: int = 42,
    device: str = "cpu",
    log_every: int = 10,
    save_dir: str | None = "checkpoints",
    save_every: int = 100,
) -> tuple[ManifoldROM, dict]:
    """
    Stage 2 physics-informed training.

        q = Encoder(U)

        U_theta = Decoder(x,q)

        F_dg = F(U_theta)

        L = λ_man L_man
          + λ_tan L_tan
          + λ_reg L_reg
          + λ_orth L_orth

    The trajectory parameter vector is NOT used anywhere in this chain.

    When use_tesseract_grad=True, the tangent loss backpropagates through
    F_dg using the analytical discrete VJP implemented in Julia.
    """

    _validate_model_dataset(
        model,
        dataset,
    )

    if lambda_schedulers is None:
        lambda_schedulers = {
            "man": lambda ep: 1.0,
            "tan": lambda ep: 1.0,
            "reg": lambda ep: 1e-3,
            "orth": lambda ep: 1e-3,
        }

    # This is a numerical regularization parameter for the tangent
    # least-squares solve. It is NOT a physical Burgers parameter.
    if tangent_param_sch is None:
        tangent_param_sch = lambda ep: 1e-6

    if orth_reg_type not in (
        "trace",
        "cond",
        None,
    ):
        raise ValueError(
            "orth_reg_type must be 'trace', 'cond', or None."
        )

    model.to(device)

    x_grid = dataset.x_coords.to(device)
    quad_weights = dataset.quad_weights.to(device)

    (
        train_loader,
        val_loader,
        n_train,
        n_val,
        n_train_sims,
        n_val_sims,
    ) = _make_grouped_loaders(
        dataset,
        train_frac,
        batch_size,
        seed,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr_model,
    )

    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr,
        )
        if use_scheduler
        else None
    )

    history: dict[str, list] = {
        "train_man": [],
        "train_tan": [],
        "train_reg": [],
        "train_orth": [],
        "train_total": [],
        "val_man": [],
        "val_tan": [],
        "val_reg": [],
        "val_orth": [],
        "val_total": [],
        "eff_lam_man": [],
        "eff_lam_tan": [],
        "eff_lam_reg": [],
        "eff_lam_orth": [],
        "eff_tan_param": [],
        "lr": [],
        "tr_gram_min_eig": [],
        "tr_gram_max_eig": [],
        "tr_gram_cond": [],
        "tr_gram_eff_rank": [],
        "val_gram_min_eig": [],
        "val_gram_max_eig": [],
        "val_gram_cond": [],
        "val_gram_eff_rank": [],
    }

    print("=" * 80)
    print("Stage 2: Physics-informed nonlinear-manifold training")
    print(
        f"  solver       : {tangent_solver}"
        f"  | orth_reg={orth_reg_type}"
    )
    print(
        f"  snapshots    : {len(dataset):,} "
        f"(train {n_train:,}, val {n_val:,})"
    )
    print(
        f"  trajectories : "
        f"(train {n_train_sims}, val {n_val_sims})"
    )
    print(f"  q_dim        : {model.q_dim}")
    print("  q            : Encoder(U)")
    print("  decoder      : psi(x,q)")
    print("  x coordinates: physical GLL coordinates")
    print("  parameters   : metadata only")
    print(
        "  DG VJP       : "
        + ("ON" if use_tesseract_grad else "OFF")
    )
    print("=" * 80)

    for epoch in range(1, n_epochs + 1):

        # --------------------------------------------------------------
        # Loss weights / tangent solver parameter
        # --------------------------------------------------------------
        eff_lm = lambda_schedulers.get(
            "man",
            lambda e: 1.0,
        )(epoch)

        eff_lt = lambda_schedulers.get(
            "tan",
            lambda e: 1.0,
        )(epoch)

        eff_lr = lambda_schedulers.get(
            "reg",
            lambda e: 1e-3,
        )(epoch)

        eff_lo = (
            lambda_schedulers.get(
                "orth",
                lambda e: 1e-3,
            )(epoch)
            if orth_reg_type is not None
            else 0.0
        )

        eff_tp = tangent_param_sch(
            epoch
        )

        history["eff_lam_man"].append(
            eff_lm
        )
        history["eff_lam_tan"].append(
            eff_lt
        )
        history["eff_lam_reg"].append(
            eff_lr
        )
        history["eff_lam_orth"].append(
            eff_lo
        )
        history["eff_tan_param"].append(
            eff_tp
        )

        # --------------------------------------------------------------
        # Training
        # --------------------------------------------------------------
        model.train()

        tr = dict(
            man=0.0,
            tan=0.0,
            reg=0.0,
            orth=0.0,
            total=0.0,
        )

        tr_g_sum = dict(
            min_eig=0.0,
            max_eig=0.0,
            cond=0.0,
            eff_rank=0.0,
        )

        n_tr_batches = 0

        for batch in train_loader:

            batch = _batch_to(
                batch,
                device,
            )

            B = batch["U"].shape[0]

            # ----------------------------------------------------------
            # U -> q -> U_pred
            #
            # Physical parameters are deliberately ignored here.
            # ----------------------------------------------------------
            q = model.encode(
                batch["U"]
            )

            U_pred, J = model.decoder_jacobian(
                x_grid,
                q,
            )

            # ----------------------------------------------------------
            # Tesseract B: F(U_pred)
            # ----------------------------------------------------------
            F_dg = _call_dg_operator(
                        tesseract_session,
                        U_pred,
                        batch["params"]
                    ).to(device)

            if not use_tesseract_grad:
                F_dg = F_dg.detach()

            # ----------------------------------------------------------
            # Losses
            # ----------------------------------------------------------
            L_man = manifold_loss(
                U_pred,
                batch["U"],
                quad_weights,
                normalize=normalize_man,
            )

            L_tan, G, v_star = tangent_loss(
                J,
                F_dg,
                quad_weights,
                solver=tangent_solver,
                solver_param=eff_tp,
                normalize=normalize_tan,
            )

            L_reg = reg_orthonormal(
                G,
                eps_reg=eps_reg,
            )

            if orth_reg_type == "trace":
                L_orth = orth_trace_loss(
                    G
                )
            elif orth_reg_type == "cond":
                L_orth = orth_cond_loss(
                    G
                )
            else:
                L_orth = torch.zeros(
                    (),
                    device=device,
                    dtype=L_man.dtype,
                )

            L_tot = (
                eff_lm * L_man
                + eff_lt * L_tan
                + eff_lr * L_reg
                + eff_lo * L_orth
            )

            # ----------------------------------------------------------
            # Backpropagation
            # ----------------------------------------------------------
            optimizer.zero_grad(
                set_to_none=True
            )

            L_tot.backward()

            if grad_clip:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

            optimizer.step()

            # ----------------------------------------------------------
            # Metrics
            # ----------------------------------------------------------
            tr["man"] += L_man.item() * B
            tr["tan"] += L_tan.item() * B
            tr["reg"] += L_reg.item() * B
            tr["orth"] += L_orth.item() * B
            tr["total"] += L_tot.item() * B

            with torch.no_grad():
                diag = gram_diagnostics(
                    G,
                    eps=eps_reg,
                )

            for key in tr_g_sum:
                tr_g_sum[key] += diag[key]

            n_tr_batches += 1

        for key in tr:
            tr[key] /= max(
                n_train,
                1,
            )

        for key in tr_g_sum:
            tr_g_sum[key] /= max(
                n_tr_batches,
                1,
            )

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------
        model.eval()

        va = dict(
            man=0.0,
            tan=0.0,
            reg=0.0,
            orth=0.0,
            total=0.0,
        )

        val_g_sum = dict(
            min_eig=0.0,
            max_eig=0.0,
            cond=0.0,
            eff_rank=0.0,
        )

        val_batches = 0

        for batch in val_loader:

            batch = _batch_to(
                batch,
                device,
            )

            B = batch["U"].shape[0]

            # We need autograd to construct J, even though validation itself
            # is not backpropagated.
            with torch.enable_grad():

                q_v = model.encode(
                    batch["U"]
                )

                U_v, J_v = model.decoder_jacobian(
                    x_grid,
                    q_v,
                )

            U_v = U_v.detach()
            J_v = J_v.detach()

            # Everything below is evaluation only.
            with torch.no_grad():

                F_dg_v = _call_dg_operator(
                        tesseract_session,
                        U_v,
                        batch["params"]
                    ).to(device)

                F_dg_v = F_dg_v.detach()

                L_man_v = manifold_loss(
                    U_v,
                    batch["U"],
                    quad_weights,
                    normalize=normalize_man,
                )

                L_tan_v, G_v, _ = tangent_loss(
                    J_v,
                    F_dg_v,
                    quad_weights,
                    solver=tangent_solver,
                    solver_param=eff_tp,
                    normalize=normalize_tan,
                )

                L_reg_v = reg_orthonormal(
                    G_v,
                    eps_reg=eps_reg,
                )

                if orth_reg_type == "trace":
                    L_orth_v = orth_trace_loss(
                        G_v
                    )
                elif orth_reg_type == "cond":
                    L_orth_v = orth_cond_loss(
                        G_v
                    )
                else:
                    L_orth_v = torch.zeros(
                        (),
                        device=device,
                        dtype=L_man_v.dtype,
                    )

                L_tot_v = (
                    eff_lm * L_man_v
                    + eff_lt * L_tan_v
                    + eff_lr * L_reg_v
                    + eff_lo * L_orth_v
                )

                va["man"] += L_man_v.item() * B
                va["tan"] += L_tan_v.item() * B
                va["reg"] += L_reg_v.item() * B
                va["orth"] += L_orth_v.item() * B
                va["total"] += L_tot_v.item() * B

                diag_v = gram_diagnostics(
                    G_v,
                    eps=eps_reg,
                )

                for key in val_g_sum:
                    val_g_sum[key] += diag_v[key]

                val_batches += 1

        if n_val > 0:
            for key in va:
                va[key] /= n_val

            for key in val_g_sum:
                val_g_sum[key] /= max(
                    val_batches,
                    1,
                )

        sched_val = (
            va["total"]
            if n_val > 0
            and np.isfinite(va["total"])
            else tr["total"]
        )

        if (
            scheduler is not None
            and np.isfinite(sched_val)
        ):
            scheduler.step(
                sched_val
            )

        cur_lr = optimizer.param_groups[0]["lr"]

        # --------------------------------------------------------------
        # History
        # --------------------------------------------------------------
        for key in (
            "man",
            "tan",
            "reg",
            "orth",
            "total",
        ):
            history[
                f"train_{key}"
            ].append(
                tr[key]
            )

            history[
                f"val_{key}"
            ].append(
                va[key]
            )

        history["lr"].append(
            cur_lr
        )

        for prefix, diag in (
            ("tr", tr_g_sum),
            ("val", val_g_sum),
        ):
            for key in diag:
                history[
                    f"{prefix}_gram_{key}"
                ].append(
                    diag[key]
                )

        # --------------------------------------------------------------
        # Checkpoint
        # --------------------------------------------------------------
        if save_dir and (
            epoch % save_every == 0
            or epoch == n_epochs
        ):
            _save_checkpoint(
                save_dir,
                epoch,
                model,
                optimizer,
                history,
                dataset,
                tag=f"stage2_ep{epoch:05d}.pt",
            )

        # --------------------------------------------------------------
        # Logging
        # --------------------------------------------------------------
        if (
            epoch % log_every == 0
            or epoch == 1
        ):
            print(
                f"Ep {epoch:5d}/{n_epochs} | "
                f"λm={eff_lm:.2e} "
                f"λt={eff_lt:.2e} "
                f"λr={eff_lr:.2e}"
                + (
                    f" λo={eff_lo:.2e}"
                    if orth_reg_type is not None
                    else ""
                )
                + (
                    f" tan_reg={eff_tp:.1e}"
                    f" lr={cur_lr:.2e}"
                )
            )

            print(
                f"{'':>12}  "
                f"man: {tr['man']:.3e}/{va['man']:.3e}   "
                f"tan: {tr['tan']:.3e}/{va['tan']:.3e}   "
                f"reg: {tr['reg']:.3e}/{va['reg']:.3e}"
            )

            if orth_reg_type is not None:
                print(
                    f"{'':>12}  "
                    f"orth: {tr['orth']:.3e}/{va['orth']:.3e}"
                )

            print(
                f"{'':>12}  "
                f"total: {tr['total']:.3e}/{va['total']:.3e}"
            )

            print(
                f"{'':>12}  "
                f"tr-Gram: "
                f"{_format_rank_status(tr_g_sum, model.q_dim)}"
            )

            print(
                f"{'':>12}  "
                f"val-Gram: "
                f"{_format_rank_status(val_g_sum, model.q_dim)}"
            )

            print("-" * 80)

    # --------------------------------------------------------------
    # Training curves
    # --------------------------------------------------------------
    components = [
        ("man", "$L_{man}$"),
        ("tan", "$L_{tan}$"),
        ("reg", "$L_{reg}$"),
        ("total", "Total"),
    ]

    if orth_reg_type is not None:
        components.insert(
            3,
            ("orth", "$L_{orth}$"),
        )

    ncols = (
        3
        if len(components) > 3
        else 2
    )

    nrows = -(
        -len(components) // ncols
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(13, 4 * nrows),
        squeeze=False,
    )

    axes_flat = axes.flat

    ep_ax = range(
        1,
        n_epochs + 1,
    )

    for ax, (key, title) in zip(
        axes_flat,
        components,
    ):
        ax.semilogy(
            ep_ax,
            history[f"train_{key}"],
            label="train",
        )

        ax.semilogy(
            ep_ax,
            history[f"val_{key}"],
            label="val",
            linestyle="--",
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(
            True,
            which="both",
            alpha=0.3,
        )

    for ax in list(axes_flat)[len(components):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()

    return model, history



def _format_rank_status(
    diag: dict,
    q_dim: int,
) -> str:
    er = diag["eff_rank"]

    frac = er / max(q_dim, 1)

    n_filled = min(
        max(int(frac * 10), 0),
        10,
    )

    bar = (
        "█" * n_filled
        +
        "░" * (10 - n_filled)
    )

    # Added minλ and maxλ to the returned string format
    return (
        f"eff_rank={er:.1f}/{q_dim} "
        f"[{bar}] cond={diag['cond']:.1e} "
        f"minλ={diag['min_eig']:.2e} maxλ={diag['max_eig']:.2e}"
    )



def train_all_losses_joint(
    model: ManifoldROM,
    dataset: BurgersSnapshotDataset,
    tesseract_session,
    lambda_schedulers: dict | None = None,
    tangent_param_sch: Callable | None = None,
    use_tesseract_grad: bool = True,
    n_epochs: int = 1000,
    ep_joint_training: int = 300,   # <--- Phase transition epoch
    lr_model: float = 1e-4,         # <--- Theta learning rate
    lr_temporal: float = 1e-3,      # <--- Phi learning rate
    orth_reg_type: str | None = "trace",
    tangent_solver: str = "tikhonov",
    eps_reg: float = 1e-8,
    normalize_man: bool = False,
    normalize_tan: bool = True,
    grad_clip: float = 1.0,
    use_scheduler: bool = True,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 20,
    scheduler_min_lr: float = 1e-6,
    batch_size: int = 256,
    train_frac: float = 0.85,
    seed: int = 42,
    device: str = "cpu",
    log_every: int = 10,
    save_dir: str | None = "checkpoints",
    save_every: int = 100,
) -> tuple[ManifoldROM, dict]:
    """
    Stage 2 physics-informed training with Temporal Network (Phi-ROM).
    """
    _validate_model_dataset(model, dataset)

    if lambda_schedulers is None:
        lambda_schedulers = {
            "man": lambda ep: 1.0,
            "tan": lambda ep: 1.0,
            "reg": lambda ep: 1e-3,
            "orth": lambda ep: 1e-3,
            "temp": lambda ep: 1.0,
        }

    if tangent_param_sch is None:
        tangent_param_sch = lambda ep: 1e-6

    if orth_reg_type not in ("trace", "cond", None):
        raise ValueError("orth_reg_type must be 'trace', 'cond', or None.")

    model.to(device)

    x_grid = dataset.x_coords.to(device)
    quad_weights = dataset.quad_weights.to(device)

    train_loader, val_loader, n_train, n_val, n_train_sims, n_val_sims = _make_grouped_loaders(
        dataset, train_frac, batch_size, seed
    )

    # ------------------------------------------------------------------
    # TWO OPTIMIZERS: Theta (Manifold) and Phi (Temporal Dynamics)
    # ------------------------------------------------------------------
    optimizer_theta = optim.Adam([
        {'params': model.encoder.parameters()},
        {'params': model.decoder.parameters()}
    ], lr=lr_model)

    optimizer_phi = optim.Adam(
        model.temporal_net.parameters(), 
        lr=lr_temporal
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_theta, mode="min", factor=scheduler_factor, 
        patience=scheduler_patience, min_lr=scheduler_min_lr,
    ) if use_scheduler else None

    history: dict[str, list] = {
        "train_man": [], "train_tan": [], "train_reg": [], "train_orth": [], "train_temp": [], "train_total": [],
        "val_man": [], "val_tan": [], "val_reg": [], "val_orth": [], "val_temp": [], "val_total": [],
        "eff_lam_man": [], "eff_lam_tan": [], "eff_lam_reg": [], "eff_lam_orth": [], "eff_tan_param": [],
        "lr": [],
        "tr_gram_min_eig": [], "tr_gram_max_eig": [], "tr_gram_cond": [], "tr_gram_eff_rank": [],
        "val_gram_min_eig": [], "val_gram_max_eig": [], "val_gram_cond": [], "val_gram_eff_rank": [],
    }

    print("=" * 80)
    print("Stage 2: Physics-informed nonlinear-manifold & Temporal training")
    print(f"  solver       : {tangent_solver}  | orth_reg={orth_reg_type}")
    print(f"  snapshots    : {len(dataset):,} (train {n_train:,}, val {n_val:,})")
    print(f"  Phase 1 (Decoupled) : Epochs 1 to {ep_joint_training}")
    print(f"  Phase 2 (Joint)     : Epochs {ep_joint_training+1} to {n_epochs}")
    print("=" * 80)

    for epoch in range(1, n_epochs + 1):

        eff_lm = lambda_schedulers.get("man", lambda e: 1.0)(epoch)
        eff_lt = lambda_schedulers.get("tan", lambda e: 1.0)(epoch)
        eff_lr = lambda_schedulers.get("reg", lambda e: 1e-3)(epoch)
        eff_lo = lambda_schedulers.get("orth", lambda e: 1e-3)(epoch) if orth_reg_type is not None else 0.0
        eff_l_temp = lambda_schedulers.get("temp", lambda e: 1.0)(epoch)
        eff_tp = tangent_param_sch(epoch)

        history["eff_lam_man"].append(eff_lm)
        history["eff_lam_tan"].append(eff_lt)
        history["eff_lam_reg"].append(eff_lr)
        history["eff_lam_orth"].append(eff_lo)
        history["eff_tan_param"].append(eff_tp)

        # --------------------------------------------------------------
        # Training
        # --------------------------------------------------------------
        model.train()
        tr = dict(man=0.0, tan=0.0, reg=0.0, orth=0.0, temp=0.0, total=0.0)
        tr_g_sum = dict(min_eig=0.0, max_eig=0.0, cond=0.0, eff_rank=0.0)
        n_tr_batches = 0

        for batch in train_loader:
            batch = _batch_to(batch, device)
            B = batch["U"].shape[0]

            q = model.encode(batch["U"])
            U_pred, J = model.decoder_jacobian(x_grid, q)

            F_dg = _call_dg_operator(tesseract_session, U_pred, batch["params"]).to(device)
            
            if not use_tesseract_grad:
                F_dg = F_dg.detach()

            # --- Spatial Losses (Theta) ---
            L_man = manifold_loss(U_pred, batch["U"], quad_weights, normalize=normalize_man)
            L_tan, G, v_star = tangent_loss(J, F_dg, quad_weights, solver=tangent_solver, solver_param=eff_tp, normalize=normalize_tan)
            L_reg = reg_orthonormal(G, eps_reg=eps_reg)
            
            if orth_reg_type == "trace":
                L_orth = orth_trace_loss(G)
            elif orth_reg_type == "cond":
                L_orth = orth_cond_loss(G)
            else:
                L_orth = torch.zeros((), device=device)

            L_theta = (eff_lm * L_man + eff_lt * L_tan + eff_lr * L_reg + eff_lo * L_orth)

            # --- Temporal Loss (Phi) ---
            v_star_detached = v_star.detach()

            # Gradient Flow Control
            if epoch <= ep_joint_training:
                # Phase 1: Detach q so temporal loss doesn't backprop into the encoder
                q_temp_in = q.detach()
            else:
                # Phase 2: Allow temporal loss to fine-tune the encoder
                q_temp_in = q

            q_dot_pred = model.temporal_net(q_temp_in, batch["params"])
            L_temp = temporal_loss(q_dot_pred, v_star_detached)

            # --- Backpropagation ---
            optimizer_theta.zero_grad(set_to_none=True)
            optimizer_phi.zero_grad(set_to_none=True)

            if epoch <= ep_joint_training:
                L_theta.backward()
                L_temp.backward()
            else:
                L_tot = L_theta + (eff_l_temp * L_temp)
                L_tot.backward()

            if grad_clip:
                nn.utils.clip_grad_norm_(model.encoder.parameters(), grad_clip)
                nn.utils.clip_grad_norm_(model.decoder.parameters(), grad_clip)
                nn.utils.clip_grad_norm_(model.temporal_net.parameters(), grad_clip)

            optimizer_theta.step()
            optimizer_phi.step()

            # --- Metrics ---
            tr["man"] += L_man.item() * B
            tr["tan"] += L_tan.item() * B
            tr["reg"] += L_reg.item() * B
            tr["orth"] += L_orth.item() * B
            tr["temp"] += L_temp.item() * B
            tr["total"] += (L_theta.item() + L_temp.item()) * B

            with torch.no_grad():
                diag = gram_diagnostics(G, eps=eps_reg)

            for key in tr_g_sum:
                tr_g_sum[key] += diag[key]

            n_tr_batches += 1

        for key in tr: 
            tr[key] /= max(n_train, 1)
        for key in tr_g_sum: 
            tr_g_sum[key] /= max(n_tr_batches, 1)

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------
        model.eval()
        va = dict(man=0.0, tan=0.0, reg=0.0, orth=0.0, temp=0.0, total=0.0)
        val_g_sum = dict(min_eig=0.0, max_eig=0.0, cond=0.0, eff_rank=0.0)
        val_batches = 0

        for batch in val_loader:
            batch = _batch_to(batch, device)
            B = batch["U"].shape[0]

            with torch.enable_grad():
                q_v = model.encode(batch["U"])
                U_v, J_v = model.decoder_jacobian(x_grid, q_v)

            U_v, J_v = U_v.detach(), J_v.detach()

            with torch.no_grad():
                F_dg_v = _call_dg_operator(tesseract_session, U_v, batch["params"]).to(device)
                F_dg_v = F_dg_v.detach()

                L_man_v = manifold_loss(U_v, batch["U"], quad_weights, normalize=normalize_man)
                L_tan_v, G_v, v_star_v = tangent_loss(J_v, F_dg_v, quad_weights, solver=tangent_solver, solver_param=eff_tp, normalize=normalize_tan)
                L_reg_v = reg_orthonormal(G_v, eps_reg=eps_reg)
                
                if orth_reg_type == "trace":
                    L_orth_v = orth_trace_loss(G_v)
                elif orth_reg_type == "cond":
                    L_orth_v = orth_cond_loss(G_v)
                else:
                    L_orth_v = torch.zeros((), device=device)

                L_theta_v = (eff_lm * L_man_v + eff_lt * L_tan_v + eff_lr * L_reg_v + eff_lo * L_orth_v)

                q_dot_pred_v = model.temporal_net(q_v, batch["params"])
                L_temp_v = temporal_loss(q_dot_pred_v, v_star_v)

                va["man"] += L_man_v.item() * B
                va["tan"] += L_tan_v.item() * B
                va["reg"] += L_reg_v.item() * B
                va["orth"] += L_orth_v.item() * B
                va["temp"] += L_temp_v.item() * B
                va["total"] += (L_theta_v.item() + L_temp_v.item()) * B

                diag_v = gram_diagnostics(G_v, eps=eps_reg)
                for key in val_g_sum:
                    val_g_sum[key] += diag_v[key]

                val_batches += 1

        if n_val > 0:
            for key in va: 
                va[key] /= n_val
            for key in val_g_sum: 
                val_g_sum[key] /= max(val_batches, 1)

        sched_val = va["total"] if n_val > 0 and np.isfinite(va["total"]) else tr["total"]
        if scheduler is not None and np.isfinite(sched_val):
            scheduler.step(sched_val)

        cur_lr = optimizer_theta.param_groups[0]["lr"]

        for key in ("man", "tan", "reg", "orth", "temp", "total"):
            history[f"train_{key}"].append(tr[key])
            history[f"val_{key}"].append(va[key])

        history["lr"].append(cur_lr)

        for prefix, diag in (("tr", tr_g_sum), ("val", val_g_sum)):
            for key in diag:
                history[f"{prefix}_gram_{key}"].append(diag[key])

        if save_dir and (epoch % save_every == 0 or epoch == n_epochs):
            _save_checkpoint(save_dir, epoch, model, optimizer_theta, history, dataset, tag=f"joint_stage2_ep{epoch:05d}.pt")

        # --------------------------------------------------------------
        # Logging Output matching the requested format
        # --------------------------------------------------------------
        if (epoch % log_every == 0 or epoch == 1):
            print(
                f"Ep {epoch:5d}/{n_epochs} | "
                f"λm={eff_lm:.2e} λt={eff_lt:.2e} λr={eff_lr:.2e} "
                f"λo={eff_lo:.2e} λ_temp={eff_l_temp:.2e} tan_reg={eff_tp:.1e} lr={cur_lr:.2e}"
            )
            print(
                f"{'':>12}  man: {tr['man']:.3e}/{va['man']:.3e}   "
                f"tan: {tr['tan']:.3e}/{va['tan']:.3e}   "
                f"reg: {tr['reg']:.3e}/{va['reg']:.3e}"
            )
            if orth_reg_type is not None:
                print(
                    f"{'':>12}  orth: {tr['orth']:.3e}/{va['orth']:.3e}"
                )
            print(
                f"{'':>12}  temp: {tr['temp']:.3e}/{va['temp']:.3e}   "
                f"total: {tr['total']:.3e}/{va['total']:.3e}"
            )

            if n_tr_batches > 0:
                print(f"{'':>12}  tr-Gram: {_format_rank_status(tr_g_sum, model.q_dim)}")
            if val_batches > 0:
                print(f"{'':>12}  val-Gram: {_format_rank_status(val_g_sum, model.q_dim)}")

            print("-" * 80)

    # --------------------------------------------------------------
    # Training curves
    # --------------------------------------------------------------
    components = [
        ("man", "$L_{man}$"),
        ("tan", "$L_{tan}$"),
        ("reg", "$L_{reg}$"),
        ("temp", "$L_{temp}$"),
        ("total", "Total"),
    ]

    if orth_reg_type is not None:
        components.insert(3, ("orth", "$L_{orth}$"))

    ncols = 3 if len(components) > 3 else 2
    nrows = -(-len(components) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows), squeeze=False)
    axes_flat = axes.flat
    ep_ax = range(1, n_epochs + 1)

    for ax, (key, title) in zip(axes_flat, components):
        ax.semilogy(ep_ax, history[f"train_{key}"], label="train")
        ax.semilogy(ep_ax, history[f"val_{key}"], label="val", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    for ax in list(axes_flat)[len(components):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()

    return model, history