"""
Loss functions for the Nonlinear Manifold ROM.

Overview of the three losses
─────────────────────────────
  L_man  — Manifold loss:  ||u_θ(·, q) - u_FOM||²_L2
           Drives the decoder to reconstruct FOM snapshots accurately.

  L_tan  — Tangent (physics) loss:
           Residual of the Galerkin projection of the PDE onto the manifold:
             J_ψ(q) dq/dt = F(u_θ(·, q))          ← tangent equation
           where J_ψ = ∂u_θ/∂q  (Tesseract A, PyTorch jacfwd)
           and   F   = DG flux   (Tesseract B, Julia/Trixi adjoint)

           Solved as a weighted least-squares problem (4 regularisation options).
           Loss = residual: ||F - J_ψ v*||²_W / ||F||²_W

  L_reg  — Regularization on the Gram matrix G = J_ψ^T W J_ψ:
           Encourages the manifold tangent directions to be well-separated
           (orthogonality and good conditioning).

All L2 norms and inner products use the GLL quadrature weights (quad_weights),
which are the exact integration weights from the DG discretization.
This ensures the losses are consistent with the tangent equation formulation.

Tesseract A / B boundary
──────────────────────────
  Tesseract A  (this file, Python):  J_ψ, all loss values, LS solve
  Tesseract B  (Julia, Trixi.jl):    F(u_θ)  — imported via src/tesseract_b.py
"""

from __future__ import annotations
import torch


# ============================================================
# Quadrature-weighted L2 norm helpers
# ============================================================

def _weighted_sq_norm(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    ||v||²_w = sum_i w_i v_i²   (per batch sample)
    v : (B, Nx)
    w : (Nx,)
    Returns : (B,)
    """
    return (w[None, :] * v.pow(2)).sum(dim=1).clamp(min=0.0)


# ============================================================
# 1. Manifold loss
# ============================================================

def manifold_loss(U_pred: torch.Tensor,
                  U_true: torch.Tensor,
                  quad_weights: torch.Tensor,
                  normalize: bool = False,
                  eps: float = 1e-16) -> torch.Tensor:
    """
    DG-quadrature-weighted L2 reconstruction error.

      L_man = mean_b  sqrt( ||u_θ(·,q_b) - u_b||²_w )           (absolute)
      L_man = mean_b  sqrt( ||u_θ-u_b||²_w / ||u_b||²_w )       (relative)

    Parameters
    ----------
    U_pred       : (B, Nx)  — decoder output
    U_true       : (B, Nx)  — FOM snapshots
    quad_weights : (Nx,)    — GLL quadrature weights w_i (sum ≈ domain length)
    normalize    : bool     — use relative error
    eps          : float    — floor to avoid sqrt(0)
    """
    diff = U_pred - U_true
    num  = torch.sqrt(_weighted_sq_norm(diff, quad_weights) + eps)   # (B,)

    if normalize:
        den = torch.sqrt(_weighted_sq_norm(U_true, quad_weights) + eps)
        return (num / den).mean()

    return num.mean()


# ============================================================
# 2. Tangent (physics) loss
# ============================================================

def _solve_ls(J: torch.Tensor,
              F: torch.Tensor,
              W: torch.Tensor,
              solver: str,
              solver_param: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Solve the Galerkin projection of the tangent equation:

      v* = argmin_{v ∈ R^{q_dim}}  ||W^{1/2} (J v - F)||²_2

    via normal equations:
      G v* = b
      G = J^T W J   ∈ R^{B × q_dim × q_dim}   (Gram matrix)
      b = J^T W F   ∈ R^{B × q_dim}

    All four variants are fully differentiable — PyTorch's autograd
    propagates through torch.linalg.solve / qr / lstsq into J and F,
    giving gradients wrt θ (via J) and φ (via the encoder → q path).

    Parameters
    ----------
    J            : (B, Nx, q_dim)  — decoder Jacobian ∂u_θ/∂q
    F            : (B, Nx)         — DG flux (from Tesseract B)
    W            : (Nx,)           — quadrature weights
    solver       : 'tikhonov' | 'adaptive_tikhonov' | 'qr' | 'svd'
    solver_param : regularisation parameter (λ for Tikhonov, rcond for svd)

    Returns
    -------
    v_star : (B, q_dim)           — optimal latent velocity
    G      : (B, q_dim, q_dim)   — Gram matrix (passed to reg losses)
    """
    B, Nx, q_dim = J.shape
    I_q = torch.eye(q_dim, device=J.device, dtype=J.dtype)

    # Weighted Jacobian and Gram matrix
    WJ = W[None, :, None] * J                            # (B, Nx, q_dim)
    G  = torch.bmm(J.transpose(1, 2), WJ)                # (B, q_dim, q_dim)
    b  = torch.einsum('bnq,bn->bq', WJ, F)               # (B, q_dim)

    if solver == 'tikhonov':
        # (G + λ I) v* = b
        # λ is a fixed regularisation strength.
        A = G + solver_param * I_q[None]
        v_star = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)

    elif solver == 'adaptive_tikhonov':
        # λ_b = λ₀ × mean_diag(G_b)
        # Scales the regularisation with the local eigenvalue magnitude,
        # making it invariant to the norm of J.
        lam = solver_param * G.diagonal(dim1=-2, dim2=-1).mean(dim=-1).detach()
        A   = G + lam.view(B, 1, 1) * I_q[None]
        v_star = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)

    elif solver == 'qr':
        # Thin QR of the pre-weighted system  A_w = W^{1/2} J:
        #   v* = R^{-1} Q^T (W^{1/2} F)
        # More numerically stable than normal equations for ill-conditioned J.
        sqrtW  = W.sqrt()                                 # (Nx,)
        A_w    = sqrtW[None, :, None] * J                # (B, Nx, q_dim)
        b_w    = sqrtW[None, :] * F                      # (B, Nx)
        Q, R   = torch.linalg.qr(A_w)                   # thin: Q(B,Nx,q), R(B,q,q)
        Qtb    = torch.bmm(Q.transpose(1, 2),
                           b_w.unsqueeze(-1))             # (B, q, 1)
        R_reg  = R + 1e-12 * I_q[None]                  # mild diagonal shift for rank-def
        v_star = torch.linalg.solve(R_reg, Qtb).squeeze(-1)

    elif solver == 'svd':
        # Minimum-norm least-squares via SVD, using rcond as the truncation threshold.
        # Handles rank-deficient J most robustly but is slowest.
        sqrtW  = W.sqrt()
        A_w    = sqrtW[None, :, None] * J                # (B, Nx, q_dim)
        b_w    = (sqrtW[None, :] * F).unsqueeze(-1)      # (B, Nx, 1)
        sol    = torch.linalg.lstsq(A_w, b_w, rcond=solver_param)
        v_star = sol.solution.squeeze(-1)                # (B, q_dim)

    else:
        raise ValueError(
            f"Unknown solver '{solver}'. "
            "Choose from: 'tikhonov', 'adaptive_tikhonov', 'qr', 'svd'."
        )

    return v_star, G


def tangent_loss(J: torch.Tensor,
                 F_dg: torch.Tensor,
                 quad_weights: torch.Tensor,
                 solver: str = 'tikhonov',
                 solver_param: float = 1e-6,
                 normalize: bool = True,
                 eps: float = 1e-16) -> tuple[torch.Tensor,
                                              torch.Tensor,
                                              torch.Tensor]:
    """
    Physics (tangent) loss — residual of the Galerkin-projected PDE.

    Tesseract A provides:  J = ∂u_θ/∂q  (via decoder_jacobian)
    Tesseract B provides:  F_dg = F(u_θ) (via DG spatial operator)

    Mathematical structure
    ──────────────────────
    Tangent equation (ODE on the manifold):
      J_ψ(q) dq/dt = F(u_θ(·, q))

    Galerkin / weighted least-squares projection:
      v* = argmin_v  ||J v - F||²_W        (W = diag(quad_weights))
      solved via normal eqs:  (J^T W J) v* = J^T W F

    Loss (normalised residual, dimensionless):
      r     = F - J v*
      L_tan = mean_b sqrt( ||r_b||²_W / ||F_b||²_W )

    Gradient flow
    ─────────────
    ∂L/∂θ flows through:
      1. J  (∂²u_θ/∂θ∂q — mixed Hessian, computed by jacfwd autograd)
      2. u_θ → F_dg (∂F/∂u · ∂u_θ/∂θ — via Tesseract B adjoint in Julia)
    ∂L/∂φ flows through q → u_θ → {J, F_dg}

    Parameters
    ----------
    J            : (B, Nx, q_dim)  — decoder Jacobian  [Tesseract A]
    F_dg         : (B, Nx)         — DG flux            [Tesseract B]
    quad_weights : (Nx,)           — GLL quadrature weights
    solver       : str             — LS solver variant
    solver_param : float           — regularisation parameter
    normalize    : bool            — use relative (dimensionless) residual
    eps          : float           — numerical floor

    Returns
    -------
    loss   : scalar
    G      : (B, q_dim, q_dim)  — Gram matrix (used by reg losses)
    v_star : (B, q_dim)         — solved latent velocities
    """
    W = quad_weights  # (Nx,)

    v_star, G = _solve_ls(J, F_dg, W, solver, solver_param)

    # Residual  r = F - J v*
    r    = F_dg - torch.einsum('bnq,bq->bn', J, v_star)    # (B, Nx)
    sq_r = _weighted_sq_norm(r, W)                           # (B,)

    if normalize:
        sq_F = _weighted_sq_norm(F_dg, W)                   # (B,)
        loss = torch.sqrt((sq_r + eps) / (sq_F + eps)).mean()
    else:
        loss = torch.sqrt(sq_r + eps).mean()

    return loss, G, v_star


# ============================================================
# 3. Regularisation / orthogonality losses on G = J^T W J
# ============================================================

def reg_orthonormal(G: torch.Tensor, eps_reg: float = 1e-8) -> torch.Tensor:
    """
    Spectral regularisation: L = -mean log(clamp(λ_i, eps, 1)).

    Interpretation: penalises near-zero eigenvalues of G, which correspond
    to near-degenerate directions in the tangent space.  Encourages the
    manifold basis vectors J[:, :, i] to span independent directions.

    G : (B, q_dim, q_dim)  — Gram matrix
    """
    eigvals = torch.linalg.eigvalsh(G)                         # (B, q_dim)
    return -torch.mean(torch.log(eigvals.clamp(min=eps_reg, max=1.0)))


def orth_trace_loss(G: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Correlation matrix loss: L = ||C - I||²_F  where C_ij = G_ij / sqrt(G_ii G_jj).

    Interpretation: off-diagonal entries of C measure cosine similarity
    between tangent directions.  L = 0 iff all directions are orthogonal.

    G : (B, q_dim, q_dim)
    """
    diag     = G.diagonal(dim1=-2, dim2=-1)                    # (B, q_dim)
    std      = torch.sqrt(diag.clamp(min=eps))                  # (B, q_dim)
    norm_mat = std.unsqueeze(-1) * std.unsqueeze(-2)            # (B, q, q)
    C        = G / norm_mat.clamp(min=eps)                      # (B, q, q)
    I        = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype).unsqueeze(0)
    return ((C - I).pow(2)).sum(dim=(-2, -1)).mean()


def orth_cond_loss(G: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Condition number loss: L = mean log(λ_max / λ_min).

    Interpretation: a large condition number means the manifold is stretched
    — some directions are much more "active" than others.  Encourages
    isotropic tangent structure.

    G : (B, q_dim, q_dim)
    """
    eigvals = torch.linalg.eigvalsh(G)                         # (B, q_dim)
    lam_min = eigvals[:, 0].clamp(min=eps)
    lam_max = eigvals[:, -1]
    return torch.log(lam_max / lam_min).mean()



# ============================================================
# 4. Temporal Dynamics Loss
# ============================================================

def temporal_loss(q_dot_pred: torch.Tensor, v_star: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
    """
    Relative L2 error between predicted latent dynamics and Galerkin projection.
    
    NOTE: v_star MUST be detached before being passed into this function during training 
    to prevent the manifold from collapsing to artificially lower this loss.
    """
    # L2 error of the residual (numerator)
    diff_sq = torch.sum((q_dot_pred - v_star) ** 2, dim=1)
    num = torch.sqrt(diff_sq + eps)
    
    # L2 norm of the target (denominator) - detached to prevent gradient scaling distortion
    v_sq = torch.sum(v_star ** 2, dim=1)
    den = torch.sqrt(v_sq + eps).detach()
    
    return (num / den).mean()