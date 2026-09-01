"""
Tesseract B Client
"""
from __future__ import annotations
import numpy as np
import torch

def _tesseract_class():
    try:
        from tesseract_core import Tesseract
        return Tesseract
    except ImportError as exc:
        raise ImportError(
            "tesseract_core is not available. Please install it."
        ) from exc

class DGFluxTesseractFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u_theta: torch.Tensor,
        mu1: torch.Tensor,
        mu2: torch.Tensor,
        tesseract,
        K: int,
        polydeg: int,
    ) -> torch.Tensor:

        ctx.save_for_backward(u_theta, mu1, mu2)
        ctx.tesseract = tesseract
        ctx.K = int(K)
        ctx.polydeg = int(polydeg)

        u_np = u_theta.detach().cpu().numpy().astype(np.float64, copy=False)
        mu1_np = mu1.detach().cpu().numpy().astype(np.float64, copy=False)
        mu2_np = mu2.detach().cpu().numpy().astype(np.float64, copy=False)

        result = tesseract.apply({
            "u_theta": u_np,
            "mu1": mu1_np,
            "mu2": mu2_np,
            "K": ctx.K,
            "polydeg": ctx.polydeg,
        })

        if "F" not in result:
            raise RuntimeError("Tesseract B /rhs output does not contain key 'F'.")

        F_np = np.asarray(result["F"], dtype=np.float64)
        return torch.from_numpy(F_np).to(dtype=u_theta.dtype, device=u_theta.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (u_theta, mu1, mu2) = ctx.saved_tensors

        u_np = u_theta.detach().cpu().numpy().astype(np.float64, copy=False)
        mu1_np = mu1.detach().cpu().numpy().astype(np.float64, copy=False)
        mu2_np = mu2.detach().cpu().numpy().astype(np.float64, copy=False)
        v_np = grad_output.detach().cpu().numpy().astype(np.float64, copy=False)

        vjp_result = ctx.tesseract.vector_jacobian_product(
            inputs={
                "u_theta": u_np,
                "mu1": mu1_np,
                "mu2": mu2_np,
                "K": ctx.K,
                "polydeg": ctx.polydeg,
            },
            vjp_inputs=["u_theta"],
            vjp_outputs=["F"],
            cotangent_vector={"F": v_np},
        )

        du_np = np.asarray(vjp_result["u_theta"], dtype=np.float64)
        du = torch.from_numpy(du_np).to(dtype=u_theta.dtype, device=u_theta.device)

        # Return gradients matching the 6 inputs to forward()
        # Gradients are only required for u_theta.
        return du, None, None, None, None, None

def compute_dg_flux(
    u_theta: torch.Tensor,
    mu1: torch.Tensor,
    mu2: torch.Tensor,
    tesseract,
    K: int,
    polydeg: int,
) -> torch.Tensor:
    return DGFluxTesseractFunction.apply(u_theta, mu1, mu2, tesseract, int(K), int(polydeg))


class TesseractBSession:
    def __init__(self, image: str = "dg_flux", K: int = 128, polydeg: int = 3):
        self.image = image
        self.K = int(K)
        self.polydeg = int(polydeg)
        self._ctx = None
        self.tess = None

    def __enter__(self):
        Tesseract = _tesseract_class()
        self._ctx = Tesseract.from_image(self.image)
        self.tess = self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._ctx is not None:
            return self._ctx.__exit__(exc_type, exc_value, traceback)
        return None

    def flux(self, u_theta: torch.Tensor, mu1: torch.Tensor, mu2: torch.Tensor) -> torch.Tensor:
        return compute_dg_flux(u_theta, mu1, mu2, self.tess, self.K, self.polydeg)