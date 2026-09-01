from __future__ import annotations
import os, sys, time, json, subprocess, atexit
import numpy as np
import requests
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Float64, Differentiable

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

_CFG = {}
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.toml")
if tomllib and os.path.exists(_CONFIG_PATH):
    with open(_CONFIG_PATH, "rb") as _f:
        _CFG = tomllib.load(_f)

def _cfg(key, default):
    parts = key.split(".")
    d = _CFG
    for p in parts[:-1]:
        d = d.get(p, {})
    return d.get(parts[-1], default)

_K_DEFAULT       = _cfg("fom.K",       64)
_POLYDEG_DEFAULT = _cfg("fom.polydeg",  2)

_JULIA_URL  = "http://127.0.0.1:8765"
_julia_proc = None

def _start_julia_server(timeout_s: int = 180) -> None:
    global _julia_proc
    julia_bin = "/usr/local/bin/julia"
    script    = "/app/burgers_server.jl"

    env = os.environ.copy()
    env.setdefault("JULIA_DEPOT_PATH", "/opt/julia_depot")
    _julia_proc = subprocess.Popen(
        [julia_bin, "-t", "auto", "--project=/app", script],
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
    )

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{_JULIA_URL}/health", timeout=2)
            if r.status_code == 200:
                print(f"[dg_flux] Julia server ready", flush=True)
                return
        except Exception:
            pass
        time.sleep(1)

    raise RuntimeError(
        f"Julia DG server did not become ready within {timeout_s}s."
    )

def _stop_julia_server() -> None:
    if _julia_proc and _julia_proc.poll() is None:
        _julia_proc.terminate()

atexit.register(_stop_julia_server)

_BUILDING = os.environ.get("_TESSERACT_IS_BUILDING") == "1"
if not _BUILDING:
    _start_julia_server()

class InputSchema(BaseModel):
    u_theta : Differentiable[Array[(None, None), Float64]] = Field(
        description="Decoded field at GLL nodes. Shape (B, Nx=K*(polydeg+1))."
    )
    mu1     : Array[(None,), Float64] = Field(
        description="Primary wave amplitude per sample. Shape (B,)."
    )
    mu2     : Array[(None,), Float64] = Field(
        description="Phase shift per sample. Shape (B,)."
    )
    K       : int = Field(default=_K_DEFAULT,       description="DG elements.")
    polydeg : int = Field(default=_POLYDEG_DEFAULT, description="Polynomial degree.")

class OutputSchema(BaseModel):
    F : Differentiable[Array[(None, None), Float64]] = Field(
        description="DG RHS F(u_theta). Shape (B, Nx)."
    )

def _matrix_to_list(arr: np.ndarray) -> list:
    return arr.tolist()

def _list_to_matrix(lst: list) -> np.ndarray:
    return np.array(lst, dtype=np.float64)

def apply(inputs: InputSchema) -> OutputSchema:
    payload = json.dumps({
        "u":       _matrix_to_list(np.asarray(inputs.u_theta, dtype=np.float64)),
        "mu1":     np.asarray(inputs.mu1, dtype=np.float64).tolist(),
        "mu2":     np.asarray(inputs.mu2, dtype=np.float64).tolist(),
        "K":       inputs.K,
        "polydeg": inputs.polydeg,
    })
    r = requests.post(f"{_JULIA_URL}/rhs", data=payload,
                      headers={"Content-Type": "application/json"})
    if not r.ok:
        raise RuntimeError(f"Julia DG server error {r.status_code} on /rhs:\n{r.text}")
    F = _list_to_matrix(r.json()["F"])
    return OutputSchema(F=F)

def vector_jacobian_product(
    inputs:           InputSchema,
    vjp_inputs:       list,
    vjp_outputs:      list,
    cotangent_vector: dict,
) -> dict:
    payload = json.dumps({
        "u":       _matrix_to_list(np.asarray(inputs.u_theta, dtype=np.float64)),
        "v":       _matrix_to_list(np.asarray(cotangent_vector["F"], dtype=np.float64)),
        "mu1":     np.asarray(inputs.mu1, dtype=np.float64).tolist(),
        "mu2":     np.asarray(inputs.mu2, dtype=np.float64).tolist(),
        "K":       inputs.K,
        "polydeg": inputs.polydeg,
    })
    r = requests.post(f"{_JULIA_URL}/vjp", data=payload,
                      headers={"Content-Type": "application/json"})
    if not r.ok:
        raise RuntimeError(f"Julia DG server error {r.status_code} on /vjp:\n{r.text}")
    du = _list_to_matrix(r.json()["du"])
    return {"u_theta": du}