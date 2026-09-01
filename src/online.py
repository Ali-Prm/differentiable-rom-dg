import os
import subprocess
import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def run_online_stage(
    model: nn.Module, 
    mu1: float, 
    mu2: float, 
    x_coords: torch.Tensor, 
    quad_weights: torch.Tensor, 
    dt: float = 0.01, 
    n_epoch: int = 200, 
    lr: float = 0.01,
    T: float = 0.5,
    K: int = 64,
    polydeg: int = 3,
    cfl: float = 0.5,
    device: str = 'cpu'
):
    model.eval()
    model.to(device)
    x_coords = x_coords.to(device)
    quad_weights = quad_weights.to(device)
    
    # =========================================================================
    # 1. Generate Ground-Truth FOM Solution via solver.jl
    # =========================================================================
    print(f"➔ Generating FOM solution for μ1={mu1}, μ2={mu2}...")
    
    jl_script = f"""
    using HDF5
    using Trixi
    using OrdinaryDiffEq
    using OrdinaryDiffEqSSPRK
    using StaticArrays

    include("solver.jl")

    sol, semi = solve_burgers({mu1}, {mu2}, {K}, {polydeg}, {T}, {cfl})

    t_save = collect(0.0:{dt}:{T})
    Nx = {K} * ({polydeg} + 1)
    u_data = zeros(Float64, length(t_save), Nx)

    for (i, t_val) in enumerate(t_save)
        u_data[i, :] .= sol(t_val)
    end

    h5open("temp_online_fom.h5", "w") do h5
        h5["u"] = u_data
        h5["t"] = t_save
    end
    """
    
    script_path = "run_single_fom.jl"
    with open(script_path, "w") as f:
        f.write(jl_script)
        
    subprocess.run(["julia", "--project=.", script_path], check=True)
    
    # Load the generated FOM data
    with h5py.File("temp_online_fom.h5", "r") as f:
        # ➔ THE FIX IS HERE: Add .T to transpose from (Nx, N_t) back to (N_t, Nx)
        U_fom_np = f["u"][:].T  
        t_grid_np = f["t"][:]
        
    os.remove(script_path)
    os.remove("temp_online_fom.h5")
    
    U_fom = torch.tensor(U_fom_np, dtype=torch.float32, device=device) # Shape: (N_t, Nx)
    
    # =========================================================================
    # 2. Decoder Projection (Optimize Initial Guess q_0)
    # =========================================================================
    print("➔ Optimizing initial latent state q_0...")
    
    # Target initial condition (1, Nx)
    U_0 = U_fom[0].unsqueeze(0)
    
    with torch.no_grad():
        q_guess = model.encode(U_0)
        
    q_opt = nn.Parameter(q_guess.clone())
    optimizer = torch.optim.Adam([q_opt], lr=lr)
    
    for epoch in range(n_epoch):
        optimizer.zero_grad()
        U_pred = model.decode(x_coords, q_opt)
        loss = (quad_weights * (U_pred - U_0)**2).sum()
        loss.backward()
        optimizer.step()
        
    q_0 = q_opt.detach()
    print(f"   Final q_0 Projection Loss: {loss.item():.4e}")
    
    # =========================================================================
    # 3. Latent Time Integration (Runge-Kutta 4)
    # =========================================================================
    print("➔ Integrating latent dynamics via RK4...")
    params_tensor = torch.tensor([[mu1, mu2]], dtype=torch.float32, device=device)
    
    q_traj = [q_0]
    q_t = q_0
    
    N_steps = len(t_grid_np) - 1
    
    with torch.no_grad():
        for _ in range(N_steps):
            k1 = model.temporal_net(q_t, params_tensor)
            k2 = model.temporal_net(q_t + 0.5 * dt * k1, params_tensor)
            k3 = model.temporal_net(q_t + 0.5 * dt * k2, params_tensor)
            k4 = model.temporal_net(q_t + dt * k3, params_tensor)
            
            q_t = q_t + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            q_traj.append(q_t)
            
    q_traj_tensor = torch.cat(q_traj, dim=0) # Shape: (N_t, q_dim)
    
    # =========================================================================
    # 4. Decode Trajectory & Compute Errors
    # =========================================================================
    print("➔ Decoding trajectory and computing errors...")
    with torch.no_grad():
        U_approx = model.decode(x_coords, q_traj_tensor) # Shape: (N_t, Nx)
        
        errors = []
        for i in range(len(t_grid_np)):
            diff_sq = quad_weights * (U_approx[i] - U_fom[i])**2
            fom_sq = quad_weights * (U_fom[i])**2
            
            rel_err = torch.sqrt(diff_sq.sum()) / torch.sqrt(fom_sq.sum())
            errors.append(rel_err.item())
            
    print(f"   Mean Relative L2 Error over T: {np.mean(errors):.4e}")
    print(f"   Max Relative L2 Error over T:  {np.max(errors):.4e}")
    
    return t_grid_np, q_traj_tensor, U_approx, U_fom, errors