using TOML
using HDF5
using Printf
using StaticArrays
using Trixi
using OrdinaryDiffEq

include("solver.jl")

function generate_fom_data()
    config_path = joinpath(@__DIR__, "config.toml")
    cfg = TOML.parsefile(config_path)

    K       = Int(cfg["fom"]["K"])
    polydeg = Int(cfg["fom"]["polydeg"])
    T       = Float64(cfg["fom"]["T"])
    cfl     = Float64(cfg["fom"]["cfl"])
    n1      = Int(cfg["fom"]["n1"])
    n2      = Int(cfg["fom"]["n2"])

    mu1_vals = range(0.5, 1.0, length=n1)
    mu2_vals = fill(pi/2, n2) 
    #mu2_vals = range(0.0, pi,  length=n2)

    n_samples = n1 * n2
    n_save    = 101  
    t_save    = range(0.0, T, length=n_save)
    Nx        = K * (polydeg + 1)

    println("="^65)
    @printf("Generating FOM Dataset (Periodic N-Wave Inviscid Burgers)\n")
    @printf("  Mesh: K=%d, p=%d (Nx=%d nodes)\n", K, polydeg, Nx)
    @printf("  Time: T=%.2f (%d uniform snapshots)\n", T, n_save)
    @printf("  Parameters: %d mu1 x %d mu2 = %d total trajectories\n", n1, n2, n_samples)
    println("="^65)

    u_data   = zeros(Float64, n_samples, n_save, Nx)
    mu1_data = zeros(Float64, n_samples)
    mu2_data = zeros(Float64, n_samples)
    x_coords = zeros(Float64, Nx)
    w_coords = zeros(Float64, Nx)  # Preallocate array for global quadrature weights

    sample_idx = 1
    for (i, m1) in enumerate(mu1_vals)
        for (j, m2) in enumerate(mu2_vals)
            @printf("Running sample [%3d/%3d]: mu1 = %.3f, mu2 = %.3f ... ",
                    sample_idx, n_samples, m1, m2)
            
            sol, semi = solve_burgers(Float64(m1), Float64(m2), K, polydeg, T, cfl)
            
            if sample_idx == 1
                _, _, solver, cache = Trixi.mesh_equations_solver_cache(semi)
                
                # 1. Extract physical GLL coordinates
                x_coords .= vec(cache.elements.node_coordinates[1, :, :])
                
                # 2. Extract and scale exact quadrature weights for [0, 1] domain
                w_ref = solver.basis.weights
                J = 1.0 / (2.0 * K) # Jacobian mapping [-1, 1] to physical element
                w_coords .= repeat(w_ref * J, K)
            end
            
            for (t_idx, t_val) in enumerate(t_save)
                u_snap = sol(t_val)
                u_data[sample_idx, t_idx, :] .= u_snap
            end
            
            mu1_data[sample_idx] = m1
            mu2_data[sample_idx] = m2
            sample_idx += 1
            println("Done ✓")
        end
    end

    data_dir = joinpath(@__DIR__, "data")
    mkpath(data_dir)
    output_file = joinpath(data_dir, "burgers_fom.h5")

    println("\nWriting dataset to: $output_file")
    h5open(output_file, "w") do h5
        h5["u"]       = u_data
        h5["t"]       = collect(t_save)
        h5["x"]       = x_coords
        h5["w"]       = w_coords      # Export global quadrature weights
        h5["mu1"]     = mu1_data
        h5["mu2"]     = mu2_data
        h5["K"]       = K
        h5["polydeg"] = polydeg
        h5["T"]       = T
        h5["cfl"]     = cfl
    end

    println("Dataset generation complete.")
end

generate_fom_data()