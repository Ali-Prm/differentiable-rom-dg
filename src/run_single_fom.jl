
    using HDF5
    using Trixi
    using OrdinaryDiffEq
    using OrdinaryDiffEqSSPRK
    using StaticArrays

    include("solver.jl")

    sol, semi = solve_burgers(0.55, 0.25, 100, 3, 0.5, 0.5)

    t_save = collect(0.0:0.01:0.5)
    Nx = 100 * (3 + 1)
    u_data = zeros(Float64, length(t_save), Nx)

    for (i, t_val) in enumerate(t_save)
        u_data[i, :] .= sol(t_val)
    end

    h5open("temp_online_fom.h5", "w") do h5
        h5["u"] = u_data
        h5["t"] = t_save
    end
    