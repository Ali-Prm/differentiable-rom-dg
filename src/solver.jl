using Trixi
using OrdinaryDiffEq
using OrdinaryDiffEqSSPRK
using StaticArrays

function initial_condition_nwave(x, t, equations::InviscidBurgersEquation1D, mu1::Float64, mu2::Float64)
    u = mu1 * sin(2.0 * pi * x[1]) + 0.3 * sin(4.0 * pi * x[1] + mu2)
    return SVector(u)
end

function solve_burgers(mu1::Float64, mu2::Float64, K::Int, polydeg::Int, T::Float64, cfl::Float64)
    equations = InviscidBurgersEquation1D()
    ic = (x, t, eq) -> initial_condition_nwave(x, t, eq, mu1, mu2)
    
    mesh = StructuredMesh((K,), (0.0,), (1.0,), periodicity=true)
    basis = LobattoLegendreBasis(polydeg)
    
    indicator_sc = IndicatorHennemannGassner(
        equations, basis;
        alpha_max    = 1.0,     
        alpha_min    = 1e-5,
        alpha_smooth = true,   
        variable     = (u, eqs) -> u[1]
    )
    
    volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                     volume_flux_dg=flux_ec,
                                                     volume_flux_fv=flux_lax_friedrichs)
    solver = DGSEM(basis, flux_lax_friedrichs, volume_integral)
    
    semi = SemidiscretizationHyperbolic(
        mesh, equations, ic, solver;
        boundary_conditions = boundary_condition_periodic
    )
    
    ode = semidiscretize(semi, (0.0, T))
    
    # 1. Properly enforce CFL-based step size control
    callbacks = CallbackSet(StepsizeCallback(cfl=cfl))
    
    # 2. Fully qualify the solver and let the CFL callback dictate adaptive dt
    sol = solve(ode, SSPRK33(), dt=1e-2, adaptive=false, 
                save_everystep=true, callback=callbacks)
    
    return sol, semi
end