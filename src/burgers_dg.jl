using Trixi
using LinearAlgebra
using StaticArrays
using TrixiBase

TrixiBase.disable_debug_timings()

# -----------------------------------------------------------------------------
# Exact semidiscretization from the user's solver.jl
# -----------------------------------------------------------------------------

const _SEMI_CACHE = Dict{Tuple{Float64, Float64, Int, Int}, Any}()

function get_or_create_semi(mu1::Float64, mu2::Float64,
                            K::Int, polydeg::Int)

    key = (mu1, mu2, K, polydeg)
    haskey(_SEMI_CACHE, key) && return _SEMI_CACHE[key]

    equations = InviscidBurgersEquation1D()

    ic_dummy(x, t, eq) = SVector(
        mu1 * sin(2.0 * pi * x[1]) +
        0.3 * sin(4.0 * pi * x[1] + mu2)
    )

    mesh = StructuredMesh((K,), (0.0,), (1.0,), periodicity=true)
    basis = LobattoLegendreBasis(polydeg)

    indicator_sc = IndicatorHennemannGassner(
        equations, basis;
        alpha_max    = 1.0,
        alpha_min    = 0.05,
        alpha_smooth = true,
        variable     = (u, eqs) -> u[1]
    )

    volume_integral = VolumeIntegralShockCapturingHG(
        indicator_sc;
        volume_flux_dg = flux_ec,
        volume_flux_fv = flux_lax_friedrichs
    )

    solver = DGSEM(
        basis,
        flux_lax_friedrichs,
        volume_integral
    )

    semi = SemidiscretizationHyperbolic(
        mesh,
        equations,
        ic_dummy,
        solver;
        boundary_conditions = boundary_condition_periodic
    )

    _SEMI_CACHE[key] = semi
    return semi
end

# -----------------------------------------------------------------------------
# Scalar Burgers fluxes used by the exact Trixi discretization
# -----------------------------------------------------------------------------

@inline burgers_physical_flux(u::Float64) = 0.5 * u * u

# Tadmor entropy-conservative two-point flux
@inline burgers_ec_flux(uL::Float64, uR::Float64) =
    (uL*uL + uL*uR + uR*uR) / 6.0

# Trixi's default LLF/Rusanov speed for Burgers:
# lambda = max(abs(uL), abs(uR))
@inline burgers_llf_flux(uL::Float64, uR::Float64) =
    0.25 * (uL*uL + uR*uR) -
    0.5 * max(abs(uL), abs(uR)) * (uR - uL)

# Exact piecewise derivative of the LLF flux.
#
# At abs(uL) == abs(uR), max(abs(uL),abs(uR)) is nondifferentiable.
# The VJP is therefore only defined away from such switching points.
@inline function burgers_llf_flux_derivatives(uL::Float64, uR::Float64)
    aL = abs(uL)
    aR = abs(uR)

    if aL > aR
        lam = aL
        dlam_dL = sign(uL)

        df_dL = 0.5*uL + 0.5*lam +
                0.5*(uL - uR)*dlam_dL
        df_dR = 0.5*uR - 0.5*lam

    elseif aR > aL
        lam = aR
        dlam_dR = sign(uR)

        df_dL = 0.5*uL + 0.5*lam
        df_dR = 0.5*uR - 0.5*lam -
                0.5*(uR - uL)*dlam_dR

    else
        # This branch is nondifferentiable. It is only a fallback so the
        # code remains defined; the test avoids these points.
        lam = aL
        df_dL = 0.5*uL + 0.5*lam
        df_dR = 0.5*uR - 0.5*lam
    end

    return df_dL, df_dR
end

# -----------------------------------------------------------------------------
# Hennemann-Gassner indicator, exactly matching the documented Trixi formula
# for the 1D scalar variable u.
#
# Important:
#   - alpha is computed element-by-element from Legendre modal coefficients.
#   - alpha_min clipping is differentiated as a constant branch.
#   - alpha_max clipping is differentiated as a constant branch.
#   - alpha_smooth=true uses max(alpha_e, 0.5*alpha_left, 0.5*alpha_right).
#     Away from ties, the gradient comes only from the winning branch.
#
# Trixi documentation:
# alpha_raw = 1 / (1 + exp(-s/T * (E-T)))
# E = max(E1,E2)
# E1 = m_N^2 / sum_{j=0}^N m_j^2
# E2 = m_{N-1}^2 / sum_{j=0}^{N-1} m_j^2
# -----------------------------------------------------------------------------

function hg_alphas_and_local_gradients(
    u::Matrix{Float64},
    polydeg::Int;
    alpha_max::Float64 = 1.0,
    alpha_min::Float64 = 0.05,
    alpha_smooth::Bool = true
)
    Np, K = size(u)

    # Trixi's basis stores the inverse Vandermonde matrix used to convert
    # nodal indicator values to Legendre modal coefficients.
    basis = LobattoLegendreBasis(polydeg)
    Vinv = basis.inverse_vandermonde_legendre

    threshold =
        0.5 * 10.0^(-1.8 * (polydeg + 1)^0.25)

    parameter_s =
        log((1.0 - 0.0001) / 0.0001)

    alpha_element = zeros(Float64, K)
    alpha_grad_local = zeros(Float64, K, Np)

    for e in 1:K

        ue = view(u, :, e)

        # Indicator variable is u itself.
        m = Vinv * ue

        total1 = sum(abs2, m)
        total2 = sum(abs2, view(m, 1:Np-1))

        if total1 == 0.0
            E1 = 0.0
        else
            E1 = m[Np]^2 / total1
        end

        if total2 == 0.0
            E2 = 0.0
        else
            E2 = m[Np-1]^2 / total2
        end

        # Gradient dE/dm for the active max branch
        dE_dm = zeros(Float64, Np)

        if E1 > E2
            E = E1

            if total1 != 0.0
                a = m[Np]
                dE_dm[Np] = 2.0*a/total1
                dE_dm .-= 2.0*a*a/total1^2 .* m
            end
        else
            # This includes the exact tie as a fallback. The test avoids ties.
            E = E2

            if total2 != 0.0
                a = m[Np-1]
                dE_dm[Np-1] = 2.0*a/total2
                for j in 1:(Np-1)
                    dE_dm[j] -= 2.0*a*a/total2^2 * m[j]
                end
            end
        end

        z =
            -(parameter_s / threshold) *
            (E - threshold)

        # Numerically stable enough for the small polynomial degrees used here.
        a_raw = 1.0 / (1.0 + exp(z))

        # d sigmoid / dE
        da_dE =
            (parameter_s / threshold) *
            a_raw * (1.0 - a_raw)

        # Convert dE/dm -> d(alpha_raw)/du.
        grad_raw =
            Vinv' * (da_dE .* dE_dm)

        # Trixi clipping logic
        if a_raw < alpha_min
            a = 0.0
            grad = zeros(Float64, Np)

        elseif a_raw > 1.0 - alpha_min
            a = 1.0
            grad = zeros(Float64, Np)

        elseif a_raw > alpha_max
            a = alpha_max
            grad = zeros(Float64, Np)

        else
            a = a_raw
            grad = grad_raw
        end

        alpha_element[e] = a
        alpha_grad_local[e, :] .= grad
    end

    if !alpha_smooth
        return alpha_element, alpha_grad_local
    end

    # Trixi copies alpha first, then applies the neighbor max.
    alpha_tmp = copy(alpha_element)
    alpha_smoothed = copy(alpha_element)

    # Return also the source element and multiplicative factor for the
    # derivative of each smoothed alpha.
    source_element = Vector{Int}(undef, K)
    source_factor = Vector{Float64}(undef, K)

    for e in 1:K
        eL = (e == 1) ? K : e - 1
        eR = (e == K) ? 1 : e + 1

        self_val = alpha_tmp[e]
        left_val = 0.5 * alpha_tmp[eL]
        right_val = 0.5 * alpha_tmp[eR]

        if self_val >= left_val && self_val >= right_val
            alpha_smoothed[e] = self_val
            source_element[e] = e
            source_factor[e] = 1.0

        elseif left_val >= right_val
            alpha_smoothed[e] = left_val
            source_element[e] = eL
            source_factor[e] = 0.5

        else
            alpha_smoothed[e] = right_val
            source_element[e] = eR
            source_factor[e] = 0.5
        end
    end

    # The local gradients returned here are gradients of the *smoothed*
    # alpha_e with respect to all u. The representation is K x K x Np in
    # principle. To avoid storing that tensor, we encode the winner and
    # recover the contribution later in eval_rhs_vjp.
    #
    # Here we return the unsmoothed local gradients plus the winner maps.
    return alpha_smoothed,
           alpha_grad_local,
           source_element,
           source_factor
end

# -----------------------------------------------------------------------------
# Element volume operators.
#
# These reproduce the two volume pieces used by
# VolumeIntegralShockCapturingHG:
#
#   R_DG = entropy-conservative split-form DG volume contribution
#          + physical flux boundary contribution
#
#   R_FV = first-order LGL subcell FV volume contribution
#          + physical flux boundary contribution
#
# The common numerical surface flux is applied separately below.
# -----------------------------------------------------------------------------

function element_volume_operators(
    ue::AbstractVector{Float64},
    D::AbstractMatrix{Float64},
    w::AbstractVector{Float64},
    J::Float64
)
    Np = length(ue)

    rd = zeros(Float64, Np)
    rf = zeros(Float64, Np)

    # ---- High-order DG flux-differencing volume term ----
    #
    # R_DG,vol(i) = -(2/J) sum_j D[i,j] f_ec(u_i,u_j)
    #
    for i in 1:Np
        s = 0.0
        ui = ue[i]

        for j in 1:Np
            s += D[i,j] * burgers_ec_flux(ui, ue[j])
        end

        rd[i] -= (2.0 / J) * s
    end

    # Strong-form physical flux contribution corresponding to the weak
    # surface integral convention used by DGSEM.
    rd[1]  -= burgers_physical_flux(ue[1])  / (J*w[1])
    rd[Np] += burgers_physical_flux(ue[Np]) / (J*w[Np])

    # ---- Low-order LGL subcell FV volume term ----
    #
    # The LGL nodal weights act as the subcell volumes.
    #
    # Left boundary subcell:
    #   -(f_1,2 - f(u_1))/(J w_1)
    #
    # Interior subcell i:
    #   +(f_{i-1,i} - f_{i,i+1})/(J w_i)
    #
    # Right boundary subcell:
    #   -(f(u_N) - f_{N-1,N})/(J w_N)
    #
    rf[1]  -= burgers_physical_flux(ue[1])  / (J*w[1])
    rf[Np] += burgers_physical_flux(ue[Np]) / (J*w[Np])

    for i in 1:(Np-1)
        fstar = burgers_llf_flux(ue[i], ue[i+1])

        rf[i]   -= fstar / (J*w[i])
        rf[i+1] += fstar / (J*w[i+1])
    end

    return rd, rf
end

# -----------------------------------------------------------------------------
# Forward clone of the spatial RHS represented by the user's solver.jl.
#
# This is used ONLY as a consistency check. The actual reference RHS in all
# verification tests is still Trixi.rhs_hyperbolic!.
# -----------------------------------------------------------------------------

function eval_rhs_clone(
    u_flat::Vector{Float64},
    mu1::Float64,
    mu2::Float64,
    K::Int,
    polydeg::Int
)
    semi = get_or_create_semi(mu1, mu2, K, polydeg)
    _, _, solver, _ = Trixi.mesh_equations_solver_cache(semi)

    Np = polydeg + 1
    D  = solver.basis.derivative_matrix
    w  = solver.basis.weights
    J  = 0.5 / K

    u = reshape(u_flat, Np, K)
    rhs = zeros(Float64, Np, K)

    hg = hg_alphas_and_local_gradients(
        u, polydeg;
        alpha_max = 1.0,
        alpha_min = 0.05,
        alpha_smooth = true
    )
    alpha = hg[1]

    # Volume terms
    for e in 1:K
        rd, rf = element_volume_operators(
            view(u, :, e), D, w, J
        )

        rhs[:,e] .+= (1.0 - alpha[e]) .* rd
        rhs[:,e] .+= alpha[e] .* rf
    end

    # Common numerical surface flux.
    #
    # At interface (e,right) <-> (e_next,left):
    #
    # right endpoint contribution: -f*/(J w_N)
    # left endpoint contribution:  +f*/(J w_1)
    #
    for e in 1:K
        e_next = (e == K) ? 1 : e + 1

        uL = u[Np,e]
        uR = u[1,e_next]

        fstar = burgers_llf_flux(uL, uR)

        rhs[Np,e]       -= fstar / (J*w[Np])
        rhs[1,e_next]   += fstar / (J*w[1])
    end

    return vec(rhs)
end

# -----------------------------------------------------------------------------
# Exact discrete VJP of the same operator.
#
# For R(u) = (1-alpha) R_DG + alpha R_FV + R_surface,
#
# dR^T v =
#   (1-alpha) J_DG^T v
# + alpha J_FV^T v
# + [v^T (R_FV-R_DG)] d(alpha)
# + J_surface^T v.
#
# The HG alpha gradient is included exactly away from the max/clipping
# switching surfaces where the piecewise definition is nondifferentiable.
# -----------------------------------------------------------------------------

function eval_rhs_vjp(
    u_flat::Vector{Float64},
    v_flat::Vector{Float64},
    mu1::Float64,
    mu2::Float64,
    K::Int,
    polydeg::Int
)
    semi = get_or_create_semi(mu1, mu2, K, polydeg)
    _, _, solver, _ = Trixi.mesh_equations_solver_cache(semi)

    Np = polydeg + 1
    D  = solver.basis.derivative_matrix
    w  = solver.basis.weights
    J  = 0.5 / K

    u = reshape(u_flat, Np, K)
    v = reshape(v_flat, Np, K)

    du = zeros(Float64, Np, K)

    hg = hg_alphas_and_local_gradients(
        u, polydeg;
        alpha_max = 1.0,
        alpha_min = 0.05,
        alpha_smooth = true
    )

    alpha = hg[1]
    alpha_grad_local = hg[2]
    source_element = hg[3]
    source_factor = hg[4]

    # g_alpha[e] = d (v_e^T R_e) / d alpha_e
    #             = v_e^T (R_FV,e - R_DG,e)
    #
    # This term is essential because the HG sensor itself depends on u.
    g_alpha = zeros(Float64, K)

    # ------------------------------------------------------------------
    # Element-local blended volume VJP
    # ------------------------------------------------------------------
    for e in 1:K

        ue = view(u, :, e)
        ve = view(v, :, e)

        rd, rf = element_volume_operators(ue, D, w, J)

        # Sensitivity of the objective with respect to alpha_e.
        g_alpha[e] = dot(ve, rf - rd)

        gDG = zeros(Float64, Np)
        gFV = zeros(Float64, Np)

        # ==============================================================
        # DG split-form volume VJP
        #
        # R_DG,vol(l) = -(2/J) sum_j D[l,j] f_ec(u_l,u_j)
        #
        # df_ec/du_l = (2*u_l + u_j)/6
        # df_ec/du_j = (u_i + 2*u_j)/6
        # ==============================================================
        for l in 1:Np

            ul = ue[l]

            term_first = 0.0
            term_second = 0.0

            # Contributions where u_l is the first argument.
            for j in 1:Np
                df_dL =
                    (2.0*ul + ue[j]) / 6.0

                term_first += D[l,j] * df_dL
            end

            # Contributions where u_l is the second argument.
            for i in 1:Np
                df_dR =
                    (ue[i] + 2.0*ul) / 6.0

                term_second +=
                    ve[i] * D[i,l] * df_dR
            end

            gDG[l] -=
                (2.0/J) *
                (
                    ve[l] * term_first +
                    term_second
                )
        end

        # DG physical-flux boundary terms.
        gDG[1] -=
            ve[1] * ue[1] / (J*w[1])

        gDG[Np] +=
            ve[Np] * ue[Np] / (J*w[Np])

        # ==============================================================
        # LGL subcell FV volume VJP
        # ==============================================================
        for i in 1:(Np-1)

            uL = ue[i]
            uR = ue[i+1]

            df_dL, df_dR =
                burgers_llf_flux_derivatives(uL, uR)

            c =
                -ve[i]   / (J*w[i]) +
                 ve[i+1] / (J*w[i+1])

            gFV[i]   += c * df_dL
            gFV[i+1] += c * df_dR
        end

        # FV physical-flux boundary terms.
        gFV[1] -=
            ve[1] * ue[1] / (J*w[1])

        gFV[Np] +=
            ve[Np] * ue[Np] / (J*w[Np])

        # Volume-integral shock capturing blends ONLY these volume terms.
        du[:,e] .+=
            (1.0 - alpha[e]) .* gDG +
            alpha[e] .* gFV
    end

    # ------------------------------------------------------------------
    # HG sensor VJP
    #
    # alpha_final[e] is the max of:
    #   alpha_tmp[e],
    #   0.5 alpha_tmp[left],
    #   0.5 alpha_tmp[right].
    #
    # Away from ties, only the winning source contributes.
    # ------------------------------------------------------------------
    for e in 1:K
        src = source_element[e]
        fac = source_factor[e]

        du[:,src] .+=
            (fac * g_alpha[e]) .* view(alpha_grad_local, src, :)
    end

    # ------------------------------------------------------------------
    # Common periodic numerical surface flux.
    #
    # For interface L=(Np,e), R=(1,e_next):
    #
    #   R[L] += -f*/(J*w[Np])
    #   R[R] += +f*/(J*w[1])
    #
    # Hence the coefficient multiplying df* is
    #
    #   -v_L/(J*w[Np]) + v_R/(J*w[1]).
    #
    # This surface operator is common to DG and FV in
    # VolumeIntegralShockCapturingHG.
    # ------------------------------------------------------------------
    for e in 1:K

        e_next = (e == K) ? 1 : e + 1

        uL = u[Np,e]
        uR = u[1,e_next]

        df_dL, df_dR =
            burgers_llf_flux_derivatives(uL, uR)

        c =
            -v[Np,e]     / (J*w[Np]) +
             v[1,e_next] / (J*w[1])

        du[Np,e]       += c * df_dL
        du[1,e_next]   += c * df_dR
    end

    return vec(du)
end

# -----------------------------------------------------------------------------
# Batch helper
# -----------------------------------------------------------------------------

# function eval_rhs_batch(
#     u_batch::Matrix{Float64},
#     mu1_batch::Vector{Float64},
#     mu2_batch::Vector{Float64},
#     K::Int,
#     polydeg::Int
# )
#     B, Nx = size(u_batch)
#     F = zeros(Float64, B, Nx)

#     for b in 1:B
#         F[b,:] .= eval_rhs_clone(
#             vec(u_batch[b,:]),
#             mu1_batch[b],
#             mu2_batch[b],
#             K,
#             polydeg
#         )
#     end

#     return F
# end

function eval_rhs_batch(
    u_batch::Matrix{Float64},
    mu1_batch::Vector{Float64},
    mu2_batch::Vector{Float64},
    K::Int,
    polydeg::Int
)
    B, Nx = size(u_batch)
    F = zeros(Float64, B, Nx)

    for b in 1:B
        semi = get_or_create_semi(mu1_batch[b], mu2_batch[b], K, polydeg)
        
        # Use Trixi's highly optimized native cache for the forward pass
        u_vec = vec(u_batch[b, :])
        F_vec = zeros(Float64, Nx)
        Trixi.rhs_hyperbolic!(F_vec, u_vec, semi, 0.0)
        
        F[b, :] .= F_vec
    end

    return F
end

function eval_rhs_vjp_batch(
    u_batch::Matrix{Float64},
    v_batch::Matrix{Float64},
    mu1_batch::Vector{Float64},
    mu2_batch::Vector{Float64},
    K::Int,
    polydeg::Int
)
    B, Nx = size(u_batch)
    dU = zeros(Float64, B, Nx)

    for b in 1:B
        dU[b,:] .= eval_rhs_vjp(
            vec(u_batch[b,:]),
            vec(v_batch[b,:]),
            mu1_batch[b],
            mu2_batch[b],
            K,
            polydeg
        )
    end

    return dU
end
