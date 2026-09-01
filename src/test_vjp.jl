using Trixi
using LinearAlgebra
using Printf
using Random

include("burgers_dg.jl")

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

function trixi_rhs(
    u::Vector{Float64},
    mu1::Float64,
    mu2::Float64,
    K::Int,
    p::Int
)
    semi = get_or_create_semi(mu1, mu2, K, p)
    F = zeros(Float64, length(u))
    Trixi.rhs_hyperbolic!(F, u, semi, 0.0)
    return F
end

function fd_directional_reference(
    u::Vector{Float64},
    v::Vector{Float64},
    d::Vector{Float64},
    mu1::Float64,
    mu2::Float64,
    K::Int,
    p::Int;
    h::Float64 = 1e-6
)
    up = u .+ h .* d
    um = u .- h .* d

    Fp = trixi_rhs(up, mu1, mu2, K, p)
    Fm = trixi_rhs(um, mu1, mu2, K, p)

    return dot(v, (Fp - Fm) ./ (2.0*h))
end

function coordinate_fd_vjp(
    u::Vector{Float64},
    v::Vector{Float64},
    mu1::Float64,
    mu2::Float64,
    K::Int,
    p::Int;
    h::Float64 = 1e-6
)
    n = length(u)
    g = zeros(Float64, n)

    for i in 1:n
        up = copy(u)
        um = copy(u)

        up[i] += h
        um[i] -= h

        Fp = trixi_rhs(up, mu1, mu2, K, p)
        Fm = trixi_rhs(um, mu1, mu2, K, p)

        g[i] = dot(v, (Fp - Fm) ./ (2.0*h))
    end

    return g
end

function check_alpha(u::Vector{Float64}, K::Int, p::Int)
    Np = p + 1
    U = reshape(u, Np, K)

    hg = hg_alphas_and_local_gradients(
        U, p;
        alpha_max = 1.0,
        alpha_min = 0.05,
        alpha_smooth = true
    )

    alpha = hg[1]

    return minimum(alpha), maximum(alpha),
           count(a -> 0.0 < a < 1.0, alpha)
end

# -----------------------------------------------------------------------------
# 1. First validate the forward clone itself.
#
# This is critical: if this fails, do NOT trust the VJP. It means the
# hand-written forward decomposition is not yet identical to Trixi's
# rhs_hyperbolic!.
# -----------------------------------------------------------------------------

function forward_consistency_test(
    u::Vector{Float64},
    mu1::Float64,
    mu2::Float64,
    K::Int,
    p::Int
)
    F_trixi = trixi_rhs(u, mu1, mu2, K, p)
    F_clone = eval_rhs_clone(u, mu1, mu2, K, p)

    abs_err = norm(F_clone - F_trixi)
    rel_err = abs_err / max(norm(F_trixi), 1e-14)

    return abs_err, rel_err
end

# -----------------------------------------------------------------------------
# 2. Directional derivative test:
#
#         d^T [J(u)^T v] = v^T J(u) d
#
# This is the most useful cheap test for a VJP.
# -----------------------------------------------------------------------------

function directional_vjp_test(
    u::Vector{Float64},
    v::Vector{Float64},
    d::Vector{Float64},
    mu1::Float64,
    mu2::Float64,
    K::Int,
    p::Int;
    h::Float64 = 1e-6
)
    g = eval_rhs_vjp(u, v, mu1, mu2, K, p)

    lhs = dot(d, g)
    rhs = fd_directional_reference(
        u, v, d, mu1, mu2, K, p;
        h = h
    )

    rel_err = abs(lhs - rhs) /
              max(abs(rhs), abs(lhs), 1e-14)

    return rel_err, lhs, rhs
end

# -----------------------------------------------------------------------------
# Test states
# -----------------------------------------------------------------------------

Random.seed!(20260829)

function make_test_state(K::Int, p::Int)
    Nx = K * (p + 1)

    x = collect(1:Nx) ./ Nx

    u = 1.0 .+
        0.20 .* sin.(2.0*pi .* x) .+
        0.08 .* cos.(6.0*pi .* x) .+
        0.025 .* sin.(17.0*pi .* x)

    return collect(Float64, u)
end

function make_test_cotangent(K::Int, p::Int)
    Nx = K * (p + 1)
    x = collect(1:Nx) ./ Nx

    v = cos.(5.0*pi .* x) .+
        0.4 .* sin.(11.0*pi .* x)

    return collect(Float64, v)
end


# -----------------------------------------------------------------------------
# Main verification
# -----------------------------------------------------------------------------

function main()

    println("="^78)
    println("  Burgers DGSEM + HG shock-capturing discrete VJP verification")
    println("="^78)
    println()
    println("Reference RHS: Trixi.rhs_hyperbolic!")
    println("Forward clone is checked before VJP verification.")
    println()

    configs = [
        (128,  2),
        (128,  3),
        (64,  3),
        (64,  4),
    ]

    mu1 = 0.7
    mu2 = 1.23

    all_forward_ok = true
    all_vjp_ok = true

    for (K, p) in configs

        Nx = K * (p + 1)

        u = make_test_state(K, p)
        v = make_test_cotangent(K, p)

        d = randn(Nx)
        d ./= norm(d)

        alpha_min_val, alpha_max_val, n_interior =
            check_alpha(u, K, p)

        forward_abs, forward_rel =
            forward_consistency_test(u, mu1, mu2, K, p)

        forward_ok = forward_rel < 1e-11
        all_forward_ok &= forward_ok

        println(
            @sprintf(
                "K=%3d p=%d Nx=%4d | alpha=[%.3e, %.3e], interior=%2d | ",
                K, p, Nx,
                alpha_min_val,
                alpha_max_val,
                n_interior
            )
        )

        println(
            @sprintf(
                "    forward clone: abs=%.3e rel=%.3e %s",
                forward_abs,
                forward_rel,
                forward_ok ? "PASS" : "FAIL"
            )
        )

        if forward_ok

            rel_errs = Float64[]

            for trial in 1:4

                d = randn(Nx)
                d ./= norm(d)

                rel_err, lhs, rhs =
                    directional_vjp_test(
                        u,
                        v,
                        d,
                        mu1,
                        mu2,
                        K,
                        p;
                        h = 1e-6
                    )

                push!(rel_errs, rel_err)

            end

            max_err = maximum(rel_errs)
            mean_err = sum(rel_errs) / length(rel_errs)

            vjp_ok = max_err < 2e-5
            all_vjp_ok &= vjp_ok

            println(
                @sprintf(
                    "    VJP directional rel errors: max=%.3e mean=%.3e %s",
                    max_err,
                    mean_err,
                    vjp_ok ? "PASS" : "FAIL"
                )
            )

        else

            println(
                "    VJP test SKIPPED because forward clone failed."
            )

            all_vjp_ok = false
        end

        println()
    end


    # -------------------------------------------------------------------------
    # Coordinate-wise test
    # -------------------------------------------------------------------------

    println("-"^78)
    println("Coordinate-wise finite-difference VJP check")
    println("-"^78)

    K = 128
    p = 3

    u = make_test_state(K, p)
    v = make_test_cotangent(K, p)

    g_an = eval_rhs_vjp(
        u,
        v,
        mu1,
        mu2,
        K,
        p
    )

    g_fd = coordinate_fd_vjp(
        u,
        v,
        mu1,
        mu2,
        K,
        p;
        h = 1e-6
    )

    coord_rel =
        norm(g_an - g_fd) /
        max(norm(g_fd), 1e-14)

    coord_abs = norm(g_an - g_fd)

    println(
        @sprintf(
            "K=%d p=%d: absolute error = %.3e, relative error = %.3e",
            K,
            p,
            coord_abs,
            coord_rel
        )
    )

    coord_ok = coord_rel < 2e-5
    all_vjp_ok &= coord_ok

    println(
        coord_ok ?
        "Coordinate test: PASS" :
        "Coordinate test: FAIL"
    )

    println()
    println("="^78)

    if all_forward_ok && all_vjp_ok

        println("ALL VJP TESTS PASSED")
        println()
        println(
            "The handwritten forward decomposition matches Trixi and the"
        )
        println(
            "analytical VJP matches the Trixi RHS under finite differences."
        )

    else

        println("VERIFICATION FAILED")
        println()
        println(
            "Do not use this VJP yet. First inspect the forward-clone error."
        )
    end

    println("="^78)

end


main()