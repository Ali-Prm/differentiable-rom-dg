using HTTP
using JSON3
using LinearAlgebra
using TrixiBase

TrixiBase.disable_debug_timings()

include(joinpath(@__DIR__, "burgers_dg.jl"))

const HOST = "127.0.0.1"
const PORT = 8765

function parse_request(body::String)
    data = JSON3.read(body)
    u_raw = data.u
    B     = length(u_raw)
    Nx    = length(u_raw[1])

    u_batch = Matrix{Float64}(undef, B, Nx)
    for (i, row) in enumerate(u_raw)
        u_batch[i, :] = Float64.(row)
    end

    mu1     = Float64.(data.mu1)
    mu2     = Float64.(data.mu2)
    K       = Int(data.K)
    polydeg = Int(data.polydeg)

    return u_batch, mu1, mu2, K, polydeg
end

function mat_to_json(M::Matrix{Float64})
    return [M[i, :] for i in 1:size(M, 1)]
end

handle_health(req::HTTP.Request) = HTTP.Response(200, "OK")

function handle_rhs(req::HTTP.Request)
    try
        u_batch, mu1, mu2, K, p = parse_request(String(req.body))
        F = eval_rhs_batch(u_batch, mu1, mu2, K, p)
        HTTP.Response(200, JSON3.write(Dict("F" => mat_to_json(F))))
    catch e
        msg = sprint(showerror, e, catch_backtrace())
        @error "Error in /rhs" msg
        HTTP.Response(500, msg)
    end
end

function handle_vjp(req::HTTP.Request)
    try
        data = JSON3.read(String(req.body))
        u_raw = data.u
        v_raw = data.v
        B, Nx = length(u_raw), length(u_raw[1])

        u_batch = Matrix{Float64}(undef, B, Nx)
        v_batch = Matrix{Float64}(undef, B, Nx)
        for (i, (ru, rv)) in enumerate(zip(u_raw, v_raw))
            u_batch[i, :] = Float64.(ru)
            v_batch[i, :] = Float64.(rv)
        end

        mu1     = Float64.(data.mu1)
        mu2     = Float64.(data.mu2)
        K       = Int(data.K)
        polydeg = Int(data.polydeg)

        du = eval_rhs_vjp_batch(u_batch, v_batch, mu1, mu2, K, polydeg)
        HTTP.Response(200, JSON3.write(Dict("du" => mat_to_json(du))))
    catch e
        msg = sprint(showerror, e, catch_backtrace())
        @error "Error in /vjp" msg
        HTTP.Response(500, msg)
    end
end

router = HTTP.Router()
HTTP.register!(router, "GET",  "/health", handle_health)
HTTP.register!(router, "POST", "/rhs",    handle_rhs)
HTTP.register!(router, "POST", "/vjp",    handle_vjp)

@info "DG flux server starting on $HOST:$PORT"
HTTP.serve(router, HOST, PORT; verbose=false)