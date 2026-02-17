
using ITensors
using ITensorMPS
using LinearAlgebra
using Printf
using JuMP
using SCS


# A "fermion operator string": coefficient × (op1(site1) op2(site2) ...)
struct FStr
  coef::ComplexF64
  ops::Vector{Tuple{String,Int}}  # (opname, site)
end

FStr() = FStr(1.0 + 0im, Tuple{String,Int}[])

cdag(is::Int...) = FStr(1.0 + 0im, [( "Cdag", i) for i in is])
c(is::Int...)    = FStr(1.0 + 0im, [( "C",    i) for i in is])

Base.:*(a::FStr, b::FStr) = FStr(a.coef*b.coef, vcat(a.ops, b.ops))
Base.:*(α::Number, a::FStr) = FStr(ComplexF64(α)*a.coef, a.ops)
Base.:*(a::FStr, α::Number) = FStr(a.coef*ComplexF64(α), a.ops)

# Infer number of sites jm the max index appearing in the string
infer_N(s::FStr) = isempty(s.ops) ? 0 : maximum(t[2] for t in s.ops)

# Convert a single string into an MPO (one-term OpSum)
function mpo(s::FStr; N::Union{Int,Nothing}=nothing, conserve_qns::Bool=false)
  N === nothing && (N = infer_N(s))
  sites = siteinds("Fermion", N; conserve_qns=conserve_qns)

  os = OpSum()
  args = Any[s.coef]
  for (name, i) in s.ops
    push!(args, name); push!(args, i)
  end
  os += args...  # same style as: os += -t,"Cdag",i,"C",i+1, ...

  return MPO(os, sites), sites
end

struct OperatorSpace{O}
    basis::Vector{O}        
    ope::Array{Float64, 3}   # kdim×kdim×kdim stack of p_k(T)
    alphas::Vector{Float64}
    betas::Vector{Float64}
end

# ------------------------- MPO helpers -------------------------
function _sites(H::MPO)
  try
    return siteinds(only, H; plev=0)  # preferred
  catch
    return [siteind(H, j; plev=0) for j in 1:length(H)]  # fallback
  end
end
_mpo_like(H::MPO) = MPO(_sites(H), "Id")

# Safe scaling without mutating shared tensor data:
# uses `*=` on a single tensor (rebind), which is safe for shallow copies
function scale_mpo(A::MPO, c::Number)
    B = copy(A)       # shallow copy ok
    B[1] *= c          # safe: rebinds tensor (avoid `. *=`)
    return B
end

identity_mpo_like(H::MPO) = MPO(_sites(H), "Id")

# Make a zero MPO with the same indices as H
function zero_mpo_like(H::MPO)
    Z = identity_mpo_like(H)
    Z[1] *= 0.0
    return Z
end

# MPO linear combination with compression knobs.
# NOTE: this demo uses plain addition; you can wire in add kwargs if desired.
function add_mpo(A::MPO, B::MPO; cutoff=1e-12, maxdim=typemax(Int), alg="directsum")
    return +(A, B)
end

function sub_mpo(A::MPO, B::MPO; kwargs...)
    return add_mpo(A, scale_mpo(B, -1); kwargs...)
end

"Exact operator composition C = A ∘ B, no truncation keywords passed to contract."
function compose_mpo_exact(A::MPO, B::MPO)
    Ap = prime(A, "Site")                 # A'
    C  = contract(Ap, B)                  # no kwargs
    C  = replaceprime(C, 2 => 1)          # restore paired prime levels
    return C
end

"Compose and then truncate."
function mul_mpo(A::MPO, B::MPO; cutoff=1e-12, maxdim=2000, truncate=true)
    C = compose_mpo_exact(A, B)
    if truncate
        truncate!(C; cutoff=cutoff, maxdim=maxdim)
    end
    return C
end

hs_sites(H::MPO) = firstsiteinds(H; plev=0)
hs_total_dim(H::MPO) = prod(dim, hs_sites(H))

hs_inner(A::MPO, B::MPO) = inner(A, B) / hs_total_dim(A)
hs_norm(A::MPO) = norm(A) / sqrt(hs_total_dim(A))

# ------------------------- MPO Krylov/Lanczos -------------------------
function krylov(H::MPO;
    tol::Float64 = 1e-8,
    max_power::Union{Int, Nothing} = nothing,
    show_progress::Bool = true,

    # Multiplication/compression controls
    mul_cutoff::Float64 = 1e-12,
    mul_maxdim::Int = 2000,
    mul_alg::String = "auto",
    mul_truncate::Bool = true,

    # Addition/compression controls used during orthogonalization
    add_cutoff::Float64 = 1e-12,
    add_maxdim::Int = 2000,
    add_alg::String = "directsum",

    full_reorthogonalize::Bool = false,
)
    max_power === nothing && (max_power = typemax(Int))

    p0 = identity_mpo_like(H)
    basis = MPO[p0]             # p₀
    alphas = Float64[]
    betas  = Float64[]

    if show_progress
        println("    k     maxlinkdim      norm")
        @printf("%2d: %10d   % .6e\n", 0, maxlinkdim(p0), hs_norm(p0))
    end

    while length(alphas) < max_power
        pk = basis[end]

        # v_new = H * p_k
        v_new = mul_mpo(H, pk; cutoff=mul_cutoff, maxdim=mul_maxdim)

        # α_k = <p_k, v_new>
        αc = hs_inner(pk, v_new)
        α  = real(αc)
        push!(alphas, α)

        # v_new ← v_new − α_k p_k − β_{k−1} p_{k−1}
        v_new = add_mpo(v_new, scale_mpo(pk, -αc); alg=add_alg)

        if !isempty(betas)
            v_new = add_mpo(v_new, scale_mpo(basis[end - 1], -betas[end]); alg=add_alg)
        end

        if full_reorthogonalize
            for j in 1:length(basis)-1
                pj = basis[j]
                cj = hs_inner(pj, v_new)
                v_new = add_mpo(v_new, scale_mpo(pj, -cj);
                                cutoff=add_cutoff, maxdim=add_maxdim, alg=add_alg)
            end
        end

        # β_k = ||v_new||
        nrm = hs_norm(v_new)

        if show_progress
            @printf("alpha=% .6e  norm=% .6e  maxlinkdim=%d\n", α, nrm, maxlinkdim(v_new))
        end

        if nrm < tol
            show_progress && println("Lanczos converged at k=$(length(alphas)) (norm < tol).")
            break
        end

        push!(betas, nrm)

        # Normalize: p_{k+1} = v_new / β_k
        v_new = scale_mpo(v_new, 1 / nrm)
        push!(basis, v_new)

        if show_progress
            @printf("%2d: %10d   % .6e\n", length(basis) - 1, maxlinkdim(v_new), nrm)
        end
    end

    # Match convention: basis length == length(alphas)
    if length(basis) > length(alphas)
        pop!(basis)
    end

    # --- Build tridiagonal T and OPE stack ---
    kdim = length(alphas)
    α = collect(alphas[1:kdim])
    β = (kdim > 1) ? collect(betas[1:kdim - 1]) : Float64[]

    Ttri = diagm(0 => α)
    if kdim > 1
        Ttri += diagm(1  => β)
        Ttri += diagm(-1 => β)
    end

    # p_1(T)=I, p_{k+1}(T) = (T - α_k I)/β_k p_k(T) - (β_{k-1}/β_k) p_{k-1}(T)
    p_prev = zeros(Float64, kdim, kdim)
    p_curr = Matrix{Float64}(I, kdim, kdim)

    ope_stack = Array{Float64,3}(undef, kdim, kdim, kdim)
    ope_stack[:, :, 1] = p_curr

    for k in 2:kdim
        rhs = Ttri * p_curr - α[k - 1] * p_curr
        if k > 2
            rhs .-= β[k - 2] .* p_prev
        end
        if abs(β[k - 1]) < 1e-12
            # remaining are zero
            for kk in k:kdim
                ope_stack[:, :, kk] .= 0.0
            end
            break
        end
        p_next = rhs / β[k - 1]
        ope_stack[:, :, k] = p_next
        p_prev, p_curr = p_curr, p_next
    end

    return OperatorSpace{MPO}(basis, ope_stack, α, β)
end

# ------------------------- Projection/reconstruction -------------------------
function represent(space::OperatorSpace{MPO}, X::MPO)::Vector{ComplexF64}
    y = ComplexF64[]
    @inbounds for p in space.basis
        push!(y, hs_inner(p, X))
    end
    return y
end

function reconstruct(space::OperatorSpace{MPO}, coeffs::AbstractVector{<:Real};
    add_cutoff::Float64 = 1e-12,
    add_maxdim::Int = 2000,
    add_alg::String = "directsum",
)
    @assert length(coeffs) == length(space.basis)
    Hlike = space.basis[1]
    acc = zero_mpo_like(Hlike)
    @inbounds for (c, p) in zip(coeffs, space.basis)
        if c != 0
            acc = add_mpo(acc, scale_mpo(p, c); cutoff=add_cutoff, maxdim=add_maxdim, alg=add_alg)
        end
    end
    return acc
end

# ------------------------- SDP Bootstrap (MPO) -------------------------
"""
qboot with two Krylov sets:

- relaxation set: {K0,...,K_{n-1}} defines PSD constraint (moment matrix size n×n)
- span/ansatz set: {K0,...,K_{2n-1}} parameterizes ρ = K0 + Σ_{t=1}^{2n-1} x_t K_t

Build OPE as kdim×kdim×kdim with kdim=2n using the Krylov recurrence (p_k(T)),
then cut to n×n×kdim for the SDP.
"""
function qboot(H::MPO;
    n::Int = 10,
    show_progress::Bool = false,
    tol::Float64 = 1e-8,

    # Krylov MPO controls
    mul_cutoff::Float64 = 1e-12,
    mul_maxdim::Int = 2000,
    mul_alg::String = "auto",
    mul_truncate::Bool = true,
    add_cutoff::Float64 = 1e-12,
    add_maxdim::Int = 2000,
    add_alg::String = "directsum",
    full_reorthogonalize::Bool = false,
)
    kdim = 2n

    space = krylov(H;
        tol=tol, max_power=kdim, show_progress=show_progress,
        mul_cutoff=mul_cutoff, mul_maxdim=mul_maxdim, mul_alg=mul_alg, mul_truncate=mul_truncate,
        add_cutoff=add_cutoff, add_maxdim=add_maxdim, add_alg=add_alg,
        full_reorthogonalize=full_reorthogonalize,
    )

    if length(space.basis) < kdim
        error("Lanczos/Krylov terminated early (got kdim=$(length(space.basis)) < 2n=$kdim). " *
              "Try lowering `tol` and/or increasing MPO accuracy (e.g. larger maxdim / smaller cutoffs).")
    end

    # Cut OPE: n×n×2n
    ope_cut = space.ope[1:n, 1:n, 1:kdim]

    # Objective coefficients: minimize Tr(ρ H)/d = hs_inner(ρ,H)
    # with ρ = K0 + Σ x_t K_t (t=1..2n-1)
    h = real.(represent(space, H))
    Econst = h[1]
    c = h[2:end]  # length kdim-1

    # SDP moment matrix from cut OPE
    F0  = ope_cut[:, :, 1]                     # corresponds to K0 coefficient fixed to 1
    Fks = [ope_cut[:, :, t] for t in 2:kdim]   # length kdim-1 (for x variables)

    model = Model(SCS.Optimizer)
    if !show_progress
        set_silent(model)
    end
    set_optimizer_attribute(model, "eps_abs", 1e-5)
    set_optimizer_attribute(model, "eps_rel", 1e-5)
    set_optimizer_attribute(model, "max_iters", 1000000)
    set_optimizer_attribute(model, "scale", 1.0)
    set_optimizer_attribute(model, "acceleration_lookback", 20)

    @variable(model, x[1:kdim-1])
    @constraint(model, F0 + sum(x[t] * Fks[t] for t in 1:kdim-1) in PSDCone())
    @objective(model, Min, Econst + sum(c[t] * x[t] for t in 1:kdim-1))
    optimize!(model)

    xval = value.(x)
    coeffs = vcat(1.0, xval)  # length kdim
    rho = reconstruct(space, coeffs; add_cutoff=add_cutoff, add_maxdim=add_maxdim, add_alg=add_alg)

    return rho
end

# -----------------------------
# Fermionic operator strings (spinful Electron sites)
# -----------------------------
FStr() = FStr(1.0 + 0im, Tuple{String,Int}[])

# creation/annihilation
cdagup(is::Int...) = FStr(1.0 + 0im, [("Cdagup", i) for i in is])
cup(is::Int...)    = FStr(1.0 + 0im, [("Cup",    i) for i in is])
cdagdn(is::Int...) = FStr(1.0 + 0im, [("Cdagdn", i) for i in is])
cdn(is::Int...)    = FStr(1.0 + 0im, [("Cdn",    i) for i in is])

nup(i::Int) = FStr(1.0 + 0im, [("Nup", i)])
ndn(i::Int) = FStr(1.0 + 0im, [("Ndn", i)])

Base.:*(a::FStr, b::FStr) = FStr(a.coef*b.coef, vcat(a.ops, b.ops))
Base.:*(α::Number, a::FStr) = FStr(ComplexF64(α)*a.coef, a.ops)
Base.:*(a::FStr, α::Number) = FStr(a.coef*ComplexF64(α), a.ops)

# Convert a list of strings into an MPO
function mpo_from_terms(terms::Vector{FStr}, sites::Vector{<:Index};
                        splitblocks::Bool=true,
                        out_eltype::Type{<:Number}=ComplexF64)
  os = OpSum()

  for s in terms
    args = Any[s.coef]
    for (name, i) in s.ops
      push!(args, name)
      push!(args, i)
    end
    os += Tuple(args)
  end

  return MPO(out_eltype, os, sites; splitblocks=splitblocks)
end

# total number operator N = Σ_i (Nup_i + Ndn_i)
function number_mpo(sites)
  N = length(sites)
  os = OpSum()
  for i in 1:N
    os += 1.0, "Nup", i
    os += 1.0, "Ndn", i
  end
  return MPO(os, sites)
end

function hubbard_opsum(N::Int; t::Real=1.0, U::Real=1.0, periodic::Bool=false)
  up(i) = i
  dn(i) = N + i
  os = OpSum()

  add_hop!(a,b,amp) = begin
    os += amp, "Cdag", a, "C", b
    os += amp, "Cdag", b, "C", a
    nothing
  end
  for i in 1:(N-1)
    j = i+1
    add_hop!(up(i), up(j), -t)
    add_hop!(dn(i), dn(j), -t)
  end
  if periodic && N > 2
    add_hop!(up(N), up(1), -t)
    add_hop!(dn(N), dn(1), -t)
  end
  for i in 1:N
    os += U, "N", up(i), "N", dn(i)
  end
  return os
end

function run_example(H1::MPO)
    start = time()
    # Choose n so that span kdim=2n matches your old max_power
    n = 10
    rho = qboot(H1; n=n, show_progress=true, tol=1e-8,     # Krylov MPO controls
    mul_cutoff = 1e-12,
    mul_maxdim = 200,
    mul_alg = "auto",
    mul_truncate = true,
    add_cutoff = 1e-12,
    add_maxdim = 2000,
    add_alg = "directsum",
    full_reorthogonalize = false)

    println("GS Energy:", hs_inner(rho, H1))
    final = time()
    println("Time:", final - start)
end


N = 5
sites = siteinds("Fermion", 2N,conserve_qns=false)

Hopen = MPO(hubbard_opsum(N; t=1.0, U=0, periodic=false), sites)
Hper  = MPO(hubbard_opsum(N; t=1.0, U=1.0, periodic=true ), sites)

run_example(Hper)
