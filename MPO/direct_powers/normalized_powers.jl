using ITensors
using ITensorMPS
using LinearAlgebra
using Printf
using JuMP
import Clarabel
import MathOptInterface as MOI


struct Pauli
  kind::UInt8
  sites::Vector{Int}
end

const PI = UInt8(0)
const PX = UInt8(1)
const PY = UInt8(2)
const PZ = UInt8(3)

@inline function _mkpauli(kind::UInt8, is::Vararg{Int})
  v = collect(is)
  sort!(v)
  unique!(v)
  return Pauli(kind, v)
end

I(is::Vararg{Int}) = _mkpauli(PI, is...)
X(is::Vararg{Int}) = _mkpauli(PX, is...)
Y(is::Vararg{Int}) = _mkpauli(PY, is...)
Z(is::Vararg{Int}) = _mkpauli(PZ, is...)

@inline function _opname_qubit(kind::UInt8)
  kind == PI && return "Id"
  kind == PX && return "X"
  kind == PY && return "Y"
  kind == PZ && return "Z"
  error("invalid Pauli kind: $kind")
end

function pauli_mpo(N::Int, p::Pauli; sites=nothing)
  if sites === nothing
    sites = siteinds("Qubit", N)
  else
    length(sites) == N || throw(ArgumentError("length(sites) must be N=$N"))
  end


  mark = falses(N)
  @inbounds for i in p.sites
    1 ≤ i ≤ N || throw(ArgumentError("site $i out of range 1:$N"))
    mark[i] = true
  end

  if N == 1
    localop = op(sites, mark[1] ? _opname_qubit(p.kind) : "Id", 1)
    return MPO([localop])
  end

  links = [Index(1, "Link,$i") for i in 1:(N-1)]
  Wtensors = Vector{ITensor}(undef, N)

  @inbounds for i in 1:N
    localop = op(sites, mark[i] ? _opname_qubit(p.kind) : "Id", i)

    # construct links, with boundaries having one bond index
    if i == 1
      r = links[1]
      gate = ITensor(r); gate[r => 1] = 1
      Wtensors[i] = localop * gate
    elseif i == N
      l = links[N-1]
      gate = ITensor(l); gate[l => 1] = 1
      Wtensors[i] = localop * gate
    else
      l = links[i-1]; r = links[i]
      gate = ITensor(l, r); gate[l => 1, r => 1] = 1
      Wtensors[i] = localop * gate
    end
  end

  return MPO(Wtensors)
end




identity_mpo(H::MPO) = MPO(firstsiteinds(H; plev=0), "Id")

function zero_mpo(H::MPO)
  Z = identity_mpo(H)
  Z[1] *= 0.0
  return Z
end

function compose_mpo_exact(A::MPO, B::MPO)
  Ap = prime(A, "Site")
  C  = contract(Ap, B)       
  C  = replaceprime(C, 2 => 1)
  return C
end

function mul_mpo(A::MPO, B::MPO; cutoff=1e-12, maxdim=2000, truncate=true)
  C = compose_mpo_exact(A, B)
  if truncate
    truncate!(C; cutoff=cutoff, maxdim=maxdim)
  end
  return C
end

@inline function mpo_maxbond(A::MPO)
  ls = linkinds(A)
  return isempty(ls) ? 1 : maximum(dim.(ls))
end

@inline function hilbert_dim(Id::MPO)
  d = real(inner(Id, Id))
  d > 0 || error("Invalid Hilbert-space dimension from Tr(Id): d=$d")
  return d
end

@inline function mpo_trace(A::MPO, Id::MPO, d::Real)
  # Normalized trace: Tr(I) = 1
  return tr(A) / d
end

@inline function mpo_inner(A::MPO, B::MPO, d::Real)
  return inner(A, B) / d
end

@inline function mpo_norm(A::MPO, d::Real)
  return sqrt(real(mpo_inner(A, A, d)))
end

@inline function normalize_mpo!(A::MPO, d::Real)
  norm = mpo_norm(A, d)
  norm > 0 || error("Zero norm encountered while normalizing an MPO.")
  A[1] *= (1 / norm)
  return norm
end


"""
    krylov_basis(H, n; cutoff, maxdim, truncate, verbose, return_aux)

Build normalized powers up to **2n** elements:

    B[1]   ≈  H^0 / ‖H^0‖
    B[p+1] ≈  H^p / ‖H^p‖     for p = 1,2,...,2n-1

Construction uses a **prefix-sum / block-doubling** schedule:

1. Compute powers of two by squaring:
       H^{2^k} = H^{2^{k-1}} * H^{2^{k-1}}
2. Fill the intermediate block:
       H^{2^k + r} = H^{2^k} * H^r,   r = 1..2^k-1

Every MPO multiplication is performed between already-normalized MPOs,
followed by optional truncation and renormalization.

All traces/inner products/norms below use the **normalized trace** (`Tr(Id)=1`).

If `return_aux=true`, also returns cached scalars:
- `trB[k] = Tr(B[k])` for k=1..2n
- `T[p,q] = Tr(B[p]† B[q])` for p,q=1..2n
- `norm_list[i,j] = ‖comp(B[i]*B[j])‖` for i,j=1..n:
    * if the product `B[i]*B[j]` was explicitly computed during basis construction,
      we store its measured norm;
    * otherwise we estimate it from tracked raw-power norms via:
          norm_list[i,j] ≈ exp(log‖H^{i+j-2}‖ - log‖H^{i-1}‖ - log‖H^{j-1}‖).
- `log_pow_norm[k] = log ‖H^{k-1}‖` (tracked multiplicatively, avoids overflow)
- `h_scale` where `H_trunc ≈ h_scale * B[2]` (exact if `truncate=false`)

Returns:
- If `return_aux=false`: `B::Vector{MPO}` (length 2n)
- If `return_aux=true`:
    `(B, trB, Id, T, norm_list, log_pow_norm, h_scale)`
"""
@inline function _exp_clamped(x::Float64)
  if x > log(floatmax(Float64))
    return floatmax(Float64)
  elseif x < log(floatmin(Float64))
    return 0.0
  else
    return exp(x)
  end
end

function krylov_basis(
  H::MPO,
  n::Integer;
  cutoff::Float64 = 1e-12,
  maxdim::Int = 256,
  truncate::Bool = true,
  verbose::Bool = false,
  return_aux::Bool = false,
)
  n = Int(n)
  n ≥ 1 || throw(ArgumentError("n must be ≥ 1"))

  span = 2n
  max_p = span - 1

  B = Vector{MPO}(undef, span)
  Id = identity_mpo(H)
  d = hilbert_dim(Id) 

  # Track raw power norms in log-scale: log_pow_norm[k] = log ||H^(k-1)||_F
  log_pow_norm = fill(-Inf, span)

  # Pair-product norms for i,j = 1..n (filled from measured products when available,
  # otherwise estimated from log_pow_norm).
  norm_list = fill(NaN, n, n)


  tmp0 = copy(Id)
  log_pow_norm[1] = log(normalize_mpo!(tmp0, d))
  B[1] = tmp0


  Hn = copy(H)
  if truncate
    truncate!(Hn; cutoff=cutoff, maxdim=maxdim)
  end
  h_scale = normalize_mpo!(Hn, d)   # ||H_trunc||
  log_pow_norm[2] = log(h_scale)
  B[2] = Hn

  # Build higher powers in doubling blocks.
  base = 2
  while base ≤ max_p
    half = base ÷ 2
    idx  = half + 1

    X  = mul_mpo(B[idx], B[idx]; cutoff=cutoff, maxdim=maxdim, truncate=truncate)
    pn = normalize_mpo!(X, d)  # ||comp(B[idx]*B[idx])||, X normalized

    B[base + 1] = X
    log_pow_norm[base + 1] = 2*log_pow_norm[idx] + log(pn)

    if idx ≤ n
      norm_list[idx, idx] = pn
    end

    verbose && @printf("B[%2d] ~ H^%2d (square): maxbond=%d\n", base + 1, base, mpo_maxbond(B[base + 1]))

    # Fill the block: H^(base + r) = H^base * H^r
    limit = min(base - 1, max_p - base)
    for r in 1:limit
      ia = base + 1
      ib = r + 1
      X = mul_mpo(B[ia], B[ib]; cutoff=cutoff, maxdim=maxdim, truncate=truncate)
      pn = normalize_mpo!(X, d)
      p = base + r
      B[p + 1] = X
      log_pow_norm[p + 1] = log_pow_norm[ia] + log_pow_norm[ib] + log(pn)
      if ia ≤ n && ib ≤ n
        norm_list[ia, ib] = pn
        norm_list[ib, ia] = pn
      end
      verbose && @printf("B[%2d] ~ H^%2d (block base=%d,r=%d): maxbond=%d\n", p + 1, p, base, r, mpo_maxbond(B[p + 1]))
    end

    base *= 2
  end


  @inbounds for i in 1:n
    norm_list[1, i] = exp(-log_pow_norm[1])
    norm_list[i, 1] = exp(-log_pow_norm[1])
  end


  # Fill any missing entries using the tracked raw-power norms.
  @inbounds for i in 2:n
    for j in 2:n
      if isnan(norm_list[i, j])
        q = i + j - 1
        loga = log_pow_norm[q] - log_pow_norm[i] - log_pow_norm[j]
        norm_list[i, j] = _exp_clamped(loga)
      end
    end
  end
 
  if !return_aux
    return B
  end

  trB = Vector{Float64}(undef, span)
  @inbounds for k in 1:span
    trB[k] = real(mpo_trace(B[k], Id, d))
  end

  T = pair_trace_matrix(B, d)

  return B, trB, Id, T, norm_list, log_pow_norm, h_scale
end

using Base.Threads

"""
    pair_trace_matrix(B, d)

Build the pair-trace (Hilbert–Schmidt) matrix using the **normalized trace**
(so `Tr(Id) = 1`):

    T[i,j] = Tr(B[i]† B[j])

For Hermitian B, `Tr(B[i] B[j])`.
"""
function pair_trace_matrix(B::Vector{MPO}, d::Real; drop_tol::Float64=1e-14)
  m = length(B)
  T = Matrix{Float64}(undef, m, m)
  for i in 1:m
    @inbounds for j in i:m
      val = real(mpo_inner(B[i], B[j], d))
      val = abs(val) < drop_tol ? 0.0 : val
      T[i, j] = val
      T[j, i] = val
    end
  end
  return T
end


function ope_tensor(
  H::MPO,
  n::Integer;
  cutoff::Float64 = 1e-12,
  maxdim::Int = 256,
  truncate::Bool = true,
  verbose::Bool = false,
  return_aux::Bool = false,
)
  n = Int(n)
  n ≥ 1 || throw(ArgumentError("n must be ≥ 1"))
  span = 2n

  B, trB, Id, T, norm_list, log_pow_norm, h_scale = krylov_basis(
    H, n;
    cutoff=cutoff, maxdim=maxdim, truncate=truncate,
    verbose=verbose, return_aux=true
  )
  A = Array{Float64}(undef, n, n, span)

  for i in 1:n
    for j in i:n
      q = i + j - 1              # index of the (i+j-2)-th power in B (1-based)
      a = norm_list[i, j]        # ≈ ||comp(B[i]*B[j])||

      @inbounds for k in 1:span
        v = a * T[q, k]          # ≈ Tr(B[i] B[j] B[k])
        A[i, j, k] = v
        A[j, i, k] = v
      end
    end

    if verbose
      println("OPE Row Finished i = $i / $n")
      flush(stdout)
    end
  end

  # Post-process: enforce exact symmetry in (i,j) and drop tiny numerical noise.
  tol = 1e-14
  @inbounds for k in 1:span
    for i in 1:n
      for j in (i+1):n
        v = 0.5 * (A[i, j, k] + A[j, i, k])
        A[i, j, k] = v
        A[j, i, k] = v
      end
    end
  end
  @inbounds for idx in eachindex(A)
    abs(A[idx]) < tol && (A[idx] = 0.0)
  end

  return return_aux ? (A, B, trB, Id, T, norm_list, log_pow_norm, h_scale) : A
end


"""
    ope_tensor_exact(H, n; cutoff, maxdim, truncate, verbose, return_aux)

Compute an (as-close-as-possible) **exact** OPE tensor

    A[i,j,k] = Tr(B[i] B[j] B[k])

by explicitly multiplying **two adjacent operators** (in a cyclic trace)
chosen to be the lowest-cost pair, then taking a Hilbert–Schmidt inner
product with the remaining operator.

We use cyclicity of trace to evaluate the same scalar via one of:
x
  1) Tr((B[i]B[j]) B[k])  = ⟨B[k], B[i]B[j]⟩
  2) Tr((B[j]B[k]) B[i])  = ⟨B[i], B[j]B[k]⟩
  3) Tr((B[k]B[i]) B[j])  = ⟨B[j], B[k]B[i]⟩

and we pick the option whose multiplied pair has smallest heuristic cost
based on MPO max-bond dimensions.

If `return_aux=true`, also returns `(B, trB, Id, T)`.
"""

function ope_tensor_exact(
  H::MPO,
  n::Integer;
  cutoff::Float64 = 1e-12,
  maxdim::Int = 256,
  truncate::Bool = true,
  verbose::Bool = false,
  return_aux::Bool = false,
)
  n = Int(n)
  n ≥ 1 || throw(ArgumentError("n must be ≥ 1"))
  span = 2n

  # Build the same Krylov basis (so the comparison is apples-to-apples).
  B, trB, Id, T, norm_list, log_pow_norm, h_scale = krylov_basis(
    H, n;
    cutoff=cutoff, maxdim=maxdim, truncate=truncate,
    verbose=false, return_aux=true
  )
  d = hilbert_dim(Id)

  # Heuristic cost for multiplying a pair.
  @inline function _pair_cost(a::Int, b::Int)
    return mpo_maxbond(B[a]) * mpo_maxbond(B[b])
  end

  # Compute Tr(B[i] B[j] B[k]) by multiplying the cheapest adjacent pair.
  function _tr3(i::Int, j::Int, k::Int)
    c1 = _pair_cost(i, j)
    c2 = _pair_cost(j, k)
    c3 = _pair_cost(k, i)

    if c1 ≤ c2 && c1 ≤ c3
      P = mul_mpo(B[i], B[j]; cutoff=cutoff, maxdim=maxdim, truncate=truncate)
      return real(inner(dag(B[k]), P)) / d
    elseif c2 ≤ c1 && c2 ≤ c3
      P = mul_mpo(B[j], B[k]; cutoff=cutoff, maxdim=maxdim, truncate=truncate)
      return real(inner(dag(B[i]), P)) / d
    else
      P = mul_mpo(B[k], B[i]; cutoff=cutoff, maxdim=maxdim, truncate=truncate)
      return real(inner(dag(B[j]), P)) / d
    end
  end

  Aex = Array{Float64}(undef, n, n, span)

  for i in 1:n
    for j in 1:n
      @inbounds for k in 1:span
        Aex[i, j, k] = _tr3(i, j, k)
      end
    end
    if verbose
      println("Exact OPE Row Finished i = $i / $n")
      flush(stdout)
    end
  end

  # Post-process: enforce exact symmetry in (i,j) and drop tiny numerical noise.
  tol = 1e-14
  @inbounds for k in 1:span
    for i in 1:n
      for j in (i+1):n
        v = 0.5 * (Aex[i, j, k] + Aex[j, i, k])
        Aex[i, j, k] = v
        Aex[j, i, k] = v
      end
    end
  end
  @inbounds for idx in eachindex(Aex)
    abs(Aex[idx]) < tol && (Aex[idx] = 0.0)
  end

  return return_aux ? (Aex, B, trB, Id, T, norm_list, log_pow_norm, h_scale) : Aex
end


"""
    compare_ope_tensors(H, n; kwargs...)

Compute `Aapprox = ope_tensor(...)` and `Aexact = ope_tensor_exact(...)`
and report error metrics including Frobenius norm of the difference.

Returns `(Aapprox, Aexact, report)` where `report` is a NamedTuple.
"""
function compare_ope_tensors(
  H::MPO,
  n::Integer;
  cutoff::Float64 = 1e-12,
  maxdim::Int = 256,
  truncate::Bool = true,
  verbose::Bool = false,
)
  Aapprox = ope_tensor(
    H, n;
    cutoff=cutoff, maxdim=maxdim, truncate=truncate,
    verbose=true, return_aux=false
  )

  Aexact = ope_tensor_exact(
    H, n;
    cutoff=cutoff, maxdim=maxdim, truncate=false,
    verbose=true, return_aux=false
  )

  Δ = Aapprox .- Aexact
  frob = norm(Δ)
  frob_exact = norm(Aexact)
  rel_frob = frob_exact > 0 ? frob / frob_exact : NaN
  max_abs = maximum(abs.(Δ))
  max_rel = begin
    denom = maximum(abs.(Aexact))
    denom > 0 ? max_abs / denom : NaN
  end

  report = (
    frobenius = frob,
    frobenius_exact = frob_exact,
    rel_frobenius = rel_frob,
    max_abs = max_abs,
    max_rel = max_rel,
    cutoff = cutoff,
    maxdim = maxdim,
    truncate = truncate,
  )

  return Aapprox, Aexact, report
end



"""
    bootstrap(H, n; cutoff, maxdim, truncate, verbose_ope, silent, return_coeffs)

`n` is the relaxation-set size; the span-set size is `2n`.

We parameterize an operator ansatz:

    ρ = Σ_{k=1}^{2n} x[k] * B[k]

with `B[k]` the normalized powers `H^(k-1) / ‖H^(k-1)‖`.

Constraints:
- `Tr(ρ) = 1`
- Moment matrix `M[a,b] = Tr(ρ * B[a] * B[b])` is PSD for a,b=1..n

Optional spectral bound (if `R` is provided):
- `L = R^2 M - Mshift` is PSD, where `Mshift[a,b] = Tr(ρ * B[a] * H^2 * B[b])`.

Objective:
- minimize `Tr(ρ * H)`.

Returns `(E, ρ)` where `ρ` is returned as an MPO.

If `return_coeffs=true`, returns `(E, x, ρ)` where `x` are the basis coefficients.
"""
function bootstrap(
  H::MPO,
  n::Integer;
  cutoff::Float64 = 1e-12,
  maxdim::Int = 256,
  truncate::Bool = true,
  verbose_ope::Bool = false,
  silent::Bool = false,
)
  n = Int(n)
  n ≥ 2 || throw(ArgumentError("n must be ≥ 2"))
  span = 2n

  # Build OPE without any MPO multiplications beyond krylov_basis():
  A, B, trB, Id, T, _norm_list, _log_pow_norm, h_scale = ope_tensor(
    H, n;
    cutoff=cutoff, maxdim=maxdim, truncate=truncate,
    verbose=verbose_ope, return_aux=true
  )


  # Objective coefficients c[k] = Tr(B[k] * H_trunc).
  # Tr(B[k] * H) =  Tr(B[k] * ||H|| B[2]) = h_scale * T[k,2].
  c = Vector{Float64}(undef, span)
  @inbounds for k in 1:span
    c[k] = h_scale * T[k, 2]
  end

  model = Model(Clarabel.Optimizer)

  # Clarabel verbosity is controlled via the "verbose" attribute.
  set_optimizer_attribute(model, "verbose", !silent)

  # Interior-point tolerances and iteration cap.
  set_optimizer_attribute(model, "tol_gap_abs", 1e-12)
  set_optimizer_attribute(model, "tol_gap_rel", 1e-12)
  set_optimizer_attribute(model, "tol_feas",    1e-12)
  set_optimizer_attribute(model, "max_iter", 4000)

  @variable(model, x[1:span])

  # Moment matrix M[a,b] = Tr(ρ B[a] B[b]) = Σ_k x[k] Tr(B[a] B[b] B[k]) is psd
  M= sum(A[:, :, k] .* x[k] for k in 1:span)
  @constraint(model,  Symmetric(M) in PSDCone())

  # Tr(ρ) = 1
  @constraint(model, sum(trB[k] * x[k] for k in 1:span) == 1)

  @objective(model, Min, sum(c[k] * x[k] for k in 1:span))

  optimize!(model)
  st = termination_status(model)
  if !(st in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED)) && verbose_ope
    @warn "bootstrap SDP not optimal" status=st primal=primal_status(model) dual=dual_status(model)
  end

  # Optimized coefficients for ρ = Σ_k x[k] B[k]
  xopt = value.(x)

  # Moment matrix M[a,b] = Tr(ρ B[a] B[b]) evaluated at xopt.
  Mval = zeros(Float64, n, n)
  @inbounds for k in 1:span
    Mval .+= xopt[k] .* A[:, :, k]
  end
  Mval = 0.5 .* (Mval .+ Mval')   # enforce symmetry
  λmin = eigmin(Symmetric(Mval))

  # Energy objective value.
  E = try
    objective_value(model)
  catch
    dot(c, xopt)
  end

  return (E=E, λmin=λmin, status=st)
end





"""
Transverse-field Ising model Hamiltonian

H = -J Σ_i Z_i Z_{i+1} - g Σ_i X_i

"""
function TFIM(L::Integer; J::Real=1.0, g::Real=1.0, periodic::Bool=true, sites=nothing)
  L = Int(L)
  sites === nothing && (sites = siteinds("Qubit", L))

  os = OpSum()

  for i in 1:L
    j = i == L ? 1 : i + 1
    if periodic || i < L
      os += (-J), "Z", i, "Z", j
    end
  end

  for i in 1:L
    os += (-g), "X", i
  end

  return MPO(os, sites)
end
