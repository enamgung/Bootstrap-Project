include("OperatorAlgebra.jl")
using LinearAlgebra
using JuMP
using SCS
using SparseArrays
using Printf
using Base.Threads
import Base:zero

const _POW2_NEG = let v = Vector{Float64}(undef, 129)
    @inbounds for i in 1:129
        v[i] = ldexp(1.0, -(i - 1))
    end
    v
end

const _LOW64_MASK = UInt128(0xFFFF_FFFF_FFFF_FFFF)

#Store indexed list of operator monomials 
struct MonoTable{K}
    keys   :: Vector{K}
    index  :: Dict{UInt128, Int}
    packed :: Vector{UInt128}
end


# "packing": label key K as UInt128 for fast hashing 
@inline packkey(m::T) where {T<:Unsigned} = UInt128(m)

@inline function packkey(k::FKey{T})::UInt128 where {T<:Unsigned}
    @assert sizeof(T) <= 8 "packkey(FKey{$T}) assumes T ≤ 64 bits (two masks packed into UInt128)."
    return (UInt128(k.c) << 64) | UInt128(k.a)
end

@inline function packkey(k::PKey{T})::UInt128 where {T<:Unsigned}
    @assert sizeof(T) <= 8 "packkey(PKey{$T}) assumes T ≤ 64 bits (two masks packed into UInt128)."
    return (UInt128(k.x) << 64) | UInt128(k.z)
end



@inline function _find_index(B::MonoTable{K}, k::K)::Int where {K}
    pkey = packkey(k)
    i = searchsortedfirst(B.packed, pkey)
    (i <= length(B.packed) && @inbounds B.packed[i] == pkey) ? i : 0
end

@inline function _find_index(B::MonoTable{FKey{T}}, C::T, A::T)::Int where {T<:Unsigned}
    @assert sizeof(T) <= 8 "MonoTable(FKey{$T}) packing assumes T ≤ 64 bits."
    pkey = (UInt128(C) << 64) | UInt128(A)
    i = searchsortedfirst(B.packed, pkey)
    (i <= length(B.packed) && @inbounds B.packed[i] == pkey) ? i : 0
end

@inline function _find_index(B::MonoTable{PKey{T}}, X::T, Z::T)::Int where {T<:Unsigned}
    @assert sizeof(T) <= 8 "MonoTable(PKey{$T}) packing assumes T ≤ 64 bits."
    pkey = (UInt128(X) << 64) | UInt128(Z)
    i = searchsortedfirst(B.packed, pkey)
    (i <= length(B.packed) && @inbounds B.packed[i] == pkey) ? i : 0
end


# Generic: Dict already has unique keys, so drop near-zeros.
function drop_near_zero(terms::Dict{K, ComplexF64}; tol::Float64=TOL)::Dict{K, ComplexF64} where {K}
    out = Dict{K, ComplexF64}()
    sizehint!(out, length(terms))
    @inbounds for (k, c) in terms
        abs(c) < tol && continue
        out[k] = c
    end
    return out
end

drop_near_zero(terms::Dict{FKey{T}, ComplexF64}; tol::Float64=TOL) where {T<:Unsigned} =
    combine_like_terms(terms)

"""
    build_mono_table(op::Operator{A,K}) -> MonoTable{K}

- sorts by `packkey(key)` (deterministic total order)
- stores `packed[i] == packkey(keys[i])`
- stores `index[packed[i]] == i`
"""

function build_mono_table(op::Operator{A, K})::MonoTable{K} where {A<:AlgTag, K}
    terms = drop_near_zero(op.terms)
    ks = collect(keys(terms))
    sort!(ks; by = packkey)

    packed = Vector{UInt128}(undef, length(ks))
    idx = Dict{UInt128, Int}()
    sizehint!(idx, length(ks))

    @inbounds for i in eachindex(ks)
        p = packkey(ks[i])
        packed[i] = p
        idx[p] = i
    end

    @assert issorted(packed)
    @assert length(idx) == length(ks)  # ensures no duplicate packed keys

    return MonoTable{K}(ks, idx, packed)
end

# =============== Vector-form operator over the frozen basis (kept) ===============
const OpVec = SparseVector{ComplexF64, Int}


# Arithmetic operations for OpVec
function Base.:*(v::OpVec, s::Number)::OpVec
    abs(s) < TOL && return OpVec(Int[], ComplexF64[])
    OpVec(copy(v.idx), [ComplexF64(s) * x for x in v.val])
end
Base.:*(s::Number, v::OpVec) = v * s
Base.:/(v::OpVec, s::Number) = v * (1 / s)
Base.:-(v::OpVec) = (-1) * v

function Base.:-(a::OpVec, b::OpVec)::OpVec
    isempty(a.idx) && return -b
    isempty(b.idx) && return OpVec(copy(a.idx), copy(a.val))
    
    # Use dense accumulator for simplicity
    N = max(maximum(a.idx; init=0), maximum(b.idx; init=0))
    N == 0 && return OpVec(Int[], ComplexF64[])
    
    acc = zeros(ComplexF64, N)
    @inbounds for (i, v) in zip(a.idx, a.val)
        acc[i] += v
    end
    @inbounds for (i, v) in zip(b.idx, b.val)
        acc[i] -= v
    end
    
    nz = findall(x -> abs2(x) >= TOL2, acc)
    isempty(nz) ? OpVec(Int[], ComplexF64[]) : OpVec(nz, acc[nz])
end

# =============== Projection w/o hashing: accumulate in a dense vector, then compress ===============
function op_to_vector(op::FermionOperator{T}, B::MonoTable{FKey{T}})::OpVec where {T<:Unsigned}
    N = length(B.keys)
    acc = zeros(ComplexF64, N)  # dense accumulator (fast for many updates)

    @inbounds for (k, c) in op.terms
        abs2(c) < TOL2 && continue
        p = _find_index(B, k.c, k.a)   # uses your fast overload for fermions
        if p == 0
            _inc_dropped!(1, abs2(c))
        else
            acc[p] += c
        end
    end

    nz = findall(v -> abs2(v) >= TOL2, acc)
    isempty(nz) ? spzeros(ComplexF64, N) : SparseVector(N, nz, acc[nz])
end

function vector_to_op(B::MonoTable{FKey{T}}, v::OpVec)::FermionOperator{T} where {T<:Unsigned}
    d = Dict{FKey{T}, ComplexF64}()
    ind = v.nzind
    val = v.nzval
    sizehint!(d, length(ind))

    @inbounds for (p, c) in zip(ind, val)
        abs2(c) < TOL2 && continue
        k = B.keys[p]  # ::FKey{T}
        d[k] = get(d, k, 0.0 + 0.0im) + c
    end

    FermionOperator{T}(d)
end

function op_to_vector(op::PauliOperator{T}, B::MonoTable{PKey{T}})::OpVec where {T<:Unsigned}
    N = length(B.keys)
    acc = zeros(ComplexF64, N)

    @inbounds for (k, c) in op.terms
        abs2(c) < TOL2 && continue
        p = _find_index(B, k)   # uses packkey/searchsorted for PKey
        if p == 0
            _inc_dropped!(1, abs2(c))
        else
            acc[p] += c
        end
    end

    nz = findall(v -> abs2(v) >= TOL2, acc)
    isempty(nz) ? spzeros(ComplexF64, N) : SparseVector(N, nz, acc[nz])
end

function vector_to_op(B::MonoTable{PKey{T}}, v::OpVec)::PauliOperator{T} where {T<:Unsigned}
    d = Dict{PKey{T}, ComplexF64}()
    ind = v.nzind
    val = v.nzval
    sizehint!(d, length(ind))

    @inbounds for (p, c) in zip(ind, val)
        abs2(c) < TOL2 && continue
        k = B.keys[p]  # ::PKey{T}
        d[k] = get(d, k, 0.0 + 0.0im) + c
    end

    PauliOperator{T}(d)
end

function op_to_vector(op::MajoranaOperator{T}, B::MonoTable{T})::OpVec where {T<:Unsigned}
    N = length(B.keys)
    acc = zeros(ComplexF64, N)

    @inbounds for (k, c) in op.terms
        abs2(c) < TOL2 && continue
        p = _find_index(B, k)   # k is mask::T
        if p == 0
            _inc_dropped!(1, abs2(c))
        else
            acc[p] += c
        end
    end

    nz = findall(v -> abs2(v) >= TOL2, acc)
    isempty(nz) ? spzeros(ComplexF64, N) : SparseVector(N, nz, acc[nz])
end

function vector_to_op(B::MonoTable{T}, v::OpVec)::MajoranaOperator{T} where {T<:Unsigned}
    d = Dict{T, ComplexF64}()
    ind = v.nzind
    val = v.nzval
    sizehint!(d, length(ind))

    @inbounds for (p, c) in zip(ind, val)
        abs2(c) < TOL2 && continue
        k = B.keys[p]  # ::T mask
        d[k] = get(d, k, 0.0 + 0.0im) + c
    end

    MajoranaOperator{T}(d)
end



@inline function prune(v::OpVec; tol2::Float64 = TOL2)::OpVec
    nnz(v) == 0 && return v
    ind = v.nzind
    val = v.nzval
    keep_i  = Int[]
    keep_v  = ComplexF64[]
    sizehint!(keep_i, length(ind))
    sizehint!(keep_v, length(ind))
    @inbounds for k in eachindex(ind)
        c = val[k]
        abs2(c) < tol2 && continue
        push!(keep_i, ind[k])
        push!(keep_v, c)
    end
    isempty(keep_i) ? spzeros(ComplexF64, length(v)) :
                      SparseVector(length(v), keep_i, keep_v)
end

# Superoperator: Left Multiplication H
# Build "Action Matrix", smaller OPE matrix specialized for H acting on monomial set.
# We still must visit each (H_term, basis_key) pair; COO triples + sparse() is the fastest way.
function build_action_matrix(H::FermionOperator{T}, B::MonoTable{FKey{T}};
                             tol2::Real = TOL2)::SparseMatrixCSC{ComplexF64,Int} where {T<:Unsigned}
    tol2 = float(tol2)
    N = length(B.keys)

    # Reserve a rough budget (helps avoid repeated resizing);
    # scale by min(#terms(H), N), tweak if you like after profiling.
    est = max(1, min(length(H.terms) * 4, 10N))
    I = Vector{Int}(undef, 0); sizehint!(I, est)
    J = Vector{Int}(undef, 0); sizehint!(J, est)
    V = Vector{ComplexF64}(undef, 0); sizehint!(V, est)

    @inbounds for j in 1:N
        kj = B.keys[j]
        for (kH, cH) in H.terms
            abs2(cH) < tol2 && continue

            r = Fkey_mul(kH, kj)  
            r === nothing && continue
            Cb, Ab, S, sgn = r
            base = (sgn == 1 ? cH : -cH)

            if S == zero(T)
                p = _find_index(B, Cb, Ab)
                if p == 0
                    _inc_dropped!(1, abs2(base))
                else
                    push!(I, p); push!(J, j); push!(V, base)
                end
            else
                s = S
                while true
                    coef = isodd(count_ones(s)) ? -base : base
                    p = _find_index(B, Cb | s, Ab | s)
                    if p == 0
                        _inc_dropped!(1, abs2(coef))
                    else
                        push!(I, p); push!(J, j); push!(V, coef)
                    end
                    s == zero(T) && break
                    s = (s - one(T)) & S
                end
            end
        end
    end

    A = sparse(I, J, V, N, N)         # sums duplicates efficiently
    # Optional pruning near-zero entries:
    if !isempty(A.nzval)
        @inbounds for k in eachindex(A.nzval)
            abs2(A.nzval[k]) < tol2 && (A.nzval[k] = 0)
        end
        dropzeros!(A)
    end
    return A
end

function build_action_matrix(H::MajoranaOperator{T}, B::MonoTable{T};
                             tol2::Real = TOL2)::SparseMatrixCSC{ComplexF64,Int} where {T<:Unsigned}
    tol2 = float(tol2)
    N = length(B.keys)

    # Precompute prefix-parity masks for every basis monomial kj
    PP = Vector{T}(undef, N)
    @inbounds for j in 1:N
        PP[j] = _prefix_parity_mask_cached(B.keys[j])
    end

    est = max(1, min(length(H.terms) * N, 10N))   # H usually sparse; tweak after profiling
    I = Int[]; sizehint!(I, est)
    J = Int[]; sizehint!(J, est)
    V = ComplexF64[]; sizehint!(V, est)

    @inbounds for j in 1:N
        kj = B.keys[j]     # ::T (mask)
        pp = PP[j]
        for (kH, cH) in H.terms
            abs2(cH) < tol2 && continue

            sgn  = _pair_sign_from_pp(kH, pp)   # ±1.0
            outk = xor(kH, kj)
            coef = cH * sgn

            p = _find_index(B, outk)
            if p == 0
                _inc_dropped!(1, abs2(coef))
            else
                push!(I, p); push!(J, j); push!(V, coef)
            end
        end
    end

    A = sparse(I, J, V, N, N)
    if !isempty(A.nzval)
        @inbounds for k in eachindex(A.nzval)
            abs2(A.nzval[k]) < tol2 && (A.nzval[k] = 0)
        end
        dropzeros!(A)
    end
    return A
end



function build_action_matrix(H::PauliOperator{T}, B::MonoTable{PKey{T}};
                             tol2::Real = TOL2)::SparseMatrixCSC{ComplexF64,Int} where {T<:Unsigned}
    tol2 = float(tol2)
    N = length(B.keys)

    est = max(1, min(length(H.terms) * N, 10N))
    I = Int[]; sizehint!(I, est)
    J = Int[]; sizehint!(J, est)
    V = ComplexF64[]; sizehint!(V, est)

    @inbounds for j in 1:N
        kj = B.keys[j]   # ::PKey{T}
        for (kH, cH) in H.terms
            abs2(cH) < tol2 && continue

            coef = cH * _pauli_phase(kH, kj)
            outk = PKey{T}(xor(kH.x, kj.x), xor(kH.z, kj.z))

            p = _find_index(B, outk)
            if p == 0
                _inc_dropped!(1, abs2(coef))
            else
                push!(I, p); push!(J, j); push!(V, coef)
            end
        end
    end

    A = sparse(I, J, V, N, N)
    if !isempty(A.nzval)
        @inbounds for k in eachindex(A.nzval)
            abs2(A.nzval[k]) < tol2 && (A.nzval[k] = 0)
        end
        dropzeros!(A)
    end
    return A
end


# Fast approach using SparseMatrixCSC
function apply_action(A::SparseMatrixCSC{ComplexF64,Int}, x::OpVec; show_progress::Bool = false)::OpVec
    isempty(x.nzind) && return spzeros(ComplexF64, size(A, 1))

    st = time()
    sy = A * x          # SparseVector{ComplexF64,Int}
    fn = time()

    # Enforce TOL2 if you want:
    return sy
end




@inline function _pack2x64(a::T, b::T) where {T<:Unsigned}
    @assert sizeof(T) <= 8 "2x64 packing assumes T ≤ 64 bits; got $(T)"
    (UInt128(a) << 64) | UInt128(b)
end

@inline function _unpack2x64(::Type{T}, p::UInt128) where {T<:Unsigned}
    @assert sizeof(T) <= 8 "2x64 unpacking assumes T ≤ 64 bits; got $(T)"
    a = T(p >> 64)
    b = T(p & _LOW64_MASK)
    return a, b
end

# Fermion key pack/unpack
@inline packkey(k::FKey{T}) where {T<:Unsigned} = _pack2x64(k.c, k.a)
@inline function unpackkey(::Type{FKey{T}}, p::UInt128) where {T<:Unsigned}
    C, A = _unpack2x64(T, p)
    return FKey{T}(C, A)
end

# Pauli key pack/unpack
@inline packkey(k::PKey{T}) where {T<:Unsigned} = _pack2x64(k.x, k.z)
@inline function unpackkey(::Type{PKey{T}}, p::UInt128) where {T<:Unsigned}
    X, Z = _unpack2x64(T, p)
    return PKey{T}(X, Z)
end

# Majorana key pack/unpack (key is the mask itself: K == T) :contentReference[oaicite:2]{index=2}
@inline packkey(m::T) where {T<:Unsigned} = UInt128(m)
@inline unpack_majorana(::Type{T}, p::UInt128) where {T<:Unsigned} = T(p)


@inline support_mask(k::FKey{T}) where {T<:Unsigned} = (k.c | k.a)
@inline support_mask(k::PKey{T}) where {T<:Unsigned} = (k.x | k.z)
@inline support_mask(m::T)       where {T<:Unsigned} = m



#Find all the monomials generated by powers of H, and store it as MonoTable. Ignore coefficient/sign for now.
function H_generated_mono(
    H::FermionOperator{T};
    max_depth::Int      = 6,
    max_size::Int       = 800_000,
    show_progress::Bool = false,
    locality::Bool      = false,
    LMAX::Int           = typemax(Int),
    locality_measure::Function = locality_measure,  
) where {T<:Unsigned}

    #Identity key
    Ikey = FKey{T}(zero(T), zero(T))
    Ipacked = packkey(Ikey)

    packed_all = UInt128[Ipacked]
    frontier   = UInt128[Ipacked]

    seen = Dict{UInt128,Bool}(Ipacked => true)

    Hkeys = collect(Base.keys(H.terms))
    if isempty(Hkeys)
        keys   = FKey{T}[Ikey]
        idx    = Dict{UInt128,Int}(Ipacked => 1)
        packed = UInt128[Ipacked]
        return MonoTable{FKey{T}}(keys, idx, packed)
    end

    depth = 0
    while depth < max_depth && !isempty(frontier) && length(packed_all) < max_size
        depth += 1
        show_progress && println("closure depth $depth: frontier=$(length(frontier)), total=$(length(packed_all))")

        new_frontier = UInt128[]
        sizehint!(new_frontier,
            min(max_size - length(packed_all),
                length(frontier) * max(length(Hkeys), 1)))

        @inbounds for p in frontier
            k = unpackkey(FKey{T}, p)

            for hk in Hkeys
                r = Fkey_mul(hk, k)   # returns (Cb,Ab,S,sgn) or nothing :contentReference[oaicite:4]{index=4}
                r === nothing && continue
                Cb, Ab, S, _ = r  # sign irrelevant for basis discovery

                if S == zero(T)
                    if locality && locality_measure(Cb | Ab) > LMAX
                        continue
                    end
                    p_new = _pack2x64(Cb, Ab)
                    if !haskey(seen, p_new)
                        seen[p_new] = true
                        push!(packed_all, p_new)
                        push!(new_frontier, p_new)
                        length(packed_all) >= max_size && break
                    end
                else
                    # (1-n) expansion: enumerate all subsets of S
                    s = S
                    while true
                        C = Cb | s
                        A = Ab | s
                        if !locality || locality_measure(C | A) <= LMAX
                            p_new = _pack2x64(C, A)
                            if !haskey(seen, p_new)
                                seen[p_new] = true
                                push!(packed_all, p_new)
                                push!(new_frontier, p_new)
                                length(packed_all) >= max_size && break
                            end
                        end
                        s == zero(T) && break
                        s = (s - one(T)) & S
                    end
                end

                length(packed_all) >= max_size && break
            end

            length(packed_all) >= max_size && break
        end

        frontier = new_frontier
    end

    show_progress && println("H-generated space: $(length(packed_all)) monomials (depth=$depth)")

    sort!(packed_all)
    N = length(packed_all)

    keys   = Vector{FKey{T}}(undef, N)
    packed = Vector{UInt128}(undef, N)
    idx    = Dict{UInt128,Int}(); sizehint!(idx, N)

    @inbounds for i in 1:N
        p = packed_all[i]
        keys[i]   = unpackkey(FKey{T}, p)
        packed[i] = p
        idx[p]    = i
    end

    return MonoTable{FKey{T}}(keys, idx, packed)
end

function H_generated_mono(
    H::PauliOperator{T};
    max_depth::Int      = 6,
    max_size::Int       = 800_000,
    show_progress::Bool = false,
    locality::Bool      = false,
    LMAX::Int           = typemax(Int),
    locality_measure::Function = (m::T)->count_ones(m),
) where {T<:Unsigned}

    Ikey = PKey{T}(zero(T), zero(T))
    Ipacked = packkey(Ikey)

    packed_all = UInt128[Ipacked]
    frontier   = UInt128[Ipacked]
    seen = Dict{UInt128,Bool}(Ipacked => true)

    Hkeys = collect(Base.keys(H.terms))
    if isempty(Hkeys)
        keys   = PKey{T}[Ikey]
        idx    = Dict{UInt128,Int}(Ipacked => 1)
        packed = UInt128[Ipacked]
        return MonoTable{PKey{T}}(keys, idx, packed)
    end

    depth = 0
    while depth < max_depth && !isempty(frontier) && length(packed_all) < max_size
        depth += 1
        show_progress && println("closure depth $depth: frontier=$(length(frontier)), total=$(length(packed_all))")

        new_frontier = UInt128[]
        sizehint!(new_frontier,
            min(max_size - length(packed_all),
                length(frontier) * max(length(Hkeys), 1)))

        @inbounds for p in frontier
            X0, Z0 = _unpack2x64(T, p)

            for hk in Hkeys
                X = xor(hk.x, X0)
                Z = xor(hk.z, Z0)

                if locality && locality_measure(X | Z) > LMAX
                    continue
                end

                p_new = _pack2x64(X, Z)
                if !haskey(seen, p_new)
                    seen[p_new] = true
                    push!(packed_all, p_new)
                    push!(new_frontier, p_new)
                    length(packed_all) >= max_size && break
                end

                length(packed_all) >= max_size && break
            end

            length(packed_all) >= max_size && break
        end

        frontier = new_frontier
    end

    show_progress && println("H-generated space: $(length(packed_all)) monomials (depth=$depth)")

    sort!(packed_all)
    N = length(packed_all)

    keys   = Vector{PKey{T}}(undef, N)
    packed = Vector{UInt128}(undef, N)
    idx    = Dict{UInt128,Int}(); sizehint!(idx, N)

    @inbounds for i in 1:N
        p = packed_all[i]
        keys[i]   = unpackkey(PKey{T}, p)
        packed[i] = p
        idx[p]    = i
    end

    return MonoTable{PKey{T}}(keys, idx, packed)
end

function H_generated_mono(
    H::MajoranaOperator{T};
    max_depth::Int      = 6,
    max_size::Int       = 800_000,
    show_progress::Bool = false,
    locality::Bool      = false,
    LMAX::Int           = typemax(Int),
    locality_measure::Function = (m::T)->count_ones(m),
) where {T<:Unsigned}

    Ikey = zero(T)
    Ipacked = packkey(Ikey)

    packed_all = UInt128[Ipacked]
    frontier   = UInt128[Ipacked]
    seen = Dict{UInt128,Bool}(Ipacked => true)

    Hkeys = collect(Base.keys(H.terms)) 
    if isempty(Hkeys)
        keys   = T[Ikey]
        idx    = Dict{UInt128,Int}(Ipacked => 1)
        packed = UInt128[Ipacked]
        return MonoTable{T}(keys, idx, packed)
    end

    depth = 0
    while depth < max_depth && !isempty(frontier) && length(packed_all) < max_size
        depth += 1
        show_progress && println("closure depth $depth: frontier=$(length(frontier)), total=$(length(packed_all))")

        new_frontier = UInt128[]
        sizehint!(new_frontier,
            min(max_size - length(packed_all),
                length(frontier) * max(length(Hkeys), 1)))

        @inbounds for p in frontier
            m0 = unpack_majorana(T, p)

            for hk in Hkeys
                m = xor(hk, m0)

                if locality && locality_measure(m) > LMAX
                    continue
                end

                p_new = UInt128(m)
                if !haskey(seen, p_new)
                    seen[p_new] = true
                    push!(packed_all, p_new)
                    push!(new_frontier, p_new)
                    length(packed_all) >= max_size && break
                end

                length(packed_all) >= max_size && break
            end

            length(packed_all) >= max_size && break
        end

        frontier = new_frontier
    end

    show_progress && println("H-generated space: $(length(packed_all)) monomials (depth=$depth)")

    sort!(packed_all)
    N = length(packed_all)

    keys   = Vector{T}(undef, N)
    packed = Vector{UInt128}(undef, N)
    idx    = Dict{UInt128,Int}(); sizehint!(idx, N)

    @inbounds for i in 1:N
        p = packed_all[i]
        keys[i]   = unpack_majorana(T, p)
        packed[i] = p
        idx[p]    = i
    end

    return MonoTable{T}(keys, idx, packed)
end




# Fast sparse dot on nzind lists (O(nnz(a)+nnz(b))), no hashing.
# Computes ∑ conj(a[i]) * b[i] over matching indices.

@inline function _sparsedot(a::OpVec, b::OpVec)::ComplexF64
    ia, va = a.nzind, a.nzval
    ib, vb = b.nzind, b.nzval
    i = 1; j = 1
    s = 0.0 + 0.0im
    @inbounds while i <= length(ia) && j <= length(ib)
        ai = ia[i]
        bj = ib[j]
        if ai == bj
            s += conj(va[i]) * vb[j]
            i += 1; j += 1
        elseif ai < bj
            i += 1
        else
            j += 1
        end
    end
    return s
end

# Majorana / Pauli: orthogonal monomial bases => dot product is enough.
# Majorana dagger includes a mask-dependent sign in H(op) definition,
# but it cancels in ⟨·,·⟩ so coefficients still pair as conj(c_a)*c_b. :contentReference[oaicite:1]{index=1}

struct NoCache end

@inline function inner_opvec(
    ::MonoTable{K}, ::NoCache, a::OpVec, b::OpVec; normalized::Bool=true
)::ComplexF64 where {K}
    # normalized flag irrelevant here because basis monomials are orthonormal under the trace convention
    return _sparsedot(a, b)
end

@inline function norm_opvec(
    B::MonoTable{K}, ::NoCache, v::OpVec
)::Float64 where {K}
    nnz(v) == 0 && return 0.0
    val = inner_opvec(B, NoCache(), v, v)
    return sqrt(max(real(val), 0.0))
end


struct FCache{T<:Unsigned}
    U  :: Vector{T}   # U[j]  = C|A
    PC :: Vector{T}   # PC[j] = C\A
    PA :: Vector{T}   # PA[j] = A\C
end

function FCache(B::MonoTable{FKey{T}}) where {T<:Unsigned}
    N = length(B.keys)
    U  = Vector{T}(undef, N)
    PC = Vector{T}(undef, N)
    PA = Vector{T}(undef, N)
    @inbounds for j in 1:N
        k = B.keys[j]
        C = k.c; A = k.a
        U[j]  = C | A
        PC[j] = C & ~A
        PA[j] = A & ~C
    end
    return FCache{T}(U, PC, PA)
end

# Same logic with Puretype
struct PureVec{T<:Unsigned}
    U    :: Vector{T}
    coef :: Vector{ComplexF64}
end

function _build_pure_buckets_vec(
    cache::FCache{T}, v::OpVec
) where {T<:Unsigned}
    ind = v.nzind
    val = v.nzval

    counts = Dict{NTuple{2,T}, Int}()
    @inbounds for j in ind
        key = (cache.PC[j], cache.PA[j])
        counts[key] = get(counts, key, 0) + 1
    end

    idx = Dict{NTuple{2,T}, PureVec{T}}()
    next_slot = Dict{NTuple{2,T}, Int}()
    @inbounds for (key, cnt) in counts
        idx[key] = PureVec{T}(Vector{T}(undef, cnt),
                                     Vector{ComplexF64}(undef, cnt))
        next_slot[key] = 1
    end

    @inbounds for (j, c) in zip(ind, val)
        key = (cache.PC[j], cache.PA[j])
        b   = idx[key]
        k   = next_slot[key]
        b.U[k]    = cache.U[j]
        b.coef[k] = c
        next_slot[key] = k + 1
    end
    return idx
end

# and the same weighting rule as your fermion inner() doc. :contentReference[oaicite:3]{index=3}
function inner_opvec(
    ::MonoTable{FKey{T}}, cache::FCache{T},
    a::OpVec, b::OpVec; normalized::Bool=true
)::ComplexF64 where {T<:Unsigned}

    (nnz(a) == 0 || nnz(b) == 0) && return 0.0 + 0.0im

    big, small = (nnz(a) >= nnz(b)) ? (a, b) : (b, a)
    conj_from_a_when_small = (small === a)

    idx = _build_pure_buckets_vec(cache, big)

    # Optional "unnormalized" scaling (your old behavior)
    scale = 1.0
    if !normalized
        total_mask = zero(T)
        @inbounds for j in a.nzind
            total_mask |= cache.U[j]
        end
        @inbounds for j in b.nzind
            total_mask |= cache.U[j]
        end
        Ltot = Int(count_ones(total_mask))
        scale = ldexp(1.0, Ltot)
    end

    s = 0.0 + 0.0im
    @inbounds for (jS, cS) in zip(small.nzind, small.nzval)
        key = (cache.PC[jS], cache.PA[jS])
        bkt = get(idx, key, nothing)
        bkt === nothing && continue

        U1 = cache.U[jS]
        Uv = bkt.U
        cv = bkt.coef

        if conj_from_a_when_small
            for k in eachindex(cv)
                m = Int(count_ones(U1 | Uv[k]))
                s += conj(cS) * cv[k] * _POW2_NEG[m + 1]
            end
        else
            for k in eachindex(cv)
                m = Int(count_ones(U1 | Uv[k]))
                s += conj(cv[k]) * cS * _POW2_NEG[m + 1]
            end
        end
    end

    return s * scale
end

@inline function norm_opvec(
    B::MonoTable{FKey{T}}, cache::FCache{T}, v::OpVec
)::Float64 where {T<:Unsigned}
    nnz(v) == 0 && return 0.0
    val = inner_opvec(B, cache, v, v)
    return sqrt(max(real(val), 0.0))
end




function weight(v::OpVec)
    s = 0.0
    @inbounds for c in v.nzval
        s += abs2(c)
    end
    s
end




"""
Project `w` onto the orthogonal complement of the *first* `nfirst` Krylov vectors in `basis`
using Modified Gram–Schmidt (optionally 2 passes), then prune.

Selective reorthogonalization to improve numerical stability, but mostly full reorthogonalization is needed for OPE.
"""
@inline function reorth(
    B::MonoTable{T}, cache::FCache{T},
    w::OpVec, basis::Vector{OpVec};
    nfirst::Int = 10,
    passes::Int = 2,
    inner_tol::Float64 = 1e-12,
    prune_tol2::Float64 = TOL2,
) where {T<:Unsigned}

    nnz(w) == 0 && return w
    m = min(nfirst, length(basis))
    m == 0 && return w

    inner_tol2 = inner_tol^2

    for _pass in 1:passes
        @inbounds for i in 1:m
            gi = inner_opvec(B, cache, basis[i], w)  # <q_i, w>
            abs2(gi) <= inner_tol2 && continue
            w = w .- (gi .* basis[i])
        end
        w = prune(w; tol2 = prune_tol2)
        nnz(w) == 0 && break
    end

    return w
end



struct OperatorSpace{O<:Operator}
    basis::Vector{O}
    ope::Array{Float64, 3}
    alphas::Vector{Float64}
    betas::Vector{Float64}
end

#Caching for Fermionic Operators Inner
krylov_cache(::FermionOperator{T}, B) where {T<:Unsigned} = FCache(B)
krylov_cache(::MajoranaOperator{T}, B) where {T<:Unsigned} = NoCache()
krylov_cache(::PauliOperator{T},   B) where {T<:Unsigned} = NoCache()


function krylov(H::O;
    tol::Float64 = 1e-8,
    max_power::Union{Int, Nothing} = nothing,
    show_progress::Bool = false,

    closure_depth::Union{Int, Nothing} = nothing,
    max_basis_size::Int = 100000000,

    enable_reorth::Bool = false,
    reorth_first::Int = 5,
    reorth_passes::Int = 2,
    reorth_inner_tol::Float64 = 1e-12,
) where {O<:Operator}

    mp = (max_power === nothing) ? typemax(Int) : max_power
    cd = (closure_depth === nothing) ? mp : closure_depth

    show_progress && println(">>> Building H-generated monomial space")
    B = H_generated_mono(H;
        max_depth     = cd,
        max_size      = max_basis_size,
        show_progress = show_progress
    )

    A = build_action_matrix(H, B; tol2 = 0.0)

    cache = krylov_cache(H, B)

    op0 = unit(H)

    basis_vecs = OpVec[]
    push!(basis_vecs, op_to_vector(op0, B))  # p₀

    alphas = Float64[]
    betas  = Float64[]

    if show_progress
        println("    k   terms   weight      norm")
        @printf("%2d: %6d %8.3f %9.3e\n",
                0,
                nnz(basis_vecs[1]),
                weight(basis_vecs[1]),
                norm_opvec(B, cache, basis_vecs[1]))
    end

    while length(alphas) < mp
        v_prev = basis_vecs[end]

        # 1) v_new = H * p_k
        v_new = apply_action(A, v_prev; show_progress = show_progress)

        # 2) α_k = ⟨p_k, v_new⟩
        αc = inner_opvec(B, cache, v_prev, v_new)
        α  = real(αc)

        # 3) 3-term recurrence orthogonalization
        v_new = v_new .- (αc .* v_prev)
        if !isempty(betas)
            v_new = v_new .- (betas[end] .* basis_vecs[end - 1])
        end

        v_new = prune(v_new; tol2 = TOL2)

        if enable_reorth
            v_new = reorth(B, cache, v_new, basis_vecs;
                nfirst     = reorth_first,
                passes     = reorth_passes,
                inner_tol  = reorth_inner_tol,
                prune_tol2 = TOL2,
            )
        end

        # 4) β_k = ||v_new||
        nrm = norm_opvec(B, cache, v_new)
        push!(alphas, α)

        show_progress && @printf("alpha=% .6e  norm=% .6e\n", α, nrm)

        if nrm < tol
            show_progress && println("Lanczos converged: ||p_k|| < tol at k=$(length(alphas))")
            break
        end

        push!(betas, nrm)
        v_new = (1 / nrm) .* v_new
        v_new = prune(v_new; tol2 = TOL2)

        if enable_reorth
            v_new = reorth(B, cache, v_new, basis_vecs;
                nfirst     = reorth_first,
                passes     = 1,
                inner_tol  = reorth_inner_tol,
                prune_tol2 = TOL2,
            )
        end

        push!(basis_vecs, v_new)

        if show_progress
            @printf("%2d: %6d %8.3f %9.3e\n",
                length(basis_vecs) - 1,
                nnz(v_new),
                weight(v_new),
                nrm)
        end
    end

    if length(basis_vecs) > length(alphas)
        pop!(basis_vecs)
    end

    # Convert back to operator struct (dispatches by algebra via vector_to_op)
    basis_out = Vector{O}(undef, length(basis_vecs))
    @inbounds for i in eachindex(basis_vecs)
        basis_out[i] = vector_to_op(B, basis_vecs[i])
    end

    # 6) Build tridiagonal T, then use three term recurrence relation on T to build OPE stack
    show_progress && (println("alphas = ", alphas); println("betas = ", betas))

    n = length(alphas)
    α = collect(alphas[1:n])
    β = (n > 1) ? collect(betas[1:n - 1]) : Float64[]

    Ttri = diagm(0 => α)
    if n > 1
        Ttri += diagm(1  => β)
        Ttri += diagm(-1 => β)
    end

    p_prev = zeros(Float64, n, n)
    p_curr = Matrix{Float64}(I, n, n)
    ope    = Vector{Matrix{Float64}}(undef, n)
    ope[1] = copy(p_curr)
    for k in 2:n
        rhs = Ttri * p_curr - α[k - 1] * p_curr
        if k > 2
            rhs .-= β[k - 2] .* p_prev
        end
        if abs(β[k - 1]) < 1e-12
            for kk in k:n
                ope[kk] = zeros(n, n)
            end
            break
        end
        p_next = rhs / β[k - 1]
        ope[k] = copy(p_next)
        p_prev, p_curr = p_curr, p_next
    end

    ope_stack = Array{Float64,3}(undef, n, n, n)
    @inbounds for k in 1:n
        ope_stack[:, :, k] = ope[k]
    end

    return OperatorSpace{O}(basis_out, ope_stack, α, β)
end

# project any operator onto the Krylov basis
function represent(space::OperatorSpace{O}, X::O)::Vector{ComplexF64} where {O<:Operator}
    y = ComplexF64[]
    @inbounds for p in space.basis
        push!(y, inner(p, X))
    end
    y
end
# reconstruct operator from coefficients in the Krylov basis
function reconstruct(space::OperatorSpace{O}, coeffs::AbstractVector{<:Real})::O where {O<:Operator}
    @assert length(coeffs) == length(space.basis)
    acc = zero(space.basis[1])  # returns O
    @inbounds for (c, p) in zip(coeffs, space.basis)
        acc = acc + c * p
    end
    acc
end


"""
Truncate operator by dropping all terms with locality measure above max_locality.
If max_locality is nothing, returns H unchanged.
Locality measure: l = r + Σ |site| where r is the number of operators.
Not working since breaks Krylov space closure, and basis orthognality.
"""
function truncate_by_locality(H::FermionOperator{T}, max_locality::Union{Int, Nothing}) where {T<:Unsigned}
    max_locality === nothing && return H
    
    truncated_terms = Dict{FKey{T}, ComplexF64}()
    dropped_count = 0
    kept_count = 0
    
    for (k, c) in H.terms
        support_mask = k.c | k.a
        loc = locality_measure(support_mask)
        
        if loc <= max_locality
            truncated_terms[k] = c
            kept_count += 1
        else
            dropped_count += 1
        end
    end
    
    if dropped_count > 0
        println("Truncated Hamiltonian: kept $kept_count terms, dropped $dropped_count terms with locality > $max_locality")
    end
    
    return FermionOperator{T}(truncated_terms)
end

# ------------------------- SDP Bootstrap by JuMP + SCS -------------------------
"""
Solve the single-block SDP:
minimize   c' x
subject to F0 + Σ_k x_k Fk  ⪰ 0

where Fk are from the OPE tensor of the Lanczos tridiagonal T.
Returns ρ (as a MajoranaOperator), along with (space, x, obj) if requested.
"""
function qboot(H::O; show_progress::Bool = false, tol::Float64 = 1e-5, max_power::Union{Int, Nothing} = nothing, closure_depth::Int = 1000, max_basis_size::Int = 10000000 , enable_reorth::Bool=false, reorth_first::Int=0) where {O<:Operator}

    space = krylov(H; tol = tol, max_power = max_power, show_progress = show_progress, closure_depth = closure_depth, max_basis_size = max_basis_size, enable_reorth = enable_reorth, reorth_first = reorth_first)
    n = length(space.basis)

    cvec = real.(represent(space, H))[2:end]            # length n-1
    F0 = space.ope[:, :, 1]                             # p0(T) = I
    Fks = [space.ope[:, :, k] for k in 2:n]            # p_k(T), k=1..n-1  (no minus)

    model = Model(SCS.Optimizer)
    if !show_progress
        set_silent(model)
    end
    # tighten SCS; default eps is loose and can give overly-aggressive minima
    set_optimizer_attribute(model, "eps_abs", 1e-8)
    set_optimizer_attribute(model, "eps_rel", 1e-8)
    set_optimizer_attribute(model, "max_iters", 1000000)
    set_optimizer_attribute(model, "scale", 1.0)
    set_optimizer_attribute(model, "acceleration_lookback", 20)

    @variable(model, x[1:n-1])
    @constraint(model, F0 + sum(x[k] * Fks[k] for k in 1:n-1) in PSDCone())

    @objective(model, Min, sum(cvec[k] * x[k] for k in 1:n-1))

    optimize!(model)

    xval = value.(x)
    coeffs = vcat(1.0, xval)            # prepend 1 (for p₀)
    rho = reconstruct(space, coeffs)  # rho::O (concrete)

    return rho
end

global locality = false
