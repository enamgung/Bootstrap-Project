using Base.Threads
import Base:zero
using Printf

if !isdefined(@__MODULE__, :AlgTag);      abstract type AlgTag end; end
if !isdefined(@__MODULE__, :MajoranaAlg); struct MajoranaAlg <: AlgTag end; end
if !isdefined(@__MODULE__, :FermionAlg);  struct FermionAlg <: AlgTag end; end
if !isdefined(@__MODULE__, :PauliAlg);  struct PauliAlg <: AlgTag end; end
if !isdefined(@__MODULE__, :Operator)
    struct Operator{A<:AlgTag,K}
        terms::Dict{K,ComplexF64}
    end
end


# Key: Fermionic monomial (∏ c†_i)(∏ c_j) 
struct FKey{T<:Unsigned}
    c::T   # creation mask
    a::T   # annihilation mask
end


# Key: Pauli monomial (∏ X_i)(∏ Z_j), (Y_i can be expressed by their product)
struct PKey{T<:Unsigned}
    x::T
    z::T
end

#Operator as dictionary from bitmask keys to complex coefficients
const MajoranaOperator = Operator{MajoranaAlg, T} where {T<:Unsigned}
const PauliOperator = Operator{PauliAlg, PKey{T}} where {T<:Unsigned}
const FermionOperator{T} = Operator{FermionAlg, FKey{T}} where {T<:Unsigned}

const TOL::Float64 = 1e-8
const TOL2::Float64 = TOL^2


#General inner product and norm for orthogonal term operators
function inner(a::Operator{A,K}, b::Operator{A,K}) where {A<:AlgTag,K}
    ta = a.terms; tb = b.terms
    small, big, conj_from_small =
        (length(ta) <= length(tb)) ? (ta, tb, true) : (tb, ta, false)

    result = 0.0 + 0.0im
    @inbounds for (k, cs) in small
        if haskey(big, k)
            cb = big[k]
            result += conj_from_small ? conj(cs) * cb : conj(cb) * cs
        end
    end
    return result
end

norm(op::Operator) = sqrt(sum(abs2, values(op.terms)))
Base.:(==)(a::Operator, b::Operator) = norm(a - b) < TOL

#Zeros and Ones for various unsigned types of bitmasks
z(::Type{UInt8}) = 0x00
z(::Type{UInt16}) = 0x0000
z(::Type{UInt32}) = 0x00000000
z(::Type{UInt64}) = 0x0000000000000000

o(::Type{UInt8}) = 0x01
o(::Type{UInt16}) = 0x0001
o(::Type{UInt32}) = 0x00000001
o(::Type{UInt64}) = 0x0000000000000001  



function commutate(A::Operator, B::Operator)
    C=A*B-B*A
    return C
end

function anticommutate(A::Operator, B::Operator)
    C=A*B+B*A
    return C
end

idkey(::Type{T}) where {T<:Unsigned} = zero(T)

idkey(::Type{PKey}) = PKey(0x0, 0x0)

# Trace: coefficient in front of identity, else 0 (fermion operator specified later) 
trace(op::Operator{A,K}) where {A<:AlgTag,K} =
    get(op.terms, idkey(K), 0.0 + 0.0im)


#------------------------- Pauli Operator -------------------------
Base.:(==)(a::PKey{T}, b::PKey{T}) where {T<:Unsigned} = (a.x == b.x) & (a.z == b.z)
Base.hash(k::PKey{T}, h::UInt) where {T<:Unsigned} = hash((k.x, k.z), h)


# Identity / zero
unit(::Type{PauliOperator{T}}) where {T<:Unsigned} =
    PauliOperator{T}(Dict(PKey{T}(zero(T), zero(T)) => 1.0 + 0im))
unit(op::PauliOperator{T}) where {T<:Unsigned} = unit(PauliOperator{T})
unit(::Type{PauliOperator}) = unit(PauliOperator{UInt64})

zero(::Type{PauliOperator{T}}) where {T<:Unsigned} =
    PauliOperator{T}(Dict{PKey{T},ComplexF64}())
zero(op::PauliOperator{T}) where {T<:Unsigned} = zero(PauliOperator{T})
zero(::Type{PauliOperator}) = zero(PauliOperator{UInt64})


# Single-site Paulis (index convention: bit j; use j-1 if you prefer 1-based sites)
function px(::Type{T}, js::Vararg{Int,N}) where {T<:Unsigned,N}
    x = zero(T)
    @inbounds for j in js
        x ⊻= (one(T) << j)          # duplicates cancel
    end
    return PauliOperator{T}(Dict(PKey{T}(x, zero(T)) => 1.0 + 0im))
end
px(js::Vararg{Int,N}) where {N} = px(UInt64, js...)
px(::Type{T}, v::AbstractVector{<:Integer}) where {T<:Unsigned} = px(T, (Int.(v))...)
px(v::AbstractVector{<:Integer}) = px(UInt64, v)

function pz(::Type{T}, js::Vararg{Int,N}) where {T<:Unsigned,N}
    z = zero(T)
    @inbounds for j in js
        z ⊻= (one(T) << j)
    end
    return PauliOperator{T}(Dict(PKey{T}(zero(T), z) => 1.0 + 0im))
end
pz(js::Vararg{Int,N}) where {N} = pz(UInt64, js...)
pz(::Type{T}, v::AbstractVector{<:Integer}) where {T<:Unsigned} = pz(T, (Int.(v))...)
pz(v::AbstractVector{<:Integer}) = pz(UInt64, v)

function py(::Type{T}, js::Vararg{Int,N}) where {T<:Unsigned,N}
    m = zero(T)
    @inbounds for j in js
        m ⊻= (one(T) << j)
    end
    return PauliOperator{T}(Dict(PKey{T}(m, m) => 1.0 + 0im))
end
py(js::Vararg{Int,N}) where {N} = py(UInt64, js...)
py(::Type{T}, v::AbstractVector{<:Integer}) where {T<:Unsigned} = py(T, (Int.(v))...)
py(v::AbstractVector{<:Integer}) = py(UInt64, v)


# Print "Xj / Yj / Zj" in ascending site order
function _pmask_repr(k::PKey{T}) where {T<:Unsigned}
    (k.x == zero(T) && k.z == zero(T)) && return "I"
    parts = String[]
    m = k.x | k.z
    while m != zero(T)
        j  = trailing_zeros(m)
        xj = ((k.x >> j) & one(T)) == one(T)
        zj = ((k.z >> j) & one(T)) == one(T)
        push!(parts, xj && zj ? "Y$(j)" : xj ? "X$(j)" : "Z$(j)")
        m &= m - one(T)
    end
    return join(parts, ' ')
end

_coef_prefix(c::ComplexF64) =
    isapprox(c, 1+0im;  atol=TOL, rtol=TOL) ? ""  :
    isapprox(c, -1+0im; atol=TOL, rtol=TOL) ? "-" :
    isapprox(c, 0+1im;  atol=TOL, rtol=TOL) ? "i" :
    isapprox(c, 0-1im;  atol=TOL, rtol=TOL) ? "-i" : string(c, " ")

function Base.show(io::IO, ::MIME"text/plain", op::PauliOperator{T}) where {T<:Unsigned}
    terms = op.terms
    if isempty(terms); print(io, "0"); return; end
    ks = sort!(collect(keys(terms)); by = k -> (k.x, k.z))
    pieces = String[]
    for k in ks
        c = terms[k]
        abs(c) < TOL && continue
        pref = _coef_prefix(c)
        txt  = _pmask_repr(k)
        if txt == "I"
            push!(pieces, pref == "" ? "I" : pref*"I")
        else
            if pref == ""
                push!(pieces, txt)
            elseif pref == "-"
                push!(pieces, "-"*txt)
            else
                push!(pieces, pref * txt)
            end
        end
    end
    print(io, isempty(pieces) ? "0" : join(pieces, " "))
end
Base.show(io::IO, op::PauliOperator{T}) where {T<:Unsigned} = show(io, MIME"text/plain"(), op)


Base.:*(a::Number, op::PauliOperator{T}) where {T<:Unsigned} = PauliOperator{T}(Dict(k => ComplexF64(a)*c for (k,c) in op.terms))
Base.:*(op::PauliOperator{T}, a::Number) where {T<:Unsigned} = a*op
Base.:/(op::PauliOperator{T}, a::Number) where {T<:Unsigned} = (1/a)*op
Base.:-(op::PauliOperator{T}) where {T<:Unsigned} = PauliOperator{T}(Dict(k => -c for (k,c) in op.terms))


# op + op  (merge dicts; drop near-zeros)
function Base.:+(a::PauliOperator{T}, b::PauliOperator{T})::PauliOperator{T} where {T<:Unsigned}
    isempty(a.terms) && return b
    isempty(b.terms) && return a
    A, B = (length(a.terms) >= length(b.terms)) ? (a.terms, b.terms) : (b.terms, a.terms)
    out = Dict{PKey{T},ComplexF64}(A)
    @inbounds for (m, c) in B
        v = get(out, m, 0.0 + 0im) + c
        if abs(v) < TOL
            haskey(out, m) && delete!(out, m)
        else
            out[m] = v
        end
    end
    return PauliOperator{T}(out)
end
Base.:-(a::PauliOperator{T}, b::PauliOperator{T}) where {T<:Unsigned} = a + (-b)

# Conjugation: Pauli monomials are Hermitian in this embedding; only conj the coeffs.
function conj_op(op::PauliOperator{T}) where {T<:Unsigned}
    out = Dict{PKey{T},ComplexF64}()
    for (m,c) in op.terms
        out[m] = conj(c)
    end
    PauliOperator{T}(out)
end
H(op::PauliOperator{T}) where {T<:Unsigned} = conj_op(op)

function weight(op::PauliOperator{T}; weight_func::Function = k->count_ones(k.x | k.z)) where {T<:Unsigned}
    numer = 0.0; denom = 0.0
    for (m,c) in op.terms
        p = abs2(c); numer += weight_func(m) * p; denom += p
    end
    return denom > 0 ? numer/denom : 0.0
end

# Pauli sign calculation
@inline function _pauli_phase(a::PKey{T}, b::PKey{T})::ComplexF64 where {T<:Unsigned}
    ax, az = a.x, a.z
    bx, bz = b.x, b.z

    e  = Int(count_ones(az & bx)) - Int(count_ones(ax & bz))
    e += 2*Int(count_ones((ax & az) & (bx ⊻ bz)))
    e += 2*Int(count_ones((bx & bz) & (ax ⊻ az)))

    r = mod(e, 4)
    return r == 0 ? (1.0 + 0im) :
           r == 1 ? (0.0 + 1im) :
           r == 2 ? (-1.0 + 0im) :
                    (0.0 - 1im)
end

function _pmul_small(a::PauliOperator{T}, b::PauliOperator{T})::PauliOperator{T} where {T<:Unsigned}
    isempty(a.terms) && return zero(a)
    isempty(b.terms) && return zero(a)

    na, nb = length(a.terms), length(b.terms)
    masks = Vector{PKey{T}}(undef, na*nb)
    coefs = Vector{ComplexF64}(undef, na*nb)

    k = 1
    @inbounds for (ma, ca) in a.terms
        for (mb, cb) in b.terms
            c = ca * cb * _pauli_phase(ma, mb)
            if abs(c) >= TOL
                masks[k] = PKey{T}(xor(ma.x, mb.x), xor(ma.z, mb.z))
                coefs[k] = c
                k += 1
            end
        end
    end
    k -= 1
    k == 0 && return zero(a)
    resize!(masks, k); resize!(coefs, k)

    p = sortperm(1:k; by = i -> (masks[i].x, masks[i].z))
    masks = masks[p]; coefs = coefs[p]

    out_k = PKey{T}[]; out_c = ComplexF64[]
    last = masks[1]; acc = coefs[1]
    @inbounds for i in 2:k
        m = masks[i]; c = coefs[i]
        if m.x == last.x && m.z == last.z
            acc += c
        else
            if abs(acc) >= TOL; push!(out_k, last); push!(out_c, acc); end
            last = m; acc = c
        end
    end
    if abs(acc) >= TOL; push!(out_k, last); push!(out_c, acc); end

    return PauliOperator{T}(Dict(out_k .=> out_c))
end

function _pmul_threaded(a::PauliOperator{T}, b::PauliOperator{T})::PauliOperator{T} where {T<:Unsigned}
    isempty(a.terms) && return zero(a)
    isempty(b.terms) && return zero(a)

    A  = collect(a.terms)                 # Vector{Pair{PKey{T},ComplexF64}}
    Bk = collect(keys(b.terms))           # Vector{PKey{T}}
    Bv = ComplexF64[b.terms[k] for k in Bk]

    nt = max(nthreads(), 1)
    partials = [Dict{PKey{T},ComplexF64}() for _ in 1:nt]

    Threads.@threads for i in 1:length(A)
        d = partials[threadid()]
        ma = A[i].first
        ca = A[i].second

        @inbounds for j in 1:length(Bk)
            coef = ca * Bv[j] * _pauli_phase(ma, Bk[j])
            if abs(coef) >= TOL
                m = PKey{T}(xor(ma.x, Bk[j].x), xor(ma.z, Bk[j].z))
                d[m] = get(d, m, 0.0 + 0im) + coef
            end
        end
    end

    out = Dict{PKey{T},ComplexF64}()
    @inbounds for d in partials
        for (m, c) in d
            v = get(out, m, 0.0 + 0im) + c
            if abs(v) < TOL
                haskey(out, m) && delete!(out, m)
            else
                out[m] = v
            end
        end
    end
    return PauliOperator{T}(out)
end

function Base.:*(a::PauliOperator{T}, b::PauliOperator{T}) where {T<:Unsigned}
    W = length(a.terms) * length(b.terms)
    return (W <= 50_000) ? _pmul_small(a, b) : _pmul_threaded(a, b)
end



# ------------------------- Majorana Operator -------------------------
# Calculate sign by number of inversion, (-1)^(# of pairs (i in a, j in b) with j < i)
@inline function majorana_sign(a::T, b::T)::Float64 where {T<:Unsigned}
    crosses = 0
    aa = a
    @inbounds while aa != zero(T)
        tz = trailing_zeros(aa)  
        lower = (tz == 0) ? zero(T) : ((one(T) << tz) - one(T))
        crosses += count_ones(b & lower)
        aa &= aa - one(T)
    end
    return isodd(crosses) ? -1.0 : 1.0
end


# conjugation sign for a Majorana monomial mask
@inline function _conj_sign(m::Unsigned)::Float64
    ((count_ones(m) ÷ 2) % 2 == 0) ? 1.0 : -1.0
end

# ---------- identity and zero ----------
unit(::Type{MajoranaOperator{T}}) where {T<:Unsigned} =
    MajoranaOperator{T}(Dict{T,ComplexF64}(zero(T) => 1.0 + 0im))
unit(op::MajoranaOperator{T}) where {T<:Unsigned} = unit(MajoranaOperator{T})
unit(::Type{MajoranaOperator}) = unit(MajoranaOperator{UInt64})

zero(::Type{MajoranaOperator{T}}) where {T<:Unsigned} =
    MajoranaOperator{T}(Dict{T,ComplexF64}())
zero(op::MajoranaOperator{T}) where {T<:Unsigned} = zero(MajoranaOperator{T})
zero(::Type{MajoranaOperator}) = zero(MajoranaOperator{UInt64})


# ---------- constructors ----------
function maj(::Type{T}, ms::Vararg{Int, N}) where {T<:Unsigned, N}
    mask = zero(T)
    @inbounds for i in ms
        mask |= (one(T) << i)
    end
    return MajoranaOperator{T}(Dict{T,ComplexF64}(mask => 1.0 + 0im))
end

# default: UInt64
maj(ms::Vararg{Int, N}) where {N} = maj(UInt64, ms...)

# vector forms
maj(::Type{T}, v::AbstractVector{<:Integer}) where {T<:Unsigned} = maj(T, (Int.(v))...)
maj(v::AbstractVector{<:Integer}) = maj(UInt64, v)


#  print a mask → "χj" ascending order
function _mask_repr(mask::T) where {T<:Unsigned}
    mask == zero(T) && return "I"
    parts = String[]
    m = mask
    while m != zero(T)
        j = trailing_zeros(m)
        push!(parts, "χ$(j)")
        m &= m - one(T)
    end
    return join(parts, ' ')
end

# coefficient text formatting 
function _coef_repr(c::ComplexF64)
    if iszero(imag(c))
        r = real(c)
        if r == round(r)
            r ==  1.0 && return " "
            r == -1.0 && return "- "
            return string(Integer(round(r)), ' ')
        else
            return @sprintf("%.3g ", r)
        end
    elseif iszero(real(c))
        i = imag(c)
        if i == round(i)
            i ==  1.0 && return "i "
            i == -1.0 && return "-i "
            return string(Integer(round(i)), "i ")
        else
            return @sprintf("%.3gi ", i)
        end
    else
        s = @sprintf("(%g%+gi) ", real(c), imag(c))
        return replace(s, "im" => "i")
    end
end


# ------------------------- Display -------------------------
function Base.show(io::IO, ::MIME"text/plain", op::MajoranaOperator{T}) where {T<:Unsigned}
    if isempty(op.terms); print(io, "0"); return; end
    ks = sort!(collect(keys(op.terms)))
    txt = IOBuffer()
    first_term = true
    for k in ks
        c = op.terms[k]
        abs(c) < TOL && continue
        ctxt = _coef_repr(c)
        ttxt = _mask_repr(k)
        if !first_term && !startswith(ctxt, '-')
            print(txt, '+')
        end
        print(txt, ctxt, ttxt)
        first_term = false
    end
    print(io, strip(String(take!(txt))))
end

function Base.show(io::IO, op::MajoranaOperator{T}) where {T<:Unsigned}
    show(io, MIME"text/plain"(), op)
end

Base.isempty(op::MajoranaOperator{T}) where {T<:Unsigned} = isempty(op.terms)
Base.length(op::MajoranaOperator{T})  where {T<:Unsigned} = length(op.terms)

# equality to 0 for convenience
Base.:(==)(op::MajoranaOperator, x::Integer) = x == 0 && isempty(op.terms)

# scalar mul/div 
function Base.:*(op::MajoranaOperator{T}, s::Number) where {T<:Unsigned}
    abs2(s) < TOL2 && return zero(op)
    out = Dict{T,ComplexF64}()
    for (m,c) in op.terms
        out[m] = c * ComplexF64(s)
    end
    return MajoranaOperator{T}(out)
end
Base.:*(s::Number, op::MajoranaOperator{T}) where {T<:Unsigned} = op * s
Base.:/(op::MajoranaOperator{T}, s::Number) where {T<:Unsigned} = op * (1/s)
Base.:-(op::MajoranaOperator{T}) where {T<:Unsigned} = op * (-1)

#addition/subtraction
function sum_maj(a::MajoranaOperator{T}, b::MajoranaOperator{T}, sgn::Int) where {T<:Unsigned}
    out = copy(a.terms)
    for (m,c) in b.terms
        v = get(out, m, 0.0 + 0im) + sgn*c
        if abs2(v) < TOL2
            haskey(out, m) && delete!(out, m)
        else
            out[m] = v
        end
    end
    return MajoranaOperator{T}(out)
end

function Base.:+(a::MajoranaOperator{T}, b::MajoranaOperator{T}) where {T<:Unsigned}
    isempty(b.terms) && return a
    isempty(a.terms) && return b
    return sum_maj(a, b, +1)
end

Base.:-(a::MajoranaOperator{T}, b::MajoranaOperator{T}) where {T<:Unsigned} = sum_maj(a, b, -1)

# number + operator, operator + number: defined by adding identity * number
Base.:+(op::MajoranaOperator{T}, s::Number) where {T<:Unsigned} = op + (s*unit(op))
Base.:+(s::Number, op::MajoranaOperator{T}) where {T<:Unsigned} = op + s
Base.:-(op::MajoranaOperator{T}, s::Number) where {T<:Unsigned} = op + (-s)
Base.:-(s::Number, op::MajoranaOperator{T}) where {T<:Unsigned} = (-op) + s

# ---------- conjugation / Hermitian ----------
function conj_op(op::MajoranaOperator{T}) where {T<:Unsigned}
    out = Dict{T,ComplexF64}()
    for (m,c) in op.terms
        out[m] = _conj_sign(m) * conj(c)
    end
    MajoranaOperator{T}(out)
end
H(op::MajoranaOperator{T}) where {T<:Unsigned} = conj_op(op)

Base.real(op::MajoranaOperator{T}) where {T<:Unsigned} = (op + H(op)) / 2
Base.imag(op::MajoranaOperator{T}) where {T<:Unsigned} = (op - H(op)) / (-2im)

# default weight: number of set bits
function weight(op::MajoranaOperator; weight_func::Function = m->count_ones(m))
    numer = 0.0
    denom = 0.0
    for (m,c) in op.terms
        p = abs2(c)
        numer += weight_func(m) * p
        denom += p
    end
    return denom > 0 ? numer/denom : 0.0
end



# ---------- Fast sign via prefix-parity mask ----------
# For each bit position i, the prefix-parity mask has bit i = 1
# iff the number of set bits in b with index < i is odd.
@inline function _prefix_parity_mask(b::T)::T where {T<:Unsigned}
    p = zero(T)
    bb = b
    @inbounds while bb != zero(T)
        j = trailing_zeros(bb)
        p ⊻= (typemax(T) << (j + 1))
        bb &= bb - one(T)
    end
    return p
end

const _PP_CACHE64 = Dict{UInt64,UInt64}()
const _PP_LOCK    = ReentrantLock()

@inline function _prefix_parity_mask_cached(b::T)::T where {T<:Unsigned}
    if T === UInt64
        bb = UInt64(b)
        m = get(_PP_CACHE64, bb, 0x0)
        if m != 0x0 || bb == 0x0
            return T(m)
        end
        m = _prefix_parity_mask(bb)
        lock(_PP_LOCK)
        try
            _PP_CACHE64[bb] = m
        finally
            unlock(_PP_LOCK)
        end
        return T(m)
    else
        return _prefix_parity_mask(b)
    end
end

@inline _pair_sign_from_pp(a::T, pp_b::T)where {T<:Unsigned} =
    isodd(count_ones(a & pp_b)) ? -1.0 : 1.0

@inline is_even_deg(mask::T) where {T<:Unsigned} = iseven(count_ones(mask))

function spin_bitselectors(::Type{T}, L::Int) where {T<:Unsigned}
    up = zero(T); dn = zero(T)
    @inbounds for i in 0:L-1
        up |= (one(T) << (4*i+0)) | (one(T) << (4*i+1))
        dn |= (one(T) << (4*i+2)) | (one(T) << (4*i+3))
    end
    return up, dn
end
spin_bitselectors(L::Int) = spin_bitselectors(UInt64, L)

# ---------- filters ----------
function filter_even_parity!(op::MajoranaOperator{T}; want_even::Bool=true) where {T<:Unsigned}
    d = op.terms
    for (m, _) in collect(d)
        keep = is_even_deg(m)
        want_even ? (keep || delete!(d,m)) : (keep && delete!(d,m))
    end
    op
end

@inline _parity_sel(mask::T, sel::T) where {T<:Unsigned} = isodd(count_ones(mask & sel))

function filter_spin_parity!(op::MajoranaOperator{T};
    up_bits::T, dn_bits::T,
    want_up_odd::Bool=false, want_dn_odd::Bool=false) where {T<:Unsigned}
    d = op.terms
    for (m, _) in collect(d)
        up_ok = (_parity_sel(m, up_bits) == want_up_odd)
        dn_ok = (_parity_sel(m, dn_bits) == want_dn_odd)
        (up_ok && dn_ok) || delete!(d, m)
    end
    op
end

function _mmul_small(a::MajoranaOperator{T}, b::MajoranaOperator{T})::MajoranaOperator{T} where {T<:Unsigned}
    isempty(a.terms) && return zero(a)
    isempty(b.terms) && return zero(a)

    A  = collect(a.terms)                          # Vector{Pair{T,ComplexF64}}
    Bk = collect(keys(b.terms))
    Bv = ComplexF64[b.terms[k] for k in Bk]
    PPB = T[_prefix_parity_mask_cached(k) for k in Bk]

    na, nb = length(A), length(Bk)
    masks = Vector{T}(undef, na*nb)
    coefs = Vector{ComplexF64}(undef, na*nb)

    k = 1
    @inbounds for i in 1:na
        ma = A[i].first; ca = A[i].second
        for j in 1:nb
            mb = Bk[j]; cb = Bv[j]
            c2 = ca * cb
            abs2(c2) < TOL2 && continue
            s  = _pair_sign_from_pp(ma, PPB[j])
            masks[k] = xor(ma, mb)
            coefs[k] = c2 * s
            k += 1
        end
    end
    k -= 1
    k <= 0 && return zero(a)
    resize!(masks, k); resize!(coefs, k)

    p = sortperm(masks; alg=QuickSort)
    masks .= @view masks[p]; coefs .= @view coefs[p]

    out_m = T[]; out_c = ComplexF64[]
    last_m = masks[1]; acc = coefs[1]
    @inbounds for i in 2:k
        m = masks[i]; c = coefs[i]
        if m == last_m
            acc += c
        else
            if abs2(acc) >= TOL2
                push!(out_m, last_m); push!(out_c, acc)
            end
            last_m = m; acc = c
        end
    end
    if abs2(acc) >= TOL2
        push!(out_m, last_m); push!(out_c, acc)
    end
    return MajoranaOperator{T}(Dict(out_m .=> out_c))
end


function _mmul_threaded(a::MajoranaOperator{T}, b::MajoranaOperator{T})::MajoranaOperator{T} where {T<:Unsigned}
    isempty(a.terms) && return zero(a)
    isempty(b.terms) && return zero(a)

    A  = collect(a.terms)          # Pair{T,ComplexF64}[]
    Bk = collect(keys(b.terms))
    Bv = ComplexF64[b.terms[k] for k in Bk]
    PPB = T[_prefix_parity_mask_cached(k) for k in Bk]

    nt = max(nthreads(), 1)
    partials = [Dict{T,ComplexF64}() for _ in 1:nt]

    Threads.@threads for i in 1:length(A)
        d = partials[threadid()]
        ma = A[i].first; ca = A[i].second

        @inbounds for j in 1:length(Bk)
            c2 = ca * Bv[j]
            abs2(c2) < TOL2 && continue

            coef = c2 * _pair_sign_from_pp(ma, PPB[j])
            m = xor(ma, Bk[j])

            d[m] = get(d, m, 0.0 + 0im) + coef
        end
    end

    out = Dict{T,ComplexF64}()
    for d in partials
        for (m,c) in d
            v = get(out, m, 0.0 + 0im) + c
            if abs2(v) < TOL2
                haskey(out, m) && delete!(out, m)
            else
                out[m] = v
            end
        end
    end
    return MajoranaOperator{T}(out)
end

function Base.:*(a::MajoranaOperator{T}, b::MajoranaOperator{T}) where {T<:Unsigned}
    W = length(a.terms) * length(b.terms)
    if W <= 1000000
        return _mmul_small(a, b)
    else
        # example default cfg (set yours)
        return _mmul_threaded(a, b)
    end
end

# ------------------------- Fermion Operator -------------------------
#  identity and zero 
unit(::Type{FermionOperator{T}}) where {T<:Unsigned} =
    FermionOperator{T}(Dict(FKey{T}(z(T), z(T)) => 1.0 + 0im))
unit(op::FermionOperator{T}) where {T} = unit(FermionOperator{T})
unit(::Type{FermionOperator}) = unit(FermionOperator{UInt64})


zero(::Type{FermionOperator{T}}) where {T<:Unsigned} =
    FermionOperator{T}(Dict{FKey{T}, ComplexF64}())
zero(op::FermionOperator{T}) where {T} = zero(FermionOperator{T})
zero(::Type{FermionOperator}) = zero(FermionOperator{UInt64})


# ------------- Constructors -------------
@inline addbit(::Type{T}, j::Integer) where {T<:Unsigned} = (one(T) << j)::T
bit(j)::UInt64 = UInt64(1) << UInt64(j)


# UInt64 as defualt unless specified
function c(::Type{T}, js::Vararg{Int, N}) where {T<:Unsigned, N}
    a = zero(T)
    @inbounds for j in js
        a |= addbit(T, j)
    end
    return FermionOperator{T}(Dict(FKey{T}(zero(T), a) => 1.0 + 0im))
end

c(js::Vararg{Int, N}) where {N} = c(UInt64, js...)
c(v::AbstractVector{<:Integer}) = c(UInt64, v)


function cdag(::Type{T}, js::Vararg{Int, N}) where {T<:Unsigned, N}
    c_mask = zero(T)
    @inbounds for j in js
        c_mask |= addbit(T, j)
    end
    return FermionOperator{T}(Dict(FKey{T}(c_mask, zero(T)) => 1.0 + 0im))
end

cdag(js::Vararg{Int, N}) where {N} = cdag(UInt64, js...)
cdag(v::AbstractVector{<:Integer}) = cdag(UInt64, v)


function n(::Type{T}, js::Vararg{Int, N}) where {T<:Unsigned, N}
    m = zero(T)
    @inbounds for j in js
        m |= addbit(T, j)
    end
    return FermionOperator{T}(Dict(FKey{T}(m, m) => 1.0 + 0im))
end

n(js::Vararg{Int, N}) where {N} = n(UInt64, js...)
n(v::AbstractVector{<:Integer}) = n(UInt64, v)


#  ac = (1 - n) expanded 
function ac(::Type{T}, js::Vararg{Int, N}) where {T<:Unsigned, N}
    m = zero(T)
    @inbounds for j in js
        m |= addbit(T, j)
    end
    return FermionOperator{T}(Dict(
        FKey{T}(zero(T), zero(T)) =>  1.0 + 0im,  # I
        FKey{T}(m, m)             => -1.0 + 0im   # -n
    ))
end

ac(js::Vararg{Int, N}) where {N} = ac(UInt64, js...)
ac(v::AbstractVector{<:Integer}) = ac(UInt64, v)

# Ascending order by sites
function _fmask_repr(k::FKey{T}) where {T<:Unsigned}
    (k.c | k.a) == zero(T) && return "I"
    parts = String[]

    n = k.c & k.a     
    cc = k.c & ~n      
    aa = k.a & ~n     

    m = cc | aa | n  

    while m != zero(T)
        j = trailing_zeros(m)                
        bit = one(T) << j

        (cc & bit) != zero(T) && push!(parts, "c†$(j)")
        (aa & bit) != zero(T) && push!(parts, "c$(j)")
        (n & bit) != zero(T) && push!(parts, "n$(j)")

        m &= m - one(T)                      
    end

    return join(parts, ' ')
end

function Base.show(io::IO, ::MIME"text/plain", op::FermionOperator{T}) where {T<:Unsigned}
    terms = op.terms
    if isempty(terms)
        print(io, "0")
        return
    end
    ks = sort!(collect(keys(terms)); by = k -> (k.c, k.a))
    first_term = true
    for k in ks
        c = terms[k]
        abs(c) < TOL && continue
        mono = _fmask_repr(k)
        pref = isapprox(c, 1 + 0im; atol = TOL, rtol = TOL) ? "" :
                isapprox(c, -1 + 0im; atol = TOL, rtol = TOL) ? "-" :
                isapprox(c, 0 + 1im; atol = TOL, rtol = TOL) ? "i" :
                isapprox(c, 0 - 1im; atol = TOL, rtol = TOL) ? "-i" : string(c, " ")
        term = mono == "I" ? (pref == "" ? "I" : pref * "I") :
                pref == "" ? mono :
                pref == "-" ? "-" * mono : pref * mono
        if first_term
            print(io, term)
            first_term = false
        else
            print(io, startswith(term, "-") ? " - $(term[2:end])" : " + $term")
        end
    end
    first_term && print(io, "0")
end
Base.show(io::IO, op::FermionOperator{T}) where {T<:Unsigned} = show(io, MIME"text/plain"(), op)



# Coefficient prefix: "", "-", "i", "-i", or general number
_coef_prefix(c::ComplexF64) =
    isapprox(c, 1 + 0im; atol = TOL, rtol = TOL) ? "" :
    isapprox(c, -1 + 0im; atol = TOL, rtol = TOL) ? "-" :
    isapprox(c, 0 + 1im; atol = TOL, rtol = TOL) ? "i" :
    isapprox(c, 0 - 1im; atol = TOL, rtol = TOL) ? "-i" : string(c, " ")

function Base.show(io::IO, ::MIME"text/plain", op::FermionOperator)
    terms = op.terms
    if isempty(terms)
        print(io, "0")
        return
    end
    ks = sort!(collect(keys(terms)); by = k -> (k.c, k.a))

    printed_any = false
    for k in ks
        coeff = terms[k]
        abs(coeff) < TOL && continue

        # monomial text
        mono = _fmask_repr(k) 

        # reduce coeff to a printable prefix and magnitude sign
        neg = real(coeff) < 0 || (iszero(real(coeff)) && imag(coeff) < 0)
        pref = _coef_prefix(coeff)  # "", "-", "i", "-i", or general
        # normalize: take sign out of the separator if you like
        term = mono == "I" ? (pref == "" ? "I" : pref * "I") :
                pref == "" ? mono :
                pref == "-" ? "-" * mono : pref * mono

        if !printed_any
            # first term: print as-is
            print(io, term)
            printed_any = true
        else
            # subsequent terms: insert explicit +/−
            if startswith(term, "-")
                print(io, " - ", term[2:end])
            else
                print(io, " + ", term)
            end
        end
    end

    printed_any || print(io, "0")
end

Base.isempty(op::FermionOperator{T}) where {T} = isempty(op.terms)
Base.length(op::FermionOperator{T}) where {T} = length(op.terms)

function Base.:*(op::FermionOperator{T}, s::Number) where {T<:Unsigned}
    abs(s) < TOL && return zero(op)
    FermionOperator{T}(Dict(k => ComplexF64(s) * c for (k, c) in op.terms))
end
Base.:*(s::Number, op::FermionOperator{T}) where {T<:Unsigned} = op * s
Base.:/(op::FermionOperator{T}, s::Number) where {T<:Unsigned} = op * (1 / s)
Base.:-(op::FermionOperator{T}) where {T<:Unsigned} = (-1) * op

@inline function _combine_terms!(out::Dict{FKey{T}, ComplexF64},
                                    src::Dict{FKey{T}, ComplexF64},
                                    α::ComplexF64) where {T<:Unsigned}
    @inbounds for (k, c) in src
        v = get(out, k, 0.0 + 0im) + α * c
        if abs(v) < TOL
            haskey(out, k) && delete!(out, k)
        else
            out[k] = v
        end
    end
    out
end

function Base.:+(a::FermionOperator{T}, b::FermionOperator{T})::FermionOperator{T} where {T<:Unsigned}
    isempty(a.terms) && return FermionOperator{T}(copy(b.terms))
    isempty(b.terms) && return FermionOperator{T}(copy(a.terms))
    if length(a.terms) >= length(b.terms)
        out = Dict(a.terms)
        _combine_terms!(out, b.terms, 1.0 + 0im)
    else
        out = Dict(b.terms)
        _combine_terms!(out, a.terms, 1.0 + 0im)
    end
    FermionOperator{T}(out)
end

function Base.:-(a::FermionOperator{T}, b::FermionOperator{T})::FermionOperator{T} where {T<:Unsigned}
    isempty(b.terms) && return FermionOperator{T}(copy(a.terms))
    if isempty(a.terms)
        out = Dict{FKey{T}, ComplexF64}()
        _combine_terms!(out, b.terms, -1.0 + 0im)
        return FermionOperator{T}(out)
    end
    out = Dict(a.terms)
    _combine_terms!(out, b.terms, -1.0 + 0im)
    FermionOperator{T}(out)
end



function weight(op::FermionOperator; weight_func::Function = k -> count_ones(k.c) + count_ones(k.a))
    numer = 0.0
    denom = 0.0
    for (m, c) in op.terms
        p = abs2(c)
        numer += weight_func(m) * p
        denom += p
    end
    return denom > 0 ? numer / denom : 0.0
end


# Combine terms with same keys and drop small terms (generic in T)
function combine_like_terms(terms::Dict{FKey{T}, ComplexF64}) where {T<:Unsigned}
    out = Dict{FKey{T}, ComplexF64}()
    sizehint!(out, length(terms))
    @inbounds for (k, c) in terms
        abs(c) < TOL && continue
        v = get(out, k, 0.0 + 0im) + c
        if abs(v) >= TOL
            out[k] = v
        elseif haskey(out, k)
            delete!(out, k)
        end
    end
    out

end

# Struct keyed by (c-only, a-only) in each monomial
struct Puretypes{T<:Unsigned}
    U :: Vector{T}            # union mask = C|A (keeps n-sites too)
    coef :: Vector{ComplexF64}
end

function build_puretypes(big_terms::Dict{FKey{T}, ComplexF64}) where {T<:Unsigned}
    # First pass: counts per (Pc, Pa) where Pc = C\A, Pa = A\C  (same-type matching)
    counts = Dict{NTuple{2, T}, Int}()
    @inbounds for (k, _) in big_terms
        C = k.c
        A = k.a
        key = (C & ~A, A & ~C)
        counts[key] = get(counts, key, 0) + 1
    end

    # Allocate pure type
    idx = Dict{NTuple{2, T}, Puretypes{T}}()
    next_slot = Dict{NTuple{2, T}, Int}()
    @inbounds for (key, cnt) in counts
        idx[key] = Puretypes{T}(Vector{T}(undef, cnt), Vector{ComplexF64}(undef, cnt))
        next_slot[key] = 1
    end

    # Fill pure type
    @inbounds for (k, cb) in big_terms
        C = k.c
        A = k.a
        key = (C & ~A, A & ~C)
        b = idx[key]
        j = next_slot[key]
        b.U[j] = (C | A) 
        b.coef[j] = cb
        next_slot[key] = j + 1
    end
    return idx
end

"""
Fermionic speciic inner ⟨a,b⟩ = Tr(a† b) / 2^L, computed from site types.

- Each monomial pair the weight is 2^{-m}, where m = popcount( (C₁|A₁) ∪ (C₂|A₂) ).
- Matching uses same-type partition: (c-only with c-only) and (a-only with a-only).
- If `normalized=false`, the final sum is multiplied by 2^{Ltot}, where
    Ltot = popcount( union of (C|A) over all terms in a and b ).
"""
function inner(a::FermionOperator{T}, b::FermionOperator{T};
                normalized::Bool = true)::ComplexF64 where {T<:Unsigned}
    ta = combine_like_terms(a.terms)
    tb = combine_like_terms(b.terms)
    (isempty(ta) || isempty(tb)) && return 0.0 + 0.0im

    # Build index on larger dict; ALWAYS conjugate coefficients from `a`
    big, small = (length(ta) >= length(tb)) ? (ta, tb) : (tb, ta)
    conj_a = (small === ta)

    idx = build_puretypes(big) 

    # For unnormalized variant, compute global Ltot as popcount of union across both ops (in T)
    scale = 1.0
    if !normalized
        total_mask = zero(T)
        @inbounds for (k, _) in ta
            total_mask |= (k.c | k.a)
        end
        @inbounds for (k, _) in tb
            total_mask |= (k.c | k.a)
        end
        Ltot = Int(count_ones(total_mask))
        scale = ldexp(1.0, Ltot)  # 2^Ltot
    end

    s = 0.0 + 0.0im
    @inbounds for (kS, cS) in small
        C1 = kS.c
        A1 = kS.a
        key = (C1 & ~A1, A1 & ~C1)   # same-type key (generic in T)
        bkt = get(idx, key, nothing)
        bkt === nothing && continue

        U1 = (C1 | A1)
        Uv = bkt.U
        cv = bkt.coef

        if conj_a
            # iterating over `a` → conjugate here
            @inbounds for j in eachindex(cv)
                m = Int(count_ones(U1 | Uv[j]))   
                w = ldexp(1.0, -m)
                s += conj(cS) * cv[j] * w
            end
        else
            # iterating over `b`; the `a`-side lives in `big` → conjugate there
            @inbounds for j in eachindex(cv)
                m = Int(count_ones(U1 | Uv[j]))
                w = ldexp(1.0, -m)
                s += conj(cv[j]) * cS * w
            end
        end
    end

    return s * scale
end


function norm(op::FermionOperator{T}; sanitized::Bool = true)::Float64 where {T<:Unsigned}
    terms = sanitized ? combine_like_terms(op.terms) : op.terms
    isempty(terms) && return 0.0

    groups = build_puretypes(terms)
    total = 0.0

    for b in values(groups)
        U  = b.U
        cs = b.coef
        nb = length(cs)
        nb <= 0 && continue

        # Precompute pc(U[i]) once
        pcU = Vector{Int}(undef, nb)
        @inbounds for i in 1:nb
            pcU[i] = count_ones(U[i])
        end

        # Diagonal: |c_i|^2 * 2^{-pc(U_i)}
        @inbounds for i in 1:nb
            total += abs2(cs[i]) * ldexp(1.0, -pcU[i])  
        end

        # Cross terms: 2 * Re(conj(ci) * cj) * 2^{-pc(Ui ∪ Uj)}
        # pc(Ui ∪ Uj) = pc(Ui) + pc(Uj) - pc(Ui ∩ Uj)
        @inbounds for i in 1:nb-1
            Ui  = U[i]
            ci  = cs[i]
            pci = pcU[i]
            for j in i+1:nb
                m = pci + pcU[j] - count_ones(Ui & U[j])
                total += 2.0 * real(conj(ci) * cs[j]) * ldexp(1.0, -m)  
            end
        end
    end

    return sqrt(total)
end


@inline _pcT(x::T) where {T<:Unsigned} = Int(count_ones(x))

# For each set bit j in Y, count set bits in X with index > j
@inline function bits_above(X::T, Y::T) where {T<:Unsigned}
    s = 0
    m = Y
    @inbounds while m != 0
        j = trailing_zeros(m)
        # bits strictly above j
        s += count_ones(X & (typemax(T) << (j + 1)))
        m &= m - one(T)
    end
    return s
end

@inline even_bits(::Type{T}) where {T<:Unsigned} = begin
    n = 8 * sizeof(T)
    m = zero(T)
    @inbounds for i in 0:2:(n - 1)
        m |= one(T) << i
    end
    m
end
@inline odd_bits(::Type{T}) where {T<:Unsigned} = begin
    n = 8 * sizeof(T)
    m = zero(T)
    @inbounds for i in 1:2:(n - 1)
        m |= one(T) << i
    end
    m
end

@inline _mask_up(::Type{T}) where {T<:Unsigned} = odd_bits(T)
@inline _mask_dn(::Type{T}) where {T<:Unsigned} = even_bits(T)

@inline _is_even_op(c::T, a::T) where {T<:Unsigned} = iseven(count_ones(c) + count_ones(a))


# locality measure: l = r + sum of # of |site|
# assuming 2 spin-orbitals per site: j = 2*site + spin
@inline function locality_measure(S::T) where {T<:Unsigned}
    r = _pcT(S)
    sumdist = 0

    m = S
    @inbounds while m != 0
        j = trailing_zeros(m)      # 0-based bit index
        site = j ÷ 2               # integer division → site index
        sumdist += abs(site)
        m &= m - one(T)
    end

    return r + sumdist
end


@inline function Fermionic_sign(lc::T, la::T, rc::T, ra::T)::Int8 where {T<:Unsigned}
    # Remove same-site a×c† pairs (handled locally by a c† = 1 - n)
    Lodd = lc | la
    e = bits_above(Lodd,  rc)  + bits_above(Lodd, ra)  
    return isodd(e) ? Int8(-1) : Int8(1)
end


# Binary calculation for Fkey, returns base key (C,A), I-n expansion set S, and global sign
@inline function Fkey_mul(x::FKey{T}, y::FKey{T})::Union{Nothing,Tuple{T,T,T,Int8}} where {T<:Unsigned}
    lc, la = x.c, x.a
    rc, ra = y.c, y.a

    L_a = la & ~lc
    L_c = lc & ~la
    L_n = lc & la
    R_a = ra & ~rc
    R_c = rc & ~ra
    R_n = rc & ra

    zmask = (L_a & R_a) | (L_c & R_c) | (L_n & R_a) | (L_c & R_n)
    zmask == zero(T) || return nothing
    S = L_a & R_c  

    Cb = (lc | rc) & ~S
    Ab = (la | ra) & ~S

    Ab &= ~(L_n & R_c)
    Cb &= ~(L_a & R_n)

    sgn = Fermionic_sign(L_c, L_a, R_c, R_a)  
    
    return (Cb, Ab, S, sgn)
end


# (1-n) subset expansion accumulator
@inline function expand_S(
    d::Dict{FKey{T},ComplexF64}, C::T, A::T, S::T, coef::ComplexF64
) where {T<:Unsigned}
    abs2(coef) < TOL2 && return

    if S == zero(T)
        k = FKey{T}(C, A)
        v = get(d, k, 0.0 + 0im) + coef
        if abs2(v) < TOL2
            delete!(d, k)
        else
            d[k] = v
        end
        return
    end

    s = S
    @inbounds while true
        vcoef = isodd(count_ones(s)) ? -coef : coef
        k = FKey{T}(C | s, A | s)
        v = get(d, k, 0.0 + 0im) + vcoef
        if abs2(v) < TOL2
            if haskey(d, k); delete!(d, k); end
        else
            d[k] = v
        end
        s == zero(T) && break
        s = (s - one(T)) & S
    end
end


function _fmul_small(a::FermionOperator{T}, b::FermionOperator{T})::FermionOperator{T} where {T<:Unsigned}
    isempty(a.terms) && return zero(a)
    isempty(b.terms) && return zero(a)

    out = Dict{FKey{T},ComplexF64}()
    na, nb = length(a.terms), length(b.terms)
    sizehint!(out, max(8, min(na*nb, 4*(na + nb))))

    Ak = collect(keys(a.terms)); Av = ComplexF64[a.terms[k] for k in Ak]
    Bk = collect(keys(b.terms)); Bv = ComplexF64[b.terms[k] for k in Bk]

    @inbounds for i in 1:length(Ak)
        ca = Av[i]
        abs2(ca) < TOL2 && continue
        ka = Ak[i]
        for j in 1:length(Bk)
            cb = Bv[j]
            c2 = ca * cb
            abs2(c2) < TOL2 && continue

            r = Fkey_mul(ka, Bk[j])
            r === nothing && continue
            Cb, Ab, S, sgn = r


            expand_S(out, Cb, Ab, S, (sgn == 1 ? c2 : -c2))
        end
    end
    FermionOperator{T}(out)
end


function _fmul_threaded(a::FermionOperator{T}, b::FermionOperator{T})::FermionOperator{T} where {T<:Unsigned}
    isempty(a.terms) && return zero(a)
    isempty(b.terms) && return zero(a)

    Ak = collect(keys(a.terms)); Av = ComplexF64[a.terms[k] for k in Ak]
    Bk = collect(keys(b.terms)); Bv = ComplexF64[b.terms[k] for k in Bk]

    nt = max(nthreads(), 1)
    partials = [Dict{FKey{T},ComplexF64}() for _ in 1:nt]
    base_hint = max(8, length(Ak)*length(Bk) ÷ (4*nt))
    for d in partials; sizehint!(d, base_hint) end

    Threads.@threads for i in 1:length(Ak)
        d = partials[threadid()]
        ca = Av[i]
        abs2(ca) < TOL2 && continue
        ka = Ak[i]

        @inbounds for j in 1:length(Bk)
            cb = Bv[j]
            c2 = ca * cb
            abs2(c2) < TOL2 && continue

            r = Fkey_mul(ka, Bk[j])
            r === nothing && continue
            Cb, Ab, S, sgn = r
            """
            # locality constraint
            if locality
                # crude but fast: use support including potential (1-n) expansion bits
                support_mask = (Cb | Ab | S)
                if locality_measure(support_mask) > LMAX
                    continue
                end
            end
        
            """
            expand_S(d, Cb, Ab, S, (sgn == 1 ? c2 : -c2))
        end
    end

    out = Dict{FKey{T},ComplexF64}()
    sizehint!(out, base_hint*nt)
    @inbounds for d in partials
        for (k,c) in d
            v = get(out, k, 0.0+0im) + c
            if abs2(v) < TOL2
                if haskey(out,k); delete!(out,k); end
            else
                out[k] = v
            end
        end
    end
    FermionOperator{T}(out)
end


function Base.:*(a::FermionOperator{T}, b::FermionOperator{T}) where {T<:Unsigned}
    W = length(a.terms) * length(b.terms)
    return (W <= 5000000) ? _fmul_small(a,b) : _fmul_threaded(a,b)
end
