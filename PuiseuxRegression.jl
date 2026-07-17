# module PuisuexRegression

using AbstractAlgebra
using Random

# export PContext, puiseux_from_poly, puiseux_from_poly_scale_shift,
#        val, trunc_terms, inv_puiseux,
#        randomPuiseuxMonomial, randomPuiseux, trueCoeffs, dataSet,
#        dataPointsToFunction, loss, lossVector


# ========
# Context
# ========

"""
Holds:
- Kt: Puiseux polynomial ring over Laurent polynomial ring
- vars: generators in Kt
- L: underlying Laurent polynomial ring
- lvars: generators in Laurent ring
"""
struct PContext
    Kt
    vars
    L
    lvars
end

function PContext(K::Field, names::Vector{String})
    Kt, vars = puiseux_polynomial_ring(K, names)
    L = base_ring(Kt)
    lvars = gens(L)
    return PContext(Kt, vars, L, lvars)
end

# ==============
# Constructors
# ==============
"""
puiseuxPolynomial(f::RingElement)
Construct AA Puiseux element from Laurent/MPoly `f` with scale 1.
"""
function puiseux_from_poly(ctx::PContext, f)
    lf = parent(f) === ctx.L ? f : ctx.L(f)
    return puiseux_polynomial_ring_elem(ctx.Kt, lf, 1)
end

"""
puiseuxPolynomial(f, N, s)
Represents sum c_i x^((k_i+s)/N).
In AA form: x^(s/N) * f(x^(1/N))  <=> store as Laurent poly with integer exponents and scale N.
So we multiply by monomial x^s in Laurent ring, then use scale N.
"""
function puiseux_from_poly_scale_shift(ctx::PContext, f, N::Int, s::Vector{Int})
    @req N > 0 "Scale N must be positive"
    lf = parent(f) === ctx.L ? f : ctx.L(f)
    @req length(s) == length(ctx.lvars) "Shift length must match number of variables"
    mon = prod(ctx.lvars[i]^s[i] for i in eachindex(s))
    return puiseux_polynomial_ring_elem(ctx.Kt, mon * lf, N)
end

# ==============
# Basic helpers
# ==============

val(f) = valuation(f)

function _is_posinf(x)
    return typeof(x).name.name == :PosInf
end

function _finite_valuation_key(v)
    _is_posinf(v) && return (true, 0//1)
    return (false, v)
end

# ==============================================================
# Truncation by number of smallest valuation terms (univariate)
# ==============================================================

function trunc_terms(f, nTerms::Int)
    @req nvars(parent(f)) == 1 "trunc_terms currently implemented for univariate Puiseux polynomials"
    @req nTerms >= 0 "nTerms must be nonnegative"

    if nTerms == 0 || iszero(f)
        return zero(parent(f))
    end

    mons = collect(monomials(f))
    coeffs = collect(coefficients(f))
    evs = collect(exponent_vectors(f))

    idx = collect(eachindex(mons))
    sort!(idx, by = i -> evs[i][1])  # ascending valuation

    keep = idx[1:min(nTerms, length(idx))]
    out = zero(parent(f))
    for i in keep
        out += coeffs[i] * mons[i]
    end
    return out
end

# ===================================
# Inverse approximation (univariate)
# ===================================

Base.@kwdef struct InvOptions
    maxValLength::Int = 5
    maxNumTerms::Int = 10
end

function _leading_term_by_smallest_exp(f)
    evs = collect(exponent_vectors(f))
    coeffs = collect(coefficients(f))
    @req !isempty(evs) "zero element has no leading term"
    idx = argmin(i -> evs[i][1], eachindex(evs))
    return evs[idx][1], coeffs[idx]
end

function inv_puiseux(f; opts::InvOptions=InvOptions())
    @req nvars(parent(f)) == 1 "inv_puiseux assumes univariate"
    @req !iszero(f) "cannot invert zero"

    Kt = parent(f)
    t = gens(Kt)[1]

    k0, c0 = _leading_term_by_smallest_exp(f)
    Pg = inv(c0) * t^(-k0)
    onep = one(Kt)

    while true
        Pr = f*Pg - onep
        iszero(Pr) && break

        k, c = _leading_term_by_smallest_exp(Pr)
        nextTerm = -(c/c0) * t^(k - k0)
        Pg = Pg + nextTerm

        vnext = valuation(nextTerm)
        vpg   = valuation(Pg)
        if !_is_posinf(vnext) && !_is_posinf(vpg)
            (vnext - vpg >= opts.maxValLength) && break
        end
        length(Pg) >= opts.maxNumTerms && break
    end
    return Pg
end

# ==========================
# Random Puiseux generation
# ==========================

Base.@kwdef struct RandomMonomialOptions
    minVal::Rational{Int} = 0//1
    maxVal::Rational{Int} = 1//1
    ReturnValuation::Bool = false
    RandomCoefficient::Bool = true
end

function _rand_coeff(K)
    try
        return rand(K, -10:10)
    catch
        return K(rand(-10:10))
    end
end

function randomPuiseuxMonomial(ctx::PContext, Lparam::Int; opts::RandomMonomialOptions=RandomMonomialOptions())
    @req nvars(ctx.Kt) == 1 "randomPuiseuxMonomial currently implemented for univariate ring"
    t = ctx.vars[1]

    q = rand()
    denomExponent = ceil(Int, log(1/q)/log(Lparam))
    numExponent = floor(Int, q * Lparam^denomExponent)
    exponent = numExponent // (Lparam^denomExponent)
    shiftedExponent = exponent * (opts.maxVal - opts.minVal) + opts.minVal

    c = opts.RandomCoefficient ? _rand_coeff(coefficient_ring(ctx.Kt)) : one(coefficient_ring(ctx.Kt))
    result = c * t^shiftedExponent

    return opts.ReturnValuation ? (result, shiftedExponent) : result
end

Base.@kwdef struct RandomPuiseuxOptions
    StoppingProbability::Float64 = 0.5
    StepSize::Rational{Int} = 2//1
    StartValuation::Rational{Int} = 0//1
    minTerms::Int = 3
end

function randomPuiseux(ctx::PContext, Lparam::Int; opts::RandomPuiseuxOptions=RandomPuiseuxOptions())
    firstTerm, currentExponent = randomPuiseuxMonomial(
        ctx, Lparam;
        opts=RandomMonomialOptions(
            minVal=opts.StartValuation,
            maxVal=opts.StartValuation + opts.StepSize,
            ReturnValuation=true,
            RandomCoefficient=true
        )
    )

    numTerms = 1
    terms = eltype([firstTerm])[]
    while true
        if numTerms >= opts.minTerms && rand() < opts.StoppingProbability
            break
        end
        nextMonomial, nextExponent = randomPuiseuxMonomial(
            ctx, Lparam;
            opts=RandomMonomialOptions(
                minVal=currentExponent,
                maxVal=currentExponent + opts.StepSize,
                ReturnValuation=true,
                RandomCoefficient=true
            )
        )
        currentExponent = nextExponent
        numTerms += 1
        push!(terms, nextMonomial)
    end

    out = firstTerm
    for tm in terms
        out += tm
    end
    return out
end

trueCoeffs(ctx::PContext, n::Int; L::Int=3) = [randomPuiseux(ctx, L) for _ in 0:n]

# =================
# Data generation
# =================


function dataSet(c::Vector, m::Int; L::Int=5, minShiftVal=3//1, maxShiftVal=10//1, truncNTerms::Int=10)
    n = length(c) - 1
    ctxKt = parent(c[1])
    D = Vector{Vector{typeof(c[1])}}()

    # derive a context-lite for random generator
    vars = gens(ctxKt)
    ctx = PContext(ctxKt, vars, base_ring(ctxKt), gens(base_ring(ctxKt)))

    for _ in 1:m
        x = [randomPuiseux(ctx, L; opts=RandomPuiseuxOptions(minTerms=5, StoppingProbability=0.9)) for _ in 1:n]
        y = c[1] + sum(c[i+1] * x[i] for i in 1:n)

        α = rationalize(rand())
        minErr = minShiftVal + (maxShiftVal - minShiftVal) * α
        r = randomPuiseux(ctx, L; opts=RandomPuiseuxOptions(StartValuation=minErr))
        y_err = trunc_terms(y + r, truncNTerms)
        push!(D, vcat(x, y_err))
    end
    return D
end


# ============================================================
# Linear solve over Puiseux (Gaussian elimination)
# ============================================================

function dataPointsToFunction(D::AbstractVector{<:AbstractVector}; truncOutput::Bool=true, truncNTerms::Int=30)
    n = length(first(D)) - 1
    Kt = parent(D[1][1])

    A = [vcat([one(Kt)], d[1:n]) for d in D]
    vals = [d[end] for d in D]

    M = [Vector{typeof(one(Kt))}() for _ in 1:(n+1)]
    for i in 1:n+1
        row = typeof(one(Kt))[]
        append!(row, A[i])
        for j in 1:n+1
            push!(row, i == j ? one(Kt) : zero(Kt))
        end
        M[i] = row
    end

    # downward
    for top in 1:n+1
        rowScalar = inv_puiseux(M[top][top])
        for col in 1:(2n+2)
            M[top][col] *= rowScalar
        end
        for bot in (top+1):(n+1)
            rowMultiple = M[bot][top]
            for col in 1:(2n+2)
                M[bot][col] -= M[top][col] * rowMultiple
            end
        end
    end

    # upward
    for i in 1:n
        bottom = n+2-i
        for j in (i+1):(n+1)
            top = n+2-j
            rowMultiple = M[top][bottom]
            for col in 1:(2n+2)
                M[top][col] -= M[bottom][col] * rowMultiple
            end
        end
    end

    result = [sum(M[i][n+1+j] * vals[j] for j in 1:n+1) for i in 1:n+1]
    return truncOutput ? [trunc_terms(r, truncNTerms) for r in result] : result
end

# ==========
# Loss
# ==========

#
# loss is given by valuation so bigger is better (closer to zero)
#

function loss(D::AbstractVector{<:AbstractVector}, S::AbstractVector{<:Integer}; 
              Verbose::Bool=true, truncOutput::Bool=true, truncNTerms::Int=30)
    Verbose && println("-- computing function through points")
    n = length(first(D)) - 1
    c = dataPointsToFunction(D[S]; truncOutput=truncOutput, truncNTerms=truncNTerms)

    Verbose && println("-- calculating loss function:")
    vals = Any[]
    for (k, dataPoint) in enumerate(D)
        Verbose && println("-- $k/$(length(D))")
        residual = c[1] + sum(c[i+1]*dataPoint[i] for i in 1:n) - dataPoint[n+1]
        push!(vals, valuation(residual))
    end

    sort!(vals, by = _finite_valuation_key)
    return vals[1]
end

function lossVector(D::AbstractVector{<:AbstractVector}, S::AbstractVector{<:Integer};
                    Verbose::Bool=true, truncOutput::Bool=true, truncNTerms::Int=30)
    Verbose && println("-- computing function through points")
    n = length(first(D)) - 1
    c = dataPointsToFunction(D[S]; truncOutput=truncOutput, truncNTerms=truncNTerms)

    Verbose && println("-- calculating loss function:")
    out = Any[]
    for (k, dataPoint) in enumerate(D)
        Verbose && println("-- $k/$(length(D))")
        residual = c[1] + sum(c[i+1]*dataPoint[i] for i in 1:n) - dataPoint[n+1]
        push!(out, valuation(residual))
    end
    return out
end

# end #end module
