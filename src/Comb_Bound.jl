# ============================================================
# Comb_Bound.jl
#
# Provides W^{k,∞} bounds on products of functions via the multinomial Leibniz
# rule. The main entry point `multinomial_sum` implements a polynomial-convolution
# scheme over derivative order, independently for each direction. Helpers support
# variable changes (r → β via tan), Bell polynomials (Faà di Bruno formula),
# and polynomial arithmetic over BigInt for exact coefficient tracking.
#
# Dependencies: My_Fourier.jl (`dt`, `dtype`, `Pi`, and `mycos`),
# Matrix_Func.jl (`Linf_to_paras`), My_Vector.jl (`VectorFunc`, `norm_lst`).
# See: docs/technical_notes/comb_bound.md for mathematical background.
# ============================================================

"""
Compute pure-direction product-derivative bounds by factorial-scaled convolution.

Require `n ≥ 0` and at least two factors. Each `a[j][m+1,l]` supplies a
nonnegative absolute derivative bound of order m in direction l, through order n.
All factors must use the same d directions and cover the same target domain.
A vector of order entries represents one direction; otherwise use an order ×
direction table whose entries are scalars or blocks, not a dense block tensor.

The first factor supplies scalar bounds, stored internally as `dtype`. Later entries
form a compatible matrix-product chain in the supplied order; scalars act as
1×1 blocks and vectors as columns. Matrix bounds are entrywise, not operator
norms. Inputs are not modified.

With `last_flag=true`, return a length-d vector of order-n bound matrices;
for d=1 return the matrix itself. With `last_flag=false`, return an
`(n+1) × d` table of matrices for orders 0:n. In either case, 1×1 matrices
are unwrapped to scalars; thus a one-direction scalar result is a scalar
when `last_flag=true`, and an `(n+1) × 1` numeric table otherwise.
See `docs/technical_notes/comb_bound.md` for the ordered product bound.
"""
function multinomial_sum(n::Int, a...; last_flag::Bool=true)
    # Outer indices: derivative order and direction; each entry holds a scalar or block.
    k = length(a)
    @assert k ≥ 2
    for j in 1:k
        @assert size(a[j], 1) ≥ n + 1
    end

    # Factorial scaling turns Leibniz coefficients into ordinary convolutions.
    fact_lst = ones(dtype, n + 1)
    @inbounds for m in 1:n
        fact_lst[m + 1] = fact_lst[m] * dt(m)
    end

    # The first factor is scalar-valued in the matrix-chain convention.
    d = size(a[1], 2)
    dp = zeros(dtype, n + 1, d)
    @inbounds for m in 0:n
        dp[m + 1, :] = a[1][m + 1, :] / fact_lst[m + 1]
    end

    r = size(a[2][1,1], 1)
    c = r
    c_next = 0

    # Convolve with the remaining block-valued factors.
    for j in 2:k
        @assert size(a[j], 2) == d "dimension mismatch a j=$j: A_j depth=$(size(a[j],4)), expected $d"
        @assert size(a[j][1,1], 1) == c "dimension mismatch a j=$j: A_j rows=$(size(a[j],2)), expected $c"
        c_next = size(a[j][1,1], 2)

        new_dp = [zeros(dtype, r, c_next) for _ in 1:(n + 1), _ in 1:d]
        @views for p in 0:n, l in 1:d
            Ajp = a[j][p + 1, l]
            αp  = inv(fact_lst[p + 1])
            @inbounds for m in 0:(n - p)
                new_dp[m + p + 1, l] .+= (dp[m + 1, l] * αp) * Ajp
            end
        end
        dp = new_dp
        c  = c_next
    end

    # Undo the factorial scaling.
    if last_flag
        out = dp[n + 1, :] .* fact_lst[n + 1]
    else
        out = [dp[m, j] .* fact_lst[m] for m in 1:n+1, j in 1:d]
    end

    # Unwrap scalar blocks and, for a single requested order, a single direction.
    if r == 1 && c == 1
        out = [only(each) for each in out]
    end
    if last_flag && d == 1
        out = only(out)
    end
    return out
end

"""
Return the derivative coefficients of a nonempty `BigInt` polynomial vector.
Coefficients are ordered from the constant term upward; a constant returns `[0]`.
The input is unchanged.
"""
function poly_deriv(c::Vector{BigInt})
    deg = length(c) - 1
    if deg == 0
        return BigInt[0]
    end
    d = Vector{BigInt}(undef, deg)
    for i in 1:deg
        d[i] = BigInt(i) * c[i+1]
    end
    return d
end

"""
Add two low-to-high `BigInt` coefficient vectors into a new vector.
Trailing zero coefficients are removed, retaining one entry for a zero polynomial
when the input vectors are nonempty.
"""
function poly_add(a::Vector{BigInt}, b::Vector{BigInt})
    n = max(length(a), length(b))
    out = fill(BigInt(0), n)

    for i in 1:length(a)
        out[i] += a[i]
    end
    for i in 1:length(b)
        out[i] += b[i]
    end

    # Trim trailing zeros, retaining one entry for the zero polynomial.
    while length(out) > 1 && out[end] == 0
        pop!(out)
    end
    return out
end

"""
Return the low-to-high `BigInt` coefficients of `x*p(x)` without changing `c`.
For nonempty `c`, trim trailing zeros while retaining one zero-polynomial entry.
"""
function poly_mul_x(c::Vector{BigInt})
    out = fill(BigInt(0), length(c) + 1)

    for i in 1:length(c)
        out[i+1] = c[i]
    end

    # Trim trailing zeros, retaining one entry for the zero polynomial.
    while length(out) > 1 && out[end] == 0
        pop!(out)
    end
    return out
end

"""
Return the low-to-high `BigInt` coefficients of `(1+x^2)*p(x)`.
For nonempty `c`, trim trailing zeros while retaining one zero-polynomial entry.
The input is unchanged.
"""
function poly_mul_1_plus_x2(c::Vector{BigInt})
    deg = length(c) - 1
    out = fill(BigInt(0), deg + 3)  # Degree increases by two.
    for i in 0:deg
        out[i+1] += c[i+1]          # Contribution from the constant term.
        out[i+3] += c[i+1]          # Contribution from the x^2 term.
    end
    # Trim trailing zeros, retaining one entry for the zero polynomial.
    while length(out) > 1 && out[end] == 0
        pop!(out)
    end
    return out
end

"""
Evaluate `sum(abs(c[i+1])*R^i)` for low-to-high `BigInt` coefficients.
For nonnegative R this bounds `abs(p(x))` on `abs(x) ≤ R`; interval mode
encloses the coefficient-sum value. Nonnegativity of R is a caller precondition.
The input is unchanged.
"""
function poly_abs_sum_bound(c::Vector{BigInt}, R::dtype)
    s = zero(dtype)
    p = one(dtype)
    for i in 0:(length(c)-1)
        s += dt(abs(c[i+1])) * p
        p *= R
    end
    return s
end

"""
Convert bounds for `f(r)` to bounds for `f(a*tan(β))` through order m.

`M[j+1]` must bound the absolute r-derivative of order j on the entire mapped
target domain. Require `0 < a < 1`, `m ≥ 0`, and at least m+1 entries in M.
These three conditions are asserted; validity of the supplied bounds is assumed.

With `β_max=nothing`, use `abs(β) ≤ atan(1/a)`, corresponding to
`abs(r) ≤ 1`. Otherwise require `0 < β_max < π/2` and use
`abs(β) ≤ β_max`; this endpoint condition is not checked here. The mapped
radius is then at most `a*tan(β_max)`. The bounds in M must already cover
any padded points in that domain; changing β_max only changes the map bounds.
Current callers use normalized radius r with `a=scl_fac/R`.

Return a new length-(m+1) `Vector{dtype}`, with entry j+1 bounding the
β-derivative of order j. M is unchanged. See
`docs/technical_notes/comb_bound.md` for the Bell-polynomial argument.
"""
function beta_all_deriv_Linf_bounds(M::AbstractVector{dtype}, a::dtype, m::Integer; β_max::Union{dtype,Nothing}=nothing)
    @assert 0 < a < 1 "Require 0 < a < 1."
    @assert m ≥ 0 "Require m ≥ 0."
    @assert length(M) ≥ m+1 "Need M[j+1] for j=0..m."

    # |tan(β)| bound on the target β-interval. Default endpoint atan(1/a) gives R = 1/a.
    # A larger β_max enlarges only this domain bound; the scale `a` in r(β)=a·tan(β)
    # (used in the derivative recurrence below) is unchanged.
    R  = β_max === nothing ? inv(a) : tan(β_max)

    # Bound derivatives of r(β)=a*tan(β) using tan-derivative polynomials.

    x = Vector{dtype}(undef, max(m,1))   # x[1]=‖r'‖∞, ..., x[m]=‖r^{(m)}‖∞
    if m ≥ 1
        P = BigInt[0, 1]                 # P_0(x)=x
        for n in 1:m
            P = poly_mul_1_plus_x2(poly_deriv(P))  # P_n = (1+x^2) * P'_{n-1}
            x[n] = a * poly_abs_sum_bound(P, R)
        end
    end

    # Evaluate partial Bell polynomials on the derivative bounds for r.

    Bbell = fill(zero(dtype), (m+1, m+1))
    Bbell[1,1] = one(dtype)  # B_{0,0}

    for n in 1:m, j in 1:n
        s = zero(dtype)
        max_i = n - j + 1
        for i in 1:max_i
            s += dt(binomial(n-1, i-1)) * x[i] * Bbell[n - i + 1, j]  # (n-i, j-1)
        end
        Bbell[n+1, j+1] = s
    end

    # Combine the Bell-polynomial coefficients with the supplied f^(j) bounds.

    out = Vector{dtype}(undef, m+1)
    out[1] = M[1]  # k=0: ‖g‖∞ ≤ ‖f‖∞

    for k in 1:m
        s = zero(dtype)
        for j in 1:k
            s += M[j+1] * Bbell[k+1, j+1]
        end
        out[k+1] = s
    end

    return out
end

"""
Return bounds for derivatives of `tan(β)/cos(β)` through order k.

Require `k ≥ 0` and `0 < β0 < π/2` (both asserted). The returned
length-(k+1) `Vector{dtype}` has entry n+1 bounding the absolute derivative
of order n on `[0,β0]`; the same bounds hold on `[-β0,β0]` by parity.
See `docs/technical_notes/comb_bound.md` for the polynomial recurrence.
"""
function tan_over_cos_deriv_Linf_bounds(β0::dtype, k::Integer)
    @assert k ≥ 0 "Require k ≥ 0."
    @assert 0 < β0 < Pi/dt(2) "Require 0 < β0 < π/2."

    # Bounds for |tan(β)| and sec(β) on [0, β0].
    T = tan(β0)
    S = inv(mycos(β0))

    # B[n+1] bounds the n-th derivative.
    B = Vector{dtype}(undef, k+1)

    # Rpoly stores R_n in f^(n)(β)=sec(β)*R_n(tan(β)).
    Rpoly = BigInt[0, 1]

    # Base case R_0(x)=x.
    B[1] = S * T

    for n in 1:k
        # Differentiate sec(β)*R_{n-1}(tan(β)) in polynomial form.
        Rp    = poly_deriv(Rpoly)
        term1 = poly_mul_x(Rpoly)
        term2 = poly_mul_1_plus_x2(Rp)
        Rpoly = poly_add(term1, term2)

        # Bound sec(β)*R_n(tan(β)) by a coefficient absolute sum.
        B[n+1] = S * poly_abs_sum_bound(Rpoly, T)
    end

    return B
end

"""
Apply `r = a0*tan(β)` to each entry of a derivative-bound sequence.

Supply a nonempty sequence A of consistently sized vectors or matrices:
`A[j]` contains entrywise absolute r-derivative bounds of order j-1.
For m×n matrix entries, return a length-`length(A)` vector of m×n matrices.
Length-m vector entries are treated as columns and return m×1 matrices.
Inputs are unchanged; scalar entries are not supported by this wrapper.

`a0` and `β_max` are passed to `beta_all_deriv_Linf_bounds`, including its
scale and mapped-domain requirements. Omitting β_max uses the normalized
radius bound `abs(r) ≤ 1`. Output entry j bounds β-derivatives of order j-1.
"""
function r_to_beta(A, a0::dtype; β_max::Union{dtype,Nothing}=nothing)
    k = length(A)
    if ndims(A[1]) == 1
        m = length(A[1])
        n = 1
    else
        m, n = size(A[1])
    end

    # Stack the order entries, treating input vectors as one-column matrices.
    bound_tensor = zeros(dtype, k, m, n)
    for j in 1:k, p in 1:m, q in 1:n
        bound_tensor[j, p, q] = A[j][p, q]
    end

    # Transform each matrix entry independently.
    for p in 1:m, q in 1:n
        bound_tensor[:, p, q] =
            beta_all_deriv_Linf_bounds(bound_tensor[:, p, q], a0, k-1; β_max=β_max)
    end

    # Return one matrix per order; vector inputs remain columns.
    out = [bound_tensor[i, :, :] for i in 1:k]
    return out
end

"""
Combine componentwise basis bounds with absolute eigenvector coefficients.

`bd_mat` has shape `(K,N,C)`: derivative orders 0:K-1, N basis elements,
and C components. `eig_vec` has shape `(N,J)`, with basis rows in exactly
the same order and one eigenfunction per column. The basis bounds must be
nonnegative and valid on the target domain; dimensions and ordering are
caller responsibilities.

Return a length-K vector of C×J matrices. Entry `result[m+1][c,j]` bounds the
order-m derivative of component c of eigenfunction j. The derivative coordinate
and domain are inherited from `bd_mat`; no coordinate conversion is performed.
Both inputs are unchanged.
"""
function eig_fun_bound(bd_mat, eig_vec)
    k = size(bd_mat, 1)
    abs_eig_vec = abs.(eig_vec)
    result = [bd_mat[i, :, :]' * abs_eig_vec for i in 1:k]
    return result
end

"""
    refine_channel_bounds(fields, priors, scl_fac, domain, order;
                          angular_frequency, padding, crop=0, label="mode")

Refine the derivative bounds of a single parity block. `fields` has axes
`(radial grid, angular grid, mode, component)` with three components. `priors`
contains 3-by-mode matrices through derivative order `order + 2`. Require
`2 <= order < 20`, `crop >= 0`, and `domain = [Beta, L1]`.
After `crop` layers are removed on each side, the grid must retain `padding`
ghost layers, at least `derivative_padding(order)`. The returned mode list
contains `(order + 1) × 2` tables of three-component bounds, with derivative
directions `(β, θ)`. The physical interval counts on both axes must be positive
multiples of 100. Input layouts are caller preconditions; the norm routines
retain their derivative-order, padding, and grid checks.

Fresh parameter packages are constructed for each mode because `norm_lst`
updates its angular-bound parameters in place.
"""
function refine_channel_bounds(fields::AbstractArray{dtype,4}, priors,
        scl_fac::dtype, domain::Vector{dtype}, order::Int;
        angular_frequency::dtype, padding::Int, crop::Int=0, label::String="mode")
    modes = size(fields, 3)
    rows = (crop + 1):(size(fields, 1) - crop)
    cols = (crop + 1):(size(fields, 2) - crop)
    result = Matrix{Vector{dtype}}[]
    for ind in 1:modes
        println("$label $ind / $modes")
        field = VectorFunc(scl_fac, "space";
            e1=fields[rows, cols, ind, 1], e2=fields[rows, cols, ind, 2],
            e3=fields[rows, cols, ind, 3])
        paras = Linf_to_paras(priors, order + 2,
            [zero(dtype), angular_frequency]; ind=ind)
        push!(result, norm_lst(field, "W$(order)Inf"; paras=paras, domain=domain, padding=padding))
    end
    return result
end

"""
Convert the β column of a mode list of derivative-bound tables into a vector
of component-by-mode matrices. Supply a nonempty list of consistently sized
order-by-direction tables, with equally sized component vectors in each entry.
The result satisfies `result[m+1][:,j] = bounds[j][m+1,1]` and has the layout
used by `multinomial_sum`. Inputs are unchanged.
"""
function beta_bound_chain(bounds)
    orders = size(first(bounds), 1)
    components = length(first(bounds)[1, 1])
    result = [zeros(dtype, components, length(bounds)) for _ in 1:orders]
    for i in 1:orders, j in eachindex(bounds)
        result[i][:, j] = bounds[j][i, 1]
    end
    return result
end
