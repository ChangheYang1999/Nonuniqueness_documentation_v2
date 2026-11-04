# ============================================================
# Matrix_Func.jl
#
# Numerical utilities for the verification pipeline:
#   - Composite closed 7-point Newton–Cotes quadrature (7 nodes per 6-subinterval panel) with a priori
#     8th-derivative error bound.
#   - Finite-difference derivative operator and W^{k,∞} norm estimation
#     via backward recursion.
#   - Interval matrix I/O helpers (save/read .mat files in lo/hi split format).
#   - LDLᵀ-based positive-definiteness certificate and generalized eigenvalue
#     verification for the main operator matrices.
#
# Numeric mode `dtype` is set by NS_Numerics.jl; `dt` is defined in My_Fourier.jl.
# See: ../docs/technical_notes/matrix_func.md for mathematical background.
# ============================================================

"""
    newton_cotes_weights(n::Int, h::dtype)

Return the composite closed 7-point Newton–Cotes weights for `n` uniform
subintervals of width `h`, with 6 subintervals per panel.

The output is a `Vector{dtype}` of length `n+1`; endpoint nodes are included.
It applies to any translated interval of length `n*h`. Adjacent panels share
a node, so their endpoint weights add at each interior panel junction.

For quadrature, supply `n > 0` and `h > 0`. Only `n % 6 == 0` is checked here.
See [the quadrature derivation](../docs/technical_notes/matrix_func.md).
"""
function newton_cotes_weights(n::Int, h::dtype)
    @assert n % 6 == 0 "n must be a multiple of 6 for Newton–Cotes's rule"
    w = zeros(dtype, n + 1)
    c = h / dt(140)
    # endpoints
    w[1]   = dt(41) * c
    w[end] = dt(41) * c
    for i in 2:n
        r = (i - 1) % 6
        if r == 1 || r == 5
            w[i] = dt(216) * c
        elseif r == 2 || r == 4
            w[i] = dt(27) * c
        elseif r == 3
            w[i] = dt(272) * c
        else
            # r == 0 : interior panel boundary, shared by two panels
            w[i] = dt(82) * c   # 41+41 from adjacent panels
        end
    end
    return w
end

"""
    newton_cotes_with_bound(f::AbstractVector{dtype}, Wk∞_bound::dtype)

Return `(S, err_bd)` for a uniform grid on `[0,1]`, where `N = length(f)-1`
is a positive multiple of 6 and `h = 1/N`. `f` contains endpoint-inclusive
samples; `Wk∞_bound` must bound the supremum of the absolute eighth
derivative on `[0,1]`.

`S` is the weighted sample sum and `err_bd = Wk∞_bound*h^8*3/2800` is the
quadrature remainder allowance. Both have type `dtype`. In interval mode,
`S` encloses the exact weighted sum, including sample and arithmetic uncertainty;
the integral is enclosed only after adding the remainder allowance, for example
with `hull(S-err_bd, S+err_bd; dec=:auto)`. The two values are returned separately.
Float64 evaluation alone is not a rigorous enclosure.

The panel-count assertion is delegated to `newton_cotes_weights`. The caller
is responsible for sample validity, smoothness, and the supplied derivative bound.
"""
function newton_cotes_with_bound(f::AbstractVector{dtype}, Wk∞_bound::dtype)
    N = length(f) - 1
    h = inv(dt(N))

    w = newton_cotes_weights(N, h)
    S = sum(w .* f)

    err_bd = Wk∞_bound * h^8 * dt(3) / dt(2800)

    return S, err_bd
end

"""
    derivative_padding(k::Int)

Return the symmetric sample padding for recursive derivative bounds through
order `k`, using two higher-order priors: `m*(m+1)÷2`, where `m=cld(k,2)`.
Require `0 ≤ k < 20`. To certify every returned derivative on an additional
`H`-layer interval, use `H + derivative_padding(k)` actual sample layers.
This recursive requirement does not apply to independent frequency estimates.
"""
function derivative_padding(k::Int)
    @assert 0 <= k < 20 "Derivative order must satisfy 0 ≤ k < 20"
    m = cld(k, 2)
    return m * (m + 1) ÷ 2
end

"""
    finite_difference(Xpad::AbstractArray{dtype}, h::dtype, k::Int)

Apply the normalized k-th difference along the first dimension; other axes
are flattened into independent columns during evaluation and then restored.

Require `0 ≤ k < 20`, `h > 0`, and at least `k+1` rows. If `NX = size(Xpad,1)`,
the output has shape `(NX-k, size(Xpad)[2:end]...)`. Row `i` uses every sample
in `Xpad[i:i+k,...]`, with anchor `ip = i + k÷2`. The stencil center is the
anchor for even k and half a mesh step to its right for odd k. These input
conditions are checked before the unchecked evaluation loops.

For `k == 0`, return the input itself (an alias), without copying or trimming.
No input is mutated. The output is a finite difference, not a certified
derivative value; `estimate_Wk∞_norm` supplies correction terms.
`Xpad` must already contain samples on the required extended grid; this
function does not construct padding or apply boundary conditions.
"""
function finite_difference(Xpad::AbstractArray{dtype}, h::dtype, k::Int)
    @assert 0 <= k < 20 "Derivative order must satisfy 0 ≤ k < 20"
    @assert h > 0 "Grid spacing must be positive"
    NX = size(Xpad, 1)
    @assert NX >= k + 1 "Not enough samples for this stencil"
    N = NX - k
    if k == 0
        return Xpad
    end

    # Even orders center on the anchor; odd orders center half a step to its right.
    shift = iseven(k) ? -(k ÷ 2) : -((k - 1) ÷ 2)

    # precompute offsets and coefficients (length k+1)
    offs  = Vector{Int}(undef, k+1)
    coeff = Vector{dtype}(undef, k+1)

    inv_scale = inv(h^dt(k))
    @inbounds for jj in 0:k
        offs[jj+1] = shift + jj
        c = ((-1)^(k - jj)) * binomial(k, jj)
        coeff[jj+1] = dt(c) * inv_scale
    end

    outsz = (N, tail(size(Xpad))...)
    Y = Array{dtype}(undef, outsz)

    X2 = reshape(Xpad, NX, :)
    Y2 = reshape(Y,     N, :)

    @inbounds @threads for col in axes(X2, 2)
        for i in 1:N
            ip = i + (k ÷ 2)
            s = zero(dtype)
            for t in eachindex(offs)
                s += coeff[t] * X2[ip + offs[t], col]
            end
            Y2[i, col] = s
        end
    end

    return Y
end


"""
    Linf_to_paras(priors, k::Int, freq_lst::Vector{dtype}; ind=nothing)

Build an `l × 2` matrix of parameter packages for `norm_lst`: rows are
components and columns are derivative directions `(β, θ)`. Each package
contains two slots: primary and transverse correction data. `freq_lst`
contains maximum wavenumbers `[Mβ, Mθ]` in the respective coordinates.

**Target order.** For a downstream target order `p`, pass `k = p+2`.
Prior entry `q+1` represents derivative order q; entries `k` and `k+1`
supply the two seeds for orders `p+1` and `p+2`.

**Input layouts.**

- Ordinary mode (`ind === nothing`): `priors` is a derivative-order-by-direction
  table of length-l component vectors. Component i in direction j uses
  `[priors[k,j][i], priors[k+1,j][i]]` as its two seeds. Slot 2 of direction j
  is `["freq", freq_lst[3-j]]`.
- Special mode (`ind !== nothing`): `priors[q+1]` is a component-by-mode matrix
  of β-derivative bounds, and `ind` selects its mode column. The β package
  uses explicit seeds and transverse frequency `Mθ`. The θ package contains
  deferred `"L∞_freq"` lists with powers `Mθ^(p+1:p+2)` and `Mθ^(0:p)`
  (entrywise).

**Deferred packages.** The ScalarFunc wrapper resolves the θ lists after
refining β: it changes their tags to `"Wk∞"` and scales them by the refined
β-order-0 and β-order-2 bounds, respectively. This special path requires
`p ≥ 2`; raw-array methods do not accept `"L∞_freq"` directly. The helper
allocates new packages without mutating `priors`, but the wrapper mutates
those packages, so build fresh ones for each refinement call.
See [the norm-refinement rationale](../docs/technical_notes/matrix_func.md).
"""
function Linf_to_paras(Wk∞_norms_lst, k::Int, freq_lst::Vector{dtype}; ind::Union{Nothing, Int}=nothing)
    if ind !== nothing
        sp_flag = true
    else
        sp_flag = false
        ind = 1
    end
    l = size(Wk∞_norms_lst[1, 1], 1)
    paras = [[] for _ in 1:l, _ in 1:2]
    for i in 1:l, j in 1:2
        if j == 2 && sp_flag
            # Deferred theta-frequency powers; scaled in the ScalarFunc wrapper.
            push!(paras[i, j], ["L∞_freq", freq_lst[2].^dt.(k-1:k)])
            push!(paras[i, j], ["L∞_freq", freq_lst[2].^dt.(0:k-2)])
        else
            push!(paras[i, j], ["Wk∞", [Wk∞_norms_lst[k, j][i, ind], Wk∞_norms_lst[k+1, j][i, ind]]])
            push!(paras[i, j], ["freq", freq_lst[3-j]])
        end
    end
    return paras
end

"""
    get_Wk∞_norm_lst(f::AbstractArray{dtype}, h::Vector{dtype}, k::Int;
                     paras=[nothing, nothing])

Return a `Vector{dtype}` of estimates for orders `0:k` along the first axis.
Entry `m+1` represents order m. `h = [h0,h1]` gives primary and transverse
mesh spacings; use `h1 = 0` when there is no transverse continuous variable.

The two slots of `paras` have different explicit-bound layouts:
- Slot 1: `["Wk∞", [Bkp1,Bkp2]]` seeds orders `k+1,k+2` and enables backward
  recursion; `["freq", M0]` instead uses the same primary frequency at every order.
- Slot 2: `["Wk∞", C]` supplies a length-`k+1` list with
  `C[m+1] ≥ ‖∂₁^m ∂₂² f‖∞`; `["freq", M1]` uses a transverse frequency.
- Either slot may be `nothing` to omit that correction. This omission requires
  a mathematical justification if a continuous bound is claimed.

The same sample array is used at every order, including every valid stencil.
In recursive mode, use at least `derivative_padding(k)` sample layers and seed
bounds covering that extended interval. Order j is certified on a nested
halo of `cld(j,2)*(cld(j,2)-1)÷2` layers, plus any extra common target halo.
Reused bounds must cover the next order's stencil domains. Independent frequency
estimates instead need ordinary stencil coverage and valid Bernstein domains.
`estimate_Wk∞_norm` checks grid length and denominator positivity at each order.
Input samples and parameter arrays are not mutated by this method.

The ScalarFunc wrapper must resolve `"L∞_freq"` tags before this raw-array call.
See [the note](../docs/technical_notes/matrix_func.md) for domain and frequency
conditions.
"""
function get_Wk∞_norm_lst(f::AbstractArray{dtype}, h::Vector{dtype}, k::Int; 
    paras::Vector=[nothing, nothing])

    @assert 0 <= k < 20 "Derivative order must satisfy 0 ≤ k < 20"
    result = zeros(dtype, k+3)
    paras_lst = [[] for _ in 1:k+1]
    rec_flag = false

    # build parameter list for each derivative order
    for i in 1:2
        if paras[i] === nothing
            for m in 1:k+1
                push!(paras_lst[m], nothing)
            end
        else
            @assert isa(paras[i], Vector) "paras[i] must be a vector when provided"
            if paras[i][1] == "freq"
                # same frequency bound for all orders
                M = paras[i][2]
                for m in 1:k+1
                    push!(paras_lst[m], ["freq", M])
                end
            elseif i == 2
                # explicit Wk∞ bounds provided for each order
                @assert paras[i][1] == "Wk∞" "paras[i][1] must be 'freq' or 'Wk∞' when provided"
                Wk∞_bound = paras[i][2]
                for m in 1:k+1
                    push!(paras_lst[m], ["Wk∞", Wk∞_bound[m]])
                end
            else
                # recursive mode: highest two bounds given
                @assert paras[i][1] == "Wk∞" "paras[i][1] must be 'freq' or 'Wk∞' when provided"
                rec_flag = true
                result[end-1: end] = paras[i][2]
            end
        end
    end

    # backward recursion from order k to 0
    for m in k+1:-1:1
        if rec_flag
            pushfirst!(paras_lst[m], ["Wk∞", result[m+2]])
        end
        result[m] = estimate_Wk∞_norm(f, h, m-1; paras=paras_lst[m])
    end

    return result[1:end-2]
end

"""
    estimate_Wk∞_norm(f::AbstractArray{dtype}, h, k::Int;
                      paras=[nothing, nothing])

Form a single-order estimate from finite differences along the first array axis.
Use `h = [h0,h1]` for primary/transverse spacings, or scalar `h0` for `h1 = 0`.
Other array axes index samples or independent columns; their geometry is not
inferred by this function.

Let `a0 = (k+3)*h0^2/24` and `a1 = h1^2/8`. Each of the two `paras` slots is:
- `["Wk∞", C]`: add `C*a0` (slot 1) or `C*a1` (slot 2).
  C bounds the supremum norm of `∂₁^(k+2)f` or `∂₁^k∂₂²f`, respectively.
- `["freq", M]`: subtract `a0*M^2` (slot 1) or `a1*M^2` (slot 2) from the denominator.
- `nothing`: omit the corresponding correction.

Return a `dtype` scalar equal to the maximum absolute finite difference plus
the explicit allowances, divided by that denominator. With both slots `nothing`,
this is only a sampled finite-difference maximum.

For a continuous bound, stencil centers must bracket the target domain and
auxiliary bounds must cover the stencils. Frequency correction requires the
Bernstein domain conditions in the note below; M is the maximum wavenumber
in the actual coordinate.

Checks `0 ≤ k < 20`, `size(f,1) ≥ 10*(k+1)`, positive primary spacing,
recognized parameter tags, and a strictly positive denominator.
No input is mutated. See [the proof](../docs/technical_notes/matrix_func.md).
"""
function estimate_Wk∞_norm(f::AbstractArray{dtype}, h::Union{dtype, Vector{dtype}}, k::Int; 
    paras::Vector=[nothing, nothing])
    @assert 0 <= k < 20 "Derivative order must satisfy 0 ≤ k < 20"
    N = size(f, 1)
    @assert N ≥ 10 * (k + 1) "N too small for this stencil"
    if isa(h, Vector{dtype})
        h0, h1 = h
    else
        h0, h1 = h, zero(dtype)
    end
    Dk_f = finite_difference(f, h0, k)
    result = maximum(abs, Dk_f)

    a0 = h0^2 * dt(k + 3) / dt(24)
    L∞_fac = one(dtype)

    if paras[1] !== nothing
        @assert isa(paras[1], Vector) "paras[1] must be a vector when provided"
        if paras[1][1] == "Wk∞"
            # Add the finite-difference and between-center interpolation allowances.
            Wk∞_bound = paras[1][2]
            result += Wk∞_bound * a0
        else
            # Move the frequency-based correction into the denominator.
            @assert paras[1][1] == "freq" "paras[1][1] must be 'freq' or 'Wk∞' when provided"
            M0 = paras[1][2]
            L∞_fac -= a0 * M0^2
        end
    end

    if paras[2] !== nothing
        @assert isa(paras[2], Vector) "paras[2] must be a vector when provided"
        if paras[2][1] == "Wk∞"
            Wk∞_bound = paras[2][2]
            result += Wk∞_bound * h1^2 / dt(8)
        else
            @assert paras[2][1] == "freq" "paras[2][1] must be 'freq' or 'Wk∞' when provided"
            M1 = paras[2][2]
            L∞_fac -= (h1 * M1)^2 / dt(8)
        end
    end

    @assert L∞_fac > 0 "L∞_fac must be positive"
    return result /L∞_fac
end

"""
    add_error(mat::AbstractMatrix{Interval{Float64}},
              err_mat::AbstractMatrix{Interval{Float64}})

Return a new matrix `hull.(mat-err_mat, mat+err_mat; dec=:auto)`; neither input is mutated.
Supply matching shapes and entrywise nonnegative error allowances.

If `mat` encloses the computed quantity and the upper endpoint of each
`err_mat` entry bounds an additional absolute error, the returned intervals
enclose the quantity with that error included. `mat` need not be a point
interval.
"""
function add_error(mat::AbstractMatrix{Interval{Float64}}, err_mat::AbstractMatrix{Interval{Float64}})
    return hull.(mat .- err_mat, mat .+ err_mat; dec=:auto)
end


"""
    save_matrix(path::String, dict::Dict{String,<:Any})

Write a MAT file. A top-level value of type `Interval{Float64}` or
`AbstractArray{Interval{Float64}}` is replaced by `key_lo` and `key_hi` fields
containing its endpoint values. Other values are passed through unchanged;
nested containers are not recursively converted.

The input dictionary is not mutated. Avoid names that collide with generated
`_lo`/`_hi` keys. The function asserts a `.mat` suffix, writes through `matwrite`,
and returns that call's result. It does not create the parent directory.
"""
function save_matrix(path::String, dict::Dict{String, <:Any})
    @assert endswith(path, ".mat") "File must be a .mat file"
    new_dict = Dict{String, Any}()
    for key in keys(dict)
        if isa(dict[key], Interval{Float64}) || isa(dict[key], AbstractArray{Interval{Float64}})
            # Split a scalar interval or interval array into endpoint fields.
            lo = inf.(dict[key])
            hi = sup.(dict[key])
            new_dict["$(key)_lo"] = lo
            new_dict["$(key)_hi"] = hi
        else
            new_dict[key] = dict[key]
        end
    end
    matwrite(path, new_dict)
end

"""
    interval_matrix(lo::Union{AbstractArray,Real},
                    hi::Union{AbstractArray,Real}; check=true)

Construct `interval.(lo,hi)`. Scalars produce a scalar interval; arrays produce
an interval array with the broadcast result shape. `size(lo) == size(hi)` is
required and a mismatch raises `DimensionMismatch`.

With `check=true`, any entry satisfying `lo > hi` raises `ErrorException`.
This is a strict comparison with no floating-point tolerance. `check=false`
skips this explicit ordering check, not the interval constructor's own behavior.
Inputs are not mutated.
"""
function interval_matrix(lo::Union{AbstractArray, Real}, hi::Union{AbstractArray, Real}; check=true)
    size(lo) == size(hi) || throw(DimensionMismatch("lo and hi size mismatch"))

    if check
        # Strict ordering check; no tolerance is added.
        bad = findall(lo .> hi)
        isempty(bad) || error("Found lo > hi at $(first(bad)) (and $(length(bad))-1 more).")
    end

    return interval.(lo, hi) 
end

"""
    read_matrix(path, key_lst)

Read the requested MAT fields in the order of `key_lst`.
A direct key takes precedence and is converted elementwise using `dt`.
Otherwise, both `key_lo` and `key_hi` must exist: interval mode reconstructs
their enclosure with `interval_matrix`, while Float64 mode returns their midpoint.

Return the object itself for one requested key, or a `Vector{Any}` of objects
for multiple keys (an empty vector for none). Objects may be scalars or arrays.
The function asserts the `.mat` suffix and the presence of both endpoint keys
when a direct key is absent; MAT reading and reconstruction can also fail.
No data are written and no file fields are changed.
"""
function read_matrix(path, key_lst)
    @assert endswith(path, ".mat") "File must be a .mat file"
    d = matread(path)
    results = []
    for key in key_lst
        if key in keys(d)
            push!(results, dt.(d[key]))
        else
            # Read endpoint fields; return intervals or midpoints according to dtype.
            @assert ("$(key)_lo" in keys(d)) && ("$(key)_hi" in keys(d)) "Key $key not found in file."
            lo, hi = d["$(key)_lo"], d["$(key)_hi"]
            if dtype == Float64
                push!(results, (lo .+ hi) / 2)
            else
                push!(results, interval_matrix(lo, hi))
            end
        end
    end
    if length(results) == 1
        return results[1]
    else
        return results
    end
end

"""
    verify_spd_by_ldlt(A::AbstractMatrix{dtype})

Attempt an unpivoted LDLᵀ certificate using A's diagonal and lower triangle.
Allocates L and D without modifying A. Checks that A is square; symmetry is
assumed, not checked, and the upper triangle is not used.

Return `true` if every pivot has finite endpoints and `inf(Dk) > 0`. Otherwise
print the first failing pivot and return `false`. In interval mode, success certifies each symmetric
point matrix enclosed by the input. Failure is inconclusive and does not prove
that a represented matrix is indefinite. Float64 arithmetic does not provide
the interval certificate.

See [the LDLᵀ argument](../docs/technical_notes/matrix_func.md).
"""
function verify_spd_by_ldlt(A::AbstractMatrix{dtype})
    n, m = size(A)
    @assert n == m "A must be square"

    L = Matrix{dtype}(undef, n, n)
    D = Vector{dtype}(undef, n)

    # Initialize L to the identity in the active numeric type.
    @inbounds for i in 1:n, j in 1:n
        L[i,j] = (i == j) ? one(dtype) : zero(dtype)
    end

    # LDLᵀ factorization (no pivoting)
    @inbounds for k in 1:n
        # Compute D[k] = A[k,k] - sum_{j<k} L[k,j]^2 * D[j]
        s = zero(dtype)
        for j in 1:k-1
            s += (L[k,j] * L[k,j]) * D[j]
        end
        Dk = A[k,k] - s
        D[k] = Dk

        if !(isfinite(inf(Dk)) && isfinite(sup(Dk)) && inf(Dk) > 0)
            println("Failed at k=$k: pivot interval D[$k] is not finite and strictly positive. D[$k]=$Dk.")
            return false
        end

        # Compute L[i,k] for i>k:
        # L[i,k] = (A[i,k] - sum_{j<k} L[i,j]*L[k,j]*D[j]) / D[k]
        for i in k+1:n
            t = zero(dtype)
            for j in 1:k-1
                t += (L[i,j] * L[k,j]) * D[j]
            end
            num = A[i,k] - t
            L[i,k] = num / Dk
        end
    end

    return true
end

"""
    verify_general_eig(A, B, eig_vec, eig_val, thr)

The last argument is the scalar candidate upper bound, denoted by τ in the
technical note. Construct `C = A*eig_vec` and
`S = thr*B - A + C*Diagonal(inv.(eig_val))*C'`, then return
`verify_spd_by_ldlt(S)`. Inputs are not mutated.

Required layouts: A and B are `n × n`, eig_vec is `n × m`, and eig_val has m
entries. For the Rayleigh-quotient interpretation, A and B represent symmetric
matrices, B is positive definite, and the eigenvalue entries exclude zero.
The application uses positive retained eigenvalues. Only the assembled S is
checked; these input hypotheses, eigenpair accuracy, and orthonormality are
not verified here.

In interval mode, success certifies τ as an upper bound on the unrestricted
deflated generalized Rayleigh quotient. This is a sufficient upper bound for
the constrained K₆ defined in
[the technical note](../docs/technical_notes/matrix_func.md); equality with
that unrestricted quotient is not assumed.
"""
function verify_general_eig(A::AbstractMatrix{dtype}, B::AbstractMatrix{dtype}, eig_vec::AbstractMatrix{dtype}, eig_val::AbstractVector{dtype}, thr::dtype)
    d = Diagonal(inv.(eig_val));
    C = A * eig_vec;
    S = thr * B - A + C * d * C'
    return verify_spd_by_ldlt(S)
end

"""
    verify_symmetric_spectrum(A::AbstractMatrix{dtype}, t::dtype)

Return the LDLᵀ test of t*I-A. For a symmetric matrix enclosed by interval inputs,
success certifies lambda_max(A)<t. This bounds the operator norm only with a
positive-semidefinite premise or a corresponding lower spectral bound.
Symmetry and input provenance are caller hypotheses. Failure is inconclusive;
inputs are unchanged. Float64 mode is exploratory.
"""
verify_symmetric_spectrum(A::AbstractMatrix{dtype}, t::dtype) = verify_spd_by_ldlt(t*one(A)-A)

"""
    verify_gram_spectrum(W::AbstractMatrix{dtype}, K::AbstractMatrix{dtype}, t::dtype)

Certify lambda_max(W^(1/2)*K*W^(1/2))<t for exact symmetric matrices enclosed by
W and K. The exact W must be a positive-semidefinite Gram matrix; it may be
singular. Equal sizes, symmetry, valid enclosures and Gram provenance are caller
hypotheses. Return false when inf(t)<=0.

For symmetric midpoint C, entrywise deviation bound E and positive weights p,
C+Diagonal((E*p)./p) is an upper matrix in PSD order. Construct V for K with unit
weights; choose weights for W using a Float64 Cholesky/eigenvalue search, then
construct U in interval arithmetic. Floating-point scores only choose weights.
Compute E from the upper endpoints of outward-rounded interval deviations
from the point midpoint, so E bounds every entrywise deviation.
The certificate consists of the interval LDL tests U>0 and t*U-U*V*U>0;
no direct positive-definite test of the input W is required.

Call with K and -K for two-sided bounds. Squared norm targets and the identity
shifts in signed quadratic applications are supplied by the caller. False is
inconclusive; inputs are unchanged. Float64 mode is exploratory. See the
[spectral certificate derivation](../docs/technical_notes/matrix_func.md#symmetric-and-gram-spectral-certificates).
"""
function verify_gram_spectrum(W::AbstractMatrix{dtype}, K::AbstractMatrix{dtype}, t::dtype)
    inf(t) > 0 || return false
    n = size(W,1); m = n÷2
    upper(C,E,p) = C + Diagonal((E*p)./p)

    c = mid.(W)
    e = sup.(abs.(W .- interval_matrix(c,c)))
    kc = mid.(K)
    ke = sup.(abs.(K .- interval_matrix(kc,kc)))
    p,best = ones(n),Inf
    V = upper(dt.(kc),dt.(ke),dt.(p))
    v = mid.(V)

    for s in range(-6,6; length=241)
        a = exp(s)
        q = vcat(fill(a,m),fill(inv(a),n-m))
        F = cholesky(Symmetric(upper(c,e,q)); check=false)
        issuccess(F) || continue
        score = eigmax(Symmetric(F.U*v*F.U'))

        if score < best
            best,p = score,q
        end
    end

    U = upper(dt.(c),dt.(e),dt.(p))
    return verify_spd_by_ldlt(U) &&
           verify_spd_by_ldlt(t*U-U*V*U)
end

# Exact range of the normalized determinant for a real symmetric 3×3 matrix.
const ACOS_INT = @interval(-1, 1)
"""
    eig_sym3(a,b,c,d,e,f)

Return eigenvalue estimates for `[a b c; b d e; c e f]`, or interval enclosures
when `dtype == Interval{Float64}`. Supply finite inputs in the active numeric
mode. Inputs are not mutated; the cubic branches follow largest, smallest,
middle order, and interval outputs may overlap.

Compute the trace center q and scale p using `abs(x)*abs(x)` for each square,
so interval square bounds stay nonnegative. If the interval p contains zero,
skip division by `2p^3` and return three copies of one Gershgorin enclosure
covering all eigenvalues. Otherwise use the trigonometric cubic formula,
intersecting its interval acos argument with `[-1,1]` using `dec=:auto`.

Float64 mode returns `[q,q,q]` when the computed p equals zero exactly;
otherwise it clips the acos argument to `[-1,1]`. Its results are approximations.
See [the derivation and interval fallback](../docs/technical_notes/matrix_func.md).
"""
function eig_sym3(a,b,c,d,e,f)
    q = (a + d + f) / dt(3)
    aq, dq, fq = a - q, d - q, f - q

    # Multiplying abs(x) by itself keeps interval square bounds nonnegative.
    aa = abs(aq)
    dd = abs(dq)
    ff = abs(fq)
    bb = abs(b)
    cc = abs(c)
    ee = abs(e)

    p = sqrt(
        (aa*aa + dd*dd + ff*ff +
         dt(2)*(bb*bb + cc*cc + ee*ee)) / dt(6)
    )

    if dtype == Interval{Float64} && in_interval(0, p)

        # Avoid division by an interval containing zero; enclose all eigenvalues
        # using the lower and upper endpoints of the Gershgorin row intervals.
        ab = abs(b)
        ac = abs(c)
        ae = abs(e)

        l1 = a - ab - ac
        u1 = a + ab + ac

        l2 = d - ab - ae
        u2 = d + ab + ae

        l3 = f - ac - ae
        u3 = f + ac + ae

        lo = min(inf(l1), inf(l2), inf(l3))
        hi = max(sup(u1), sup(u2), sup(u3))

        λ = interval(lo, hi)
        # Repeat the common enclosure; this does not imply equal eigenvalues.
        return [λ, λ, λ]

    elseif dtype == Float64 && p == 0
        # Exact floating-point zero test; no small-scale tolerance is used.
        return [q, q, q]
    
    end

    detAqI =
        aq * (dq*fq - e*e) -
        b  * (b*fq - c*e) +
        c  * (b*e  - c*dq)

    two_p = dt(2)*p
    r = detAqI / (two_p*p*p)

    if dtype == Interval{Float64}
        r = intersect_interval(r, ACOS_INT; dec=:auto)
    else
        r = max(-1, min(r, 1))
    end

    ϕ = acos(r) / dt(3) / Pi

    λ1 = q + two_p * cospi(ϕ)
    λ2 = q + two_p * cospi(ϕ + dt(2)/dt(3))
    λ3 = q + two_p * cospi(ϕ + dt(4)/dt(3))

    return [λ1, λ2, λ3]
end

"""
    vec_to_mat(bound_lst::Matrix{dtype})

Return a `Vector{Any}` of newly allocated `3 × 3` symmetric matrices, one per
input row. Each row must have six entries ordered
`(a11,a12,a13,a22,a23,a33)`; the layout is `[a11 a12 a13; a12 a22 a23; a13 a23 a33]`.
The input is not mutated.
"""
function vec_to_mat(bound_lst::Matrix{dtype})
    result = []
    for each in eachrow(bound_lst)
        a11, a12, a13, a22, a23, a33 = each
        mat = [
            a11  a12  a13
            a12  a22  a23
            a13  a23  a33
        ]
        push!(result, mat)
    end
    return result
end
