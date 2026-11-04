# ============================================================
# My_Bessel.jl
#
# Implements the Bessel / radial-basis layer: half-integer Bessel evaluation,
# root equations, Y/X/Z radial channels and derivative bounds, and the radial
# sign extension. Key capabilities:
#   - `my_besselj`: dispatches on Float64, Arb, or Interval{Float64} to produce
#     a provably correct enclosure when dtype = Interval{Float64}.
#   - `bessel_W2_bound_interval`: analytic supremum bound on |J_{l+1/2}''(r)|.
#   - `rigorous_root_interval`: local root enclosure on a supplied continuous, pole-free bracket.
#   - `func_Z`, `func_Y`: radial boundary-condition equations whose roots are the
#     spectral parameters α.
#   - `Ur_func`, `eval_Ur`: construct the radial Y/X/Z basis channels on a grid.
#   - `eval_Uθ`: evaluates the angular basis rows.
#   - `Ur_bound`, `Uθ_bound`, `Ur_bound_lst`: individual-channel derivative bounds.
# Finite-rank packing, reconstruction, Q multiplication, and Gram matrices are in My_Assembly.jl.
#
# Mathematical background (full derivations live in the Markdown docs, not here):
#   - docs/technical_notes/bessel_basis.md   : Bessel / radial-basis layer, packing and parity conventions.
#   - docs/technical_notes/finite_rank_assembly.md : Gram-matrix assembly and finite-rank reconstruction.
#
# Dependencies: My_Fourier.jl, Matrix_Func.jl.
# Comments below are intentionally short "road signs" pointing into those docs;
# keep derivations in the Markdown, not in the source.
# ============================================================

# --- 1. Bessel evaluation ---

const DEFAULT_PREC = 128

"""
Return a certified upper bound for `sup |J_{l+1/2}''(r)|` over `r ∈ [a, b]`.

The bound evaluates the two decreasing terms at `a` (worst case) and the increasing term
at `b`. Used by the interval path of `my_besselj` to bound the chord remainder. All
arithmetic is performed in `Arb` at `prec` bits, so the result is a provably correct upper
bound. See `docs/technical_notes/bessel_basis.md`, section "Bessel derivative enclosure —
`bessel_W2_bound_interval`", for the derivation from the paper's integral representation (`int rep`) and the constants.

# Arguments
- `ν::Real`: Bessel order; must be a half-integer `ν = l + 1/2` with `l ≥ 0`.
- `a::Arb`: lower endpoint of the interval (`a > 0`).
- `b::Arb`: upper endpoint (`b ≥ a`).
- `prec::Int`: working precision in bits; default `DEFAULT_PREC = 128`.

# Returns
- `Arb`: provably correct upper bound for `sup_{r ∈ [a,b]} |J_{l+1/2}''(r)|`.

# Throws
- `AssertionError`: if `a ≤ 0`, `b ≤ 0`, or `a > b`.
"""
function bessel_W2_bound_interval(ν::Real, a::Arb, b::Arb; prec::Int = DEFAULT_PREC)
    # interval must satisfy 0 < a ≤ b
    @assert Arblib.is_positive(a) "need a > 0"
    @assert Arblib.is_positive(b) "need b > 0"
    @assert a <= b

    # ν = l + 1/2
    l = round(Int, ν - 0.5)
    @assert ν == l + 0.5 && l ≥ 0

    # π
    pi_arb = Arb(prec=prec)
    Arblib.const_pi!(pi_arb)

    # sup over [a,b]:
    # decreasing terms → evaluate at a
    # increasing term  → evaluate at b
    return sqrt(Arb(2)/(pi_arb*(2l+1))) *
           (inv(Arb(4)*a^(Arb(3)/2)) + inv(sqrt(Arb(3))*a^(Arb(1)/2)) + b^(Arb(1)/2)/sqrt(Arb(5)))
end

"""
Evaluate `J_ν(r)` by dispatching on the argument type.

`Float64` returns a floating-point approximation; `Arb` returns a ball enclosure.
The `Interval{Float64}` method uses Arb endpoint evaluations and a second-derivative
chord remainder to enclose all interior values. It supports the project's positive
half-integer orders, with an exact-zero argument or a strictly positive interval.
See `docs/technical_notes/bessel_basis.md`, sections 3–4.

# Arguments
- `ν::Real`: Bessel order; for interval input, `ν = l + 1/2`, `l ≥ 0`.
- `r`: `Float64`, `Arb`, or `Interval{Float64}` argument.
- `prec::Int`: Arb precision for the Arb/interval methods; default `DEFAULT_PREC = 128`.

# Returns
- The Bessel value in the argument's numeric type. Both Arb and interval results
  are enclosures; Float64 results are approximations.
"""
function my_besselj(ν::Real, r::Interval{Float64}; prec::Int = DEFAULT_PREC)
    # If r = 0, return 0
    if is_zero(r)
        return @interval(0)
    end
    @assert inf(r) > 0 "Interval must be either [0,0] or strictly positive"

    # Convert Float64 interval endpoints to Arb (rigorous enclosures at higher precision)
    lo = Arb(inf(r); prec = prec)
    hi = Arb(sup(r); prec = prec)

    # Evaluate J_ν at both endpoints with Arb
    val_lo = my_besselj(ν, lo; prec = prec)
    val_hi = my_besselj(ν, hi; prec = prec)

    # Upper bound on sup_{x ∈ [lo, hi]} |d²/dr² J_ν(x)|.
    bound = bessel_W2_bound_interval(ν, lo, hi; prec = prec)

    # Remainder bound for enclosing J_ν on [lo, hi] using endpoints:
    #   max deviation from the chord ≤ (sup |J''|) * (hi-lo)^2 / 8
    err = bound * ((hi - lo) ^ 2) / Arb(8)

    # Enlarge endpoint enclosures by the remainder to cover interior points
    Arblib.add_error!(val_lo, err)
    Arblib.add_error!(val_hi, err)

    # Convert Arb enclosures back to Float64 endpoints
    val_lo = Interval{Float64}(val_lo)
    val_hi = Interval{Float64}(val_hi)

    # Final Float64 interval enclosure of J_ν([inf(r), sup(r)])
    return hull(val_lo, val_hi; dec=:auto)
end

function my_besselj(ν::Real, r::Arb; prec::Int = DEFAULT_PREC)
    # Convert ν to Arb
    ν_arb = Arb(ν; prec = prec)

    # Compute Bessel function value
    val_arb = Arb(prec = prec)
    Arblib.hypgeom_bessel_j!(val_arb, ν_arb, r; prec = prec)
    return val_arb
end

function my_besselj(ν::Real, r::Float64)
    # Compute Bessel function value directly for Float64 input
    return besselj(ν, r)
end

# --- 2. Root equations and loading ---

"""
Compute the toroidal (Z-family) boundary-condition function `f_Z(l, x)`.

`f_Z(l, x) = x · J_{l−1/2}(x) / J_{l+1/2}(x) − (l+1)`. Its positive zeros are the toroidal
spectral parameters `α`. Comes from the paper's boundary condition `eq:alphaZ` after the Bessel
recurrence removes `J'`; the ratio is only interpreted on pole-free brackets where
`J_{l+1/2} ≠ 0`. See `docs/technical_notes/bessel_basis.md`, section "Root equations and root completeness".

# Arguments
- `l::Int`: spherical-harmonic degree.
- `x::dtype`: evaluation point.

# Returns
- `dtype` value of `f_Z(l, x)`.
"""
function func_Z(l::Int, x::dtype)
    # For integer l, the expression (l - 0.5) is exact (no floating-point rounding).
    return x * my_besselj(l - 0.5, x) / my_besselj(l + 0.5, x) - dt(l + 1)
end

"""
Compute the poloidal (Y-family) boundary-condition function `f_Y(l, x)`.

Implements the paper's poloidal boundary condition `eq:alphaY` in recurrence form. It has two pole families: the
zeros of `J_{l+1/2}` (shared with `func_Z`) and an extra rational pole at the radius
`γ = √((l−1)l(l+2))`, so the root search must bracket around `γ` as well. For `l = 1` the
correction degenerates to `+ x²`. Zeros of `f_Y` are the poloidal spectral parameters `α`
shared by the Y- and X-channels. See `docs/technical_notes/bessel_basis.md`, section "Root equations and root
completeness".

# Arguments
- `l::Int`: spherical-harmonic degree.
- `x::dtype`: evaluation point.

# Returns
- `dtype` value of `f_Y(l, x)`.
"""
function func_Y(l::Int, x::dtype)
    # For integer l, the expression (l - 0.5) is exact (no floating-point rounding).
    if l >= 2
        return x * my_besselj(l - 0.5, x) / my_besselj(l + 0.5, x) - dt(l + 2) + 
               (x^2 - dt((l - 1) * l)) * (x^2 - dt((l - 1) * (l + 2))) / (x^2 - dt((l - 1) * l * (l + 2)))
    else
        return x * my_besselj(l - 0.5, x) / my_besselj(l + 0.5, x) - dt(l + 2) + x^2
    end
end


"""
Return the rigorous sign of an interval `v`: `+1` if `v ⊂ (0, ∞)`, `-1` if `v ⊂ (−∞, 0)`,
and `0` if `v` straddles or touches zero (sign undetermined). Used by the bracketing root
finder to test interval sign changes without ever guessing across zero.
"""
@inline rsign(v) = inf(v) > 0 ? 1 : (sup(v) < 0 ? -1 : 0)

"""
Return whether `fa` and `fb` have strictly opposite interval signs.

Returns `false` if either interval touches zero. Opposite signs imply a root only
when the underlying function is continuous between the evaluation points.
"""
@inline sign_change(fa::Interval{Float64}, fb::Interval{Float64}) =
    rsign(fa) * rsign(fb) == -1

"""
Enclose a root by interval bisection on a supplied continuous, pole-free bracket.

The caller must exclude poles from the full expanded bracket
`[inf(x0) - δ, sup(x0) + δ]`; this routine checks endpoint signs, not continuity.
It preserves a strict sign change while bisecting. An undecidable midpoint sign
returns the current bracket, which may be wider than `tol`. This local enclosure
does not establish uniqueness or completeness of a root list; see
`docs/technical_notes/bessel_zeros_certificate.md` for those conditions.

# Arguments
- `l::Real`: parameter passed to `func`.
- `x0::Interval{Float64}`: candidate interval; exact zero is returned unchanged
  without evaluating `func`, for the known zero-mode convention.
- `func::Function`: interval-valued function `(l, x)` with rigorous evaluations.
- `δ`: expansion at each endpoint; default `dt("1e-10")`.
- `prec`: accepted but unused; it does not set Bessel-evaluation precision.
- `tol::Real`: stopping threshold; default `1e-20`. The loop tests
  `inf(abs(b - a))`, so no final-diameter guarantee is made.

# Returns
- An interval containing at least one root under the stated assumptions.

# Throws
- `AssertionError`: if the initial endpoint signs are not strictly opposite.
"""
function rigorous_root_interval(l::Real, x0::Interval{Float64}, func::Function; δ=dt("1e-10"), prec=200, tol=1e-20)
    if is_zero(x0)
        return zero(dtype) # Preserve the known zero-mode entry without evaluating func.
    end

    # Check for sign change in the initial interval
    a = dt(inf(x0)) - δ
    b = dt(sup(x0)) + δ
    fa = func(l, a)
    fb = func(l, b)
    @assert sign_change(fa, fb) "No sign change in initial interval! Increase δ."

    # Bisection method
    while inf(abs(b - a)) >= tol
        m = (a + b) / dt(2)
        fm = func(l, m)
        if sign_change(fa, fm)
            b, fb = m, fm
        elseif sign_change(fm, fb)
            a, fa = m, fm
        else
            # midpoint sign is inconclusive; keep current certified bracket
            break
        end
    end

    # Return the final interval enclosure
    m = hull(a, b; dec=:auto)
    return m
end

"""
Check a floating-point sign change around a positive candidate `x0`.

A cheaper, non-interval sign-change check (used when reconstructing the root lists in
`mat_to_lst`): confirms `func(l, x0·(1−δ))` and `func(l, x0·(1+δ))` have opposite signs.
When `x0 ≤ 0`, return it without evaluating `func`; this retains the known zero mode.
Unlike `rigorous_root_interval`, it does not produce a verified enclosure. See
`docs/technical_notes/bessel_basis.md`, section "Root equations and root completeness".

# Arguments
- `l::Real`: parameter passed to `func`.
- `x0::Float64`: candidate root.
- `func::Function`: function `(l, x)` returning a real value.
- `δ::Real`: relative perturbation size; default `1e-10`.

# Returns
- `x0::Float64` unchanged, after a successful check when positive or directly otherwise.

# Throws
- `AssertionError`: if `x0 > 0` and no sign change is detected.
"""
function verify_root(l::Real, x0::Float64, func::Function; δ::Real=1e-10)
    if x0 > 0
        left  = func(l, x0 * (1 - δ))
        right = func(l, x0 * (1 + δ))
        @assert left * right < 0 "Sign change expected around l=$l, x0=$x0; left=$left, right=$right"
    end
    return x0
end

"""
Convert a matrix of candidate roots into a nested list of verified roots.

Reverses the list→matrix packing done in `get_bessel_zeros.ipynb`: each row `l` becomes a
list of `[α]` entries passed through `verify_func`. The first padding zero after
column 1 ends the row; a first-column zero-mode entry is retained.
This repacks and checks each candidate only; it does not establish root-list completeness
(see `docs/technical_notes/bessel_basis.md`, section "Root equations and root completeness").

# Arguments
- `mat::AbstractMatrix`: input matrix of candidate root values (row = degree `l`).
- `verify_func::Function`: function `(l, value, eq_func) -> verified_value`.
- `eq_func::Function`: equation function passed to `verify_func`.

# Returns
- A `Vector{Any}` of row lists, each holding `[α]` entries.

# Throws
- `AssertionError`: if any row produces an empty list.
"""
function mat_to_lst(mat, verify_func, eq_func)
    result = []
    for l in axes(mat, 1)
        lst = []
        for j in axes(mat, 2)
            # Early termination if zero entry is found
            if(j > 1 && is_zero(mat[l, j]))
                break
            end

            # Verify and refine each candidate root.
            zero_l = verify_func(l, dt(mat[l, j]), eq_func)

            # push a list of roots since there are other parameters related to
            # each root to be added into the list later
            push!(lst, [zero_l])
        end
        @assert !isempty(lst) "empty list when l=$l"
        push!(result, lst)
    end
    return result
end

"""
Augment parameter lists with the radial-channel coefficients `(A, B)`.

For `vari = "Y"` (poloidal: the coupled Y and X channels share one `(α, A, B)` triple),
appends the matching `(A, B)` to each `paras[l][j]` where `α = paras[l][j][1]`:
```
A = −(α² − (l+2)(l−1)) · J_{l+1/2}(α) / α^{3/2}
B =  (α² − l(l−1)(l+2)) / α^{3/2}
```
The `α = 0` zero mode (the `l = 1` constant mode) is handled separately: `A = B = 1`. For
`vari = "Z"` (toroidal), appends a single trailing `1`, i.e. the convention `A ≡ 1`.
See `docs/technical_notes/bessel_basis.md`, section "Y/X/Z radial formulas and coefficients".

# Arguments
- `paras::Vector`: nested list; each `paras[l][j][1]` is the root `α`.
- `vari::String`: `"Y"` or `"Z"`.

# Returns
- The mutated `paras` (modified in-place and returned).
"""
function preprocess_para!(paras::Vector, vari::String)
    if vari == "Y"
        for l in eachindex(paras), j in eachindex(paras[l])
            α = paras[l][j][1]

            # Calculate the coefficients for X- and Y-type bases.
            if is_zero(α)
                A, B = one(dtype), one(dtype)
            else
                A = -(α^2 - dt((l+2)*(l-1))) * my_besselj(l + 0.5, α) / α^dt(3/2)
                B = (α^2 - dt(l*(l-1)*(l+2))) / α^dt(3/2)
            end

            # Add the coefficients to the list
            append!(paras[l][j], [A, B])
            if is_zero(α)
                continue
            end
        end
    else
        for l in eachindex(paras), j in eachindex(paras[l])
            push!(paras[l][j], one(dtype)) # For Z, we set the coefficient to be 1
        end
    end
    return paras
end

"""
Load Bessel zeros from a `.mat` file and return processed Y/Z parameter lists.

Reads `"Y_zeros"` and `"Z_zeros"` matrices, converts them via `mat_to_lst`,
and augments with `(A, B)` coefficients via `preprocess_para!`.

# Arguments
- `file_path::String`: path to `.mat` file containing `"Y_zeros"` and `"Z_zeros"`.
- `refine::Bool`: if `true`, enclose the root with `rigorous_root_interval`;
  if `false`, use `verify_root`.
  Defaults to `true` in interval mode and `false` in Float64 mode.
  Refinement assumes the expanded brackets are pole-free; the loader does not check this.

# Returns
- `(Y_paras, Z_paras)`: nested parameter lists ready for use in `Ur_func`.
"""
function load_besselj_zeros(file_path::String; refine::Bool=(dtype != Float64))
    # Load Bessel zeros from a file
    Y_zeros, Z_zeros = read_matrix(file_path, ["Y_zeros", "Z_zeros"])

    # Refine or just verify the roots
    if refine
        verify_func = rigorous_root_interval
    else
        verify_func = verify_root
    end

    # Convert the matrix to the nested lists
    Y_paras = mat_to_lst(Y_zeros, verify_func, func_Y)
    Z_paras = mat_to_lst(Z_zeros, verify_func, func_Z)

    # # Preprocessing
    Y_paras = preprocess_para!(Y_paras, "Y")
    Z_paras = preprocess_para!(Z_paras, "Z")
    return Y_paras, Z_paras
end

# --- 3. Basis evaluation ---

# EPS shifts only the odd-parity sign factor sign(r + EPS).
# The analytic origin value is selected separately by is_zero(r).
const EPS = dt("1e-16")
"""
Evaluate one radial channel `R^Y`, `R^X`, or `R^Z` for a single spherical-harmonic degree.

`"Y"`, `"X"`, `"Z"` are channel/component labels, **not** three independent basis families:
a Y-root yields one *poloidal* basis element whose Y and X channels are coupled (sharing one
`(α, A, B)` triple); a Z-root yields one *toroidal* basis element with a single Z channel.
The half-integer Bessel formula (paper label `lem:lap eig`) is extended to signed grids via `abs(r)` plus
a parity sign, and the `r = 0` value is an intentional analytic limit. See `docs/technical_notes/bessel_basis.md`,
sections "Basis elements vs. component labels", "Y/X/Z radial formulas and coefficients", and
"Sign / parity extension, and field parity".

# Arguments
- `l::Int`: spherical-harmonic degree.
- `paras::Vector`: basis parameters, `(α, A)` for `"Z"` and `(α, A, B)` for
  `"X"` or `"Y"`.
- `r::Vector{dtype}`: radial evaluation points, possibly signed for parity
  extension.
- `vari::String`: radial channel selector, one of `"X"`, `"Y"`, or `"Z"`.

# Returns
- `Vector{dtype}`: values of `Uᵣ` at the requested points.

# Throws
- `AssertionError`: if `vari` is not a supported channel or the parameter types
  do not match `dtype`.

# Notes
- Interval samples must be exact zero or lie strictly on one side of zero;
  negative samples must remain negative after adding `EPS` for the parity sign.
  `is_zero` selects the analytic origin value: exact `[0,0]` in interval mode,
  `abs(r) < 1e-14` in Float64 mode. `EPS` is not this zero-test threshold.
"""
function Ur_func(l::Int, paras::Vector, r::Vector{dtype}, vari::String)
    # The signed extension lets the radial formulas be evaluated on `abs(r)` while preserving parity.
    # For "Z": U_r is even if l is even, odd if l is odd. For "X","Y": the opposite.
    # See docs/technical_notes/bessel_basis.md, "Sign / parity extension, and field parity".
    n = (vari == "Z") ? l : (l - 1)
    sign_r = isodd(n) ? dt.(sign.(r .+ EPS)) : ones(dtype, length(r))
    # Angular X is -S_l'; compensate to obtain the physical component.
    if vari == "X"
        sign_r .*= -one(dtype)
    end
    r = abs.(r)

    # The channel selector fixes both the boundary condition and the parameter layout.
    @assert vari in ["X", "Y", "Z"]
    if vari == "Z"
        α, A = paras
        @assert isa(α, dtype) && isa(A, dtype)
    else
        # paras for vari="X" shares the same parameter layout as "Y" (α, A, B), 
        # and is typically provided via Y_paras.
        α, A, B = paras
        @assert isa(α, dtype) && isa(A, dtype) && isa(B, dtype)
    end

    # evaluate formula depending on variant
    if l == 1 && is_zero(α)
        return ones(dtype, length(r)) .* sign_r
    elseif vari == "X"
        result = A * (r.^dt(l-1)) + (B/dt(l*(l+1))) * (α*my_besselj.(l - 0.5, α*r)
                - dt(l) * my_besselj.(l + 0.5, α*r)./r) ./ sqrt.(r)
    elseif vari == "Y"
        result = dt(l) * A * (r.^dt(l-1)) + B * my_besselj.(l + 0.5, α*r) ./ (r.^dt(3/2))
    else
        result = A * my_besselj.(l + 0.5, α*r) ./ sqrt.(r)
    end

    # Handle r=0 case. The value at `r = 0` is set by a special-case limit:
    # only the `(l == 1, vari in ("X","Y"))` branch yields a nonzero value;
    # all other cases are set to zero at `r = 0`.
    mask = is_zero.(r)
    if l == 1 && vari in ["X", "Y"]
        result[mask] .= A + sqrt(dt(2)/Pi)/dt(3)*B*α^dt(3/2)
    else
        result[mask] .= zero(dtype)
    end
    return result .* sign_r
end

"""
Evaluate the radial basis matrices `Uᵣ` for all spherical degrees and roots.

One matrix per degree `l`; column `j` corresponds to root index `j`. With the same `Y_paras`,
`Ur_Y_lst[l][:,j]` and `Ur_X_lst[l][:,j]` are the two coupled radial channels of one *poloidal*
element, while `Ur_Z_lst[l][:,j]` is the *toroidal* element's Z channel. See `docs/technical_notes/bessel_basis.md`,
section "Indexing and packing conventions".

# Arguments
- `r::Vector{dtype}`: radial grid.
- `paras_lst::Vector`: nested parameter list; `paras_lst[l][j]` contains `(α, A[, B])`.
- `vari::String`: `"X"`, `"Y"`, or `"Z"`.

# Returns
- A `Vector{Any}` holding one `Matrix{dtype}` per degree `l`; each matrix has shape
  `(length(r), number_of_zeros_for_l)`.
"""
function eval_Ur(r::Vector{dtype}, paras_lst::Vector, vari::String)
    Ur_lst = []
    N_r = length(r)
    for l in eachindex(paras_lst)
        lst = zeros(dtype, N_r, length(paras_lst[l]))
        for j in eachindex(paras_lst[l])
            lst[:, j] = Ur_func(l, paras_lst[l][j], r, vari)
        end
        push!(Ur_lst, lst)
    end
    return Ur_lst
end

"""
Sample spherical-harmonic angular rows on `θ` using `conversion_mat` and `trig_func`.

Boundary-code order is ["00", "01", "10", "11"]. Cosine blocks contain S_l=P_l(cos θ);
sine blocks contain H_l=−S_l′ for both X/Z channels. These are scalar angular factors.
Ur_func already negates the poloidal X radial coefficient, so their product is the
physical θ component. Use physical Q samples and coefficient rows matching the complete
basis, including the toroidal whole-mode sign. No Q-frame conversion is performed here.
See docs/technical_notes/bessel_basis.md, sections 6 and 9.

# Arguments
- `θ::Vector{dtype}`: angular grid.
- `l::Int`: maximum degree to support (`l ≥ 1`).

# Returns
- A `Vector{Any}` of four matrices, each of shape `((l+1)÷2, length(θ))`.
  Within the selected block, degree `l′` uses row `(l′+1)÷2`.
"""
function eval_Uθ(θ::Vector{dtype}, l::Int)
    bdry_lst = ["00", "01", "10", "11"]
    res_lst = []

    # Allocate lh=(l+1)÷2 rows in each degree-parity block.
    # For odd l, the even-degree blocks also include degree l+1.
    lh = (l+1)÷2
    for bdry in bdry_lst
        conv_mat = conversion_mat(lh, bdry, 1)
        res = conv_mat * trig_func(lh+(bdry=="11"), θ, "I", bdry)'
        push!(res_lst, res)
    end
    return res_lst
end

# --- 4. Basis derivative bounds ---

"""
Analytic W^{k,∞} bound for one radial channel `U_r` (Y, X, or Z) of degree `l`.

Returns a bound for ‖∂_r^k U_r‖_{L^∞} on normalized radii `|r| ≤ 1.1`,
using the signed regular extension. Padded domains must remain within this
radius because the polynomial terms use a fixed `1.1` allowance. "Y"/"X"
are coupled poloidal channels and "Z" is toroidal. The formulas and derivative
bounds are explained in `docs/technical_notes/bessel_basis.md`, section 6.
The Z bound assumes the standard `A = 1` normalization from `preprocess_para!`
and includes the scaling factor `α^(k+1/2)` for derivatives in normalized radius.

# Arguments
- `paras`: parameters for the radial basis.
    • `vari == "Z"`: `(α, A)`
    • `vari ∈ {"X","Y"}`: `(α, A, B)`
- `vari`: channel label ("X", "Y", or "Z") — Y/X are the two channels of one poloidal element, Z the toroidal channel.
- `l`: spherical degree index.
- `k`: derivative order (k ≥ 0).

# Returns
- A scalar upper bound (dtype) for `‖∂_r^k U_r‖_{L^∞}` on the stated domain.
"""
function Ur_bound(paras::Vector, vari::String, l::Int, k::Int)
    if vari == "Z"
        α, A = paras
        @assert isa(α, dtype) && isa(A, dtype)
    else
        α, A, B = paras
        @assert isa(α, dtype) && isa(A, dtype) && isa(B, dtype)
    end

    # `interval_prod` below bounds the polynomial-term derivative without Int64 overflow and
    # returns 0 once the derivative order exceeds the polynomial degree.
    if α == 0
        return A
    elseif vari == "X"
        return sqrt(dt(2)) * α^dt(3/2 + k) * abs(B) / (sqrt(Pi * dt(2k + 1)) * dt((2l + 1) * l * (l+1))) * 
            (dt(l + 1) / sqrt(dt(2l - 1)) + dt(l) / sqrt(dt(2l + 3)) ) + interval_prod(l-k, l-1) * abs(A) * (dt("1.1") ^ dt(l-k))
    elseif vari == "Y"
        return sqrt(dt(2)) * α^dt(3/2 + k) * abs(B) / (sqrt(Pi * dt(2k + 1)) * dt(2l + 1)) * 
            (dt(1) / sqrt(dt(2l - 1)) + dt(1) / sqrt(dt(2l + 3))) + interval_prod(l-k, l) * abs(A) * (dt("1.1") ^ dt(l-k))
    else
        return sqrt(dt(2) / (Pi * dt((2k + 1) * (2l + 1)))) * α^dt(k) * sqrt(α)
    end
end

# Product ∏_{n=a}^{b} n accumulated in dtype: avoids Int64 overflow at high degree. An empty range
# (a > b) returns 1; a range that contains 0 returns 0, so vanishing higher derivatives are handled.
function interval_prod(a::Integer, b::Integer)
    if a > b
        return dt(1)
    end

    p = dt(1)
    for n in a:b
        p *= dt(n)
    end
    return p
end

"""
Bound angular derivatives of `S_i(θ)=P_i(cos θ)` on θ ∈ [0, π/2].

The table satisfies `‖∂_θ^m S_i‖_{L^∞} ≤ i^m`. For the X/Z factor
`H_i=-∂_θ S_i`, the order-m bound uses column `m+2` and equals `i^(m+1)`.
See `docs/technical_notes/bessel_basis.md` for the angular bounds.

# Arguments
- `l::Int`: maximum spherical degree to bound (degrees `1, 2, ..., l`).
- `k::Int`: derivative order, with `0 ≤ k < 5`; columns cover `m = 0, 1, ..., k+1`.
  Integer powers must fit in `Int`; the guard on `k` alone does not check the degree.

# Returns
- `Matrix{Int}` of shape `l × (k+2)`. Row `i` corresponds to spherical degree `i`
  (the table starts at degree 1, so row `i` is degree `i` — not row `l+1` ↔ degree `l`);
  column `m+1` corresponds to derivative order `m`; entry `B[i, m+1] = i^m`.
"""
function Uθ_bound(l::Int, k::Int)
    @assert k<5 "Int64 is not safe for a large k=$k."
    l_lst = reshape(collect(1:l), :, 1)
    k_lst = reshape(collect(0:k+1), 1, :)
    return l_lst.^(k_lst)
end

"""
Generate W^{k,∞}-style radial bound tables for the Y/X/Z radial channels from parameter lists.
For each degree index i and each root j within that degree, this routine evaluates
Ur_bound(paras, vari, i, m) for m = 0..k and stores the results column-wise.
# Arguments
- Y_paras: parameter lists for the poloidal Y/X channels (shared parameters), grouped by degree i
- Z_paras: parameter lists for the toroidal Z channel, grouped by degree i
- k: maximum derivative/order index (computes m = 0..k)
# Returns
- Y_bd_lst, X_bd_lst, Z_bd_lst: lists of matrices (k+1)×n_basis(i), grouped by degree
"""
function Ur_bound_lst(Y_paras::Vector, Z_paras::Vector, k::Int)
    LY = length(Y_paras)
    LZ = length(Z_paras)
    Y_bd_lst = []
    X_bd_lst = []
    Z_bd_lst = []

    # part1: (Y, X) contributions. Y and X share Y_paras[i][j] (two channels of one poloidal
    # element). Each table is (k+1)×n: rows = derivative order m, columns = root j.
    for i in 1:LY
        Li = length(Y_paras[i]) # number of basis functions in block i
        cur_r_bd_Y = zeros(dtype, k+1, Li)
        cur_r_bd_X = zeros(dtype, k+1, Li)
        for j in 1:Li
            for m in 0:k
                cur_r_bd_Y[m+1, j] = Ur_bound(Y_paras[i][j], "Y", i, m)
                cur_r_bd_X[m+1, j] = Ur_bound(Y_paras[i][j], "X", i, m)
            end
        end
        push!(Y_bd_lst, cur_r_bd_Y)
        push!(X_bd_lst, cur_r_bd_X)
    end

    # part2: Z contributions, similar as above
    for i in 1:LZ
        Li = length(Z_paras[i])
        cur_r_bd = zeros(dtype, k+1, Li)
        for j in 1:Li
            for m in 0:k
                cur_r_bd[m+1, j] = Ur_bound(Z_paras[i][j], "Z", i, m)
            end
        end
        push!(Z_bd_lst, cur_r_bd)
    end
    return Y_bd_lst, X_bd_lst, Z_bd_lst
end
