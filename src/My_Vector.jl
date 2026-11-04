# ============================================================
# My_Vector.jl
#
# Defines VectorFunc: an ordered collection of named ScalarFunc components.
# Used for velocity/pressure fields, residuals, full gradients, and symmetric
# tensors such as Q. Provides MAT I/O, divergence-free reconstruction, component
# evaluation, inner products, derivative-bound tables, and arithmetic.
#
# Dependencies: My_Scalar.jl (and transitively My_Fourier.jl, Matrix_Func.jl).
# Numeric arrays use dtype. Interval mode encloses arithmetic; continuous bounds
# also require the documented domain, frequency, derivative, and quadrature data.
# See docs/technical_notes/scalar_vector.md for the field workflow and
# docs/technical_notes/matrix_func.md for the estimates.
# ============================================================

# Each configuration lists u1,u2,u3,p in order; each descriptor is [radial, angular].
# In the odd block, ["10","00"] assigns the first radial code to l=1 and the
# second to l=3,5,... . Other radial descriptors use one code for all columns.
const BDRY_LST_SET = [
    [[["10", "00"], "10"], [["10", "00"], "01"], ["00", "00"], ["00", "10"]],
    [["01", "11"], ["01", "00"], ["01", "01"], ["10", "11"]],
    [["00", "11"], ["00", "00"], ["00", "01"], ["10", "11"]]
]

"""
Ordered collection of named `ScalarFunc` components with a shared scale.

Can hold velocity and pressure, three residual components, nine full-gradient
entries, or six symmetric-tensor entries (including Q and the symmetric gradient).
The container does not encode their geometric interpretation or tensor weights.

# Fields
- `keys::Vector{String}`: component order used by integer indexing, operators,
  and the component vectors returned by norm routines.
- `dict::Dict{String,ScalarFunc}`: components accessible by name.
- `scl_fac::dtype`: scale supplied to the components when constructed.
- `bdry_index::Union{Int,Nothing}`: selected `BDRY_LST_SET` configuration.
  `nothing` permits explicit component boundary descriptors; it does not imply
  that those descriptors are absent.

See `docs/technical_notes/scalar_vector.md` for representations and field conventions.
"""
mutable struct VectorFunc
    keys::Vector{String}
    dict::Dict{String, ScalarFunc}
    scl_fac::dtype
    bdry_index::Union{Int, Nothing}
end

"""
Construct an ordered container with unset coefficient and sample arrays.

`keys` gives component order; `scl_fac` is passed to each scalar component.
With `bdry_index=nothing`, component boundary descriptors are initially unset.
Otherwise the index must select an entry of `BDRY_LST_SET`, and `keys` must
have four entries. Those boundary lists are assigned positionally, so use the
intended order `u1,u2,u3,p`. Returns the new `VectorFunc`.
"""
function VectorFunc(keys::Vector{String}; bdry_index::Union{Int, Nothing}=nothing, scl_fac::dtype=1.0)
    dict = Dict{String, ScalarFunc}()
    if bdry_index !== nothing
        @assert 1 <= bdry_index <= length(BDRY_LST_SET) "Invalid bdry_index"
        @assert length(BDRY_LST_SET[bdry_index]) == length(keys) "Length of bdry_lst does not match number of keys"
    end
    for (i, key) in enumerate(keys)
        bdry = (bdry_index === nothing) ? nothing : BDRY_LST_SET[bdry_index][i]
        dict[key] = ScalarFunc(key; bdry=bdry, scl_fac=scl_fac)
    end
    return VectorFunc(keys, dict, scl_fac, bdry_index)
end

"""
Construct components from keyword-argument matrices in keyword order.

`domain="space"` attaches samples and `domain="frequency"` attaches coefficients;
values must be `AbstractMatrix{dtype}` and are stored by reference via `set_func!`.
`scl_fac` is passed to each component.

`bdry_lst`, when supplied, must contain one descriptor per component in that
order. A supplied `bdry_index` selects `BDRY_LST_SET` and overrides `bdry_lst`.
The result stores only the supplied representation; no transform is performed.
"""
function VectorFunc(scl_fac::dtype, domain::String; bdry_index::Union{Int, Nothing}=nothing, 
    bdry_lst::Union{Vector, Nothing}=nothing, kwargs...)
    key_lst = string.(collect(keys(kwargs)))
    result = VectorFunc(key_lst; scl_fac=scl_fac)
    for (key, value) in kwargs
        set_func!(result[string(key)], domain, value)
    end
    if bdry_index !== nothing
        result.bdry_index = bdry_index
        bdry_lst = BDRY_LST_SET[bdry_index]
    end
    if bdry_lst !== nothing
        @assert length(bdry_lst) == length(result.keys) "Length of bdry_lst does not match number of keys"
        for (i, key) in enumerate(result.keys)
            set_bdry!(result.dict[key], bdry_lst[i])
        end
    end
    return result
end

"""
Load the coefficient field `u1,u2,u3,p` and its scale from a MAT file.

Use `domain="frequency"` for the project's coefficient inputs. `read_matrix`
converts the arrays to `dtype`; `bdry_index` selects their boundary configuration.
The loader attaches angular evaluation conversion matrices to `u1` and `u2`
without rewriting their coefficients, then evaluates samples using the saved
`shape` (point counts). Although `domain` is forwarded to construction and
transformation, the conversion setup requires coefficient arrays to exist.

With the default `divfree=false`, `set_div_free!` subsequently replaces
`u2.freq_func` from `u1.freq_func`. Existing `u2.space_func` is not refreshed;
use `interpolate` or `transform!` when samples of the reconstructed field are
needed. Set `divfree=true` to skip reconstruction when loading a corrected file.
The exact coefficient relation is idempotent, but repeating interval evaluation
need not reproduce identical interval endpoints.

Returns `(up, lambda)` if reading and indexing the optional eigenvalue succeeds;
otherwise returns `up`. The optional read is caught; required-input errors propagate.
"""
function from_mat_file(filename::String, bdry_index::Int; domain::String="frequency", divfree::Bool=false)
    u1, u2, u3, p, scl_fac, shape = read_matrix(filename, ["u1", "u2", "u3", "p", "scl_fac", "shape"])
    N0, N1 = dtype == Float64 ? Int64.(shape) : Int64.(inf.(shape))

    up = VectorFunc(scl_fac, domain; bdry_index=bdry_index, u1=u1, u2=u2, u3=u3, p=p)

    # Attach spherical-to-trigonometric evaluation matrices; keep the stored coefficients.
    set_conv_mat!(up["u1"])
    set_conv_mat!(up["u2"])

    transform!(up, domain; shape=(N0, N1))
    divfree || set_div_free!(up)  # impose divergence-free condition (skip if file already stores it)

    try
        lambda = read_matrix(filename, ["lambda"])
        return up, lambda[1]
    catch
        return up
    end
end

"""
Load the six independent entries of the auxiliary symmetric tensor Q.

Reads `A11,A12,A13,A22,A23,A33`, `scl_fac`, and `shape`, then constructs a
`VectorFunc` in that component order with explicit boundary descriptors.
`domain="frequency"` (default) stores coefficients; `domain="space"` stores
supplied samples. No transform or grid evaluation is performed.
The saved shape is read but does not determine any newly evaluated grid.

Returns the container with only the selected representation set. The Q tensor
shares component names with the symmetric gradient; this loader does not compute
that gradient.
"""
function load_grad_U(file_path::String; domain::String="frequency")
    A11, A12, A13, A22, A23, A33, R_max, shape = read_matrix(
        file_path, ["A11","A12","A13","A22","A23","A33","scl_fac","shape"])
    N0, N1 = dtype == Float64 ? Int64.(shape) : Int64.(inf.(shape))
    # bdry_lst = [["10", "11"], ["10", "00"], ["00", "01"], ["10", "11"], ["00", "10"], ["00", "11"]]
    bdry_lst = [
        ["10", "11"],  # Q11
        ["10", "00"],  # Q12
        ["10", "01"],  # Q13
        ["10", "11"],  # Q22
        ["10", "10"],  # Q23
        ["10", "11"],  # Q33
    ]

    Q = VectorFunc(R_max, domain; bdry_lst=bdry_lst,
            A11=A11, A12=A12, A13=A13, A22=A22, A23=A23, A33=A33)

    return Q
end

"""
Return the stored `ScalarFunc` by component name or integer position.

Integer indexing resolves through `self.keys`; out-of-range positions raise
`BoundsError`, and an unknown name fails the membership assertion.
The component is returned by reference, not copied.
"""
function getindex(self::VectorFunc, key::Union{Int,String})
    isa(key, Int) && (key = self.keys[key])
    @assert key in self.keys "Invalid key"
    return self.dict[key]
end

"""
Assign a `ScalarFunc` to a component by name or position.

# Arguments
- `self::VectorFunc`: the vector function container (modified in-place).
- `value::ScalarFunc`: the new component to assign.
- `key::Union{Int,String}`: integer index into `self.keys` or the component name string.

# Returns
- The assigned `ScalarFunc`, following Julia `setindex!` convention.

# Throws
- `BoundsError`: if an integer index is outside `self.keys`.
- `AssertionError`: if a component name is not found.

# Notes
- This replaces the component reference in-place and does not copy `value`.
"""
function setindex!(self::VectorFunc, value::ScalarFunc, key::Union{Int,String})
    isa(key, Int) && (key = self.keys[key])
    @assert key in self.keys "Invalid key"
    self.dict[key] = value
end

"""
Transform each component in place and return the same `VectorFunc`.

`domain` names the source: `"frequency"` evaluates coefficients into samples,
while `"space"` transforms samples into coefficients. `shape=(Nβ,Nθ)` counts
grid points and is needed for evaluation when samples are unset; otherwise the
scalar methods infer it from their sample arrays.

The space-to-frequency branch enables spherical-harmonic conversion for `u1`
and `u2`. Frequency-to-space evaluation uses each component's existing `conv_mat`.
The source representation remains stored; see scalar `transform!` for preconditions.
"""
function transform!(self::VectorFunc, domain::String; shape=nothing)
    for key in self.keys
        # Enable spherical conversion for u1/u2 in the space-to-frequency branch.
        conv_flag = key in ["u1", "u2"]
        transform!(self.dict[key], domain; shape=shape, conv_flag=conv_flag)
    end
    return self
end

"""
Evaluate every component from coefficients into a new `VectorFunc`.

Supply coefficient arrays, boundary descriptors, and the desired `r_pt, θ_pt`.
For compactified profile/eigenpair fields, `r_pt` contains β values.
`keys=["I","I"]` evaluates the stored functions; other keys select basis operators.

The returned container has samples of shape `length(r_pt) × length(θ_pt)`,
unset coefficients and conversion matrices, and the source component order and
scale. Component boundary descriptors are attached only for identity keys;
`bdry_index` remains unset. The source object is unchanged.

With `pt_flag=true`, the stored arrays instead contain radial samples and
angular trigonometric coefficients, after any angular conversion. No angular
sampling has occurred; see scalar `interpolate`.
"""
function interpolate(self::VectorFunc, r_pt::Vector{dtype}, θ_pt::Vector{dtype}; keys::Vector{String}=["I","I"], pt_flag=false)
    result = VectorFunc(self.keys; scl_fac=self.scl_fac)
    @threads for key in self.keys
        set_func!(result.dict[key], "space", interpolate(self[key], r_pt, θ_pt; keys=keys, pt_flag=pt_flag))
        if keys == ["I","I"]
            set_bdry!(result.dict[key], self.dict[key].bdry)
        end
    end
    return result
end

"""
Reconstruct `u2.freq_func` from `u1.freq_func` with `calc_Utheta_divfree`.

Uses the compactified coefficient convention so that the represented physical
velocity is divergence free. Requires both coefficient arrays and `u1.bdry`.
The two-entry radial descriptor is reversed to match the helper's ordering.
Only `u2.freq_func` is replaced; existing samples are not refreshed.
See Section 5 of `docs/technical_notes/fourier_basis.md` for the modewise relation
and the scalar angular table `-S'_l`, with its matching coefficient sign.

`print_flag=true` prints the maximum absolute coefficient change.
Returns the helper's intermediate matrices `(DL[, DL2], DR)`, excluding the
reconstructed array that has already been assigned to `u2`.
"""
function set_div_free!(self::VectorFunc; print_flag::Bool=false)
    u1, u2 = self["u1"].freq_func, self["u2"].freq_func
    @assert self["u1"].bdry !== nothing "Boundary conditions must be set to enforce divergence-free condition"
    bdry = self["u1"].bdry[1]
    bdry = isa(bdry, String) ? [bdry] : reverse(bdry)
    result = calc_Utheta_divfree(u1, bdry)
    u2_tld = result[end]# The last entry is the new u2
    set_func!(self["u2"], "frequency", u2_tld)
    print_flag && println("u2 difference: ", norm(vec(u2 - u2_tld), Inf))
    return front(result) # Exclude the last entry, which is u2_tld
end


"""
Sum the scalar-component quadrature inner products, with optional error bounds.

Both fields must have identical ordered keys, matching unpadded uniform grids,
and the same scale. Only the key condition is checked here. `domain=[Lβ,Lθ]`
gives coordinate interval lengths, defaulting to `[π/2,π/2]`; grid interval
counts must be positive multiples of six. Samples must already contain any
radial weight required for the physical inner product.

Every stored component contributes, including pressure when present. Six-entry
symmetric tensors receive no extra factor on off-diagonal terms.

`W_norms_lst=[Wself,Wother]` supplies pure derivative bounds through order 8.
Each table has two direction columns and component-vector entries in `keys`
order: `W[m+1,j][i]` bounds order m of component i in direction j.
These bounds must apply to the weighted fields on the integration domain.
In interval mode, the method adds the quadrature-error allowance, including
the same `4π*scl_fac^3` factor as the scalar sums. Float64 mode returns the
quadrature sum without that enlargement. Omitting the tables also omits it.

Returns a `dtype` scalar. See `docs/technical_notes/matrix_func.md` for the
quadrature-error argument.
"""
function inner_product(self::VectorFunc, other::VectorFunc; W_norms_lst=nothing, domain::Vector{dtype}=[Pi/dt(2), Pi/dt(2)])
    result = zero(dtype)
    k = 8
    @assert all(self.keys == other.keys) "Keys do not match for space domain"
    for i in 1:length(self.keys)
        result += inner_product(self[i], other[i]; domain=domain)
    end
    if W_norms_lst !== nothing
        w_bds = zeros(dtype, k + 1, 2)
        w_bds[:, 2] .= one(dtype)
        w_bds[1, 1] = one(dtype)
        W_normsT = [W' for W in W_norms_lst[1]]
        total_W_norm = multinomial_sum(k, w_bds, W_normsT, W_norms_lst[2])

        L0, L1 = domain
        N0, N1 = size(self[1].space_func) .- 1
        h0, h1 = L0 / dt(N0), L1 / dt(N1)
        # Scale the quadrature allowance by the same 4π·scl_fac³ as the sum.
        err = (total_W_norm[1] * h0^dt(k) + total_W_norm[2] * h1^dt(k)) * dt(3) / dt(2800) * L0 * L1 * dt(4) * Pi * self.scl_fac^3
        if dtype == Interval{Float64}
            result = hull(result - err, result + err; dec=:auto)
        end
    end
    return result  
end

"""
Compute component supremum estimates, derivative-bound tables, or an L2 norm.

All modes read `space_func`; `domain=[Lβ,Lθ]` gives the unpadded coordinate
lengths and defaults to `[π/2,π/2]`. Let d be the number of ordered components.

# Modes and returns
- `"LInf"`: length-d vector in `self.keys` order. `paras=[Mβ,Mθ]` supplies
  frequencies for continuous corrections; `paras=nothing` gives sample maxima
  without between-point corrections.
- `"W8Inf"`, or `"W"` followed by an integer k and `"Inf"`: a `(k+1) × 2`
  table of length-d vectors. Entry `[m+1,j][i]` bounds the pure order-m derivative
  of component i in direction j (β, θ for compactified fields). It is not a
  single Sobolev norm and contains no separate mixed-derivative rows.
- `"L2"`: one `dtype` scalar, the square root of the sum of all component
  inner products. `paras=W` supplies derivative bounds through order 8 for
  quadrature error; without it, the result uses only the quadrature sum.

# W-mode parameters and grids
A two-entry `Vector{dtype}` selects corrected component supremum estimates
followed by Bernstein frequency powers; use an unpadded grid and justified
frequency ceilings on a domain supporting that inequality.
A d × 2 package matrix from `Linf_to_paras(priors,k+2,freqs)` selects directional
finite-difference refinement for `0 < k < 20`. Pass the actual `padding` at
each end of both axes; recursive refinement requires at least
`derivative_padding(k)` layers. Special deferred packages are mutated by the
scalar wrapper, so build fresh ones for each call. `paras` is required for W modes.

The LInf and W paths require positive interior interval counts divisible by
100. L2 instead requires the unpadded quadrature grid and the weighting and
compatibility conditions of `inner_product`. In interval mode its error
allowance is applied only when bounds are supplied. See
`docs/technical_notes/scalar_vector.md` and `docs/technical_notes/matrix_func.md`.
"""
function norm_lst(self::VectorFunc, mode::String; paras=nothing,
    domain::Vector{dtype}=[Pi/dt(2), Pi/dt(2)], padding::Int=0)
    if mode == "L2"
        W_norms_lst = paras === nothing ? nothing : [paras, paras]
        return sqrt(inner_product(self, self; W_norms_lst=W_norms_lst, domain=domain))
    elseif mode == "LInf"
        @assert padding == 0 "Frequency-based estimates require unpadded samples"
        l = length(self.keys)
        results = dtype[]
        if paras === nothing
            paras = [nothing, nothing]
        else
            M0, M1 = paras
            paras = [["freq", M0], ["freq", M1]]
        end
        for i in 1:l
            push!(results, get_Wk∞_norm_lst(self[i], 0; paras_lst=paras, L=domain))
        end
        return results
    else
        @assert mode[1] == 'W' && mode[end-2:end] == "Inf" "Invalid norm mode"
        k = parse(Int, mode[2:end-3])
        l = length(self.keys)

        if isa(paras, Vector{dtype})
            @assert padding == 0 "Frequency-based estimates require unpadded samples"
            L∞_norms = norm_lst(self, "LInf"; paras=paras, domain=domain)
            return [L∞_norms .* paras[j]^dt(i) for i in 0:k, j in 1:2]
        end

        @assert 0 < k < 20 "Refinement requires 0 < k < 20; use LInf for order zero"
        Wk∞_norms_lst = []
        for i in 1:l
            push!(Wk∞_norms_lst, get_Wk∞_norm_lst(self[i], k; paras_lst=paras[i, :], L=domain, padding=padding))
        end
        results = [zeros(dtype, l) for _ in 1:k+1, _ in 1:2]

        for i in 1:l, j in 1:2, m in 0:k
            results[m+1, j][i] = Wk∞_norms_lst[i][m+1, j]
        end
        return results
    end
end

"""
Trim the spatial-domain border of every component in a `VectorFunc` in-place.

Calls `trim_border!` on each `ScalarFunc` with the same `num` argument.

# Arguments
- `self::VectorFunc`: container to modify (modified in-place).
- `num::Union{Int, Tuple{Int,Int}}`: rows/cols to remove from each side; see `ScalarFunc.trim_border!`.

# Returns
- `self` (the modified container).
"""
function trim_border!(self::VectorFunc, num::Union{Int, Tuple{Int,Int}})
    for k in self.keys
        trim_border!(self.dict[k], num)
    end
    return self
end


"""
Save selected field arrays and metadata to a MAT file; return the constructed dictionary.

`domain="frequency"` writes each component's `freq_func`; other domain values
select `space_func`. Supply arrays in the selected representation. Any set
`conv_mat` is also included, along with `scl_fac` and optional `lambda`.
`save_matrix` writes scalar intervals and interval arrays as `_lo`/`_hi` pairs;
the returned dictionary still contains the unsplit `dtype` values.

If `shape=nothing`, nonempty stored samples are required and their first
component's dimensions are written as `shape`. If `shape` is supplied, this
implementation does not write a `shape` field. The parent directory must exist;
file validation and writing are delegated to `save_matrix`.
"""
function save(self::VectorFunc, filename::String; domain::String="frequency", shape::Union{Tuple{Int,Int}, Nothing}=nothing, lambda::Union{dtype, Nothing}=nothing)
    data = Dict{String,Any}()
    for k in self.keys
        scalar = self.dict[k]
        data[k] = (domain == "frequency") ? scalar.freq_func : scalar.space_func 
        scalar.conv_mat !== nothing && (data[k * "_conv_mat"] = scalar.conv_mat)
    end
    # Keep scalar dtype values; save_matrix splits scalar intervals into _lo/_hi too.
    data["scl_fac"] = self.scl_fac
    if shape === nothing
        @assert !isempty(self.keys) && self[1].space_func !==
            nothing "No keys in solution or no space function available"
        N0, N1 = size(self[1].space_func)
        data["shape"] = [N0, N1]
    end
    lambda !== nothing && (data["lambda"] = lambda)
    save_matrix(filename, data)
    return data
end

"""
Copy selected components into a new `VectorFunc`, preserving their order.

`keys` accepts an integer, string, vector of either, `UnitRange`, or `nothing`
for all components. Each selected scalar's numerical arrays are copied;
its boundary descriptor remains shared as described by scalar `copy`.
The scale and `bdry_index` are retained.

With a non-`nothing` `bdry_index`, the constructor still requires four keys,
so selecting fewer components fails its length assertion. Use existing keys;
invalid positions, names, or selector types raise indexing, assertion, or
argument errors according to the branch.
"""
function copy(self::VectorFunc; keys=nothing)
    copied_keys = keys === nothing ? copy(self.keys) : begin
        if isa(keys, Int)
            [self.keys[keys]]
        elseif isa(keys, String)
            @assert keys in self.keys "Key not found in solution"
            [keys]
        elseif isa(keys, Vector)
            [isa(k, Int) ? self.keys[k] : k for k in keys]
        elseif isa(keys, UnitRange)
            self.keys[keys]
        else
            error("Invalid keys argument")
        end
    end
    new = VectorFunc(copied_keys; bdry_index=self.bdry_index, scl_fac=self.scl_fac)
    for k in copied_keys
        new[k] = copy(self.dict[k])
    end
    return new
end

"""
Add `other` to `self` in place and return `self`.

Matching names use scalar `add_equal!`, with its representation and grid/basis
compatibility requirements. New names are appended in `other.keys` order and
their scalar components are copied. Container scale and `bdry_index` are unchanged;
compatible component meanings and scales are the caller's responsibility.
`self += other` uses `+` and reassigns a new container instead.
"""
function add_equal!(self::VectorFunc, other::VectorFunc)
    for k in self.keys
        if haskey(other.dict, k)
            add_equal!(self.dict[k], other.dict[k])
        end
    end
    for k in other.keys
        if !(k in self.keys)
            push!(self.keys, k)
            self.dict[k] = copy(other.dict[k])
        end
    end
    return self
end

"""
Return a new `VectorFunc` equal to `self + other`.

# Arguments
- `self::VectorFunc`: first operand.
- `other::VectorFunc`: second operand.

# Returns
- New `VectorFunc` holding the componentwise sum.
"""
function +(self::VectorFunc, other::VectorFunc)
    result = copy(self)
    add_equal!(result, other)
    return result
end

"""
Multiply every component by `scalar::Union{dtype,Int}` in place; return `self`.

Each scalar method updates its set representations and retains metadata.
`self *= scalar` uses `*` and reassigns a new container instead.
"""
function multiply_equal!(self::VectorFunc, scalar::Union{dtype,Int})
    for k in keys(self.dict)
        multiply_equal!(self.dict[k], scalar)
    end
    return self
end

"""
Return a new `VectorFunc` equal to `self * scalar`.

# Arguments
- `self::VectorFunc`: source container.
- `scalar::Union{dtype,Int}`: scalar multiplier.

# Returns
- New `VectorFunc` with all components scaled by `scalar`.
"""
function *(self::VectorFunc, scalar::Union{dtype,Int})
    result = copy(self)
    multiply_equal!(result, scalar)
    return result
end

"""
Subtract `other` from `self` in place and return `self`.

Uses `add_equal!(self, other * dt(-1))`, with the same component and
representation requirements as `add_equal!`. `self -= other` uses `-`
and reassigns a new container instead.
"""
function minus_equal!(self::VectorFunc, other::VectorFunc)
    add_equal!(self, other * dt(-1))
    return self
end

"""
Return a new `VectorFunc` equal to `self - other`.

# Arguments
- `self::VectorFunc`: minuend.
- `other::VectorFunc`: subtrahend.

# Returns
- New `VectorFunc` holding the componentwise difference.
"""
function -(self::VectorFunc, other::VectorFunc)
    return self + other * dt(-1)
end

"""
Divide every component by `scalar::Union{dtype,Int}` in place; return `self`.

Asserts `scalar != 0` and calls `multiply_equal!` with its reciprocal.
`self /= scalar` uses `/` and reassigns a new container instead.
"""
function divide_equal!(self::VectorFunc, scalar::Union{dtype,Int})
    @assert scalar != 0 "Division by zero"
    multiply_equal!(self, inv(scalar))
    return self
end

"""
Return a new `VectorFunc` equal to `self / scalar`.

# Arguments
- `self::VectorFunc`: source container.
- `scalar::Union{dtype,Int}`: nonzero scalar divisor.

# Returns
- New `VectorFunc` with all components scaled by `1/scalar`.

# Throws
- `AssertionError`: if `scalar == 0`.
"""
function /(self::VectorFunc, scalar::Union{dtype,Int})
    @assert scalar != 0 "Division by zero"
    return self * inv(scalar)
end
