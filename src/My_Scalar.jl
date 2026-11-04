# ============================================================
# My_Scalar.jl
#
# Defines ScalarFunc: one scalar component with independently stored coefficients
# and grid samples. Components can represent velocity, pressure, residuals, or
# tensor entries. Provides basis evaluation, transforms, weighted quadrature,
# derivative-bound refinement, and arithmetic.
#
# Dependencies: My_Fourier.jl, Matrix_Func.jl.
# Numeric arrays use dtype. Interval mode encloses arithmetic; continuous bounds
# also require the documented domain, frequency, derivative, and quadrature data.
# See docs/technical_notes/scalar_vector.md for the field workflow and
# docs/technical_notes/matrix_func.md for the estimates.
# ============================================================

"""
    grid(L, N, k=0)

Uniform grid on `[0,L]` with `N` intervals and `k` padding points at each end.
The spacing is `L/N` and the returned vector has `N+1+2k` entries.
"""
function grid(L, N::Int, k::Int=0)
    @assert N > 0 "N must be a positive interval count"
    @assert k >= 0 "Padding must be nonnegative"
    return dt.(-k:N+k) .* (L/dt(N))
end

# BDRY_LIST: list of valid boundary conditions, 0 for Dirichlet,
#  1 for Neumann, two entries for two boundaries.
const BDRY_LIST = ["00", "01", "10", "11"]

"""
Validate `[bdry_r, bdry_θ]` and return `nothing`.

Each boundary code is one of `"00"`, `"01"`, `"10"`, or `"11"`: its characters
specify Dirichlet (`0`) or Neumann (`1`) conditions at 0 and π/2.
`bdry_θ` is one code. `bdry_r` is either one code for all columns or
`[first_column, remaining_columns]`. In the odd spherical-harmonic block,
these correspond to l=1 and l=3,5,..., respectively.

Throws an error for an invalid descriptor. This checks its format, not whether
existing coefficients or samples satisfy the stated boundary conditions.
"""
function check_bdry(bdry::Vector)
    if length(bdry)==2 && bdry[2] in BDRY_LIST
        # ok
    else
        error("Invalid boundary: $bdry")
    end

    if bdry[1] ∉ BDRY_LIST
        if isa(bdry[1], Vector) && length(bdry[1])==2 && all(x->x in BDRY_LIST, bdry[1])
            # ok
        else
            error("Invalid boundary: $(bdry[1])")
        end
    end
end

"""
Mutable representation of one scalar component, including residual or tensor entries.

`freq_func` holds coefficients; `space_func` holds samples. Either can be
`nothing`, and updating one does not refresh or invalidate the other.
The object does not store grid coordinates or identify physical/radial weights.

# Fields
- `name::String`: component label.
- `bdry::Union{Vector,Nothing}`: `[bdry_r, bdry_θ]`; see `check_bdry`.
- `space_func::Union{AbstractMatrix{dtype},Nothing}`: rows are radial grid points;
  columns are angular grid points.
- `freq_func::Union{AbstractMatrix{dtype},Nothing}`: rows are radial modes;
  columns are angular modes. Its shape can differ from `space_func`.
- `conv_mat::Union{AbstractMatrix{dtype},Nothing}`: right-acting angular
  conversion from spherical-harmonic coefficients to trigonometric coefficients.
- `scl_fac::dtype`: scale used in physical operators and the inner-product measure.

See `docs/technical_notes/scalar_vector.md` for the field conventions.
"""
mutable struct ScalarFunc
    name::String
    bdry::Union{Vector, Nothing}
    space_func::Union{AbstractMatrix{dtype},Nothing}
    freq_func::Union{AbstractMatrix{dtype},Nothing}
    conv_mat::Union{AbstractMatrix{dtype},Nothing}
    scl_fac::dtype
end

"""
Construct a `ScalarFunc` with a name, optional boundary descriptor, and scale.

`bdry` defaults to `nothing`; otherwise `check_bdry` validates it.
`scl_fac` defaults to `one(dtype)`. The returned object's `space_func`,
`freq_func`, and `conv_mat` are all `nothing`.
Use `set_func!` to attach arrays and `set_conv_mat!` to attach angular conversion;
`interpolate` returns evaluated values and `transform!` stores the transformed array.
"""
function ScalarFunc(name::String; bdry::Union{Vector, Nothing}=nothing, scl_fac::dtype=one(dtype))
    # Check the validity of boundary conditions
    bdry !== nothing && check_bdry(bdry)
    return ScalarFunc(name, bdry, nothing, nothing, nothing, scl_fac)
end

"""
Assign `func::AbstractMatrix{dtype}` to the selected representation and return it.

`domain="space"` sets `space_func`; `domain="frequency"` sets `freq_func`.
The supplied array is stored by reference, without copying. The other array,
boundary descriptor, and conversion matrix are unchanged.
An unrecognized domain raises an error.
"""
function set_func!(self::ScalarFunc, domain::String, func::AbstractMatrix{dtype})
    if domain == "space"
        self.space_func = func
    elseif domain == "frequency"
        self.freq_func = func
    else
        error("Invalid domain: $domain")
    end
end

"""
Set and return the boundary descriptor of a `ScalarFunc`.

Requires `self.bdry === nothing` and a descriptor accepted by `check_bdry`.
The supplied vector is stored by reference; neither representation is changed.
"""
function set_bdry!(self::ScalarFunc, bdry::Vector)
    @assert self.bdry === nothing "Boundary already set"
    check_bdry(bdry)
    self.bdry = bdry
end

"""
Build, store, and return the angular evaluation conversion matrix.

Requires `freq_func` and `bdry` to be set. Uses
`conversion_mat(size(self.freq_func, 2), self.bdry[2], 1)`, mapping
spherical-harmonic coefficient rows to trigonometric coefficient rows.
The coefficient and sample arrays themselves are unchanged.
"""
function set_conv_mat!(self::ScalarFunc)
    @assert self.freq_func !== nothing "freq_func not set"
    @assert self.bdry !== nothing "boundary not set"
    self.conv_mat = conversion_mat(size(self.freq_func, 2), self.bdry[2], 1)
end

"""
Return spectral coefficients computed from the stored samples.

Requires `space_func` and `bdry`. The order is angular DFT, optional
trigonometric-to-spherical conversion, then radial DFT. A two-entry radial
boundary descriptor treats the first coefficient column separately.

`conv_flag=false` leaves the object unchanged. With `conv_flag=true`, the
routine also calls `set_conv_mat!`, updating `self.conv_mat`; that call requires
an existing `freq_func` and sizes the conversion matrix from it. The returned
coefficient array is not assigned to `self.freq_func` here; `transform!` does that.
"""
function Fourier_transform(self::ScalarFunc; conv_flag::Bool=false)
    # Fourier transform procedure: angular DFT -> conversion to 
    # spherical harmonic basis (if needed) -> radial DFT

    @assert self.space_func !== nothing && self.bdry !== nothing

    # Apply Fourier transform in θ direction
    N0, N1 = size(self.space_func)
    Fθ = Fourier_mat(N1-1, self.bdry[2])
    tmp = self.space_func * Fθ'

    # Apply conversion to spherical harmonic basis if needed
    if conv_flag
        conv_mat = conversion_mat(size(tmp,2), self.bdry[2], 0)
        tmp = tmp * conv_mat
        set_conv_mat!(self)
    end

    # Apply Fourier transform in radial direction
    if isa(self.bdry[1], String)
        Fr = Fourier_mat(N0-1, self.bdry[1])
        return Fr * tmp
    else
        # If there are two boundary conditions, we need to treat
        # the first column and the rest columns separately
        Fr1 = Fourier_mat(N0-1, self.bdry[1][1])
        Fr2 = Fourier_mat(N0-1, self.bdry[1][2])
        freq_func1 = Fr1 * tmp[:,1:1]
        freq_func2 = Fr2 * tmp[:,2:end]
        m1 = size(freq_func1, 1)
        m2 = size(freq_func2, 1)
        if m2 < m1
            freq_func2 = vcat(freq_func2, zeros(dtype, m1 - m2, size(freq_func2,2)))
        end
        return hcat(freq_func1, freq_func2)
    end
end

"""
Evaluate coefficients at the supplied radial and angular grid points.

Requires `freq_func` and `bdry`. The default `keys=["I","I"]` evaluates the
stored function; other keys apply the corresponding `trig_func` basis operators.
The radial coordinate is β for compactified profile/eigenpair fields.
No physical-field factor is inserted by the default identity keys.

Radial evaluation is followed by right multiplication by `conv_mat`, if set,
then angular evaluation. With `pt_flag=false`, the returned matrix has size
`length(r_pt) × length(θ_pt)`. With `pt_flag=true`, evaluation stops after
angular conversion: rows are radial samples and columns remain angular
trigonometric coefficients. The source object is unchanged.
"""
function interpolate(self::ScalarFunc, r_pt::Vector{dtype}, θ_pt::Vector{dtype};
        keys::Vector{String}=["I","I"], pt_flag=false)
    M0 = size(self.freq_func, 1)
    # Interpolation procedure: radial transform -> conversion to
    # Fourier basis (if needed) -> angular transform, which is
    # the reverse order of Fourier transform.

    # Apply radial transform
    if isa(self.bdry[1], String)
        u_r = trig_func(M0, r_pt, keys[1], self.bdry[1])
        tmp = u_r * self.freq_func
    else
        # If there are two boundary conditions, we need to treat
        # the first column and the rest columns separately
        u_r1 = trig_func(M0, r_pt, keys[1], self.bdry[1][1])
        u_r2 = trig_func(M0, r_pt, keys[1], self.bdry[1][2])
        t1 = u_r1 * self.freq_func[:,1:1]
        t2 = u_r2 * self.freq_func[:,2:end]
        tmp = hcat(t1, t2)
    end

    # Apply conversion to Fourier basis if needed
    self.conv_mat !== nothing && (tmp *= self.conv_mat)
    if pt_flag
        return tmp
    else
        u_θt = trig_func(size(tmp,2), θ_pt, keys[2], self.bdry[2])'
        return tmp * u_θt
    end
end

"""
Transform one representation and store the resulting array in `self`.

`domain` names the source: `"space"` assigns `Fourier_transform(self)` to
`freq_func`; `"frequency"` assigns interpolated values to `space_func`.
Returns the assigned array, not `self`; the source array remains stored.

`shape=(Nβ,Nθ)` counts points on the uniform `[0,π/2]²` evaluation grid.
If omitted, it is inferred from existing samples; supply it when samples are
unset. The space-to-frequency branch uses the stored sample shape.
`conv_flag` controls angular conversion only in that branch; frequency-to-space
uses the existing `conv_mat`. An unrecognized domain raises an error.
"""
function transform!(self::ScalarFunc, domain::String;
    shape::Union{Tuple{Int,Int},Nothing}=nothing, conv_flag::Bool=false)
    # Determine shape of the function matrix
    if shape === nothing
        @assert self.space_func !== nothing "No space function"
        N0, N1 = size(self.space_func)
    else
        N0, N1 = shape
    end

    # For space->frequency transform, use the Fourier transform
    if domain == "space"
        self.freq_func = Fourier_transform(self; conv_flag=conv_flag)
    # For frequency->space transform, use interpolation
    elseif domain == "frequency"
        # build radial and angular grids
        r_pt = grid(Pi/dt(2), N0-1)
        θ_pt = grid(Pi/dt(2), N1-1)
        # invert transform
        self.space_func = interpolate(self, r_pt, θ_pt)
    else
        error("Invalid domain: $domain")
    end
end

"""
Compute the weighted quadrature sum from two scalar sample arrays.

Supply matching samples on the same unpadded uniform `[0,L0] × [0,L1]` grid
and the same scale; these compatibility conditions are not checked here.
Both interval counts must be positive multiples of six. `domain=[L0,L1]`
defaults to `[π/2,π/2]`; `space_func` must be set for both operands.

The sum includes `sin(θ)` and `4π*self.scl_fac^3`. To represent a physical
whole-space inner product under the project's symmetry conventions, each input
must already include the radial weight `sin(β)/cos(β)^2`.

Returns a `dtype` scalar enclosing the quadrature arithmetic in interval mode.
No quadrature-error allowance is added here; the `VectorFunc` method adds it
when derivative-bound tables are supplied. See
`docs/technical_notes/matrix_func.md` for the weighted-integral derivation.
"""
function inner_product(self::ScalarFunc, other::ScalarFunc; domain::Vector{dtype}=[Pi/dt(2), Pi/dt(2)])
    A = self.space_func
    B = other.space_func

    A === nothing && error("self.space_func is nothing")
    B === nothing && error("other.space_func is nothing")

    return _inner_product_space(A, B, self.scl_fac, domain)
end

"""
The `space_func` field can hold different matrix types, so Julia cannot determine
the concrete type from `ScalarFunc` alone. Passing the arrays to a separate helper
lets Julia compile the loop specifically for their actual types, avoiding repeated
method lookups during the quadrature calculation.
"""
function _inner_product_space(
    A::AbstractMatrix{dtype},
    B::AbstractMatrix{dtype},
    scl_fac,
    domain::AbstractVector{dtype},
)
    N0, N1 = size(A)

    L0, L1 = domain
    hβ = L0 / dt(N0 - 1)
    hθ = L1 / dt(N1 - 1)

    wβ = newton_cotes_weights(N0 - 1, hβ)
    wθ = newton_cotes_weights(N1 - 1, hθ)

    s = zero(dtype)

    @inbounds for j in 1:N1
        tj = wθ[j] * mysin(dt(j - 1) * hθ)
        @simd for i in 1:N0
            s += A[i,j] * B[i,j] * wβ[i] * tj
        end
    end

    return s * dt(4) * Pi * scl_fac^3
end

"""
Estimate pure derivative bounds from a scalar component's samples.

For target order `k > 0`, supply actual `padding=p` and samples of shape
`(Nβ+1+2p, Nθ+1+2p)`. Recursive refinement requires at least
`derivative_padding(k)` layers; larger padding may certify a common extended
β domain without changing the physical lengths or mesh widths.
`L=[Lβ,Lθ]` gives the unpadded lengths. The wrapper asserts that `Nβ` and `Nθ`
are positive multiples of 100 and computes spacings from these interior counts.
It removes transverse padding for each calculation and transposes for θ.

For `k > 0`, `paras_lst=[Pβ,Pθ]` supplies two directional packages for the
raw-array `get_Wk∞_norm_lst`. The wrapper reads `Pθ[1][1]`, so the primary θ
slot must be a tagged vector rather than `nothing`. Packages from
`Linf_to_paras(priors,k+2,freqs)` seed orders k+1 and k+2. Priors must cover the stencil regions. If a θ package
uses deferred `"L∞_freq"` tags, k must be at least 2: the wrapper changes these
tags to `"Wk∞"` and scales their lists by the refined β-order-0 and β-order-2
bounds. This mutates the nested packages; prepare fresh ones for another call.
The sample array is unchanged.

Returns a `(k+1) × 2` numeric table: entry `[m+1,j]` bounds derivative order m
in coordinate j (β, θ for compactified fields). For `k == 0`, use unpadded
samples and pass the two correction slots of `estimate_Wk∞_norm` directly,
for example `[nothing,nothing]` for a sample maximum; the return is one scalar
estimate. Supply `paras_lst` explicitly: the nested default is unusable in either branch. See `docs/technical_notes/matrix_func.md` for the correction
assumptions; omitting corrections does not by itself give a continuous bound.
"""
function get_Wk∞_norm_lst(f::ScalarFunc, k::Int; 
    paras_lst::Vector=[[nothing, nothing], [nothing, nothing]],
    L::Vector{dtype}=[Pi/dt(2), Pi/dt(2)], padding::Int=0)
    @assert 0 <= k < 20 "Derivative order must satisfy 0 ≤ k < 20"
    @assert padding >= 0 "Padding must be nonnegative"
    L0, L1 = L
    A = f.space_func
    N1, N2 = size(A) .- 1
    N1_phys = N1 - 2 * padding
    N2_phys = N2 - 2 * padding
    @assert N1_phys % 100 == 0 "Expected physical radial grid count to be a multiple of 100"
    @assert N2_phys % 100 == 0 "Expected physical angular grid count to be a multiple of 100"
    @assert N1_phys > 0 "Not enough radial interior points after removing padding"
    @assert N2_phys > 0 "Not enough angular interior points after removing padding"
    # space_func includes `padding` ghost layers on both axes. The Taylor/finite-difference
    # spacing must use the physical interior grid; padding must not shrink the
    # mesh size. Using the padded transverse count can understate error terms.
    h = [L0 / dt(N1_phys), L1 / dt(N2_phys)]
    if k == 0 # Unpadded samples; the supplied slots determine the corrections.
        @assert padding == 0 "Order-zero estimates require unpadded samples"
        return estimate_Wk∞_norm(A, h, k; paras=paras_lst)
    end
    for package in paras_lst
        primary = package[1]
        recursive = primary !== nothing && primary[1] in ("Wk∞", "L∞_freq")
        required_padding = recursive ? derivative_padding(k) : cld(k, 2)
        @assert padding >= required_padding "Insufficient padding for derivative refinement"
    end
    result = zeros(dtype, k+1, 2)
    @views begin # @views can speed up slicing
        # Note that space_func stores `padding` extra grid layers on each side in both directions,
        # used for finite-difference evaluation near the boundary. When computing
        # directional derivative bounds, the redundant layers in the transverse direction
        # are removed before estimation.
        result[:, 1] = get_Wk∞_norm_lst(A[:, padding+1:end-padding], h, k; paras=paras_lst[1])
        # The second direction is transposed, so reverse h
        # Resolve deferred θ lists with the refined β-order-0 and β-order-2 bounds.
        if paras_lst[2][1][1] == "L∞_freq"
            paras_lst[2][1][1] = "Wk∞"
            paras_lst[2][1][2] .*= result[1, 1]
            paras_lst[2][2][1] = "Wk∞"
            paras_lst[2][2][2] .*= result[3, 1]
        end
        result[:, 2] = get_Wk∞_norm_lst(A[padding+1:end-padding, :]', reverse(h), k; paras=paras_lst[2])
    end 
    return result
end

"""
Replace `space_func` by a copy of its trimmed interior and return that array.

`num::Int` removes that many rows and columns from each end;
`num=(nβ,nθ)` specifies the two counts separately. Supply nonnegative counts.
The method asserts that samples exist and that at least one row and column remain.
Coefficients and metadata are unchanged.
"""
function trim_border!(self::ScalarFunc, num::Union{Int, Tuple{Int,Int}})
    @assert self.space_func !== nothing "No space function"
    if isa(num, Int)
        @assert num*2 < size(self.space_func, 1) "Trim size too large"
        @assert num*2 < size(self.space_func, 2) "Trim size too large"
        self.space_func = self.space_func[num+1:end-num, num+1:end-num]
    else
        num1, num2 = num
        @assert num1*2 < size(self.space_func, 1) "Trim size too large"
        @assert num2*2 < size(self.space_func, 2) "Trim size too large"
        self.space_func = self.space_func[num1+1:end-num1, num2+1:end-num2]
    end
end

"""
Copy the numerical arrays of a `ScalarFunc` into a new object.

Each non-`nothing` `space_func`, `freq_func`, and `conv_mat` is copied.
The boundary descriptor `bdry` is shared by reference; the name and scale
are retained. This is not a recursive deep copy of all metadata.
"""
function copy(self::ScalarFunc)
    new_space_func = (self.space_func === nothing) ? nothing : copy(self.space_func)
    new_freq_func = (self.freq_func === nothing) ? nothing : copy(self.freq_func)
    new_conv_mat = (self.conv_mat === nothing) ? nothing : copy(self.conv_mat)
    return ScalarFunc(self.name, self.bdry, new_space_func, new_freq_func,
        new_conv_mat, self.scl_fac)
end

"""
Add `other` to `self` in place and return `self`.

Each representation is updated only when that array exists in both operands.
Sample arrays must use a compatible grid; coefficient arrays must use compatible
bases and are zero-padded to the larger row and column counts when needed.
Grid, basis, and scale compatibility are the caller's responsibility.
Metadata is unchanged, and missing representations are not constructed.

`add_equal!` mutates the existing object. `self += other` instead uses `+`
and reassigns its newly allocated result.
"""
function add_equal!(self::ScalarFunc, other::ScalarFunc)
    # space_func part
    if self.space_func !== nothing && other.space_func !== nothing
        self.space_func .+= other.space_func
    end

    # freq_func part
    if self.freq_func !== nothing && other.freq_func !== nothing
        m1, n1 = size(self.freq_func)
        m2, n2 = size(other.freq_func)

        if m1 == m2 && n1 == n2
            self.freq_func .+= other.freq_func
        elseif m1 > m2 && n1 > n2
            # Add into the matching coefficient block.
            self.freq_func[1:m2, 1:n2] .+= other.freq_func
        else
            # Zero-pad both coefficient arrays to their common size.
            m = max(m1, m2)
            n = max(n1, n2)
            result = zeros(dtype, m, n)

            result[1:m1, 1:n1] .+= self.freq_func
            result[1:m2, 1:n2] .+= other.freq_func

            self.freq_func = result
        end
    end

    return self
end

"""
Return a new `ScalarFunc` equal to `self + other`.

# Arguments
- `self::ScalarFunc`: first operand.
- `other::ScalarFunc`: second operand.

# Returns
- New `ScalarFunc` holding the elementwise sum.
"""
function +(self::ScalarFunc, other::ScalarFunc)
    result = copy(self); add_equal!(result, other); return result
end

"""
Multiply every set numerical representation by `s::Real` in place; return `self`.

Metadata and unset representations are unchanged. `multiply_equal!` mutates
the existing object; `self *= s` uses `*` and reassigns a new object.
"""
function multiply_equal!(self::ScalarFunc, s::Real)
    self.space_func !== nothing && (self.space_func .*= s)
    self.freq_func !== nothing && (self.freq_func .*= s)
    return self
end

"""
Return a new `ScalarFunc` equal to `self * s`.

# Arguments
- `self::ScalarFunc`: source object.
- `s::Real`: scalar multiplier.

# Returns
- New `ScalarFunc` scaled by `s`.
"""
function *(self::ScalarFunc, s::Real)
    result = copy(self); multiply_equal!(result, s); return result
end

"""
Subtract `other` from `self` in place and return `self`.

Uses `add_equal!(self, other * dt(-1))`, with the same representation and
compatibility requirements as `add_equal!`. `self -= other` instead uses `-`
and reassigns a new object.
"""
function minus_equal!(self::ScalarFunc, other::ScalarFunc)
    add_equal!(self, other * dt(-1))
    return self
end

"""
Return a new `ScalarFunc` equal to `self - other`.

# Arguments
- `self::ScalarFunc`: minuend.
- `other::ScalarFunc`: subtrahend.

# Returns
- New `ScalarFunc` holding the elementwise difference.
"""
function -(self::ScalarFunc, other::ScalarFunc)
    return self + other * dt(-1)
end

"""
Divide every set numerical representation by `s::Real` in place; return `self`.

Asserts `s != 0` and multiplies by `inv(dt(s))`. Metadata is unchanged.
`self /= s` instead uses `/` and reassigns a new object.
"""
function divide_equal!(self::ScalarFunc, s::Real)
    @assert s != 0 "Division by zero"
    multiply_equal!(self, inv(dt(s)))
    return self
end

"""
Return a new `ScalarFunc` equal to `self / s`.

# Arguments
- `self::ScalarFunc`: source object.
- `s::Real`: nonzero scalar divisor.

# Returns
- New `ScalarFunc` scaled by `1/s`.

# Throws
- `AssertionError`: if `s == 0`.
"""
function /(self::ScalarFunc, s::Real)
    @assert s != 0 "Division by zero"
    return self * inv(dt(s))
end
