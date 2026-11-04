# ============================================================
# NS_Terms.jl
#
# Builds the Navier-Stokes operator terms used by the verification notebooks.
# Inputs are stored compactified unknowns: U_phys = cos(beta) * U_hat and
# P_phys = cos(beta) * P_hat. The Julia grid argument `r` is beta. The scaled
# radius is rho = tan(beta); the physical radius is R = scl_fac * tan(beta).
# The weight w(beta)=sin(beta)/cos(beta)^2 is the square root of the scaled
# radial Jacobian; physical L2 integration also includes the scl_fac^3 factor.
#
# Dependencies: My_Operator.jl, My_Assembly.jl, My_Fourier.jl.
# See: docs/technical_notes/ns_terms.md for mathematical background.
# ============================================================

"""
Construct a separated-variable operator `op = Operator(U_left, U_right')`.

The radial matrix `U_left` is the linear combination of
`trig_func(M0, r, key_r, bd_r)` entries selected by `key_r_lst`. The angular
matrix `U_right` is `trig_func(M1, θ, key_θ, bd_θ)`, transposed for right
multiplication. If the radial boundary descriptor is `[bd_r1, bd_r]`, a special
first-column left matrix `L1` is built and installed with `set_L1!`.

See `docs/technical_notes/ns_terms.md` for the compactified coordinates
and the `trig_func` key map.

# Arguments
- `M0::Int`: number of radial modes.
- `M1::Int`: number of angular modes.
- `r::Vector{dtype}`: compactified radial grid, as a column vector.
- `θ::Vector{dtype}`: angular grid (must be a column vector).
- `bdry::Vector`: boundary condition specification `[bd0, bd_θ]`.
  - `bd0`: either a `String` (single radial boundary) or a 2-element
    `Vector{String}` giving `[bd_r1, bd_r]`.
  - `bd_θ::String`: angular boundary condition.
- `key_r_lst::Vector`: list of `(key_r, α)` pairs for radial factors.
- `key_θ::String`: key specifying the angular basis.
- `key_r_lst1` (keyword): first-column radial key list; defaults to `key_r_lst`.
  Used only when `bd0` supplies two radial boundary codes; otherwise ignored.

# Returns
- `Operator`: separated block with `U_left`, `U_right'`, and optional `L1`.
"""
function build_operator(M0::Int, M1::Int, r::Vector{dtype}, θ::Vector{dtype}, bdry::Vector, key_r_lst::Vector,
    key_θ::String; key_r_lst1::Union{Vector, Nothing}=nothing)
    @assert length(bdry)==2 "bdry must be length 2"
    bd0, bd_θ = bdry
    bd_r1, bd_r = (isa(bd0, String) ? (nothing, bd0) : bd0)
    @assert size(r,2)==1 && size(θ,2)==1 "r and θ must be column vectors"

    # construct left multiplication matrix
    U_left = zeros(dtype, length(r), M0)
    for (key_r, alpha) in key_r_lst
        @assert isa(alpha, Union{dtype, Int}) "alpha must be a scalar"
        @assert isa(key_r, String) "key_r must be a String"
        U_left .+= alpha * trig_func(M0, r, key_r, bd_r)
    end

    # construct right multiplication matrix
    U_right = trig_func(M1, θ, key_θ, bd_θ)
    op = Operator(U_left, U_right')

    # construct L1 if two radial boundaries are given
    if bd_r1 !== nothing # If bd_r1 is nothing, even if key_r_lst1 is provided, it is ignored
        # use key_r_lst if key_r_lst1 is not provided
        if key_r_lst1 === nothing
            key_r_lst1 = key_r_lst
        end
        # construct and set L1
        U_left1 = zeros(dtype, length(r), M0)
        for (key_r1, alpha1) in key_r_lst1
            @assert isa(alpha1, Union{dtype, Int}) "alpha1 must be a scalar"
            @assert isa(key_r1, String) "key_r1 must be a String"
            U_left1 .+= alpha1 * trig_func(M0, r, key_r1, bd_r1)
        end
        set_L1!(op, U_left1)
    end

    return op
end

"""
Construct weighted physical linear terms from stored compactified unknowns.

The input columns are `u1`, `u2`, `u3`, and `p`, representing stored
`(U_hat, P_hat)`. Reconstruct `U_phys = cos(beta)*U_hat` and
`P_phys = cos(beta)*P_hat`. With physical position `x` and
`D = (I + x·∇)/2`, the returned `3×4` matrix evaluates
`w * (D*U_phys + ΔU_phys + ∇P_phys)`.

Here `w = sin(beta)/cos(beta)^2`. The keys use scaled coordinates
`y = x/scl_fac`, with `|y| = tan(beta)`. Diffusion is multiplied by
`scl_fac^(-2)` and the pressure gradient by `inv(scl_fac)` to give
physical derivatives. The drift needs no scale factor.

The notebooks subtract this output in the residual, including its pressure
term. Their stationary eigenvalue `λ ≈ -0.113` uses the convention of the
paper's numerical results; the evolution generator has eigenvalue `-λ > 0`.

With `mode="transpose"`, `Scale_T` reverses the drift sign, diffusion is
retained, and pressure blocks are omitted: the result is
`w * (ΔU_phys - D*U_phys)`. This branch performs no projection. See
Section 10 of `docs/technical_notes/ns_terms.md` for the adjoint shift and
its use in the projected estimate.

With `mode="pressure"`, only the weighted physical pressure gradient
`w * ∇P_phys` is constructed; the velocity columns remain empty.

# Arguments
- `M0::Int`: radial mode count.
- `M1::Int`: angular mode count.
- `r::Vector{dtype}`: compactified radial grid.
- `θ::Vector{dtype}`: angular grid.
- `scl_fac::dtype`: physical radial scale; diffusion uses its inverse square,
  and the pressure gradient uses its inverse.
- `bdry_index::Int`: boundary-condition set index.
- `mode::String="linear"`: `"linear"` includes drift, diffusion, and pressure;
  `"transpose"` reverses drift and omits pressure; `"laplacian"` keeps only diffusion.
  `"pressure"` keeps only the pressure gradient.

# Returns
- `Operator_matrix`: `3×4` block matrix with velocity output rows and
  velocity/pressure input columns.
"""
function get_linear(M0::Int, M1::Int, r::Vector{dtype}, θ::Vector{dtype},
                    scl_fac::dtype, bdry_index::Int; mode::String="linear")
    mode in ("linear", "transpose", "laplacian", "pressure") || throw(ArgumentError("Invalid linear mode: $mode"))
    op_mat  = Operator_matrix(3,4)
    bdry_r, bdry_θ, bdry_φ, bdry_p = BDRY_LST_SET[bdry_index]

    # Pressure-only mode returns before constructing any velocity blocks.
    if mode in ("linear", "pressure")
        beta = inv(scl_fac)# pressure scaling factor. order -1 because of only one derivative

        t14a = @spawn build_operator(M0, M1, r, θ, bdry_p, [["D_p_r", beta]], "I")

        t24a = @spawn build_operator(M0, M1, r, θ, bdry_p, [["I", beta]], "D")

        op_mat[1,4] = [fetch(t14a)]
        op_mat[2,4] = [fetch(t24a)]
    end
    mode == "pressure" && return op_mat

    # Scale applies w*D to the reconstructed field; Scale_T applies -w*D.
    # Laplacian mode keeps diffusion only.
    scale_key = mode == "laplacian" ? "zero" : (mode == "transpose" ? "Scale_T" : "Scale")

    alpha = scl_fac^dt(-2) # diffusion scaling factor. order -2 because of two derivatives
    M1p     = (bdry_index > 1) ? M1+1 : M1 
    # For bdry_index 2 or 3, u1 has one more (constant) mode in θ due to Neumann BC on both ends

    # For bdry_index=1, the l=1 blocks use the divergence-free relation.
    # The *_odd keys combine coupled terms; the corresponding separate
    # first-column blocks are zero. See Section 8 of docs/technical_notes/ns_terms.md.
    t11a = @spawn build_operator(M0,   M1p, r, θ, bdry_r, [["Cot", alpha]], "Laplacian_r_θ";
                                 key_r_lst1=[["zero", zero(dtype)]])

    t11b = @spawn build_operator(M0,   M1p, r, θ, bdry_r,
                                 [["Laplacian_r_r", alpha], [scale_key, one(dtype)]], "I";
                                 key_r_lst1=[["Laplacian_r_odd", alpha], [scale_key, one(dtype)]])

    t12a = @spawn build_operator(M0+1, M1,  r, θ, bdry_θ, [["Cot", alpha]], "Laplacian_θ_r_r";
                                 key_r_lst1=[["zero", zero(dtype)]])

    t22a = @spawn build_operator(M0+1, M1,  r, θ, bdry_θ, [["Cot", alpha]], "Laplacian_θ_θ";
                                 key_r_lst1=[["zero", zero(dtype)]])

    t22b = @spawn build_operator(M0+1, M1,  r, θ, bdry_θ,
                                 [["Laplacian_θ_r", alpha], [scale_key, one(dtype)]], "I";
                                 key_r_lst1=[["Laplacian_θ_odd", alpha], [scale_key, one(dtype)]])

    t21a = @spawn build_operator(M0,   M1p, r, θ, bdry_r, [["Cot", dt(2)*alpha]], "D";
                                 key_r_lst1=[["Laplacian_inter_odd", dt(2)*alpha]])

    t33a = @spawn build_operator(M0,   M1,  r, θ, bdry_φ, [["Cot", alpha]], "Laplacian_ϕ_θ")

    t33b = @spawn build_operator(M0,   M1,  r, θ, bdry_φ,
                                 [["Laplacian_ϕ_r", alpha], [scale_key, one(dtype)]], "I")
    
    # assemble
    op_mat[1,1] = [fetch(t11a), fetch(t11b)]
    op_mat[1,2] = [fetch(t12a)]

    op_mat[2,2] = [fetch(t22a), fetch(t22b)]
    op_mat[2,1] = [fetch(t21a)]

    op_mat[3,3] = [fetch(t33a), fetch(t33b)]

    return op_mat
end

"""
Build the identity operator matrix used for the `λI` residual term.

The input is the stored unknown `U_hat`. `weight_flag=false` uses `"Cos"` to
map `U_hat` to the physical field `U_phys = cos(beta)*U_hat`. `weight_flag=true`
uses `"Tan"` to map `U_hat` to the weighted physical field
`w U_phys = tan(beta)*U_hat`, where `w = sin(beta)/cos(beta)^2` is the square
root of the scaled radial Jacobian. The `u1` angular block gets the extra
mode when `bdry_index > 1`.

# Arguments
- `M0::Int`: radial mode count.
- `M1::Int`: angular mode count.
- `r::Vector{dtype}`: compactified radial grid.
- `θ::Vector{dtype}`: angular grid.
- `bdry_index::Int`: boundary-condition set index.
- `weight_flag::Bool=true`: choose `"Tan"` when true, `"Cos"` when false.

# Returns
- `Operator_matrix`: `3×4` diagonal velocity-block identity matrix.
"""
function get_identity(M0::Int, M1::Int, r::Vector{dtype}, θ::Vector{dtype},
                      bdry_index::Int; weight_flag::Bool=true)
    @assert size(r, 2) == 1 "r must be a column vector"
    @assert size(θ, 2) == 1 "θ must be a column vector"

    op_mat = Operator_matrix(3, 4)
    bdry_r, bdry_θ, bdry_φ, _ = BDRY_LST_SET[bdry_index]
    # For angular boundary condition "11", u₁ includes an extra constant mode in θ
    M1p = (bdry_index > 1) ? M1+1 : M1

    key_r = weight_flag ? "Tan" : "Cos"
    t11 = @spawn build_operator(M0,   M1p, r, θ, bdry_r, [[key_r, one(dtype)]], "I")
    t22 = @spawn build_operator(M0+1, M1,  r, θ, bdry_θ, [[key_r, one(dtype)]], "I")
    t33 = @spawn build_operator(M0,   M1,  r, θ, bdry_φ, [[key_r, one(dtype)]], "I")

    op_mat[1,1] = [fetch(t11)]
    op_mat[2,2] = [fetch(t22)]
    op_mat[3,3] = [fetch(t33)]

    return op_mat
end

"""
Build the symmetric-gradient operator matrix.

The input is stored `U_hat`; the keys compute physical symmetric-gradient
components for `U_phys = cos(beta)*U_hat`. The mathematical target is
`sym ∇U_phys = (∇U_phys + (∇U_phys)^T)/2`. The output has six rows, ordered as
`(11)`, `(12)`, `(13)`, `(22)`, `(23)`, `(33)`.

Row/key packing:
- row 1 `(11)`: `D11_r_sym`;
- row 2 `(12)`: `D12_r_sym` and `D12_θ_sym`;
- row 3 `(13)`: `D13_ϕ_sym`;
- row 4 `(22)`: `D22_r_sym` and `D22_θ_sym`;
- row 5 `(23)`: `D23_ϕ_sym` with angular key `D23_ϕ_θ_sym`;
- row 6 `(33)`: `D33_r_sym` and `D33_θ_sym`.

It is used in the positivity-compensation term involving
`Q + (∇U + ∇ᵀU)/2 + η₁I`. See `docs/technical_notes/ns_terms.md`
for the longer guide.

# Arguments
- `M0::Int`: radial mode count.
- `M1::Int`: angular mode count.
- `r::Vector{dtype}`: compactified radial grid.
- `θ::Vector{dtype}`: angular grid.
- `scl_fac::dtype`: scale factor; gradient blocks use `inv(scl_fac)`.
- `bdry_index::Int`: boundary-condition set index.

# Returns
- `Operator_matrix`: `6×4` symmetric-gradient block matrix.
"""
function get_gradient_sym(M0::Int, M1::Int, r::Vector{dtype}, θ::Vector{dtype},
                          scl_fac::dtype, bdry_index::Int)
    op_mat = Operator_matrix(6, 4)
    bdry_r, bdry_θ, bdry_φ, _ = BDRY_LST_SET[bdry_index]

    # Scaling factor
    alpha  = inv(scl_fac)
    # For angular boundary condition "11", u₁ includes an extra constant mode in θ
    M1p    = (bdry_index > 1) ? M1+1 : M1

    t11 = @spawn build_operator(M0,   M1p, r, θ, bdry_r, [["D11_r_sym", alpha]], "I")

    t21 = @spawn build_operator(M0,   M1p, r, θ, bdry_r, [["D12_r_sym", alpha]], "D")

    t22 = @spawn build_operator(M0+1, M1,  r, θ, bdry_θ, [["D12_θ_sym", alpha]], "I")

    t33 = @spawn build_operator(M0,   M1,  r, θ, bdry_φ, [["D13_ϕ_sym", alpha]], "I")

    t41 = @spawn build_operator(M0,   M1p, r, θ, bdry_r, [["D22_r_sym", alpha]], "I")

    t42 = @spawn build_operator(M0+1, M1,  r, θ, bdry_θ, [["D22_θ_sym", alpha]], "D")

    t53 = @spawn build_operator(M0,   M1,  r, θ, bdry_φ, [["D23_ϕ_sym", alpha]], "D23_ϕ_θ_sym")

    t61 = @spawn build_operator(M0,   M1p, r, θ, bdry_r, [["D33_r_sym", alpha]], "I")

    t62 = @spawn build_operator(M0+1, M1,  r, θ, bdry_θ, [["D33_θ_sym", alpha]], "Cot")

    op_mat[1,1] = [fetch(t11)]
    op_mat[2,1] = [fetch(t21)]

    op_mat[2,2] = [fetch(t22)]

    op_mat[3,3] = [fetch(t33)]

    op_mat[4,1] = [fetch(t41)]
    op_mat[4,2] = [fetch(t42)]

    op_mat[5,3] = [fetch(t53)]

    op_mat[6,1] = [fetch(t61)]
    op_mat[6,2] = [fetch(t62)]

    return op_mat
end

"""
Build the full gradient operator matrix in spherical coordinates.

The input is a stored field `B_hat`; unweighted gradient keys compute components
of `∇B_phys` for `B_phys = cos(beta)*B_hat`. The output rows are later named
`A11`, `A12`, `A13`, `A21`, `A22`, `A23`, `A31`, `A32`, `A33`. The local
contraction convention is `(A·∇B)_i = Σ_j A_j * grad_B["Aji"]`, so the first
index is the contracted direction and the second index is the output component.

If `weight_flag=true`, radial keys receive `"_w"`, meaning multiplication by
`tan(beta)` relative to the unweighted physical-gradient key. This lets
`mul_A_gradB` compute `U_hat_A · (tan(beta) ∇B_phys)`, which equals
`w A_phys · ∇B_phys` because `A_phys = cos(beta)*U_hat_A`.

# Arguments
- `M0::Int`: radial mode count.
- `M1::Int`: angular mode count.
- `r::Vector{dtype}`: compactified radial grid.
- `θ::Vector{dtype}`: angular grid.
- `scl_fac::dtype`: scale factor; gradient blocks use `inv(scl_fac)`.
- `bdry_index::Int`: boundary-condition set index.
- `weight_flag::Bool=false`: append `"_w"` to radial keys when true.

# Returns
- `Operator_matrix`: `9×4` full-gradient block matrix.
"""
function get_gradient(M0::Int, M1::Int, r::Vector{dtype}, θ::Vector{dtype}, 
    scl_fac::dtype, bdry_index::Int; weight_flag::Bool=false)
    op_mat = Operator_matrix(9, 4)
    bdry_r, bdry_θ, bdry_φ, _ = BDRY_LST_SET[bdry_index]

    # Scaling factor
    alpha = inv(scl_fac)
    # For angular boundary condition "11", u₁ includes an extra constant mode in θ
    M1p = (bdry_index > 1) ? M1+1 : M1
    # Append "_w" to operator keys if weighted form is requested
    suffix = weight_flag ? "_w" : ""
    # In the odd l=1 mode, incompressibility combines the u1/u2 contributions
    # to A21, A22, and A33. k1 carries the combined action; k0 removes the
    # separate u2 action. See Section 8 of docs/technical_notes/ns_terms.md.
    k1 = (bdry_index > 1) ? nothing : [["D11_r" * suffix, -alpha/dt(2)]]
    k0 = (bdry_index > 1) ? nothing : [["zero", zero(dtype)]]

    t11 = @spawn build_operator(M0,   M1p, r, θ, bdry_r, [["D11_r" * suffix,  alpha]], "I")

    t22 = @spawn build_operator(M0+1, M1,  r, θ, bdry_θ, [["D21_θ" * suffix,  alpha]], "I")

    t33 = @spawn build_operator(M0,   M1,  r, θ, bdry_φ, [["D31_ϕ" * suffix,  alpha]], "I")

    t41 = @spawn build_operator(M0,   M1p, r, θ, bdry_r, [["D12_r" * suffix,  alpha]], "D"; key_r_lst1 = k1)

    t42 = @spawn build_operator(M0+1, M1,  r, θ, bdry_θ, [["D12_θ" * suffix, -alpha]], "I"; key_r_lst1 = k0)

    t51 = @spawn build_operator(M0,   M1p, r, θ, bdry_r, [["D22_r" * suffix,  alpha]], "I"; key_r_lst1 = k1)

    t52 = @spawn build_operator(M0+1, M1,  r, θ, bdry_θ, [["D22_θ" * suffix,  alpha]], "D"; key_r_lst1 = k0)

    t63 = @spawn build_operator(M0,   M1,  r, θ, bdry_φ, [["D32_ϕ" * suffix,  alpha]], "D")

    t73 = @spawn build_operator(M0,   M1,  r, θ, bdry_φ, [["D13_ϕ" * suffix, -alpha]], "I")

    t83 = @spawn build_operator(M0,   M1,  r, θ, bdry_φ, [["D23_ϕ" * suffix, -alpha]], "Cot")

    t91 = @spawn build_operator(M0,   M1p, r, θ, bdry_r, [["D33_r" * suffix,  alpha]], "I";   key_r_lst1 = k1)

    t92 = @spawn build_operator(M0+1, M1,  r, θ, bdry_θ, [["D33_θ" * suffix,  alpha]], "Cot"; key_r_lst1 = k0)

    op_mat[1,1] = [fetch(t11)]
    op_mat[2,2] = [fetch(t22)]
    op_mat[3,3] = [fetch(t33)]

    op_mat[4,1] = [fetch(t41)]
    op_mat[4,2] = [fetch(t42)]

    op_mat[5,1] = [fetch(t51)]
    op_mat[5,2] = [fetch(t52)]

    op_mat[6,3] = [fetch(t63)]
    op_mat[7,3] = [fetch(t73)]
    op_mat[8,3] = [fetch(t83)]

    op_mat[9,1] = [fetch(t91)]
    op_mat[9,2] = [fetch(t92)]

    return op_mat
end

"""
Construct an operator matrix for a `VectorFunc` and convert it to that basis.

The `key` dispatches to `"linear"`, `"transpose"`, `"laplacian"`, `"pressure"`, `"identity"`,
`"gradient_sym"`, or `"gradient"`. Mode sizes, scaling, and boundary conditions are read from `sol`.
After construction, `apply_convmat_right!(op_mat, sol)` is called so the
operator acts on the internal coefficient basis of `sol`.

# Requirements
- `sol["u3"].freq_func` must be set.
- `sol.scl_fac` and `sol.bdry_index` must be valid.

# Arguments
- `sol::VectorFunc`: object providing mode sizes, scaling, and boundary data.
- `r::Vector{dtype}`: compactified radial grid.
- `θ::Vector{dtype}`: angular grid.
- `key::String`: operator family:
  - `"linear"`, `"transpose"`, `"laplacian"`, `"pressure"`: `get_linear(...; mode=key)`
  - `"identity"`: `get_identity(...; weight_flag=...)`
  - `"gradient_sym"`: `get_gradient_sym(...)`
  - `"gradient"`: `get_gradient(...; weight_flag=...)`
- `kwargs`: passed through to the underlying constructor:
  - For `"identity"`: `weight_flag::Bool`
  - For `"gradient"`: `weight_flag::Bool`

# Returns
- `Operator_matrix`: basis-compatible operator matrix for `apply(op_mat, sol)`.
"""
function get_operator_matrix(sol::VectorFunc, r::Vector{dtype}, θ::Vector{dtype}, key::String; kwargs...)
    M0, M1 = size(sol["u3"].freq_func)
    scl_fac = sol.scl_fac
    bdry_index = sol.bdry_index
    if key in ("linear", "transpose", "laplacian", "pressure")
        isempty(kwargs) || throw(ArgumentError("Select the linear mode with key, without extra keywords"))
        op_mat = get_linear(M0, M1, r, θ, scl_fac, bdry_index; mode=key)
    elseif key == "identity"
        weight_flag = get(kwargs, :weight_flag, true)
        op_mat = get_identity(M0, M1, r, θ, bdry_index; weight_flag=weight_flag)
    elseif key == "gradient_sym"
        op_mat = get_gradient_sym(M0, M1, r, θ, scl_fac, bdry_index)
    elseif key == "gradient"
        weight_flag = get(kwargs, :weight_flag, false)
        op_mat = get_gradient(M0, M1, r, θ, scl_fac, bdry_index; weight_flag=weight_flag)
    else
        error("Invalid operator key: $key")
    end
    apply_convmat_right!(op_mat, sol)
    return op_mat
end

"""
Assemble the pointwise contraction between `A` and `∇B` in space form.

For `transpose=false`, the contraction is
`(A·∇B)_i = Σ_j A_j * grad_B["Aji"]`. In the nonlinear weighted use, `A` is the
stored compactified velocity and `grad_B` is produced by `_w` gradient keys, so
the product represents `w A_phys·∇B_phys`. For `transpose=true`, the gradient
indexing is swapped to compute the transpose-gradient contraction in the code's
convention.

# Arguments
- `A::VectorFunc`: space-domain velocity with keys `u1`, `u2`, `u3`.
- `grad_B::VectorFunc`: space-domain gradient with keys `A11`...`A33`.
- `transpose::Bool=false`: use the transpose-gradient contraction when true.

# Returns
- `VectorFunc`: space-domain vector with keys `e1`, `e2`, `e3`.
"""
function mul_A_gradB_local(A::VectorFunc, grad_B::VectorFunc; transpose::Bool=false)
    if transpose
        e1 = A["u1"].space_func .* grad_B["A11"].space_func .+ 
             A["u2"].space_func .* grad_B["A12"].space_func .+ 
             A["u3"].space_func .* grad_B["A13"].space_func
        e2 = A["u1"].space_func .* grad_B["A21"].space_func .+ 
             A["u2"].space_func .* grad_B["A22"].space_func .+ 
             A["u3"].space_func .* grad_B["A23"].space_func
        e3 = A["u1"].space_func .* grad_B["A31"].space_func .+ 
             A["u2"].space_func .* grad_B["A32"].space_func .+ 
             A["u3"].space_func .* grad_B["A33"].space_func
    else
        e1 = A["u1"].space_func .* grad_B["A11"].space_func .+ 
             A["u2"].space_func .* grad_B["A21"].space_func .+ 
             A["u3"].space_func .* grad_B["A31"].space_func
        e2 = A["u1"].space_func .* grad_B["A12"].space_func .+ 
             A["u2"].space_func .* grad_B["A22"].space_func .+ 
             A["u3"].space_func .* grad_B["A32"].space_func
        e3 = A["u1"].space_func .* grad_B["A13"].space_func .+ 
             A["u2"].space_func .* grad_B["A23"].space_func .+ 
             A["u3"].space_func .* grad_B["A33"].space_func
    end
    result = VectorFunc(A.scl_fac, "space"; e1=e1, e2=e2, e3=e3)
    return result
end

"""
Evaluate `A·(∇B)` or the transpose-gradient contraction on a grid.

The routine interpolates stored `A_hat`, builds the weighted gradient operator
from the first `B_hat`, applies it to every B input with row keys `A11`...`A33`,
and then calls
`mul_A_gradB_local`. The weighted gradient supplies the factor
`tan(beta)∇B_phys`, so the contraction represents
`w A_phys·∇B_phys`. List inputs produce the pairwise matrix of nonlinear terms.
In the single-input `transpose=true` case, only the transpose-gradient
contraction `A·(∇B)^T` is computed and returned as a `VectorFunc`.

See `docs/technical_notes/ns_terms.md` for the weighting and
gradient-index conventions.

# Arguments
- `A_lst`: `VectorFunc` or list of `VectorFunc`s with keys `u1`, `u2`, `u3`.
- `B_lst`: `VectorFunc` or nonempty list in the ordered layout `u1,u2,u3,p`;
  the pressure operator column is empty but retained. All B inputs must share
  the first B's coefficient dimensions and angular conversions. List inputs
  must be nonempty, with the common scale and boundary index asserted below.
- `r::Vector{dtype}`: compactified radial grid.
- `θ::Vector{dtype}`: angular grid.
- `transpose::Bool=false`:
  - If `false`, compute `A·(∇B)`.
  - If `true`, this option is only supported when both inputs are single `VectorFunc`s. 
    In that case, compute and return only `A·(∇B)^T`.

# Returns
- If `A_lst` and `B_lst` are single `VectorFunc` and `transpose=false`:
  - `VectorFunc` with keys `"e1","e2","e3"`.
- If `A_lst` and/or `B_lst` are vectors and `transpose=false`:
  - `Array{VectorFunc,2}` whose `(i,j)` entry corresponds to `A_lst[i]·∇B_lst[j]`.
- If `transpose=true` (single-single case only):
  - `VectorFunc` with keys `"e1","e2","e3"` for `A·(∇B)^T`.

# Throws
- `AssertionError` if component keys are missing, scaling factors mismatch, boundary
  indices mismatch, or if `transpose=true` is requested for non-single inputs.
"""
function mul_A_gradB(A_lst::Union{VectorFunc, Vector}, B_lst::Union{VectorFunc, Vector}, r::Vector{dtype}, θ::Vector{dtype}; transpose::Bool=false)
    only_flag = isa(A_lst, VectorFunc) && isa(B_lst, VectorFunc)
    if isa(A_lst, VectorFunc)
        A_lst = [A_lst]
    end
    if isa(B_lst, VectorFunc)
        B_lst = [B_lst]
    end

    # Validate A, B list: same scaling factor and correct component keys
    for A in A_lst
        @assert A.scl_fac == A_lst[1].scl_fac "All VectorFuncs must have the same scaling factor"
        @assert all(u -> u in A.keys, ["u1", "u2", "u3"]) "Each VectorFunc in A_lst must have keys u1, u2, u3"
    end
    for B in B_lst
        @assert B.scl_fac == A_lst[1].scl_fac "All VectorFuncs must have the same scaling factor"
        @assert all(u -> u in B.keys, ["u1", "u2", "u3"]) "Each VectorFunc in B_lst must have keys u1, u2, u3"
        @assert B.bdry_index == B_lst[1].bdry_index "All VectorFuncs must have the same boundary condition index"
    end

    A_lst = [interpolate(A, r, θ) for A in A_lst]
    grad_mat = get_operator_matrix(B_lst[1], r, θ, "gradient"; weight_flag=true)
    grad_B_lst = [apply(grad_mat, B; new_keys=["A11", "A12", "A13", "A21", "A22", "A23", "A31", "A32", "A33"]) for B in B_lst]
    if transpose
        @assert only_flag "Transpose option is only supported for single VectorFunc input"
        return mul_A_gradB_local(A_lst[1], grad_B_lst[1]; transpose=true)
    else
        result = [mul_A_gradB_local(A, grad_B) for A in A_lst, grad_B in grad_B_lst]
        if only_flag
            result = only(result)
        end
        return result
    end
end
