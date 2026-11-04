# ============================================================
# My_Operator.jl
#
# Defines two structs used to represent linear operators on the spectral
# coefficient matrices:
#   - `Operator`: a single two-sided matrix operator `L · mat · R` with an
#     optional pointwise multiplier `T` and an alternative left matrix `L1`
#     for the first coefficient column (the l=1 mode in NS_Terms.jl).
#   - `Operator_matrix`: a block M×N grid of `Operator` lists, mapping a
#     `VectorFunc` (N input components) to another `VectorFunc` (M output
#     components).
# Provides arithmetic, application, variable-change, and eigenvalue utilities.
#
# Dependencies: My_Vector.jl.
# Numeric mode: dtype selects interval enclosures of the algebraic operations
# or floating-point approximations; continuous bounds need separate estimates.
# See: docs/technical_notes/ns_terms.md, Section 7, for the operator overview.
# ============================================================

"""
Spectral operator for a scalar function in the separated-variable form `L · mat · R`.

Acts on a coefficient matrix `mat` (rows = radial modes, cols = angular modes):
1. Form `L * mat`; if `L1` is set, replace column 1 by `L1 * mat[:,1]`.
2. Right-multiply by `R`.
3. Pointwise-multiply by `T` if set.

The input basis must match `L` and `R`; `R` may include angular basis conversion.
In the Navier-Stokes operators, `L1` handles the special l=1 radial formula.
# Fields
- `L::AbstractMatrix{dtype}`: left radial matrix.
- `R::AbstractMatrix{dtype}`: right angular matrix.
- `T::Union{AbstractMatrix{dtype}, Nothing}`: optional pointwise multiplier.
- `L1::Union{AbstractMatrix{dtype}, Nothing}`: optional replacement left matrix for
  the first coefficient column.
"""
mutable struct Operator
    L::AbstractMatrix{dtype}                 
    R::AbstractMatrix{dtype}                 
    T::Union{AbstractMatrix{dtype}, Nothing}          
    L1::Union{AbstractMatrix{dtype}, Nothing}           
end

"""
Construct an `Operator` with optional `T` and `L1` fields.

# Arguments
- `L::AbstractMatrix{dtype}`: left radial matrix.
- `R::AbstractMatrix{dtype}`: right angular matrix.
- `T::Union{AbstractMatrix{dtype}, Nothing}`: (optional) pointwise multiplier; default `nothing`.
- `L1::Union{AbstractMatrix{dtype}, Nothing}`: optional replacement left matrix for
  the first coefficient column; default `nothing`.

# Returns
- Constructed `Operator`.

"""
function Operator(L::AbstractMatrix{dtype}, R::AbstractMatrix{dtype}; 
    T::Union{AbstractMatrix{dtype}, Nothing}=nothing, L1::Union{AbstractMatrix{dtype}, Nothing}=nothing)
    op = Operator(L, R, T, L1)
    return op
end

"""
Scale an `Operator` in place with `multiply_equal!(op, scalar)`.

Scales only the right matrix `R`; `L`, `T`, and `L1` are unchanged.

# Arguments
- `op::Operator`: operator to modify in-place.
- `scalar::Union{dtype,Int}`: multiplier.

# Returns
- `op` (the modified operator).

"""
function multiply_equal!(op::Operator, scalar::Union{dtype,Int})
    op.R .*= scalar
    return op
end

"""
Return a new `Operator` equal to `op * scalar`.

Copies the operator before scaling. Consequently, `op *= scalar` reassigns
`op` to a scaled copy; use `multiply_equal!` to modify the existing object.

# Arguments
- `op::Operator`: source operator.
- `scalar::Union{dtype,Int}`: multiplier.

# Returns
- New `Operator` with `R` scaled by `scalar`.
"""
function *(op::Operator, scalar::Union{dtype,Int})
    result = copy(op)
    multiply_equal!(result, scalar)
    return result
end

"""
Set the pointwise multiplier `T` of an `Operator` in-place.

# Arguments
- `op::Operator`: target operator (modified in-place).
- `T::AbstractMatrix{dtype}`: pointwise multiplier matrix.

# Returns
- `Nothing`; mutates `op.T`.

# Notes
- `T` is applied pointwise after the radial and angular matrix actions.
"""
set_T!(op::Operator, T::AbstractMatrix{dtype}) = (op.T = T; nothing)

"""
Set the alternative first-column left matrix `L1` of an `Operator` in-place.

# Arguments
- `op::Operator`: target operator (modified in-place).
- `L1::AbstractMatrix{dtype}`: replacement left matrix for the first coefficient column.

# Returns
- `Nothing`; mutates `op.L1`.

# Notes
- `apply` replaces column 1 of `L * mat` before multiplying by `R`.
  The Navier-Stokes operators use this for the l=1 radial formula.
"""
set_L1!(op::Operator, L1::AbstractMatrix{dtype}) = (op.L1 = L1; nothing)

"""
Apply an `Operator` to a coefficient matrix.

Forms `L * mat`, replaces its first column by `L1 * mat[:,1]` when `L1` is set,
then right-multiplies by `R` and pointwise-multiplies by `T` if present.

# Arguments
- `op::Operator`: operator to apply.
- `mat::AbstractMatrix{dtype}`: input coefficient matrix.

# Returns
- `AbstractMatrix{dtype}` of size `(size(op.L, 1), size(op.R, 2))`.
  For basis-evaluation matrices, its entries are grid samples.

# Throws
- `AssertionError`: if the dimensions of `L` or `R` are incompatible with `mat`.

"""
function apply(op::Operator, mat::AbstractMatrix{dtype})
    @assert size(op.L, 2) == size(mat, 1) "Left matrix size mismatch: $(size(op.L, 2)) vs $(size(mat, 1))"
    @assert size(op.R, 1) == size(mat, 2) "Right matrix size mismatch: $(size(op.R, 1)) vs $(size(mat, 2))"

    result = op.L * mat
    op.L1 !== nothing && (result[:,1] .= op.L1 * mat[:,1])
    result *= op.R
    op.T  !== nothing && (result .*= op.T)

    return result
end

"""
Update the matrices of an `Operator` in-place to compose a variable change.

When `L` is provided: `op.L ← op.L · L` (post-multiply).
When `R` is provided: `op.R ← R · op.R` (pre-multiply).
When `L1` is provided: `op.L1 ← op.L1 · L1` (post-multiply).

# Arguments
- `op::Operator`: operator to modify in-place.
- `L`, `R`, `L1`: (optional) transformation matrices; `nothing` leaves the field unchanged.

# Returns
- `op` (the modified operator).

"""
function change_variable!(op::Operator; L::Union{AbstractMatrix{dtype}, Nothing}=nothing,
    R::Union{AbstractMatrix{dtype}, Nothing}=nothing, L1::Union{AbstractMatrix{dtype}, Nothing}=nothing)
    L  !== nothing && (op.L  *= L)
    L1 !== nothing && (op.L1 *= L1)
    R  !== nothing && (op.R   =  R * op.R)
    return op
end

"""
Return a deep copy of an `Operator`, duplicating all non-`nothing` fields.

# Arguments
- `op::Operator`: source operator.

# Returns
- New `Operator` with independent copies of `L`, `R`, and optionally `T` and `L1`.

"""
function copy(op::Operator)
    new = Operator(copy(op.L), copy(op.R))
    op.T  !== nothing && set_T!(new,  copy(op.T))
    op.L1 !== nothing && set_L1!(new, copy(op.L1))
    return new
end

"""
Block-structured container of `Operator` lists mapping a `VectorFunc` to another `VectorFunc`.

Entry `(i, j)` holds a list of `Operator`s each mapping the `j`-th input component to the
`i`-th output component. When `apply` is called, the contributions from all operators in each
list are accumulated by summing.

# Fields
- `M::Int`: number of output rows.
- `N::Int`: number of input columns.
- `op_lst::AbstractMatrix{Vector{Operator}}`: block grid where `op_lst[i,j]` is the
  list of operators from column `j` to row `i`.
"""
mutable struct Operator_matrix
    M::Int
    N::Int
    op_lst::AbstractMatrix{Vector{Operator}}
end


"""
Construct an `Operator_matrix` of size `M × N` with empty operator lists.

# Arguments
- `M::Int`: number of output rows.
- `N::Int`: number of input columns.

# Returns
- `Operator_matrix` with `Operator[]` in every entry.

"""
function Operator_matrix(M::Int, N::Int)
    op_lst = [Operator[] for i in 1:M, j in 1:N]
    return Operator_matrix(M, N, op_lst)
end

"""
Slice an `Operator_matrix` using a `UnitRange` for each axis.

# Arguments
- `mat::Operator_matrix`: source container.
- `rows::Union{Int,UnitRange}`: use a row range, including `i:i` for one row.
- `cols::Union{Int,UnitRange}`: use a column range, including `j:j` for one column.

# Returns
- New `Operator_matrix` sharing (not copying) the inner operator lists.

Integer selectors drop dimensions and do not produce the matrix required by
the container. Use singleton ranges to retain both dimensions.

# Throws
- `BoundsError`: if the requested row or column selector is outside the matrix.

# Notes
- The returned container shares inner lists with `mat`; use `copy` when independent
  operator objects are needed.
"""
function getindex(mat::Operator_matrix, rows::Union{Int,UnitRange}, cols::Union{Int,UnitRange})
    new_op_lst = mat.op_lst[rows, cols]
    return Operator_matrix(length(rows), length(cols), new_op_lst)
end

"""
Assign an operator list to entry `(i, j)` of an `Operator_matrix`.

# Arguments
- `mat::Operator_matrix`: target container (modified in-place).
- `ops::Vector{Operator}`: operator list to store.
- `i::Int`: row index.
- `j::Int`: column index.

# Returns
- The assigned operator list, following Julia `setindex!` convention.

# Throws
- `BoundsError`: if `(i, j)` is outside the matrix.

# Notes
- This mutates only the block reference; it does not copy the operators in `ops`.
"""
function setindex!(mat::Operator_matrix, ops::Vector{Operator}, i::Int, j::Int)
    mat.op_lst[i, j] = ops
end

"""
Scale every block in place with `multiply_equal!(mat, scalar)`.

# Arguments
- `mat::Operator_matrix`: target container (modified in-place).
- `scalar::Union{dtype,Int}`: multiplier applied to every `Operator.R`.

# Returns
- `mat` (the modified container).

"""
function multiply_equal!(mat::Operator_matrix, scalar::Union{dtype,Int})
    for i in 1:mat.M, j in 1:mat.N, op in mat.op_lst[i, j]
        multiply_equal!(op, scalar)
    end
    return mat
end

"""
Return a new `Operator_matrix` equal to `mat * scalar`.

Copies the container and its operators before scaling. `mat *= scalar`
reassigns `mat` to this copy; `multiply_equal!` modifies the existing blocks.

# Arguments
- `mat::Operator_matrix`: source container.
- `scalar::Union{dtype,Int}`: multiplier.

# Returns
- Deep copy of `mat` with every `Operator.R` scaled by `scalar`.
"""
function *(mat::Operator_matrix, scalar::Union{dtype,Int})
    mat2 = copy(mat)
    multiply_equal!(mat2, scalar)
    return mat2
end

"""
Return a new `Operator_matrix` equal to `A + B`.

The output size is `max(A.M, B.M) × max(A.N, B.N)`; operator lists at each entry
are the concatenation of the corresponding lists from `A` and `B`.

# Arguments
- `A::Operator_matrix`: first operand.
- `B::Operator_matrix`: second operand.

# Returns
- New `Operator_matrix` with copied operators from both grids.
"""
function +(A::Operator_matrix, B::Operator_matrix)
    # determine the new size
    M = max(A.M, B.M)
    N = max(A.N, B.N)
    C = Operator_matrix(M, N)
    # add operators from A and B into C
    for i in 1:M, j in 1:N
        if i <= A.M && j <= A.N
            for op in A.op_lst[i, j]
                push!(C.op_lst[i, j], copy(op))
            end
        end
        if i <= B.M && j <= B.N
            for op in B.op_lst[i, j]
                push!(C.op_lst[i, j], copy(op))
            end
        end
    end
    return C
end

"""
Return a new `Operator_matrix` equal to `A - B`.

# Arguments
- `A::Operator_matrix`: minuend.
- `B::Operator_matrix`: subtrahend.

# Returns
- New `Operator_matrix` equal to `A + (-1)·B`.
"""
function -(A::Operator_matrix, B::Operator_matrix)
    return A + B * dt(-1)
end

"""
Append one or more `Operator`s to entry `(i, j)` of an `Operator_matrix`.

# Arguments
- `mat::Operator_matrix`: target container (modified in-place).
- `i::Int`: row index.
- `j::Int`: column index.
- `ops::Operator...`: one or more operators to append.

# Returns
- `mat` (the modified container).

"""
function add_operator!(mat::Operator_matrix, i::Int, j::Int, ops::Operator...)
    append!(mat.op_lst[i, j], ops)
    return mat
end

"""
Apply an `Operator_matrix` to a `VectorFunc`, producing a new `VectorFunc`.

For each output row `i`, accumulates `∑_j ∑_{op ∈ op_lst[i,j]} apply(op, vec[j].freq_func)`
and stores the result as a spatial-domain `ScalarFunc` with the given output key.
Input components follow `vec.keys` order; their coefficients are read from
`freq_func`. Every block must produce the same output grid shape.

# Arguments
- `mat::Operator_matrix`: operator to apply.
- `vec::VectorFunc`: input; must have `length(vec.keys) == mat.N`.
- `new_keys::Vector{String}`: output component names; length must equal `mat.M`;
  default `["e1","e2","e3"]`.

# Returns
- New `VectorFunc` with `space_func` set for each output component.

# Throws
- `AssertionError`: if input/output size mismatches.

"""
function apply(mat::Operator_matrix, vec::VectorFunc; new_keys::Vector{String}=["e1","e2","e3"])
    M, N = mat.M, mat.N
    @assert length(vec.keys) == N "Input size mismatch"
    @assert length(new_keys) == M "Output keys mismatch"

    # Determine the common output grid shape (m, n) from L rows and R columns.
    # All blocks must have this same shape.
    m, n = 0, 0
    for i in 1:M, j in 1:N
        for op in mat.op_lst[i, j]
            if op.L !== nothing
                if m == 0
                    m = size(op.L, 1)
                else
                    @assert m == size(op.L, 1) "Inconsistent output size in L across operators"
                end
            end
            if op.L1 !== nothing
                @assert m == size(op.L1, 1) "Inconsistent output size in L1 across operators"
            end
            if op.R !== nothing
                if n == 0
                    n = size(op.R, 2)
                else
                    @assert n == size(op.R, 2) "Inconsistent input size in R across operators"
                end
            end
        end
    end

    result = VectorFunc(new_keys; scl_fac=vec.scl_fac)
    for i in 1:M
        # Accumulate this output component on the common grid.
        cum = zeros(dtype, m, n)
        for j in 1:N
            for op in mat.op_lst[i, j]
                cum .+= apply(op, vec[j].freq_func)
            end
        end
        set_func!(result[new_keys[i]], "space", cum)
    end
    return result
end

"""
Return a deep copy of an `Operator_matrix` with all `Operator` objects duplicated.

# Arguments
- `mat::Operator_matrix`: source container.

# Returns
- New `Operator_matrix` with independent copies of every `Operator`.

"""
function copy(mat::Operator_matrix)
    new = Operator_matrix(mat.M, mat.N)
    for i in 1:mat.M, j in 1:mat.N
        for op in mat.op_lst[i, j]
            add_operator!(new, i, j, copy(op))
        end
    end
    return new
end

"""
Update every operator in column `j` of an `Operator_matrix` by pre-composing `vec[j].conv_mat`.

For each column `j` where `vec[j].conv_mat ≠ nothing`, updates every operator:
`op.R ← vec[j].conv_mat · op.R`. This is needed after the angular conversion matrix
is loaded for the `u1`/`u2` components.

# Arguments
- `mat::Operator_matrix`: target container (modified in-place).
- `vec::VectorFunc`: source of `conv_mat` per input column.

# Returns
- `mat` (the modified container).

"""
function apply_convmat_right!(mat::Operator_matrix, vec::VectorFunc)
    for j in 1:mat.N, i in 1:mat.M
        if vec[j].conv_mat !== nothing
            for op in mat.op_lst[i, j]
                change_variable!(op; L=nothing, R=vec[j].conv_mat, L1=nothing)
            end
        end
    end
    return mat
end

"""
Compute pointwise eigenvalues of a symmetric 3×3 tensor field.

Calls `eig_sym3` at every grid point `(i, j)` in parallel.

# Arguments
- `eigen_mat::VectorFunc`: must have keys `"A11","A12","A13","A22","A23","A33"` with
  `space_func` set; each is an `N0 × N1` spatial array.

# Returns
- `Array{dtype,3}` of shape `(N0, N1, 3)`: eigenvalue estimates or interval
  enclosures at each grid point, in the order returned by `eig_sym3`. When
  its scale interval contains zero, it repeats one enclosure covering all eigenvalues.

"""
function eigen_calc(eigen_mat::VectorFunc)
    A11 = eigen_mat["A11"].space_func
    A12 = eigen_mat["A12"].space_func
    A13 = eigen_mat["A13"].space_func
    A22 = eigen_mat["A22"].space_func
    A23 = eigen_mat["A23"].space_func
    A33 = eigen_mat["A33"].space_func

    N0, N1 = size(A11)
    eig_val_mat = Array{dtype}(undef, N0, N1, 3)

    @threads for j in 1:N1
        @inbounds for i in 1:N0
            λ1, λ2, λ3 = eig_sym3(A11[i,j], A12[i,j], A13[i,j],
                                  A22[i,j], A23[i,j], A33[i,j])
            eig_val_mat[i, j, 1] = λ1
            eig_val_mat[i, j, 2] = λ2
            eig_val_mat[i, j, 3] = λ3
        end
    end

    return eig_val_mat
end
