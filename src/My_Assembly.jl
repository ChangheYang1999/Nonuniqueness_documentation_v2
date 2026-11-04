# ============================================================
# My_Assembly.jl
#
# Combines the Y/X/Z basis and derivative bounds from My_Bessel.jl into
# finite-rank matrices, packed bounds, and reconstructed fields.
# Read each driver before its helpers:
#   - `integrate_r`: combines θ-integrated tensors with radial Newton–Cotes quadrature.
#   - `get_lap_matrix`: builds diagonal energy matrices from weighted squared L² norms.
#   - `lst_to_mat`: packs per-degree bounds into finite-rank parity blocks.
#   - `eval_eig_function`, `apply_Q_to_channel_fields`: reconstruct fields and multiply by Q.
# The two Gram-matrix workflows use `_r_sq_bound_table` for the radial r² weight.
#
# Dependencies: My_Bessel.jl, Trig_Poly.jl, Comb_Bound.jl, Matrix_Func.jl.
# Matrix assembly and basis ordering: docs/technical_notes/finite_rank_assembly.md.
# Bound packing, reconstruction, and Q multiplication: docs/technical_notes/finite_rank_rhs.md.
# Individual basis functions and angular conventions: docs/technical_notes/bessel_basis.md.
# Y/X are coupled poloidal channels; Z is toroidal. Matrix rows index basis elements.
# ============================================================

# --- 1. Shared parity and angular conventions ---

# Field parity: 1 = even (even-degree poloidal, odd-degree toroidal); 2 = odd (the converse).
_field_parity_poloidal(l::Int) = mod(l, 2) + 1
_field_parity_toroidal(l::Int) = mod(l + 1, 2) + 1

# Angular blocks for the (Y, X, Z) channels. The integers index arrays in Uθ_lst,
# stored in boundary-code order ["00", "01", "10", "11"]:
#   index 1 field-even: Y="11", X="00", Z="01"
#   index 2 field-odd:  Y="10", X="01", Z="00"
_angular_code_indices_by_field_parity() = [[4, 1, 2], [3, 2, 1]]

# --- 2. Q-weighted Gram matrix: integrate_r ---

"""
Assemble the Q-weighted Gram matrices by radial Newton–Cotes quadrature,
including the r² weight, from the tensors returned by `integrate_θ`.
Y and X are components of the same poloidal element; Z is toroidal.

# Arguments
- `int_θ_lst`: `[field-even, field-odd]` angular tensors, each with axes
  `(extended radial sample, left compressed degree, right compressed degree,
  left component, right component)`. Components 1, 2, 3 are Y, X, Z.
  The first axis has `length(r)+16` samples: eight extra points at each end
  on the same uniform grid. Cropping `9:end-8` aligns it with `r`.
- `Ur_Y_lst`, `Ur_X_lst`, `Ur_Z_lst`: per-degree radial sample matrices.
  `Ur_*_lst[l]` has shape `(length(r), n_l)`, with one column per root.
  Y and X must share the same degree/root layout.
- `Y_bound_lst`, `X_bound_lst`, `Z_bound_lst`: matching radial derivative
  bounds through order 8; row `m+1` bounds derivative order `m` for each root.
- `r`: uniform, endpoint-inclusive grid on `[0,1]`; `length(r)-1` must be
  a positive multiple of 6. The quadrature infers its spacing from this count.
- `M`: Q's radial frequency cutoff. Because Q's argument is `(Pi/2)*r`,
  `int_θ_bound` uses the normalized-radius frequency bound `M*Pi/2`.

# Returns
`(A_even, A_odd, A_even_err, A_odd_err)`: two square matrices of quadrature
values and their separate entrywise remainder allowances. Within each parity,
rows and columns pack poloidal elements first, then toroidal elements, ordered
by increasing degree and then by root column. The three assembly helpers fill
the poloidal–poloidal, toroidal–toroidal, and mixed blocks in this order.

In interval mode, the value matrices enclose the weighted sums; the caller
uses `add_error` to enclose the integrals, then multiplies by `4πR³` for the
physical matrices. Float64 values alone are not rigorous enclosures.
See `docs/technical_notes/finite_rank_assembly.md#basis-ordering` for packing
and sections 4 and 6 for the integration and final scaling.
"""
function integrate_r(int_θ_lst::Vector, Ur_Y_lst::Vector, Ur_X_lst::Vector, Ur_Z_lst::Vector,
    Y_bound_lst::Vector, X_bound_lst::Vector, Z_bound_lst::Vector, r::Vector{dtype}, M::Int)
    k = 8
    h = r[2] - r[1]
    r_sq = r .^ 2
    r_sq_bound = _r_sq_bound_table(k)

    poloidal_counts, toroidal_counts = _count_packed_basis_sizes(Ur_Y_lst, Ur_Z_lst)
    result_lst, err_lst = _allocate_gram_blocks(poloidal_counts, toroidal_counts)
    # Replace the radial-sample axis by derivative orders 0:8; preserve the other axes.
    Q_bound_lst = int_θ_bound.(int_θ_lst, Ref(h), Ref(k), Ref(M))

    _assemble_poloidal_poloidal_blocks!(result_lst, err_lst, int_θ_lst, Q_bound_lst,
        Ur_Y_lst, Ur_X_lst, Y_bound_lst, X_bound_lst, poloidal_counts, r_sq, r_sq_bound, k)
    _assemble_toroidal_toroidal_blocks!(result_lst, err_lst, int_θ_lst, Q_bound_lst,
        Ur_Z_lst, Z_bound_lst, poloidal_counts, toroidal_counts, r_sq, r_sq_bound, k)
    _assemble_poloidal_toroidal_blocks!(result_lst, err_lst, int_θ_lst, Q_bound_lst,
        Ur_Y_lst, Ur_X_lst, Ur_Z_lst, Y_bound_lst, X_bound_lst, Z_bound_lst,
        poloidal_counts, r_sq, r_sq_bound, k)

    return result_lst[1], result_lst[2], err_lst[1], err_lst[2]
end

# Helpers for integrate_r

function _count_packed_basis_sizes(Ur_Y_lst, Ur_Z_lst)
    poloidal_counts = [0, 0]   # each Y/X pair counts as one basis element
    toroidal_counts = [0, 0]   # Z basis elements
    for i in 1:length(Ur_Y_lst)
        poloidal_counts[_field_parity_poloidal(i)] += size(Ur_Y_lst[i], 2)
    end
    for i in 1:length(Ur_Z_lst)
        toroidal_counts[_field_parity_toroidal(i)] += size(Ur_Z_lst[i], 2)
    end
    return poloidal_counts, toroidal_counts
end

function _allocate_gram_blocks(poloidal_counts, toroidal_counts)
    # Each parity block packs [poloidal elements; toroidal elements] on both matrix axes.
    n_even = poloidal_counts[1] + toroidal_counts[1]
    n_odd  = poloidal_counts[2] + toroidal_counts[2]
    result_lst = [zeros(dtype, n_even, n_even), zeros(dtype, n_odd, n_odd)]
    err_lst    = [zeros(dtype, n_even, n_even), zeros(dtype, n_odd, n_odd)]
    return result_lst, err_lst
end

# Indices shared by the three assembly helpers:
# ind: field parity; i,j: degrees; ii,jj: root columns at those degrees.
# ih=(i+1)÷2 and jh=(j+1)÷2: compressed degree indices in the angular tensor.
# cur_i,cur_j: counts of preceding roots within the respective family/parity;
# NY: total poloidal count, added to each toroidal matrix index.
# Thus a poloidal row is cur_i+ii; a toroidal row is NY+cur_i+ii.
# cur_mat axes: (extended radial sample, ih, jh, left component, right component).
# cur_bound replaces its sample axis by derivative orders 0:k. With k=8,
# cur_mat[k+1:end-k,...] removes the padding and aligns with the radial samples.

# Part 1: poloidal–poloidal block. Each poloidal element has two radial channels (Y, X), so a
# single matrix entry receives YY, XY, YX, and XX contributions (Q11, Q21, Q12, Q22).
function _assemble_poloidal_poloidal_blocks!(result_lst, err_lst, int_θ_lst, Q_bound_lst,
        Ur_Y_lst, Ur_X_lst, Y_bound_lst, X_bound_lst, poloidal_counts, r_sq, r_sq_bound, k)
    tmp = similar(r_sq)
    LY = length(Ur_Y_lst)
    # offsets[ind] counts poloidal roots already packed in this parity.
    offsets = [0, 0]
    for i in 1:LY
        ind = _field_parity_poloidal(i)
        cur_i = offsets[ind]
        Li = size(Ur_Y_lst[i], 2) # root count at degree i

        ih = (i+1)÷2
        cur_mat = int_θ_lst[ind]
        cur_bound = Q_bound_lst[ind]
        cur_j = cur_i

        for j in i:2:LY
            Lj = size(Ur_Y_lst[j], 2)

            jh = (j+1)÷2
            r_sq_Q11 = r_sq .* cur_mat[k+1:end-k, ih, jh, 1, 1] # r² times the YY angular entry
            r_sq_Q12 = r_sq .* cur_mat[k+1:end-k, ih, jh, 1, 2]
            r_sq_Q21 = r_sq .* cur_mat[k+1:end-k, ih, jh, 2, 1]
            r_sq_Q22 = r_sq .* cur_mat[k+1:end-k, ih, jh, 2, 2]
            println("Integrating r-blocks (Y,X) $i and $j")

            for ii in 1:Li, jj in 1:Lj
                @views begin
                    yi = Ur_Y_lst[i][:, ii]
                    xi = Ur_X_lst[i][:, ii]
                    yj = Ur_Y_lst[j][:, jj]
                    xj = Ur_X_lst[j][:, jj]

                    b_yi = Y_bound_lst[i][:, ii]
                    b_xi = X_bound_lst[i][:, ii]
                    b_yj = Y_bound_lst[j][:, jj]
                    b_xj = X_bound_lst[j][:, jj]
                end

                # Product-bound factors: r², left radial channel, angular entry, right radial channel.
                # multinomial_sum bounds derivative order k=8 for the quadrature remainder.
                # YY   (yi · yj · Q11)
                @. tmp = yi * yj * r_sq_Q11
                pt1, err1 = newton_cotes_with_bound(tmp, multinomial_sum(k, r_sq_bound, b_yi, cur_bound[:, ih, jh, 1, 1], b_yj))

                # XY   (xi · yj · Q21)
                @. tmp = xi * yj * r_sq_Q21
                pt2, err2 = newton_cotes_with_bound(tmp, multinomial_sum(k, r_sq_bound, b_xi, cur_bound[:, ih, jh, 2, 1], b_yj))

                # YX   (yi · xj · Q12)
                @. tmp = yi * xj * r_sq_Q12
                pt3, err3 = newton_cotes_with_bound(tmp, multinomial_sum(k, r_sq_bound, b_yi, cur_bound[:, ih, jh, 1, 2], b_xj))

                # XX   (xi · xj · Q22)
                @. tmp = xi * xj * r_sq_Q22
                pt4, err4 = newton_cotes_with_bound(tmp, multinomial_sum(k, r_sq_bound, b_xi, cur_bound[:, ih, jh, 2, 2], b_xj))

                result_lst[ind][cur_i+ii, cur_j+jj] = pt1 + pt2 + pt3 + pt4
                err_lst[ind][cur_i+ii, cur_j+jj] = err1 + err2 + err3 + err4
            end

            # All root pairs are evaluated when i=j; for distinct degrees, copy the transpose.
            if j > i
                result_lst[ind][cur_j+1: cur_j+Lj, cur_i+1: cur_i+Li] =
                    result_lst[ind][cur_i+1: cur_i+Li, cur_j+1: cur_j+Lj]'

                err_lst[ind][cur_j+1: cur_j+Lj, cur_i+1: cur_i+Li] =
                    err_lst[ind][cur_i+1: cur_i+Li, cur_j+1: cur_j+Lj]'
            end

            cur_j += Lj
        end
        offsets[ind] += Li
    end
    @assert offsets == poloidal_counts "offsets = $offsets, poloidal_counts = $poloidal_counts"
    return nothing
end

# Part 2: toroidal–toroidal block. Only the Z channel is nonzero; uses the Q33 θ-integrated entry.
# Rows/cols start after the whole poloidal block, at offset NY = poloidal_counts[ind].
function _assemble_toroidal_toroidal_blocks!(result_lst, err_lst, int_θ_lst, Q_bound_lst,
        Ur_Z_lst, Z_bound_lst, poloidal_counts, toroidal_counts, r_sq, r_sq_bound, k)
    tmp = similar(r_sq)
    LZ = length(Ur_Z_lst)
    offsets = [0, 0]   # toroidal elements already packed in parity block ind
    for i in 1:LZ
        ind = _field_parity_toroidal(i)
        cur_i = offsets[ind]
        Li = size(Ur_Z_lst[i], 2)
        ih = (i+1)÷2
        cur_mat = int_θ_lst[ind]
        cur_bound = Q_bound_lst[ind]
        cur_j = cur_i
        NY = poloidal_counts[ind]   # shift both toroidal matrix indices
        for j in i:2:LZ
            println("Integrating r-blocks Z $i and $j")
            Lj = size(Ur_Z_lst[j], 2)
            jh = (j+1)÷2
            r_sq_Q33 = r_sq .* cur_mat[k+1:end-k, ih, jh, 3, 3]
            for ii in 1:Li, jj in 1:Lj
                @views @. tmp = Ur_Z_lst[i][:, ii] * Ur_Z_lst[j][:, jj] * r_sq_Q33
                pt, err = newton_cotes_with_bound(tmp,
                    multinomial_sum(k, r_sq_bound, Z_bound_lst[i][:, ii], cur_bound[:, ih, jh, 3, 3], Z_bound_lst[j][:, jj]))
                result_lst[ind][NY+cur_i+ii, NY+cur_j+jj] = pt
                err_lst[ind][NY+cur_i+ii, NY+cur_j+jj] = err
            end
            if j > i
                result_lst[ind][NY+ cur_j+1: NY+ cur_j+Lj, NY+ cur_i+1: NY+ cur_i+Li] =
                    result_lst[ind][NY+ cur_i+1: NY+ cur_i+Li, NY+ cur_j+1: NY+ cur_j+Lj]'
                
                err_lst[ind][NY+cur_j+1: NY+cur_j+Lj, NY+cur_i+1: NY+cur_i+Li] =
                    err_lst[ind][NY+cur_i+1: NY+cur_i+Li, NY+cur_j+1: NY+cur_j+Lj]'
            end
            cur_j += Lj
        end
        offsets[ind] += Li
    end
    @assert offsets == toroidal_counts "offsets = $offsets, toroidal_counts = $toroidal_counts"
    return nothing
end

# Part 3: (Y,X)–Z cross block between poloidal and toroidal elements; uses Q13 and Q23. Stores the
# poloidal→Z block, then fills the Z→poloidal block by transpose; only the Z index gets the NY offset.
function _assemble_poloidal_toroidal_blocks!(result_lst, err_lst, int_θ_lst, Q_bound_lst,
        Ur_Y_lst, Ur_X_lst, Ur_Z_lst, Y_bound_lst, X_bound_lst, Z_bound_lst,
        poloidal_counts, r_sq, r_sq_bound, k)
    tmp = similar(r_sq)
    LY = length(Ur_Y_lst)
    LZ = length(Ur_Z_lst)
    offsets = [0, 0]
    for i in 1:LY
        ind = _field_parity_poloidal(i)
        cur_i = offsets[ind]
        Li = size(Ur_Y_lst[i], 2)
        ih = (i+1)÷2
        cur_mat = int_θ_lst[ind]
        cur_bound = Q_bound_lst[ind]
        cur_j = 0   # local offset on the toroidal (column) side
        NY = poloidal_counts[ind]   # start of the toroidal column block = #poloidal elements
        # Same field parity pairs opposite degree parities, so Z degrees start at ind.
        for j in ind:2:LZ
            println("Integrating r-blocks (Y,X)/Z $i and $j")
            Lj = size(Ur_Z_lst[j], 2)
            jh = (j+1)÷2
            r_sq_Q13 = r_sq .* cur_mat[k+1:end-k, ih, jh, 1, 3]
            r_sq_Q23 = r_sq .* cur_mat[k+1:end-k, ih, jh, 2, 3]
            for ii in 1:Li, jj in 1:Lj
                @views @. tmp = Ur_Y_lst[i][:, ii] * Ur_Z_lst[j][:, jj] * r_sq_Q13
                pt1, err1 = newton_cotes_with_bound(tmp,
                    multinomial_sum(k, r_sq_bound, Y_bound_lst[i][:, ii], cur_bound[:, ih, jh, 1, 3], Z_bound_lst[j][:, jj]))

                @views @. tmp = Ur_X_lst[i][:, ii] * Ur_Z_lst[j][:, jj] * r_sq_Q23
                pt2, err2 = newton_cotes_with_bound(tmp,
                    multinomial_sum(k, r_sq_bound, X_bound_lst[i][:, ii], cur_bound[:, ih, jh, 2, 3], Z_bound_lst[j][:, jj]))

                # Write the poloidal→Z block (poloidal rows, Z columns at offset NY).
                result_lst[ind][cur_i+ii, NY+cur_j+jj] = pt1 + pt2
                err_lst[ind][cur_i+ii, NY+cur_j+jj] = err1 + err2
            end
            # Fill the Z→poloidal block by transpose of the poloidal→Z block just written.
            result_lst[ind][NY+cur_j+1: NY+cur_j+Lj, cur_i+1: cur_i+Li] =
                result_lst[ind][cur_i+1: cur_i+Li, NY+cur_j+1: NY+cur_j+Lj]'

            err_lst[ind][NY+cur_j+1: NY+cur_j+Lj, cur_i+1: cur_i+Li] =
                err_lst[ind][cur_i+1: cur_i+Li, NY+cur_j+1: NY+cur_j+Lj]'
            cur_j += Lj
        end
        offsets[ind] += Li
    end
    @assert offsets == poloidal_counts "offsets = $offsets, poloidal_counts = $poloidal_counts"
    return nothing
end

# --- 3. Diagonal energy matrix: get_lap_matrix ---

"""
Assemble the diagonal energy Gram matrices B from squared L² basis norms.
Analytic orthogonality supplies the zero off-diagonal entries.

# Arguments
- `Ur_Y_lst`, `Ur_X_lst`, `Ur_Z_lst`: per-degree radial samples, each block
  shaped `(length(r), n_l)`; Y and X share the poloidal root layout.
- `Y_bound_lst`, `X_bound_lst`, `Z_bound_lst`: matching radial derivative
  tables through order 8, with row `m+1` bounding derivative order `m`.
- `Y_paras`, `Z_paras`: per-degree root parameters; `[l][j][1]` is α.
- `r`: uniform, endpoint-inclusive grid on `[0,1]`, with `length(r)-1`
  a positive multiple of 6.
- `R`, `β`: physical radius and constant energy shift in `α²/R² + β`.
  Here `β` is the notebook's `eta`, not the compactified radial coordinate.

# Returns
`(B_even, B_odd, B_even_err, B_odd_err)`: diagonal energy values and separate
quadrature allowances, packed in the same field-parity order as `integrate_r`.
No square root is taken. In interval mode, the caller applies `add_error`,
then the physical factor `4πR³`, to obtain the integral enclosures.
See `docs/technical_notes/finite_rank_assembly.md`, sections 1, 5, and 6.
"""
function get_lap_matrix(Ur_Y_lst::Vector, Ur_X_lst::Vector, Ur_Z_lst::Vector,
    Y_bound_lst::Vector, X_bound_lst::Vector, Z_bound_lst::Vector,
    Y_paras::Vector, Z_paras::Vector, r::Vector{dtype}, R::dtype, β::dtype)
    # Bound the eighth derivatives of r²U², then integrate the squared basis norms.
    Y_bound_lst = Ur_sq_Wk∞_bound_lists(Y_bound_lst)
    X_bound_lst = Ur_sq_Wk∞_bound_lists(X_bound_lst)
    Z_bound_lst = Ur_sq_Wk∞_bound_lists(Z_bound_lst);
    Y_int_lst, Z_int_lst, Y_err_lst, Z_err_lst = eval_l2_inpd(Ur_Y_lst, Ur_X_lst, Ur_Z_lst, Y_bound_lst, X_bound_lst, Z_bound_lst, r)

    # Apply the energy factor to squared norms and their quadrature allowances.
    Y_norm_even, Y_norm_odd = eval_h1_norm(Y_int_lst, Y_paras, R, β)
    Z_norm_even, Z_norm_odd = eval_h1_norm(Z_int_lst, Z_paras, R, β)
    Y_err_even, Y_err_odd = eval_h1_norm(Y_err_lst, Y_paras, R, β)
    Z_err_even, Z_err_odd = eval_h1_norm(Z_err_lst, Z_paras, R, β)

    # eval_h1_norm splits by degree parity; recombine into field parity:
    # even-degree poloidal + odd-degree toroidal gives field-even, and conversely.
    norm_vec_even = [Y_norm_even; Z_norm_odd]
    norm_vec_odd = [Y_norm_odd; Z_norm_even]
    norm_err_even = [Y_err_even; Z_err_odd]
    norm_err_odd = [Y_err_odd; Z_err_even]

    B_even = Matrix(Diagonal(norm_vec_even))
    B_odd = Matrix(Diagonal(norm_vec_odd))
    B_even_err = Matrix(Diagonal(norm_err_even))
    B_odd_err = Matrix(Diagonal(norm_err_odd))
    return B_even, B_odd, B_even_err, B_odd_err
end

# Helpers for get_lap_matrix

"""
Bound derivative order `k` of each weighted square `r²U(r)²` using
`multinomial_sum` and `_r_sq_bound_table(k)`; the default is `k=8`.

`bound_lst[l]` stores radial derivative bounds through order `k`, with
row `m+1` for order `m` and one column per root. Supply `k ≥ 2` for the
radial-weight table. The return value is a per-degree list of vectors:
`result[l][j]` is one scalar bound for the kth derivative at root `j`,
not a table of bounds for all orders. `eval_l2_inpd` uses the order-8
results as quadrature remainder inputs.
See `docs/technical_notes/finite_rank_assembly.md`, section 5.2.
"""
function Ur_sq_Wk∞_bound_lists(bound_lst::Vector; k::Int=8)
    r_sq_bound = _r_sq_bound_table(k)

    sq_bound_lst = []
    for each_lst in bound_lst
        tmp_lst = zeros(dtype, size(each_lst, 2))
        @views for j in axes(each_lst, 2)
            tmp_lst[j] = multinomial_sum(k, r_sq_bound, each_lst[:, j], each_lst[:, j])
        end
        push!(sq_bound_lst, tmp_lst)
    end
    return sq_bound_lst
end

"""
Compute diagonal squared L² basis norms using angular identities and radial
Newton–Cotes quadrature. This evaluates the diagonal entries required by
analytic orthogonality; it does not test orthogonality numerically.

For each degree/root pair, let `I_Y`, `I_X`, `I_Z` denote the integrals of
`r²U_Y²`, `r²U_X²`, `r²U_Z²` over `[0,1]`. The poloidal value is
`(I_Y + l(l+1)*I_X)/(2l+1)`; the toroidal value is `l(l+1)*I_Z/(2l+1)`.
These include the half-meridian angular integrals but omit `4πR³`.

# Arguments
- `Ur_Y_lst`, `Ur_X_lst`, `Ur_Z_lst`: per-degree radial samples, each block
  shaped `(length(r), n_l)`; Y and X share the same root columns.
- `Y_bound_lst`, `X_bound_lst`, `Z_bound_lst`: outputs of
  `Ur_sq_Wk∞_bound_lists(...; k=8)`. Each `[l][j]` is a single bound on
  the eighth derivative of the weighted square, not a raw derivative table.
- `r`: uniform, endpoint-inclusive grid on `[0,1]`, with `length(r)-1`
  a positive multiple of 6.

# Returns
`(Y_int_lst, Z_int_lst, Y_err_lst, Z_err_lst)`: per-degree vectors of
quadrature values and separate allowances, one entry per root. `Y_int_lst`
contains the complete poloidal squared norm, including the X contribution.
The caller must add the allowances to enclose the integrals in interval mode.
See `docs/technical_notes/finite_rank_assembly.md`, section 5.2.
"""
function eval_l2_inpd(Ur_Y_lst::Vector, Ur_X_lst::Vector, Ur_Z_lst::Vector, 
    Y_bound_lst::Vector, X_bound_lst::Vector, Z_bound_lst::Vector, r::Vector{dtype})
    Y_int_lst = []
    Z_int_lst = []
    Y_err_lst = []
    Z_err_lst = []
    r_sq = r .^ 2

    for l in eachindex(Ur_Y_lst)
        # Both radial components contribute to the same poloidal squared norm.
        fX = Ur_X_lst[l] .* Ur_X_lst[l] .* r_sq
        fY = Ur_Y_lst[l] .* Ur_Y_lst[l] .* r_sq

        res_Y = newton_cotes_with_bound.(eachcol(fX), X_bound_lst[l])
        Y_int1 = first.(res_Y)
        Y_err1 = last.(res_Y)

        res_Y = newton_cotes_with_bound.(eachcol(fY), Y_bound_lst[l])
        Y_int2 = first.(res_Y)
        Y_err2 = last.(res_Y)

        Y_int = (Y_int1 .* dt(l*(l+1)) .+ Y_int2) ./ dt(2l+1)
        Y_err = (Y_err1 .* dt(l*(l+1)) .+ Y_err2) ./ dt(2l+1)

        push!(Y_int_lst, Y_int)
        push!(Y_err_lst, Y_err)
    end
    for l in eachindex(Ur_Z_lst)
        fZ = Ur_Z_lst[l] .* Ur_Z_lst[l] .* r_sq

        res_Z = newton_cotes_with_bound.(eachcol(fZ), Z_bound_lst[l])
        Z_int = first.(res_Z)
        Z_err = last.(res_Z)

        Z_int .= Z_int .* dt(l*(l+1)) ./ dt(2l+1)
        Z_err .= Z_err .* dt(l*(l+1)) ./ dt(2l+1)

        push!(Z_int_lst, Z_int)
        push!(Z_err_lst, Z_err)
    end
    return Y_int_lst, Z_int_lst, Y_err_lst, Z_err_lst
end

"""
Multiply per-root squared L² values by the energy factor `α²/R² + β`,
then split one basis family by degree parity. `norm_vec[l][j]` supplies
the value and `paras[l][j][1]` supplies α. The same operation propagates
quadrature allowances when the energy factor is nonnegative.

`R` is the physical radius; `β` is the constant called `eta` in the
notebook, not its compactified radial coordinate. No square root is taken,
and the physical factor `4πR³` is left to the caller.

Returns `(res_even, res_odd)` for even and odd degrees, respectively,
ordered by increasing degree and then root. These are degree-parity
vectors for one family; `get_lap_matrix` combines the poloidal and
toroidal families into field-parity matrices.
See `docs/technical_notes/finite_rank_assembly.md`, section 5.3.
"""
function eval_h1_norm(norm_vec::Vector, paras::Vector, R::dtype, β::dtype)
    # Here ind=1 means even degree and ind=2 means odd degree, for one family.
    N_lst = [0, 0]
    L = length(paras)
    for l in 1:L
        N_lst[mod(l, 2) + 1] += length(paras[l])
    end
    result_lst = [zeros(dtype, N_lst[1]), zeros(dtype, N_lst[2])]
    cur_l_lst = [0, 0]

    for l in 1:L
        ind = mod(l, 2) + 1
        for j in 1:length(norm_vec[l])
            result_lst[ind][cur_l_lst[ind]+j] = norm_vec[l][j] * (paras[l][j][1]^2/R^2 + β)
        end

        # Advance the root-count offset for this degree parity.
        cur_l_lst[ind] += length(norm_vec[l]) 
    end
    return result_lst[1], result_lst[2]
end

# --- 4. Packing derivative bounds ---

"""
Pack radial derivative bounds for full basis components, uniform in θ.
The input tables from `Ur_bound_lst` have shape `(K+1, n_l)` at degree `l`:
row `m+1` bounds radial derivative order `m`, for `m=0:K`. Y and X must
share the poloidal root layout.

Returns `(U_even, U_odd)` with axes `(derivative order + 1, packed basis
element, component)` and shape `(K+1, N_parity, 3)`. The basis axis matches
the eigenvector rows: poloidal elements first, then toroidal, ordered by
degree and root within each family/parity.

Components 1, 2, 3 are Y, X, Z. The Y angular factor has magnitude bound 1;
X and Z receive the angular magnitude bound `l` from `Uθ_bound(l,2)`.
Thus these are bounds for radial derivatives of the full components,
not just their radial factors. Unused component slots remain zero.
See `docs/technical_notes/finite_rank_rhs.md#packed-field-bounds`.
"""
function lst_to_mat(Y_bound_lst, X_bound_lst, Z_bound_lst)
    poloidal_counts = [0, 0]
    toroidal_counts = [0, 0]
    LY = length(Y_bound_lst)
    LZ = length(Z_bound_lst)
    k = size(Y_bound_lst[1], 1)   # row count K+1, not the maximum derivative order
    for i in 1:LY
        ind = mod(i, 2) + 1
        poloidal_counts[ind] += size(Y_bound_lst[i], 2)
    end
    for i in 1:LZ
        ind = mod(i+1, 2) + 1
        toroidal_counts[ind] += size(Z_bound_lst[i], 2)
    end

    # [field-even, field-odd]; axes are derivative order, packed basis element, component.
    result_lst = [zeros(dtype, k, poloidal_counts[1] + toroidal_counts[1], 3),
                  zeros(dtype, k, poloidal_counts[2] + toroidal_counts[2], 3)]

    # offsets[ind] counts roots already placed on the packed basis axis for this family.
    offsets = [0, 0]  

    l = max(length(Y_bound_lst), length(Z_bound_lst))
    Uθ_bound_lst = dt.(Uθ_bound(l, 2))

    # Poloidal elements occupy both Y and X component slots.
    for i in 1:LY
        ind = mod(i, 2) + 1
        cur_i = offsets[ind]
        Li = size(Y_bound_lst[i], 2) # root count at degree i

        # Angular magnitude bounds: 1 for Y, degree i for X (and Z below).
        result_lst[ind][:, cur_i+1: cur_i+Li, 1] = Y_bound_lst[i]
        result_lst[ind][:, cur_i+1: cur_i+Li, 2] = X_bound_lst[i] * Uθ_bound_lst[i, 2]
        offsets[ind] += Li
    end

    # Check that all poloidal roots were packed.
    @assert offsets == poloidal_counts "offsets = $offsets, poloidal_counts = $poloidal_counts"

    offsets = [0, 0]

    # Toroidal elements occupy only the Z component slot.
    for i in 1:LZ
        ind = mod(i+1, 2) + 1
        cur_i = offsets[ind]
        Li = size(Z_bound_lst[i], 2)
        # Start Z after all poloidal elements on the packed basis axis.
        cur_N = poloidal_counts[ind]

        result_lst[ind][:, cur_N+cur_i+1 : cur_N+cur_i+Li, 3] = Z_bound_lst[i] * Uθ_bound_lst[i, 2]
        offsets[ind] += Li
    end
    @assert offsets == toroidal_counts "offsets = $offsets, toroidal_counts = $toroidal_counts"
    return result_lst[1], result_lst[2]
end

# --- 5. Field reconstruction and Q multiplication ---

"""
Reconstruct finite-rank eigenfunctions ψ on the `(r, θ)` grid.

Rows of each coefficient matrix index packed basis elements; columns index
retained modes. One poloidal coefficient contributes to both Y and X channels;
one toroidal coefficient contributes to Z. This function reconstructs ψ only.
To multiply these fields by Q, use `apply_Q_to_channel_fields` with separately
sampled Q.
See `docs/technical_notes/finite_rank_rhs.md`, sections 1–2, for reconstruction;
`docs/technical_notes/finite_rank_assembly.md#basis-ordering` for packing; and
`docs/technical_notes/bessel_basis.md` for the angular convention.

# Arguments
- `eig_vec_odd::AbstractMatrix{dtype}`: `(N_odd, L_odd)` coefficients.
- `eig_vec_even::AbstractMatrix{dtype}`: `(N_even, L_even)` coefficients.
- `Y_paras::Vector`, `Z_paras::Vector`: radial parameters from `load_besselj_zeros`.
- `r::Vector{dtype}`, `θ::Vector{dtype}`: normalized-radius and angular grids.

# Returns
- A two-element vector `[ψ_even, ψ_odd]`, although inputs list odd before even.
  Each array has axes `(radial sample, angular sample, mode, component)` and
  shape `(length(r), length(θ), L_even_or_odd, 3)`; components are Y, X, Z.
  Values are physical spherical components in (e_r,e_θ,e_ϕ) order; the Bessel X radial sign compensates the angular factor H_l=−S_l′.
"""
function eval_eig_function(eig_vec_odd::AbstractMatrix{dtype}, eig_vec_even::AbstractMatrix{dtype},
    Y_paras::Vector, Z_paras::Vector, r::Vector{dtype}, θ::Vector{dtype})
    # Y and X share poloidal parameters and root columns; Z has its own toroidal parameters.
    Ur_Y_lst = eval_Ur(r, Y_paras, "Y")
    Ur_X_lst = eval_Ur(r, Y_paras, "X")
    Ur_Z_lst = eval_Ur(r, Z_paras, "Z")

    # precompute angular rows for the four boundary-code blocks ["00","01","10","11"]
    Uθ_lst = eval_Uθ(θ, max(length(Ur_Y_lst), length(Ur_Z_lst)))
    ind_lst = _angular_code_indices_by_field_parity()

    N_r = length(r)
    N_θ = length(θ)
    N_even, L_even = size(eig_vec_even)
    N_odd, L_odd = size(eig_vec_odd)

    # These lists are indexed by field parity: index 1 = field-even, index 2 = field-odd.
    eig_vec_blocks = [eig_vec_even, eig_vec_odd]
    channel_field_blocks = _allocate_channel_field_blocks(N_r, N_θ, L_even, L_odd)

    # offsets[ind] = number of packed basis elements already consumed in parity block ind.
    # It advances through all poloidal Y/X elements first, then the toroidal Z elements.
    offsets = [0, 0]
    _accumulate_poloidal_channels!(channel_field_blocks, eig_vec_blocks,
        Ur_Y_lst, Ur_X_lst, Uθ_lst, ind_lst, N_r, offsets)
    _accumulate_toroidal_channel!(channel_field_blocks, eig_vec_blocks,
        Ur_Z_lst, Uθ_lst, ind_lst, N_r, offsets)

    # check that every packed basis element was consumed
    @assert offsets[1] == N_even "offsets[1] = $offsets[1], N_even = $N_even"
    @assert offsets[2] == N_odd "offsets[2] = $offsets[2], N_odd = $N_odd"
    
    return channel_field_blocks
end

"""
Multiply each reconstructed field by sampled Q at every grid point:
output component `i` is `sum_j Q_ij * ψ_j` for each parity and retained mode.

Q_pts supplies the six physical symmetric entries A11, A12, A13, A22, A23, A33, with
space_func arrays of shape (N_r,N_θ) on the same grid as
channel_field_blocks=[ψ_even,ψ_odd]. Both use physical spherical components; do not
conjugate Q by D. No interpolation or coordinate conversion is performed. Returns
(Qψ_even,Qψ_odd), preserving the input shape (N_r,N_θ,L,3).

The caller applies the radial weight separately; see
`docs/technical_notes/finite_rank_rhs.md`, section 5.
"""
function apply_Q_to_channel_fields(Q_pts, channel_field_blocks)
    result_even = zeros(dtype, size(channel_field_blocks[1]))
    result_odd = zeros(dtype, size(channel_field_blocks[2]))
    # symmetric 3×3 Q field: key_mat[i][j] names the Q_ij component ("Aij")
    key_mat = [["A11", "A12", "A13"],
               ["A12", "A22", "A23"],
               ["A13", "A23", "A33"]]
    for i in 1:3, j in 1:3
        result_even[:, :, :, i] += Q_pts[key_mat[i][j]].space_func .* channel_field_blocks[1][:, :, :, j]
        result_odd[:, :, :, i] += Q_pts[key_mat[i][j]].space_func .* channel_field_blocks[2][:, :, :, j]
    end
    return result_even, result_odd
end

# Reconstruction helpers

function _allocate_channel_field_blocks(N_r::Int, N_θ::Int, L_even::Int, L_odd::Int)
    # [field-even, field-odd]; last dim = the three vector components (Y, X, Z channels)
    return [zeros(dtype, N_r, N_θ, L_even, 3), zeros(dtype, N_r, N_θ, L_odd, 3)]
end

# Contract: field-parity indexing (ind 1 = field-even, 2 = field-odd); Ur_Y_lst/Ur_X_lst are
# degree-ordered. Mutates channel_field_blocks[ind][:,:,:,1:2] (the Y and X slots). One eigenvector
# coefficient per poloidal element drives BOTH the Y and X channels. Advances offsets[ind],
# preserving packing order (poloidal elements come first within each parity block).
function _accumulate_poloidal_channels!(channel_field_blocks, eig_vec_blocks,
        Ur_Y_lst, Ur_X_lst, Uθ_lst, ind_lst, N_r, offsets)
    for l in eachindex(Ur_Y_lst)
        println("ψ YX $l/$(length(Ur_Y_lst))")
        ind = _field_parity_poloidal(l)
        Ll = size(Ur_Y_lst[l], 2) # L_l: number of radial columns (roots) at degree l

        # Angular row for the Y and X channels (boundary-code block chosen by parity), shape (N_θ,)
        Uθ_Y = Uθ_lst[ind_lst[ind][1]][(l+1)÷2, :]
        Uθ_X = Uθ_lst[ind_lst[ind][2]][(l+1)÷2, :]

        # (N_r, L_l) * (L_l, N_modes) → (N_r, N_modes): this degree's radial columns combined with
        # its rows (offsets[ind]+1 : offsets[ind]+L_l) of the parity block's eigenvector coefficients.
        Ur_Y_mat = Ur_Y_lst[l] * eig_vec_blocks[ind][offsets[ind]+1: offsets[ind]+Ll, :]
        Ur_X_mat = Ur_X_lst[l] * eig_vec_blocks[ind][offsets[ind]+1: offsets[ind]+Ll, :]

        # Outer product of angular row and radial field:
        # Uθ_Y (N_θ,) ⊗ Ur_Y_mat (N_r, N_modes) → U_Y (N_r, N_θ, N_modes).
        U_Y = reshape(Uθ_Y, 1, :, 1) .* reshape(Ur_Y_mat, N_r, 1, :)
        U_X = reshape(Uθ_X, 1, :, 1) .* reshape(Ur_X_mat, N_r, 1, :)

        channel_field_blocks[ind][:, :, :, 1] += U_Y
        channel_field_blocks[ind][:, :, :, 2] += U_X

        offsets[ind] += Ll
    end
    return nothing
end

# Contract: toroidal analogue of _accumulate_poloidal_channels!. Mutates
# channel_field_blocks[ind][:,:,:,3] (the Z slot). One Z-root → one toroidal element with a single
# Z channel. Advances offsets[ind] through the toroidal elements (packed after the poloidal ones).
function _accumulate_toroidal_channel!(channel_field_blocks, eig_vec_blocks,
        Ur_Z_lst, Uθ_lst, ind_lst, N_r, offsets)
    for l in eachindex(Ur_Z_lst)
        println("ψ Z $l/$(length(Ur_Z_lst))")
        ind = _field_parity_toroidal(l)
        Ll = size(Ur_Z_lst[l], 2)                        # L_l: radial columns (roots) at degree l
        Uθ_Z = Uθ_lst[ind_lst[ind][3]][(l+1)÷2, :]       # angular row, shape (N_θ,)
        Ur_Z_mat = Ur_Z_lst[l] * eig_vec_blocks[ind][offsets[ind]+1: offsets[ind]+Ll, :]  # (N_r, N_modes)
        U_Z = reshape(Uθ_Z, 1, :, 1) .* reshape(Ur_Z_mat, N_r, 1, :)                       # (N_r, N_θ, N_modes)
        channel_field_blocks[ind][:, :, :, 3] += U_Z
        offsets[ind] += Ll
    end
    return nothing
end

# --- 6. Shared radial-weight bound ---

function _r_sq_bound_table(k::Int)
    # Bounds for r² and its derivatives (k ≥ 2), covering [0,1] and the notebook
    # padding; see docs/technical_notes/finite_rank_assembly.md, section 4.5.
    r_sq_bound = zeros(dtype, k + 1)
    r_sq_bound[1] = dt("1.1")
    r_sq_bound[2] = dt("2.2")
    r_sq_bound[3] = dt("2.2")
    return r_sq_bound
end
