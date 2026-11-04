# ============================================================
# Trig_Poly.jl
#
# Trigonometric-polynomial algebra for the angular part of Gram-matrix assembly.
# Products use product-to-sum identities; mode integrals use integer endpoint
# values of sin(kπ/2) and cos(kπ/2), with no angular quadrature remainder.
# Arithmetic uses dtype: interval mode encloses rounding, while Float64 mode
# also skips coefficients with abs(value) < 1e-14 through is_zero.
#
# Pipeline: basis lists → tensor_int_basis → contract_QT_blocks → split_tensor.
# integrate_θ drives angular integration; int_θ_bound bounds radial derivatives
# of its output. Radial quadrature and matrix assembly are in My_Assembly.jl.
# Dependencies: My_Fourier.jl (dtype helpers and conversion_mat),
#   My_Vector.jl (VectorFunc and interpolate), Matrix_Func.jl (get_Wk∞_norm_lst).
# See docs/technical_notes/finite_rank_assembly.md for the assembly workflow
# and docs/src.md for the source-reading guide.
# ============================================================

"""
Container for a single-variable trigonometric polynomial.

Stores the polynomial as separate cosine and sine coefficient vectors:
```
P(x) = sum_{k=0}^{...} c[k+1] cos(kx) + sum_{k=1}^{...} s[k] sin(kx)
```

# Fields
- `c::Vector{dtype}`: cosine coefficients; `c[k+1]` multiplies `cos(kx)` for `k ≥ 0`.
- `s::Vector{dtype}`: sine coefficients; `s[k]` multiplies `sin(kx)` for `k ≥ 1`.
"""
struct TrigPoly
    c::Vector{dtype}  # cos coeffs: c[k+1] for cos(kx), k>=0
    s::Vector{dtype}  # sin coeffs: s[k]   for sin(kx), k>=1
end

############################
# Multiplication
############################
"""
Add `val*cos(kx)` to `c` in place, using `cos(-kx) = cos(kx)`.
Extend the coefficient vector with zeros when needed.
"""
@inline function _add_cos!(c::Vector{dtype}, k::Int, val::dtype)
    if k < 0
        k = -k
    end
    idx = k + 1
    if idx > length(c)
        old = length(c)
        resize!(c, idx)
        fill!(view(c, old+1:idx), zero(dtype))
    end
    c[idx] += val
end

"""
Add `val*sin(kx)` to `s` in place, reversing the sign when `k < 0`.
Ignore `k == 0`; extend the coefficient vector with zeros when needed.
"""
@inline function _add_sin!(s::Vector{dtype}, k::Int, val::dtype)
    if k == 0
        return
    elseif k < 0
        k = -k
        val = -val
    end
    idx = k  # s[k] corresponds to sin(kx)
    if idx > length(s)
        old = length(s)
        resize!(s, idx)
        fill!(view(s, old+1:idx), zero(dtype))
    end
    s[idx] += val
end

"""
Multiply two trigonometric polynomials using product-to-sum identities:
- cos(mx)cos(nx) = (cos((m-n)x) + cos((m+n)x))/2
- sin(mx)sin(nx) = (cos((m-n)x) - cos((m+n)x))/2
- sin(mx)cos(nx) = (sin((m+n)x) + sin((m-n)x))/2

# Degree convention (storage-based)
Let
    deg(P) = max( length(P.c)-1, length(P.s) ).
Then `deg(A*B) ≤ deg(A) + deg(B)`, and with full storage allocation we create
arrays up to that maximum degree.

# Arguments
- `A::TrigPoly`: first trigonometric polynomial
- `B::TrigPoly`: second trigonometric polynomial

# Returns
- A new `TrigPoly` representing the product; inputs are not mutated.

Arithmetic and coefficient skipping follow the numeric-mode convention at the top of this file.
"""
function trig_mul(A::TrigPoly, B::TrigPoly)
    degA = max(length(A.c) - 1, length(A.s))
    degB = max(length(B.c) - 1, length(B.s))
    deg  = degA + degB

    c = zeros(dtype, deg + 1)  # cos(0..deg)
    s = zeros(dtype, deg)      # sin(1..deg)

    # cos*cos
    for m0 in 0:(length(A.c)-1)
        am = A.c[m0+1]
        is_zero(am) && continue
        for n0 in 0:(length(B.c)-1)
            bn = B.c[n0+1]
            is_zero(bn) && continue
            v = am * bn / dt(2)
            _add_cos!(c, abs(m0 - n0), v)
            _add_cos!(c, m0 + n0, v)
        end
    end

    # sin*sin
    for m in 1:length(A.s)
        am = A.s[m]
        is_zero(am) && continue
        for n in 1:length(B.s)
            bn = B.s[n]
            is_zero(bn) && continue
            v = am * bn / dt(2)
            _add_cos!(c, abs(m - n),  v)
            _add_cos!(c, m + n,      -v)
        end
    end

    # sin*cos
    for m in 1:length(A.s)
        am = A.s[m]
        is_zero(am) && continue
        for n0 in 0:(length(B.c)-1)
            bn = B.c[n0+1]
            is_zero(bn) && continue
            v = am * bn / dt(2)
            _add_sin!(s, m + n0, v)
            _add_sin!(s, m - n0, v)
        end
    end

    # cos*sin
    for m0 in 0:(length(A.c)-1)
        am = A.c[m0+1]
        is_zero(am) && continue
        for n in 1:length(B.s)
            bn = B.s[n]
            is_zero(bn) && continue
            v = am * bn / dt(2)
            _add_sin!(s, n + m0, v)
            _add_sin!(s, n - m0, v)
        end
    end

    return TrigPoly(c, s)
end

############################
# Integral on [0, Pi/2]
############################
"""
Return `sin(kπ/2)` exactly as an integer in `{-1, 0, 1}`, using `k mod 4`.
Used in the mode integral `∫₀^{π/2} cos(kx) dx = sin(kπ/2)/k` for `k ≥ 1`.
"""
@inline _sin_kPi_over2(k::Int)::Int =
    (r = mod(k,4); r==0 ? 0 : r==1 ? 1 : r==2 ? 0 : -1)

"""
Return `cos(kπ/2)` exactly as an integer in `{-1, 0, 1}`, using `k mod 4`.
Used in the mode integral `∫₀^{π/2} sin(kx) dx = (1-cos(kπ/2))/k` for `k ≥ 1`.
"""
@inline _cos_kPi_over2(k::Int)::Int =
    (r = mod(k,4); r==0 ? 1 : r==1 ? 0 : r==2 ? -1 : 0)

"""
Integrate `P::TrigPoly` over `[0, π/2]` mode by mode, returning a `dtype` scalar.

Use the integer sine/cosine endpoint values above; the constant cosine mode
contributes `P.c[1]*Pi/2`. There is no quadrature remainder. Interval arithmetic
encloses coefficient and arithmetic uncertainty; Float64 uses rounded arithmetic
and the `is_zero` coefficient threshold. The input is not mutated.
"""
function trig_int_0_pi2(P::TrigPoly)
    res = zero(dtype)

    # cos part
    if length(P.c)  > 0
        res += P.c[1] * (Pi / dt(2))
        for k in 1:(length(P.c)-1)
            ck = P.c[k+1]
            is_zero(ck) && continue
            res += ck * dt(_sin_kPi_over2(k)) / dt(k)
        end
    end

    # sin part
    for k in 1:length(P.s)
        sk = P.s[k]
        is_zero(sk) && continue
        res += sk * dt(1 - _cos_kPi_over2(k)) / dt(k)
    end

    return res
end

"""
Construct cosine and sine basis lists up to order l.
Each element is a TrigPoly with exactly one nonzero basis coefficient equal to 1.

# Arguments
- l: nonnegative integer

# Returns
- two lists of TrigPoly:
    cos_list[j+1] = cos(jx) for j = 0..l (so cos_list[1] = 1)
    sin_list[j]   = sin(jx) for j = 1..l
"""
function trig_basis_lists(l::Int)
    @assert l >= 0

    cos_list = Vector{TrigPoly}(undef, l + 1)
    for j in 0:l
        c = zeros(dtype, j+1)  # need slots up to cos(jx)
        c[j + 1] = one(dtype)
        cos_list[j + 1] = TrigPoly(c, dtype[])  # no sin part
    end

    sin_list = Vector{TrigPoly}(undef, l)
    for j in 1:l
        s = zeros(dtype, j)      # need slots up to sin(jx)
        s[j] = one(dtype)
        sin_list[j] = TrigPoly(dtype[], s)  # no cos part
    end
   
    return cos_list, sin_list
end

"""
Construct the angular factors as trigonometric polynomials for degrees `1:l`.

With `S_j(θ) = P_j(cosθ)` and `H_j(θ) = -dS_j/dθ`, return `(P_list, dP_list)`:
- `P_list[j]` represents `S_j`, using cosine coefficients.
- `dP_list[j]` represents `H_j`, using sine coefficients (including the minus sign).

`conversion_mat` supplies the coefficients. Boundary codes `"10"` and `"11"`
select odd and even degrees of S; `"01"` and `"00"` select odd and even degrees
of H. Each length-`l` list interleaves both parities by degree and excludes degree 0.
The function checks that `l` is nonnegative and even.
See `docs/technical_notes/fourier_basis.md` for the conversion convention.
"""
function sph_basis_lists(l::Int)
    @assert l >= 0 && iseven(l)
    dP_list = Vector{TrigPoly}(undef, l)
    P_list = Vector{TrigPoly}(undef, l)
    for j in 1:l
        s = zeros(dtype, j)
        c = zeros(dtype, j + 1)
        dP_list[j] = TrigPoly(dtype[], s)  # no cos part
        P_list[j] = TrigPoly(c, dtype[])  # no sin part
    end

    bdry_lst = ["00", "01"]
    lh = l÷2
    for bdry in bdry_lst
        conv_mat = conversion_mat(lh, bdry, 1)
        id = bdry == "00" ? 2 : 1
        for i in 1:lh
            dP_list[2*i+id-2].s[id:2:end] = conv_mat[i, 1:i]
        end
    end

    bdry_lst = ["10", "11"]
    for bdry in bdry_lst
        conv_mat = conversion_mat(lh, bdry, 1)
        id = bdry == "10" ? 0 : 1
        for i in 1:lh
            P_list[2*i+id-1].c[2-id:2:end] = conv_mat[i, 1:i+id]
        end
    end

    return P_list, dP_list
end

"""
Precompute angular integrals of two spherical factors and one trigonometric mode:

    T[u,v,n] = ∫₀^{π/2} f_u(θ) f_v(θ) h_n(θ) sin(θ) dθ.

`trig_mul` expands the product and `trig_int_0_pi2` integrates each mode
analytically, including the spherical weight `sin(θ)`. There is no angular
quadrature remainder; arithmetic follows the numeric-mode convention above.

# Inputs and packed axes
- `trig_lists = (cos_list, sin_list)`, from `trig_basis_lists(Nmax)`.
  Write `Nmax1 = length(sin_list)`, `Nmax2 = length(cos_list)`.
  `sin_list[n]` represents `sin(nθ)` for `n = 1:Nmax1`;
  `cos_list[k+1]` represents `cos(kθ)` for `k = 0:Nmax2-1`.
  The third axis stacks `sin_list` followed by `cos_list`. `sin_list[1]`
  must exist and represent `sin(θ)` for the integration weight.
- `sph_lists = (p_list, dp_list)`, from `sph_basis_lists(Lmax)`.
  These represent `S_l = P_l(cosθ)` and `H_l = -dS_l/dθ`, respectively.
  Write `Lmax1 = length(dp_list)`, `Lmax2 = length(p_list)`.
  Each of the first two axes stacks `dp_list` followed by `p_list`.

Return a `dtype` tensor of shape `(Lmax1+Lmax2, Lmax1+Lmax2, Nmax1+Nmax2)`.
For the standard basis lists this is `(2Lmax, 2Lmax, 2Nmax+1)`. Packed indices
select basis factors, not radial roots. See §3.2 of
`docs/technical_notes/finite_rank_assembly.md` for an index example.
"""
function tensor_int_basis(trig_lists, sph_lists)
    cos_list, sin_list = trig_lists
    p_list, dp_list = sph_lists
    Lmax1 = length(dp_list)
    Lmax2 = length(p_list)
    Nmax1 = length(sin_list)
    Nmax2 = length(cos_list)

    # Include the spherical integration weight sin(θ).
    sin1 = (Nmax1 >= 1) ? sin_list[1] : error("sin_list must have at least sin(x).")

    T = zeros(dtype, Lmax1+Lmax2, Lmax1+Lmax2, Nmax1+Nmax2)

    for l in 1:Lmax1+Lmax2
        f1 = l > Lmax1 ? p_list[l-Lmax1] : dp_list[l]
        f1 = trig_mul(f1, sin1)  # weight the first angular factor

        for m in 1:Lmax1+Lmax2
            g = m > Lmax1 ? p_list[m-Lmax1] : dp_list[m]
            f2 = trig_mul(f1, g)

            for n in 1:Nmax1+Nmax2
                h = n > Nmax1 ? cos_list[n-Nmax1] : sin_list[n]
                f3 = trig_mul(f2, h)
                T[l, m, n] = trig_int_0_pi2(f3)
            end
        end
    end

    return T
end

"""
Contract Q's angular coefficients with the integral table `T`.

`Q::VectorFunc` contains the six symmetric entries `"A11"`, `"A12"`, `"A13"`,
`"A22"`, `"A23"`, `"A33"`; off-diagonal entries are reused by symmetry.
Every `Q[key].space_func` must have the same shape `(N,M)`: rows are radial
samples and columns are angular coefficients, not θ samples.

Let `Lh` be the maximum degree. `T` must have shape `(2Lh,2Lh,4M+1)`:
its first two axes stack `[H₁,…,H_Lh,S₁,…,S_Lh]`, where
`S_l = P_l(cosθ)` and `H_l = -dS_l/dθ`. Its third axis stacks sine modes
`1:2M` followed by cosine modes `0:2M` (build it with `Nmax = 2M`).

The output entry is

    out[a,l,m,i,j] = sum(Q[key_mat[i,j]].space_func[a,n] *
                        T[l+s_i,m+s_j,ν(n)] for n in 1:M).

Here `a` is a radial sample, `l,m` are degrees, and components `i,j = 1,2,3`
are Y,X,Z. The offset `s_i` is `Lh` for component 1 (S) and zero for
components 2 and 3 (H); the same rule applies independently to `s_j`.
For the selected entry's angular boundary code, column `n` maps to:

| Code | Angular mode | Third-axis index ν(n) |
|------|--------------|-----------------------|
| `"00"` | `sin(2nθ)` | `2n` |
| `"01"` | `sin((2n-1)θ)` | `2n-1` |
| `"10"` | `cos((2n-1)θ)` | `2M+2n` |
| `"11"` | `cos((2n-2)θ)` | `2M+2n-1` |

Return a new tensor of shape `(N,Lh,Lh,3,3)`; the angular coefficient index
has been summed out. Inputs are not mutated. See §3.3 of
`docs/technical_notes/finite_rank_assembly.md` for a worked contraction.
"""
function contract_QT_blocks(Q::VectorFunc, T::Array{dtype,3})
    key_mat = ["A11" "A12" "A13";
               "A12" "A22" "A23";
               "A13" "A23" "A33"]
    N, M = size(Q[1].space_func)

    # The two stacked families each contain Lh degrees.
    L = size(T, 1)
    Lh = L ÷ 2
    @assert iseven(L) "L must be even"
    @assert size(T) == (L, L, M*4+1) "Size of T does not match" 

    result = zeros(dtype, N, Lh, Lh, 3, 3)
    for i in 1:3, j in 1:3
        # Select S for each Y component and H for each X or Z component.
        st_i = i==1 ? Lh : 0
        st_j = j==1 ? Lh : 0

        key = key_mat[i,j]
        mat = Q[key].space_func

        # Map stored column m to the packed trig index using the table above.
        bdry = Q[key].bdry[2]
        add_m = (bdry[1] == '1') * 2M - (bdry[2] == '1')
        for m in 1:M
            idm = 2m + add_m
            @views result[:, :, :, i, j] .+= reshape(mat[:, m], :, 1, 1) .* 
                reshape(T[st_i+1:st_i+Lh, st_j+1:st_j+Lh, idm], 1, Lh, Lh)
        end
    end
    return result
end

"""
Split a degree-indexed tensor into field-even and field-odd blocks.

Supply `T` of shape `(N,l,l,3,3)` with even maximum degree `l`. Axes 2 and 3
contain full degrees `1:l`, not packed matrix rows or radial-root indices.
Each degree axis is selected according to its component:
- Components 1 and 2 (Y,X): the field-even block takes degrees `2,4,6,…`;
  the field-odd block takes `1,3,5,…`.
- Component 3 (Z): the field-even block takes degrees `1,3,5,…`;
  the field-odd block takes `2,4,6,…`.

Return `(result_even, result_odd)`, each of shape `(N,l÷2,l÷2,3,3)`.
Selected degree d occupies compressed index `(d+1)÷2`; radial samples and
component axes are preserved. The input is not mutated. The dimensions and
evenness of `l` are caller preconditions, not checked here.
See §3.4 of `docs/technical_notes/finite_rank_assembly.md` for index examples.
"""
function split_tensor(T::Array{dtype, 5}, l::Int)
    N = size(T, 1)
    lh = l ÷ 2
    result_even = zeros(dtype, N, lh, lh, 3, 3)
    result_odd = zeros(dtype, N, lh, lh, 3, 3)
    for i in 1:3, j in 1:3
        st_i = i==3 ? 1 : 2
        st_j = j==3 ? 1 : 2
        result_even[:, :, :, i, j] = T[:, st_i:2:l, st_j:2:l, i, j]
        result_odd[:, :, :, i, j] = T[:, (3-st_i):2:l, (3-st_j):2:l, i, j]
    end
    return result_even, result_odd
end

"""
Compute the angular integrals of symmetric Q on the supplied radial grid.

Before splitting by field parity, each entry is

    T[a,l,m,i,j] = ∫₀^{π/2} Q_ij(r_pt[a],θ) Θ_l^i(θ) Θ_m^j(θ) sin(θ) dθ,

where `Θ_l^1 = S_l = P_l(cosθ)` and `Θ_l^2 = Θ_l^3 = H_l = -dS_l/dθ`.

The driver builds the basis lists and `tensor_int_basis`, samples Q's radial
coefficient functions with `interpolate(..., pt_flag=true)`, contracts them
with `contract_QT_blocks`, and splits the result with `split_tensor`.
The intermediate sample arrays retain angular coefficients in their columns.

# Arguments
- `Q::VectorFunc`: symmetric matrix function with the six keys required by
  `contract_QT_blocks` and matching coefficient-array sizes.
- `r_pt`: arguments in Q's stored radial coordinate. In Step 4A the caller
  supplies `(Pi/2)*r_ext`, where `r_ext` is the extended normalized-radius grid.
  This function does not apply that scaling or remove extended-grid samples.
- `Lmax`: nonnegative even maximum degree, used by `sph_basis_lists`.
- `Nmax`: angular cutoff for the integral table; must equal `2M`, where `M`
  is the number of stored angular coefficient columns in each Q entry.

Return `(result_even, result_odd)`, each of shape
`(length(r_pt),Lmax÷2,Lmax÷2,3,3)`. Axes are radial samples, two compressed
degree indices, and two Y/X/Z component indices. This is the field-parity
order expected by `integrate_r`. Radial integration and the `4πR³` volume
factor are applied later. See §3 of `docs/technical_notes/finite_rank_assembly.md`.
"""
function integrate_θ(Q::VectorFunc, r_pt::Vector{dtype}, Lmax::Int, Nmax::Int)
    trig_lists = trig_basis_lists(Nmax)
    sph_lists = sph_basis_lists(Lmax)
    T = tensor_int_basis(trig_lists, sph_lists)
    Q_part = interpolate(Q, r_pt, dtype[]; keys=["I","I"], pt_flag=true)
    contracted_T = contract_QT_blocks(Q_part, T)
    int_θ_lst = split_tensor(contracted_T, Lmax)
    return int_θ_lst
end

"""
Bound radial derivatives of an angular-integrated tensor, for orders `0:k`.

`X` has uniformly spaced samples along its first axis; all remaining axes
label independent entries. Each sample column is passed to `get_Wk∞_norm_lst`
with `paras = [["freq", M*Pi/2], nothing]` and spacings `[h,0]`.

Here `h` is the spacing in normalized radius r and `M` bounds Q's radial
wavenumbers in its stored coordinate `x_Q = (π/2)r`. The frequency in r is
therefore `M*π/2`. This M is independent of the angular coefficient count
used by `contract_QT_blocks`. There is no remaining continuous θ variable,
so the transverse correction is omitted.

Supply `k ≥ 0`, `h > 0`, and an already extended grid whose finite-difference
centers cover the target domain through order k. No padding is added here.
The estimator checks `size(X,1) ≥ 10*(k+1)` and a positive frequency-correction
denominator; its Bernstein domain conditions must also hold.

Return a new tensor of shape `(k+1,size(X)[2:end]...)`: entry `Y[j+1,…]`
bounds the j-th radial derivative of the corresponding entry of X.
The trailing indices are preserved and X is not mutated. See §4.5 of
`docs/technical_notes/finite_rank_assembly.md` for the coordinate scaling
and `docs/technical_notes/matrix_func.md` for the estimator's domain conditions.
"""
function int_θ_bound(X::AbstractArray{dtype}, h::dtype, k::Int, M::Int)
    paras = [["freq", dt(M)*Pi/dt(2)] , nothing]
    h = [h, zero(dtype)]
    N = size(X, 1)
    outsz = (k+1, tail(size(X))...)
    Y = Array{dtype}(undef, outsz)

    X2 = reshape(X, N, :)
    Y2 = reshape(Y, k+1, :)

    @threads for col in axes(X2, 2)
        @inbounds begin
            xcol = @view X2[:, col]
            Y2[:, col] .= get_Wk∞_norm_lst(xcol, h, k; paras=paras)
        end
    end

    return Y
end
