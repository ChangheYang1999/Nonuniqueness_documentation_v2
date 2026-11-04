

# Finite-rank assembly: constructing and verifying A and B

This note follows Step 4A in [`finite_rank_construction.ipynb`](../../notebooks/finite_rank_construction.ipynb), using [`My_Assembly.jl`](../../src/My_Assembly.jl) and [`Trig_Poly.jl`](../../src/Trig_Poly.jl). Read [the Bessel basis note](bessel_basis.md) for the individual basis functions. Here the question is how those functions become matrix entries with rigorous integral bounds.

For basis elements $\Phi_p$, we construct

$$
A_{pq}=\langle\Phi_p,Q\Phi_q\rangle_{L^2(B_R)},\qquad
B_{pq}=\langle\Phi_p,\Phi_q\rangle_E,
$$

where

$$
\langle u,v\rangle_E
=\int_{B_R}\bigl(\nabla u:\nabla v+\eta\,u\cdot v\bigr)\,dx.
$$

The notebook uses $R=32$ and $\eta=\eta_2=0.005$. Its finite-rank construction uses this ball; the underlying Navier–Stokes problem remains on the whole space. The argument named `β` in `get_lap_matrix` and `eval_h1_norm` is the constant $\eta$, not a coordinate.

The six steps below form the reading order:

1. [Fix the basis ordering and matrix indices.](#basis-ordering)
2. [Prepare radial samples and derivative bounds.](#assembly-inputs)
3. [Perform the angular integration for A.](#angular-integration)
4. [Assemble A by radial integration, with error bounds.](#radial-assembly)
5. [Construct the diagonal energy matrix B.](#energy-matrix)
6. [Complete the enclosures and verify the imported eigenpairs.](#assembly-verification)

Steps 4 and 5 share the radial inputs but are independent: B does not use Q or the angular tensors. The notebook evaluates B before A. This note explains A first so that its angular and radial calculations can be read together. Interval evaluation encloses arithmetic and function values; the separate quadrature allowances are needed to enclose integrals. Float64 runs give numerical approximations.

<a id="field-reconstruction"></a>

Reconstructing finite-rank fields from the imported coefficients, packing their derivative bounds with `lst_to_mat`, and forming the RHS belong to [Step 4B](finite_rank_rhs.md). That calculation uses the same basis ordering but does not read Step 4A's saved matrices.

<a id="basis-ordering"></a>
<a id="2-fix-the-meaning-of-one-matrix-row"></a>

## 1. Fix the basis ordering and matrix indices

### 1.1 One basis element may have two nonzero components

Write $S_l(\theta)=P_l(\cos\theta)$ and $H_l(\theta)=-S_l'(\theta)$, following [the scalar angular convention](bessel_basis.md#angular-rows). In physical spherical component order `(Y, X, Z)`, the two families are

$$
\Phi^P_{l,j}
=\bigl(R^Y_{l,j}S_l,\ R^X_{l,j}H_l,\ 0\bigr),\qquad
\Phi^Z_{l,j}
=\bigl(0,\ 0,\ R^Z_{l,j}H_l\bigr).
$$

The returned X radial channel is the negative of the physical poloidal X coefficient, so its product with $H_l=-S_l'$ has the physical sign. The toroidal whole-mode sign is part of the basis normalization. These fields contract directly with physical Q, without a $DQD$ conversion; see [the signed radial formulas](bessel_basis.md#62-evaluate-each-channel).

Here $l$ is the spherical degree and $j$ is a root index **within that degree and family**. A Y-root supplies one poloidal element with both Y and X channels; a Z-root supplies one toroidal element. Consequently, `Ur_Y_lst[2][:,2]` and `Ur_X_lst[2][:,2]` are the two radial factors of the same element $\Phi^P_{2,2}$. They must contribute to one matrix row, not two.

### 1.2 Split by field parity, then pack the rows

The helpers `_field_parity_poloidal` and `_field_parity_toroidal` choose the group:

| Group index `g` | Field parity | Poloidal degrees | Toroidal degrees |
|---|---|---|---|
| 1 | even | $2,4,6,\ldots$ | $1,3,5,\ldots$ |
| 2 | odd | $1,3,5,\ldots$ | $2,4,6,\ldots$ |

For example, degree 2 belongs to group 1 for a poloidal element but group 2 for a toroidal element. Field parity is therefore different from degree parity.

Within each group, pack **all poloidal elements first, then all toroidal elements**. Within a family, increase the degree, then the root index. `_count_packed_basis_sizes` counts these elements; `_allocate_gram_blocks` allocates a square value matrix and a matching error matrix for each group.

We use six degrees with unequal root counts in both families. Each field-parity group then contains three degree blocks per family, and some blocks contain several roots. These counts illustrate the layout; they are not the notebook's actual truncation:

```text
Degree l:          1  2  3  4  5  6
Poloidal roots:    2  3  1  2  2  1
Toroidal roots:    1  2  2  1  1  1
```

| Packed row or column | Field-even block | Field-odd block |
|---|---|---|
| 1 | $\Phi^P_{2,1}$ | $\Phi^P_{1,1}$ |
| 2 | $\Phi^P_{2,2}$ | $\Phi^P_{1,2}$ |
| 3 | $\Phi^P_{2,3}$ | $\Phi^P_{3,1}$ |
| 4 | $\Phi^P_{4,1}$ | $\Phi^P_{5,1}$ |
| 5 | $\Phi^P_{4,2}$ | $\Phi^P_{5,2}$ |
| 6 | $\Phi^P_{6,1}$ | $\Phi^Z_{2,1}$ |
| 7 | $\Phi^Z_{1,1}$ | $\Phi^Z_{2,2}$ |
| 8 | $\Phi^Z_{3,1}$ | $\Phi^Z_{4,1}$ |
| 9 | $\Phi^Z_{3,2}$ | $\Phi^Z_{6,1}$ |
| 10 | $\Phi^Z_{5,1}$ | — |

Thus `poloidal_counts = [6,5]`, `toroidal_counts = [4,4]`, and the even and odd matrices have sizes `10 × 10` and `9 × 9`. In particular, `result_lst[1][6,9]` pairs $\Phi^P_{6,1}$ with $\Phi^Z_{3,2}$, whereas `result_lst[2][6,9]` pairs $\Phi^Z_{2,1}$ with $\Phi^Z_{6,1}$.

The even poloidal degree blocks occupy rows `1:3`, `4:5`, and `6:6`; the toroidal degree blocks occupy `7:7`, `8:9`, and `10:10`. The starting row of a block is determined by the total number of earlier roots in that group, not by the degree itself.

### 1.3 Distinguish a packed matrix row from an angular index

All roots of degree $l$ share the same angular factor. After the parity split, its angular index is

$$
h(l)=\left\lfloor\frac{l+1}{2}\right\rfloor.
$$

| Element in the even block | Degree $l$ | Root $j$ | Packed matrix row | Compressed angular index $h(l)$ |
|---|---|---|---|---|
| $\Phi^P_{2,1}$ | 2 | 1 | 1 | 1 |
| $\Phi^P_{2,2}$ | 2 | 2 | 2 | 1 |
| $\Phi^P_{2,3}$ | 2 | 3 | 3 | 1 |
| $\Phi^P_{4,1}$ | 4 | 1 | 4 | 2 |
| $\Phi^P_{4,2}$ | 4 | 2 | 5 | 2 |
| $\Phi^P_{6,1}$ | 6 | 1 | 6 | 3 |
| $\Phi^Z_{1,1}$ | 1 | 1 | 7 | 1 |
| $\Phi^Z_{3,1}$ | 3 | 1 | 8 | 2 |
| $\Phi^Z_{3,2}$ | 3 | 2 | 9 | 2 |
| $\Phi^Z_{5,1}$ | 5 | 1 | 10 | 3 |

For example, rows 1–3 use angular row 1 but three different poloidal radial columns; rows 8–9 use angular row 2 but two different toroidal radial columns. Also, compressed angular index 1 means degree 2 for the poloidal channels of an even field and degree 1 for its toroidal channel. The group **and component** determine its meaning.

<a id="assembly-inputs"></a>
<a id="6-radial-derivative-bounds-ur_bound-and-the-bound-tables"></a>

## 2. Prepare radial samples and derivative bounds

### 2.1 Match the radial coordinates and sample indices

Let $\rho$ denote physical radius. Basis evaluation and radial quadrature use $r=\rho/R$ on $[0,1]$. The stored trigonometric representation of Q uses the argument

$$
x_Q=\frac{\pi}{2}r.
$$

This argument is named `r_Q` in the notebook. For example, $r=1/2$ represents physical radius 16 and Q argument $\pi/4$ when $R=32$.

With `N0 = 18000` subintervals and derivative order `k = 8`, the notebook prepares

```julia
r_ext = dt.(-k:N0+k) ./ dt(N0)
r = r_ext[k+1:end-k]
r_Q = r_ext .* (Pi/dt(2))
```

The angular tensors use all `18017` extended samples, because radial finite differences need points beyond each endpoint. The Bessel factors use the `18001` samples in `r`.

| Physical-grid sample index `t` | Value `r[t]` | Corresponding extended-grid index `t+8` |
|---|---|---|
| 1 | 0 | 9 |
| 2 | $1/18000$ | 10 |
| 18001 | 1 | 18009 |

Thus `theta_block_even[9:end-8,...]` aligns exactly with `r` and the radial basis samples. In an expression using the full tensor, sample 10 means $r=1/18000$; in a cropped array it is sample 2.

The quadrature rule assumes a uniform grid on $[0,1]$, with the number of subintervals divisible by 6. It infers the spacing from the sample count. The derivative-based refinement and quadrature themselves are explained in [the matrix-functions note](matrix_func.md).

### 2.2 Read each per-degree input table

Let $n_r=N_0+1$, and let $n_l^P,n_l^Z$ be the root counts at degree $l$.

| Input | Shape at degree $l$ | Axes |
|---|---|---|
| `Ur_Y_lst[l]`, `Ur_X_lst[l]` | $(n_r,n_l^P)$ | radial sample, poloidal root |
| `Ur_Z_lst[l]` | $(n_r,n_l^Z)$ | radial sample, toroidal root |
| `Y_bound_lst[l]`, `X_bound_lst[l]` | $(9,n_l^P)$ | derivative order $0{:}8$, poloidal root |
| `Z_bound_lst[l]` | $(9,n_l^Z)$ | derivative order $0{:}8$, toroidal root |

For our root-count example, `Ur_X_lst[2]` has shape `18001 × 3`. The entry `Ur_X_lst[2][2,2]` is the X radial factor of $\Phi^P_{2,2}$ at $r=1/18000$. In contrast, `X_bound_lst[2][4,2]` bounds its **third radial derivative** uniformly over the required domain. Derivative order $m$ occupies row `m+1`; sample indices have no role in this bound table.

The corresponding spectral parameter is `Y_paras[2][2][1]`: degree 2, second poloidal root, then the parameter slot containing $\alpha$. The meaning of the other parameter slots and the construction of `eval_Ur` and `Ur_bound_lst` are given in [the Bessel note](bessel_basis.md#radial-bound-tables).

These raw radial bound tables feed both `integrate_r` and `get_lap_matrix`. They are not passed through `lst_to_mat` for matrix assembly; that packing belongs to [bounds for reconstructed fields](finite_rank_rhs.md#packed-field-bounds).

<a id="angular-integration"></a>

## 3. Perform the angular integration for A

### 3.1 Remove the angular variable before radial quadrature

For components $a,b\in\{1,2,3\}$, define

$$
\mathcal T^{ab}_{ll'}(r)
=\int_0^{\pi/2}\Theta_l^a(\theta)\,
Q_{ab}(r,\theta)\,\Theta_{l'}^b(\theta)\sin\theta\,d\theta,
$$

where $\Theta_l^1=S_l$ and $\Theta_l^2=\Theta_l^3=H_l$. Here Q is written as a function of normalized radius; its stored representation receives $x_Q=(\pi/2)r$.

The root indices do not occur in this tensor. Precomputing it avoids repeating the same angular integral for every root pair. Q need not separate into one radial factor and one angular factor: its **finite angular expansion** is what makes this calculation possible.

The driver `integrate_θ` first builds an integral table independent of Q's coefficients, inserts those coefficients, then splits the result by field parity.

### 3.2 Build the angular integral table

`TrigPoly` stores cosine and sine coefficients separately: `c[n+1]` multiplies $\cos(n\theta)$, while `s[n]` multiplies $\sin(n\theta)$. For example,

$$
S_2=\tfrac14+\tfrac34\cos(2\theta),\qquad
H_2=\tfrac32\sin(2\theta).
$$

Thus `P_list[2]` has cosine coefficients `[1/4, 0, 3/4]`, and `dP_list[2]` has sine coefficients `[0, 3/2]`. The list index 2 is the degree; coefficient position 3 in `c` is frequency 2. The name `dP_list` denotes the scalar H family, with the sign specified in §1.1.

`trig_basis_lists(Nmax)` supplies `cos_list[n+1]` and `sin_list[n]`; for example, `cos_list[1]` is the constant 1 and `sin_list[2]` is $\sin(2\theta)$. `sph_basis_lists(L)` supplies the S and H lists for degrees `1:L`. The underlying trigonometric and spherical basis conversion is explained in [the Fourier note](fourier_basis.md).

`trig_mul` uses product-to-sum identities, and `trig_int_0_pi2` integrates each resulting mode analytically. `tensor_int_basis` combines them into

$$
\mathcal I_{uv\nu}
=\int_0^{\pi/2} f_u(\theta)f_v(\theta)h_\nu(\theta)\sin\theta\,d\theta.
$$

The first two axes each stack `[H₁,…,H_L,S₁,…,S_L]`. The third stacks sine modes `1:Nmax`, then cosine modes `0:Nmax`. The table has shape `(2L, 2L, 2Nmax+1)` and already includes the spherical weight $\sin\theta$.

**Index example.** Take `L = 6` and `Nmax = 4`. The first two axes are `[H₁,H₂,H₃,H₄,H₅,H₆,S₁,S₂,S₃,S₄,S₅,S₆]`, and the third is

```text
Index ν:  1     2     3     4     5     6     7     8     9
Mode:     sinθ  sin2θ sin3θ sin4θ 1     cosθ  cos2θ cos3θ cos4θ
```

The table has shape `12 × 12 × 9`. Its entry `I[12,3,3]` is the integral of $S_6H_3\sin(3\theta)\sin\theta$. Its indices specify two angular factors and one Q mode; none counts roots or radial samples.

There is no angular quadrature remainder: a finite trigonometric expansion is integrated algebraically. Interval arithmetic still encloses coefficients and rounding.

### 3.3 Insert Q's angular coefficients

At a fixed extended radial sample $r_t$, `contract_QT_blocks` computes the angular integral

$$
\texttt{contracted}[t,l,l',a,b]
=\int_0^{\pi/2}\Theta_l^a(\theta)\,
Q_{ab}(r_t,\theta)\,\Theta_{l'}^b(\theta)\sin\theta\,d\theta.
$$

Here $l,l'$ are the left and right spherical degrees, and $a,b$ are the left and right component indices: 1 = Y, 2 = X, 3 = Z. Thus $\Theta_l^1=S_l$, while $\Theta_l^2=\Theta_l^3=H_l$. As in §3.1, $Q_{ab}(r_t,\theta)$ denotes Q after inserting its stored radial argument $(\pi/2)r_t$. The final $\sin\theta$ is the spherical integration weight.

The calculation uses linearity: expand Q in its angular modes, multiply each coefficient by the corresponding precomputed angular integral, and sum over those modes.

**The coefficient array: radial sample and stored angular column.**

The radial-only interpolation inside `integrate_θ` leaves Q expanded in $\theta$. Let $C_{ab}(t,m)$ denote `Q_part[key].space_func[t,m]` for the key corresponding to $(a,b)$. If each entry has M stored angular columns, then

$$
Q_{ab}(r_t,\theta)=\sum_{m=1}^{M}C_{ab}(t,m)\,g_m^{ab}(\theta).
$$

The component pair $(a,b)$ selects `key` and its angular basis $g_m^{ab}$:

```text
        b=1   b=2   b=3
a=1     A11   A12   A13
a=2     A12   A22   A23
a=3     A13   A23   A33
```

For instance, `(a,b)=(1,3)` selects `A13`; `(3,1)` selects the same stored entry because Q is symmetric. `A13` names an entry of Q, not a row or column of the assembled matrix A.

Each coefficient array has shape `(n_r+16, M)`. Its row `t` selects the extended radial point, and its column `m` selects a stored angular mode. There is no angular-sample index here, despite the name `space_func`. With `N0=18000` and eight padding points, $r_t=(t-9)/18000$: row 9 is $r=0$, and row 10 is $r=1/18000$. The same `t` is used in the coefficient array and the output tensor.

**The mode index: a stored column is not necessarily a frequency or a table index.**

The boundary code determines the function represented by column `m`. The notebook builds I with `Nmax = 2M`, so its third axis has length `4M+1`: sine frequencies `1:2M`, followed by cosine frequencies `0:2M`. The index `ν` selects a position in that combined list.

| Angular boundary code | Stored column `m` represents | Index `ν` in I | Example with `M=2`: columns 1 and 2 |
|---|---|---|---|
| `"00"` | $\sin(2m\theta)$ | `2m` | indices 2, 4: $\sin(2\theta),\sin(4\theta)$ |
| `"01"` | $\sin((2m-1)\theta)$ | `2m-1` | indices 1, 3: $\sin\theta,\sin(3\theta)$ |
| `"10"` | $\cos((2m-1)\theta)$ | `2M+2m` | indices 6, 8: $\cos\theta,\cos(3\theta)$ |
| `"11"` | $\cos((2m-2)\theta)$ | `2M+2m-1` | indices 5, 7: $1,\cos(2\theta)$ |

For example, column `m=2` with code `"11"` and `M=2` represents frequency 2, but uses table position `ν=7`. In the code, `add_m` is the offset in `idm = 2m + add_m`, and `idm` is this `ν`. For `"01"`, `add_m=-1`; for `"11"` with `M=2`, it is 3.

**The integral table: two angular-factor indices and one mode index.**

The first two axes of I each list `[H₁,…,H_L,S₁,…,S_L]`. To select the angular factor of component $a$ and degree $l$, define

$$
u_a(l)=\begin{cases}L+l,&a=1\quad\text{(Y uses }S_l\text{)},\\l,&a=2\text{ or }3\quad\text{(X and Z use }H_l\text{)}.\end{cases}
$$

Here the symbol $u_a(l)$ names a position on the first or second axis; it is distinct from the mode index $\nu$. With $\nu_{ab}(m)$ given by the boundary-code table, the contraction is

$$
\texttt{contracted}[t,l,l',a,b]
=\sum_{m=1}^{M}C_{ab}(t,m)\,
\mathcal I_{u_a(l),\,u_b(l'),\,\nu_{ab}(m)}.
$$

For a fixed output entry, only `m` is summed. The indices `t,l,l′,a,b` stay fixed. I has no radial index because its entries are integrals of angular basis functions only; all radial dependence in this calculation is carried by C.

**Worked example: degree-6 Y coupled to degree-3 Z.**

Take `L=6`, `M=2`, so I has shape `12 × 12 × 9`. The loaded `A13` entry uses angular code `"01"`. At radial sample `t`, define

```text
q1 = Q_part["A13"].space_func[t,1]
q2 = Q_part["A13"].space_func[t,2].
```

Its angular expansion is therefore

$$
Q_{13}(r_t,\theta)=q_1\sin\theta+q_2\sin(3\theta).
$$

The desired integral is

$$
\int_0^{\pi/2}S_6(\theta)Q_{13}(r_t,\theta)H_3(\theta)\sin\theta\,d\theta.
$$

Substituting the expansion gives

$$
q_1\underbrace{\int_0^{\pi/2}S_6H_3\sin\theta\,\sin\theta\,d\theta}_{\mathcal I_{12,3,1}}
+q_2\underbrace{\int_0^{\pi/2}S_6H_3\sin(3\theta)\,\sin\theta\,d\theta}_{\mathcal I_{12,3,3}}.
$$

In each integral, the first sine factor comes from Q's angular mode; the last $\sin\theta$ is the integration weight already included in I. In particular, the two sine factors in the first term have different roles.

Every index in the resulting array expression can now be read separately:

| Array expression | Meaning of its indices |
|---|---|
| `Q_part["A13"].space_func[t,1]` | Q entry `(1,3)`; extended radial sample `t`; stored angular column 1, representing $\sin\theta$. |
| `Q_part["A13"].space_func[t,2]` | The same entry and radial sample; stored angular column 2, representing $\sin(3\theta)$. |
| `I[12,3,1]` | First axis: `12=6+6`, selecting $S_6$. Second: 3, selecting $H_3$. Third: 1, selecting $\sin\theta$. |
| `I[12,3,3]` | The same $S_6$ and $H_3$; third-axis index 3 selects $\sin(3\theta)$. The two occurrences of 3 mean degree and mode-table position, respectively. |

Thus

```text
contracted[t,6,3,1,3]
    = Q_part["A13"].space_func[t,1] * I[12,3,1]
    + Q_part["A13"].space_func[t,2] * I[12,3,3].
```

The five output axes have the following meanings:

| Axis | Index in this example | Meaning | Range |
|---|---|---|---|
| 1 | `t` | extended radial sample | `1:(n_r+16)` |
| 2 | `6` | degree of the left angular factor | `1:L` |
| 3 | `3` | degree of the right angular factor | `1:L` |
| 4 | `1` | left component, Y | `1:3` |
| 5 | `3` | right component, Z | `1:3` |

The full output has shape `(n_r+16, L, L, 3, 3)`, or `(18017,6,6,3,3)` in this example. The angular-mode column has been summed out. The degree indices are still the original degrees; §3.4 compresses them by field parity.

**Match these indices to the implementation.** In `contract_QT_blocks`, this note's I is the argument `T`, and `contracted` is the returned array `result`. The local variable `L = size(T,1)` counts both angular families: it equals 12 here. The local `Lh = L÷2` equals 6, the maximum degree called L in this note. Component indices `a,b` are named `i,j` in the loop. For `(i,j)=(1,3)`, `st_i=6` and `st_j=0`; degree pair `(6,3)` therefore selects table indices `(st_i+6,st_j+3)=(12,3)`.

The implementation handles all radial samples and degree pairs together. `mat[:,m]` holds that mode's coefficient at every radial point, while `T[st_i+1:st_i+Lh,st_j+1:st_j+Lh,idm]` holds the matching degree-pair integrals. The two `reshape` calls give shapes `(N,1,1)` and `(1,Lh,Lh)`, with `N=n_r+16`. Their elementwise product supplies the contribution to every `(t,l,l′)` entry for the fixed components `(i,j)`; `.+=` accumulates it over `m`.

Root indices do not appear in this computation: different roots of the same degree share the angular factors. Later, radial assembly chooses a root on each side and multiplies this angular integral by the corresponding radial factors and $r^2$. For example, $\Phi^P_{6,1}$ and $\Phi^Z_{3,2}$ reuse the angular entry above in the Y–Z term of `A_even[6,9]`; the X–Z term uses the component pair `(2,3)`.

### 3.4 Compress the degree axes by field parity

The tensor from §3.3 is indexed by the original degrees:

```text
contracted[t,l,l′,a,b].
```

Here `t` is the extended radial-sample index, `l,l′` are the left and right degrees, and `a,b` are the left and right components. `split_tensor` extracts the degree pairs needed for each field-parity group and numbers the selected degrees consecutively. Each entry retains the angular integral already computed in §3.3; this step selects and copies entries, with no new integration or summation.

**Choose degrees using the field parity and the component.**

The two groups use the basis ordering from §1: an even field contains even-degree poloidal elements and odd-degree toroidal elements; an odd field uses the complementary degrees. Components Y and X belong to the same poloidal element, so both select the poloidal degree list. Component Z selects the toroidal degree list. Thus “even block” refers to the whole field, not to every degree in its components being even.

For `L=6`, each component has three allowed degrees in each group:

| Field group | Component `a` or `b` | Family | Original degrees in selected order | New angular indices |
|---|---|---|---|---|
| even | 1 = Y or 2 = X | poloidal | `2,4,6` | `1,2,3` |
| even | 3 = Z | toroidal | `1,3,5` | `1,2,3` |
| odd | 1 = Y or 2 = X | poloidal | `1,3,5` | `1,2,3` |
| odd | 3 = Z | toroidal | `2,4,6` | `1,2,3` |

Apply this rule **independently to both degree axes**. Component `a` selects the list for the left degree `l`; component `b` selects the list for the right degree `l′`. For example, in the even Y–Z block, the left degrees are `2,4,6` and the right degrees are `1,3,5`.

**Translate between original degrees and compressed indices.**

Write `h,h′` for positions in those two selected lists. The inverse mapping is:

| Group and component | Original degree at compressed index `h` |
|---|---|
| even, Y or X | $2h$ |
| even, Z | $2h-1$ |
| odd, Y or X | $2h-1$ |
| odd, Z | $2h$ |

For an allowed degree $l$, the forward mapping in every case is $h=\lfloor(l+1)/2\rfloor$. Both degrees 5 and 6 map to compressed index 3, but the group and component determine which degree that index represents. The right axis follows the same rule with `h′` and `l′`.

For the even Y–Z component pair, the full degree-pair lookup is:

| Left index `h` | Right index `h′=1` (Z degree 1) | Right index `h′=2` (Z degree 3) | Right index `h′=3` (Z degree 5) |
|---|---|---|---|
| 1 (Y degree 2) | `(l,l′)=(2,1)` | `(2,3)` | `(2,5)` |
| 2 (Y degree 4) | `(4,1)` | `(4,3)` | `(4,5)` |
| 3 (Y degree 6) | `(6,1)` | `(6,3)` | `(6,5)` |

Every cell is a pair of original degrees, not a pair of matrix rows or root indices. In particular, compressed position `(3,2)` selects original degrees `(6,3)`.

**Continue the contraction example from §3.3.**

The entry there has left degree 6 and component 1 (Y), and right degree 3 and component 3 (Z). These both belong to the even field group. Degree 6 is third in `[2,4,6]`; degree 3 is second in `[1,3,5]`. Therefore

```text
theta_block_even[t,3,2,1,3] = contracted[t,6,3,1,3].
```

The table below accounts for all five indices:

| Axis | Input index | Output index | Meaning |
|---|---|---|---|
| 1 | `t` | `t` | Same extended radial sample. For example, `t=10` still means $r=1/18000$. |
| 2 | `6` | `3` | Left degree 6 becomes position 3 in the even poloidal degree list. |
| 3 | `3` | `2` | Right degree 3 becomes position 2 in the even toroidal degree list. |
| 4 | `1` | `1` | Left component remains Y. |
| 5 | `3` | `3` | Right component remains Z. |

Thus `theta_block_even[t,3,2,1,3]` still represents

$$
\int_0^{\pi/2}S_6(\theta)Q_{13}(r_t,\theta)H_3(\theta)\sin\theta\,d\theta.
$$

Only the two degree labels have been replaced by their positions in the selected lists. The field group is carried by the array name, rather than by another tensor axis.

**Examples with other components and with the odd group.**

| Output entry | Corresponding input entry | Angular factors and Q entry |
|---|---|---|
| `theta_block_even[t,2,3,1,2]` | `contracted[t,4,6,1,2]` | $S_4\,Q_{12}\,H_6$ |
| `theta_block_even[t,2,3,3,3]` | `contracted[t,3,5,3,3]` | $H_3\,Q_{33}\,H_5$ |
| `theta_block_even[t,2,3,3,2]` | `contracted[t,3,6,3,2]` | $H_3\,Q_{32}\,H_6$ |
| `theta_block_odd[t,3,3,1,3]` | `contracted[t,5,6,1,3]` | $S_5\,Q_{13}\,H_6$ |

Each row denotes the angular integral of the displayed product with weight $\sin\theta$. The last row shows why equal compressed indices need not mean equal degrees: index 3 selects degree 5 for the odd Y component and degree 6 for the odd Z component.

**Match the selection to the code.**

Inside `split_tensor`, the argument `T` is the five-dimensional `contracted` array, not the three-dimensional angular integral table I used in §3.3. The scalar argument `l` is the maximum degree L (6 here), `lh=l÷2` is the number of selected degrees (3), and `N=size(T,1)` counts the extended radial samples (18017). Loop variables `i,j` denote component indices `a,b`, each ranging over `1:3`.

The code chooses the first allowed degree for the even group with

```julia
st_i = i == 3 ? 1 : 2
st_j = j == 3 ? 1 : 2
```

The range `st_i:2:l` means start at `st_i` and advance by two without exceeding `l`; for example, `1:2:6` selects `[1,3,5]`. It selects one degree parity. The odd group replaces `st_i` by `3-st_i`, exchanging starts 1 and 2. The same operation is applied independently on the right using `st_j`.

For `(i,j)=(1,3)`, `st_i=2` and `st_j=1`. With `l=6`, the two assignments become

```julia
result_even[:, :, :, 1, 3] = T[:, 2:2:6, 1:2:6, 1, 3]
result_odd[:, :, :, 1, 3]  = T[:, 1:2:6, 2:2:6, 1, 3]
```

On the right, the first `:` keeps every radial sample, including padding. The two degree ranges select three degrees each. On the left, the next two `:` positions receive these degrees as indices `1:3`. Components 1 and 3 remain fixed on both sides. For each component pair, this copies a block of shape `(18017,3,3)`.

The full input shape `(18017,6,6,3,3)` becomes two arrays, each of shape `(18017,3,3,3,3)`. Their axes are now `(radial sample, left compressed degree, right compressed degree, left component, right component)`. Both degree axes have length `L/2`; the radial and component axes retain their sizes. The routine extracts the entries within each field-parity group; it does not return the entries coupling different field groups.

**Hand the tensors to radial assembly.**

`integrate_θ` returns `(theta_block_even, theta_block_odd)`. The notebook passes `[theta_block_even, theta_block_odd]` to `integrate_r`, so `theta_blocks[1]` selects the entire even tensor and `theta_blocks[2]` selects the odd tensor. This outer group index is separate from the five indices inside either tensor.

The compressed degree indices still do not count roots. In our basis example, $\Phi^P_{6,1}$ is matrix row 6, while $\Phi^Z_{3,1}$ and $\Phi^Z_{3,2}$ are rows 8 and 9. Both `A_even[6,8]` and `A_even[6,9]` use `theta_block_even[:,3,2,1,3]` for their Y–Z contribution, but multiply it by different radial Z columns. Section 4 explains this final conversion from angular degree indices to matrix indices.

<a id="radial-assembly"></a>

## 4. Assemble A by radial integration, with error bounds

### 4.1 From an angular tensor to a matrix entry

**The matrix indices p and q label complete basis elements.**

First fix a field-parity group `g`: 1 for even, 2 for odd. Within that group, $p$ is the row index of the left basis element $\Phi_p$, and $q$ is the column index of the right basis element $\Phi_q$. They use the ordering in §1.2: poloidal elements first, then toroidal elements; within a family, degree first, then root. In our example, $p,q$ range over `1:10` for the even group and `1:9` for the odd group. The numbering starts again in each group.

Thus, after fixing `g`, each matrix index identifies three pieces of information:

```text
p → (family of Φ_p, degree l(p), root index j_p)
q → (family of Φ_q, degree l(q), root index j_q).
```

For example, even-group index $p=5$ identifies $\Phi^P_{4,2}$: poloidal family, degree 4, second root. The degree $l(p)=4$ and root index $j_p=2$ are different from the matrix index 5. A poloidal element's Y and X components share that single matrix index.

The matrix position is obtained by counting the roots at all earlier degrees of the same family and field group, then adding the root index within the current degree. For a toroidal element, also add the total number of poloidal elements in that group. Section 3.4's compressed angular index counts only the selected degrees; it does not include these root counts or the poloidal offset.

**Recover the angular and radial factors from p and q.**

The degrees determine the two compressed angular indices

$$
h_p=\left\lfloor\frac{l(p)+1}{2}\right\rfloor,\qquad
h_q=\left\lfloor\frac{l(q)+1}{2}\right\rfloor.
$$

For a component pair $(a,b)$, the angular integral is read from

```text
theta_blocks[g][t,h_p,h_q,a,b].
```

These are exactly the indices from §3.4: `g` chooses the field group; `t` chooses the extended radial sample; `h_p,h_q` choose the compressed degrees; `a,b` choose the left and right components. The root indices $j_p,j_q$ select columns of the radial sample matrices instead of axes of the angular tensor.

Write $R_p^a(r)$ for component $a$'s radial factor in $\Phi_p$. For a poloidal element it is the Y factor when $a=1$, the X factor when $a=2$, and zero when $a=3$. For a toroidal element only $a=3$ is nonzero, giving its Z factor. The family, degree, and root identified by p specify which radial function this is; the same rule defines $R_q^b$ on the right.

The reduced matrix entry is therefore

$$
\widehat A_{pq}
=\sum_{a=1}^{3}\sum_{b=1}^{3}\int_0^1 r^2 R_p^a(r)
\mathcal T^{ab}_{l(p),l(q)}(r)R_q^b(r)\,dr.
$$

The group is fixed and omitted from this formula. The tensor $\mathcal T$ is written with its **original degree labels**; its sampled values are stored using compressed indices `h_p,h_q` as above. The component indices $a,b$ are summed, while p and q remain fixed. Radial quadrature uses the samples indexed by t, after removing the padding to match the radial basis grid. The hat indicates that the physical factor $4\pi R^3$ has not yet been applied.

**Continue the example from §3.4: `A_even[6,9]`.**

Here `g=1`, $p=6$, and $q=9$. The two matrix indices decode as follows:

| Information | Left element, p = 6 | Right element, q = 9 |
|---|---|---|
| Complete basis element | $\Phi^P_{6,1}$ | $\Phi^Z_{3,2}$ |
| Family | poloidal | toroidal |
| Original degree | $l(p)=6$ | $l(q)=3$ |
| Root within that degree | $j_p=1$ | $j_q=2$ |
| Compressed angular index | $h_p=3$ | $h_q=2$ |
| Nonzero components | Y and X (`a=1,2`) | Z (`b=3`) |
| Radial sample columns | `Ur_Y_lst[6][:,1]`, `Ur_X_lst[6][:,1]` | `Ur_Z_lst[3][:,2]` |

Only two component pairs contribute. The Y–Z term uses `theta_block_even[t,3,2,1,3]`, the entry traced through §§3.3–3.4. The X–Z term uses `theta_block_even[t,3,2,2,3]`. Both contributions belong to the same matrix entry `(p,q)=(6,9)` and are added after radial integration.

Changing q from 9 to 8 selects $\Phi^Z_{3,1}$ instead of $\Phi^Z_{3,2}$. Its degree and compressed angular index remain 3 and 2, so the angular entries stay the same; the right radial column changes from 2 to 1. This is why matrix indices cannot be used directly as angular-tensor indices.

**Match p and q to the assembly loops.**

`integrate_r` allocates the two parity matrices, prepares derivative bounds for the angular tensors, and calls three helpers for the three family pairings. In their loops, the code uses these names. Here `i,j` are degree indices; in `split_tensor` they named component indices instead:

| Code index or count | Meaning | Example: $\Phi^P_{4,2}$ paired with $\Phi^P_{6,1}$ |
|---|---|---|
| `i`, `j` | left and right degrees | 4, 6 |
| `ii`, `jj` | root columns within those degrees | 2, 1 |
| `Li`, `Lj` | root counts at those degrees | 2, 1 |
| `ih`, `jh` | compressed angular indices | 2, 3 |
| `cur_i`, `cur_j` | counts of earlier roots in the respective family and parity | 3, 5 |
| `cur_i+ii`, `cur_j+jj` | packed poloidal row and column | 5, 6 |

For toroidal rows or columns, the helper additionally adds `NY = poloidal_counts[g]`. These offsets count basis elements; they do not count degrees.

### 4.2 Poloidal–poloidal entries combine four terms

`_assemble_poloidal_poloidal_blocks!` computes

$$
\widehat A^{PP}_{pq}
=\int_0^1 r^2\bigl(
Y_p\mathcal T^{11}Y_q
+X_p\mathcal T^{21}Y_q
+Y_p\mathcal T^{12}X_q
+X_p\mathcal T^{22}X_q\bigr)\,dr.
$$

Here $Y_p=R_p^Y$, $X_p=R_p^X$, and the tensor degree indices are implicit. The four terms are `pt1`–`pt4`; each gets its own quadrature allowance.

For `A_even[5,6]`, use degree/root pairs `(4,2)` and `(6,1)`. The YY term takes `Ur_Y_lst[4][:,2]`, `Ur_Y_lst[6][:,1]`, and `theta_block_even[9:end-8,2,3,1,1]`. The other terms keep angular indices `(2,3)` and replace the component pair by `(2,1)`, `(1,2)`, or `(2,2)`, together with the matching radial channels.

The helper copies this degree block's transpose, including its errors, to the reversed block. Within a same-degree block it evaluates all root pairs: for example, `A_even[1,2]` uses angular indices `(1,1)` but radial columns 1 and 2 of degree 2.

### 4.3 Toroidal–toroidal entries shift both matrix indices

`_assemble_toroidal_toroidal_blocks!` computes

$$
\widehat A^{ZZ}_{pq}=\int_0^1 r^2Z_p\mathcal T^{33}Z_q\,dr.
$$

For $\Phi^Z_{3,2}$ paired with $\Phi^Z_{5,1}$ in the even block, `NY=6`, `cur_i=1`, `cur_j=3`, `ii=2`, and `jj=1`. The toroidal offsets count the one degree-1 root before degree 3, and the three roots at degrees 1 and 3 before degree 5. The matrix indices are therefore

```text
row = NY + cur_i + ii = 6 + 1 + 2 = 9
col = NY + cur_j + jj = 6 + 3 + 1 = 10.
```

This entry uses `Ur_Z_lst[3][:,2]`, `Ur_Z_lst[5][:,1]`, and angular slice `theta_block_even[9:end-8,2,3,3,3]`. The reverse entry is filled by transposition.

### 4.4 Mixed entries shift only the toroidal index

`_assemble_poloidal_toroidal_blocks!` computes

$$
\widehat A^{PZ}_{pq}
=\int_0^1 r^2\bigl(Y_p\mathcal T^{13}+X_p\mathcal T^{23}\bigr)Z_q\,dr.
$$

The two degrees have opposite parity within the same field-parity group. For the even example $\Phi^P_{6,1}$ and $\Phi^Z_{3,2}$, `cur_i=5`, `cur_j=1`, `NY=6`, `ii=1`, and `jj=2`. The left side has five earlier poloidal roots, while the right side has one earlier toroidal root. Thus

```text
row = cur_i + ii      = 5 + 1     = 6
col = NY + cur_j + jj = 6 + 1 + 2 = 9.
```

The entry `A_even[6,9]` uses compressed angular indices `(3,2)`. Its combined integrand is

```julia
r.^2 .* (
    Ur_Y_lst[6][:,1] .* theta_block_even[9:end-8,3,2,1,3] .+
    Ur_X_lst[6][:,1] .* theta_block_even[9:end-8,3,2,2,3]
) .* Ur_Z_lst[3][:,2]
```

The helper integrates the two terms separately, so each has its own derivative bound. Symmetry of Q supplies `A_even[9,6]` by transposition. The same symmetry justifies the copies in the other helpers.

For an odd example, $\Phi^P_{5,2}$ and $\Phi^Z_{6,1}$ occupy `A_odd[5,9]`. Here `NY=5`, `cur_i=cur_j=3`, `ii=2`, and `jj=1`: the row is `3+2=5` and the column is `5+3+1=9`. Both compressed angular indices are 3. The Y–Z term uses `theta_block_odd[9:end-8,3,3,1,3]`, matching the last example in §3.4.

### 4.5 Match derivative-bound indices to each integrand

After angular integration, $\mathcal T(r)$ still has a finite radial trigonometric expansion. `int_θ_bound` estimates its radial derivatives by flattening all trailing axes, applying `get_Wk∞_norm_lst` to each column, then restoring the axes. It does not differentiate in $\theta$.

For `L=6`, the shape changes from `(18017,3,3,3,3)` to `(9,3,3,3,3)`. Let `Q_bound_lst[1]` denote this even-group result, as in `integrate_r`. Then

```text
Q_bound_lst[1][4,3,2,1,3]
```

bounds the **third radial derivative** of the same angular entry whose samples are `theta_block_even[:,3,2,1,3]`. The first axis has changed from sample index to derivative-order index; the remaining four retain their meaning.

In the internal column-major reshape, trailing indices `(3,2,1,3)` become column

```text
3 + 3*(2-1) + 9*(1-1) + 27*(3-1) = 60.
```

The routine sends sample column 60 to the bound estimator and places the returned orders `0:8` in bound column 60. Reshaping back therefore preserves the correspondence.

The frequency parameter `M0` comes from Q's **radial** coefficient count. Since Q is evaluated at $(\pi/2)r$, the normalized-radius frequency bound is `M0 * Pi / 2`. `M1`, used in `integrate_θ`, instead comes from the angular coefficient count. For an illustrative `10 × 2` Q coefficient array, the notebook obtains `M0=20`, `M1=4`: the refinement frequency is $10\pi$, while the angular integral table uses `Nmax=4`.

For the Y–Z term of `A_even[6,9]`, the four derivative-bound vectors are

```text
_r_sq_bound_table(8)
Y_bound_lst[6][:,1]
Q_bound_lst[1][:,3,2,1,3]
Z_bound_lst[3][:,2].
```

Each has nine entries, for orders `0:8`. `multinomial_sum` combines them into a bound $D_8$ for the eighth derivative of $r^2Y_p\mathcal T^{13}Z_q$. Its general product formula is explained in [the combinatorial-bounds note](comb_bound.md).

`_r_sq_bound_table(8)` supplies `[1.1,2.2,2.2,0,…,0]`. Entries 1, 2, and 3 bound $r^2$, $2r$, and 2; entries 4–9 vanish. The constants also cover the extended grid, since $1+8/18000<\sqrt{1.1}$.

### 4.6 Keep the quadrature value and error together

`newton_cotes_with_bound` returns a quadrature value and a separate error allowance. On the unit interval, its closed 7-point rule gives

$$
E_{\mathrm{NC}}=\frac{3}{2800}h^8D_8,\qquad h=1/N_0.
$$

The proof is in [the matrix-functions note](matrix_func.md). For a four-term poloidal entry, the helper sums four quadrature values and four allowances. The same packed indices identify both outputs: `A_even_err[5,6]` accompanies `A_even[5,6]`, for example.

`integrate_r` returns `(A_even, A_odd, A_even_err, A_odd_err)`. In interval mode, the value matrix encloses the quadrature sums; §6 adds the symmetric remainder intervals to enclose the integrals.

<a id="energy-matrix"></a>

## 5. Construct the diagonal energy matrix B

### 5.1 Use orthogonality to reduce B to basis norms

The spectral basis is orthogonal for the L² and energy forms. Therefore only diagonal entries need integration. The zero off-diagonal entries follow from the analytic basis construction, not from a numerical smallness test.

<a id="natural-stokes-boundary-conditions"></a>

#### Natural Stokes boundary conditions and the diagonal energy matrix

The basis is associated with $\int_{B_R}|\nabla u|^2$ on regular axisymmetric divergence-free $H^1$ vector fields with no essential boundary condition. On the unit ball its modes satisfy

$$
-\Delta\Phi_i+\nabla p_i=\alpha_i^2\Phi_i,\qquad \nabla\!\cdot\Phi_i=0,\qquad \partial_n\Phi_i-p_i n=0.
$$

For a poloidal mode with physical radial coefficients $(Y,X)$, $p=\alpha^2Ar^lS_l$ and the boundary conditions are $Y'(1)=\alpha^2A$, $X'(1)=0$. For toroidal modes, $p=0$ and $Z'(1)=0$. These are the natural boundary conditions of the full-gradient energy, not componentwise homogeneous Neumann conditions or a condition on the symmetric strain tensor.

For any divergence-free test field v, integration by parts cancels the pressure and boundary terms:

$$
\int_{B_R}\nabla\Phi_i:\nabla v\,dx=\frac{\alpha_i^2}{R^2}\int_{B_R}\Phi_i\cdot v\,dx.
$$

Distinct eigenvalues are L²-orthogonal by symmetry of this identity. Distinct angular degrees are orthogonal by Legendre identities, and poloidal/toroidal fields are componentwise orthogonal. This also handles equal eigenvalues in different angular sectors. The raw modes are orthogonal but are not individually normalized to unit L² norm. Therefore

$$
B_{ij}=\delta_{ij}\left(\frac{\alpha_i^2}{R^2}+\eta_2\right)\|\Phi_i\|_{L^2(B_R)}^2,
\qquad A_{ij}=\int_{B_R}\Phi_i\cdot Q\Phi_j\,dx.
$$

The raw squared norms include $4\pi R^3$ and the reduced angular factors $1/(2l+1)$ and $l(l+1)/(2l+1)$. For imported retained coefficients V, energy normalization means $V^TBV\approx I$, which must be certified by the Gram diagnostics; it is not a normalization assumption on each raw basis element.

#### Evaluate the squared norms

The driver `get_lap_matrix` follows this chain:

```text
raw radial derivative bounds → Ur_sq_Wk∞_bound_lists → bounds for weighted squares
radial samples + these bounds → eval_l2_inpd → squared L² values and errors by degree
values/errors + spectral parameters → eval_h1_norm → energy values/errors by degree parity
                                    → pack by field parity → diagonal B and B_err
```

### 5.2 Integrate the weighted squares

For each radial channel U, `Ur_sq_Wk∞_bound_lists` bounds the eighth derivative of $r^2U^2$ using the radial-weight table and two copies of U's derivative-bound vector.

**Index example.** The input `X_bound_lst[2]` has shape `9 × 3`. The output at degree 2 is a vector of length 3; its second entry is the single eighth-derivative bound for $r^2(R^X_{2,2})^2$. It no longer has a derivative-order axis. `eval_l2_inpd` pairs that scalar bound with column 2 of the sampled weighted square.

For a fixed degree/root pair, let $I_Y=\int_0^1r^2(R^Y_{l,j})^2\,dr$, and define $I_X,I_Z$ similarly. The angular identities are

$$
\int_0^{\pi/2}S_l^2\sin\theta\,d\theta=\frac1{2l+1},\qquad
\int_0^{\pi/2}H_l^2\sin\theta\,d\theta=\frac{l(l+1)}{2l+1}.
$$

They follow from Legendre orthogonality and its derivative identity, with reflection halving the full-meridian integrals. Consequently,

$$
\widehat n^P_{l,j}=\frac{I_Y+l(l+1)I_X}{2l+1},\qquad
\widehat n^Z_{l,j}=\frac{l(l+1)I_Z}{2l+1}.
$$

There is no Y–X cross term, because these are orthogonal vector components. `eval_l2_inpd` applies the same positive angular factors to the quadrature allowances and returns values and errors grouped by degree and root.

For example, `Y_int_lst[2][2]` is $(I_Y+6I_X)/5$ for $\Phi^P_{2,2}$, and `Z_int_lst[3][1]` is $12I_Z/7$ for $\Phi^Z_{3,1}$. These are squared norms; no square root is taken. The `Y_int_lst` name denotes the complete poloidal norm, including its X contribution.

### 5.3 Apply the energy factor, then restore field parity

For the spectral parameter $\alpha_{l,j}$, `eval_h1_norm` multiplies the squared norm by

$$
\frac{\alpha_{l,j}^2}{R^2}+\eta.
$$

The driver applies this operation to the error lists as well. This spectral identity avoids separately integrating derivatives of every basis element.

`eval_h1_norm` returns **degree-even and degree-odd** vectors for one family. In our example, their entries are ordered as follows:

| Intermediate vector | Entries, identified by basis element |
|---|---|
| `Y_norm_even` | $\Phi^P_{2,1},\Phi^P_{2,2},\Phi^P_{2,3},\Phi^P_{4,1},\Phi^P_{4,2},\Phi^P_{6,1}$ |
| `Y_norm_odd` | $\Phi^P_{1,1},\Phi^P_{1,2},\Phi^P_{3,1},\Phi^P_{5,1},\Phi^P_{5,2}$ |
| `Z_norm_even` | $\Phi^Z_{2,1},\Phi^Z_{2,2},\Phi^Z_{4,1},\Phi^Z_{6,1}$ |
| `Z_norm_odd` | $\Phi^Z_{1,1},\Phi^Z_{3,1},\Phi^Z_{3,2},\Phi^Z_{5,1}$ |

Inside that helper, `cur_l_lst` counts roots already placed in each degree-parity vector. After the three degree-2 poloidal roots, its even counter is 3; the degree-4 roots go to positions `3+1=4` and `3+2=5`. The counter then becomes 5, so the degree-6 root goes to position `5+1=6`.

`get_lap_matrix` then packs by **field parity**:

```julia
norm_vec_even = [Y_norm_even; Z_norm_odd]
norm_vec_odd  = [Y_norm_odd;  Z_norm_even]
```

Thus `Y_norm_even[5]` becomes `B_even[5,5]`, `Z_norm_odd[3]` becomes `B_even[9,9]`, and `Z_norm_even[4]` becomes `B_odd[9,9]`. These entries correspond to $\Phi^P_{4,2}$, $\Phi^Z_{3,2}$, and $\Phi^Z_{6,1}$, respectively. They match the rows fixed in §1.2. In particular,

$$
\widehat B_{\mathrm{even},55}
=\left(\frac{(\alpha^Y_{4,2})^2}{R^2}+\eta\right)
\frac{I_Y+20I_X}{9}.
$$

The same packing is used for the errors. `get_lap_matrix` returns `(B_even, B_odd, B_even_err, B_odd_err)`.

<a id="assembly-verification"></a>

## 6. Complete the enclosures and verify the imported eigenpairs

### 6.1 Add the integral errors and physical scaling

Both drivers return normalized radial integrals over the half-meridian. In interval mode, the notebook first calls `add_error` on each value/error pair, then multiplies the resulting matrices by

$$
4\pi R^3
=\underbrace{2}_{\text{meridian reflection}}
\underbrace{2\pi}_{\text{azimuthal integration}}
\underbrace{R^3}_{\rho^2d\rho=R^3r^2dr}.
$$

The allowances receive the same scaling because they were added first. The derivative scaling $R^{-2}$ is already included in B's energy factor. For example, the physical `A_even[6,9]` encloses $4\pi R^3$ times the mixed integral from §4.4.

### 6.2 Keep basis indices separate from retained-field indices

The notebook loads eigenvalues and coefficient matrices from `data/external/eig.mat`; it does not solve for new eigenvectors. For each parity, write V for the imported coefficient matrix and $\Lambda=\operatorname{diag}(\nu_s)$ for the positive retained generalized eigenvalues, distinct from the unstable eigenvalue. Column $s$ represents

$$
\psi_s=\sum_pV_{ps}\Phi_p.
$$

**Index example.** If three even fields are retained in our ten-element even block, V has shape `10 × 3`. `V[2,1]` is the coefficient of $\Phi^P_{2,2}$ in the first field; `V[9,3]` is the coefficient of $\Phi^Z_{3,2}$ in the third. The three entries of the eigenvalue vector are indexed by retained field, not by basis row.

The notebook forms

$$
G_E=V^TBV,\qquad G_Q=V^TAV,\qquad
D=AV\Lambda^{-1}-BV.
$$

In this example, $G_E,G_Q$ have size `3 × 3`, while D has size `10 × 3`. Thus `GE[1,3]` pairs the first and third retained fields, whereas `A[1,3]` pairs the first and third basis elements. `D[9,3]` is the basis-row-9 component of the third field's eigenpair residual. The local variable `Lambda` in `eigen_diagnostics` stores $\Lambda^{-1}$, despite its name.

### 6.3 Bounds for the retained eigensystem

`verify_general_eig` checks positive definiteness of

$$
S=\tau B-A+(AV)\Lambda^{-1}(AV)^T.
$$

The current acceptance candidates are $\tau_U=0.449$ (even) and
$\tau_v=0.445$ (odd), each asserted immediately after printing. These are
candidate upper bounds, not measured eigenvalues or evidence of a completed run.
The supplied columns need not be exact eigenvectors for this matrix check.

The manuscript's $K_6$ is the Q Rayleigh quotient on the Q-orthogonal complement
of the retained fields, measured in the energy norm. On that subspace,
$V^TAz=0$, so the deflation term vanishes and $S\succ0$ proves $K_6\le\tau$.
The unrestricted deflated supremum is a sufficient upper bound, not an identity
with the constrained supremum. See [the matrix-functions derivation](matrix_func.md#generalized-eigenvalue-certificate--verify_general_eig).

The following arguments explain the separate $\varepsilon_3$ and $\varepsilon_4$ calculations in [finite_rank_construction.ipynb](../../notebooks/finite_rank_construction.ipynb), in `eigen_diagnostics`, which forms `GE`, `GQ`, `eQ`, `eE`, and `eD` and then reports their bounds. Each error bound is immediately asserted to have upper endpoint at most `2.8e-7`, in each parity block; see the [current acceptance table](../notebooks.md). Here $\|\cdot\|_2$ denotes the Euclidean vector norm or matrix spectral norm, and $\|\cdot\|_F$ denotes the matrix Frobenius norm. For any matrix, $\|M\|_2\leq\|M\|_F$; the notebook computes the latter as `sqrt(sum(abs2, M))`.

These are conditional matrix certificates: the computed enclosures must contain the intended physical bilinear forms A and B, including quadrature and scaling errors, and the imported columns of V must use the same basis. The formulas below refer to these exact matrices. Assume $A=A^T$, $B=B^T\succ0$, and $\Lambda=\operatorname{diag}(\nu_1,\ldots,\nu_m)\succ0$. All interval divisions and square roots below require certified positive denominators. The paper's use of the Q-inner product also requires a separate certificate that the matrix field $Q(x)\succeq0$; neither the finite-dimensional checks nor positivity of $Q+\operatorname{sym}\nabla U+\eta_1I$ proves that pointwise hypothesis.

<a id="eps3-bound"></a>

#### 6.3.1 The normalized Q Gram error: $\varepsilon_3$

The paper defines $\varepsilon_3$ in `def-eps3` and proves its projection estimate in `lem:quasi ort`; see the paper. With $G_Q\succeq0$, its definition is

$$
\varepsilon_3
=\left\|I-G_Q^{1/2}\Lambda^{-1}G_Q^{1/2}\right\|_2.
$$

The notebook constructs a different-looking symmetric matrix,

$$
e_Q=I-\Lambda^{-1/2}G_Q\Lambda^{-1/2}.
$$

This is `eQ_even` or `eQ_odd`: the code's `Lambda_*` already stores $\Lambda^{-1}$, so `sqrt.(Lambda_*)` is the diagonal matrix $\Lambda^{-1/2}$. The elementwise square root is appropriate because this matrix is diagonal.

To prove that the two spectral norms agree, set $C=\Lambda^{-1/2}G_Q^{1/2}$ and take a singular value decomposition $C=U\Sigma W^T$. Then

$$
I-G_Q^{1/2}\Lambda^{-1}G_Q^{1/2}
=I-C^TC=W(I-\Sigma^2)W^T,
$$

$$
e_Q=I-CC^T=U(I-\Sigma^2)U^T.
$$

Both matrices therefore have the same eigenvalues, and orthogonal invariance gives equality of their spectral norms and of their Frobenius norms. No commutation of $G_Q$ and $\Lambda$ is assumed. In particular,

$$
\boxed{\varepsilon_3=\|e_Q\|_2\leq\|e_Q\|_F.}
$$

Thus the upper endpoint of the interval result `sqrt(sum(abs2, eQ))` is a valid upper bound for $\varepsilon_3$. If this bound is less than one, it also certifies $G_Q\succ0$: every eigenvalue of $\Lambda^{-1/2}G_Q\Lambda^{-1/2}=I-e_Q$ is positive. This implication needs only symmetry of $G_Q$ and positive $\Lambda$, so it can establish the Gram-matrix positivity needed for the square root above. It does not establish pointwise positivity of Q.

For completeness, write the Q-projection as $f_{\mathrm{lg}}=\Psi a$, where $\Psi=(\psi_1,\ldots,\psi_m)$ and $\langle f-f_{\mathrm{lg}},Q\psi_j\rangle=0$. The two quantities in the paper's projection estimate are $a^TG_Qa$ and $a^TG_Q\Lambda^{-1}G_Qa$. Their difference is bounded by $\varepsilon_3\|Q^{1/2}f_{\mathrm{lg}}\|_{L^2}^2$. When $Q\succeq0$, orthogonal projection in the Q-geometry gives

$$
\|Q^{1/2}f_{\mathrm{lg}}\|_{L^2}^2
\leq\|Q^{1/2}f\|_{L^2}^2
\leq\|\lambda_{\max}(Q)\|_{L^\infty}\|f\|_{L^2}^2,
$$

which completes the stated $\varepsilon_3$ estimate.

<a id="eps4-bound"></a>

#### 6.3.2 Energy-weighted eigen-residual bound: $\varepsilon_4$

Let $\Psi=(\psi_1,\ldots,\psi_m)$ be the retained fields with coefficient matrix V, and set

$$
G_E=V^TBV,\qquad D=AV\Lambda^{-1}-BV,
\qquad d=\|B^{-1/2}D\|_F,\qquad \gamma=\|I-G_E\|_F<1.
$$

The weak residuals $r_j^e\in V_N$ satisfy

$$
\langle\psi_j,Qv\rangle
=\nu_j\langle\psi_j,v\rangle_E+\langle r_j^e,v\rangle_E,
\qquad v\in V_N.
$$

Since the energy inner product in coefficient coordinates is $x^TBy$, the coefficients of $r_j^e/\nu_j$ are $B^{-1}D_j$. Thus

$$
d^2=\sum_j\frac{\|r_j^e\|_E^2}{\nu_j^2}.
$$

For $f=\Psi a$, the Gram error gives

$$
\|f\|_E^2=a^TG_Ea\geq(1-\gamma)\|a\|_2^2.
$$

If $u\in V_N$ is Q-orthogonal to every $\psi_j$, the residual identity and Cauchy–Schwarz imply

$$
\begin{aligned}
|\langle f,u\rangle_E|
&=\left|\sum_j\frac{a_j}{\nu_j}\langle r_j^e,u\rangle_E\right|\\
&\leq\|a\|_2\,d\,\|u\|_E\\
&\leq\frac{d}{\sqrt{1-\gamma}}\|f\|_E\|u\|_E.
\end{aligned}
$$

Hence a certified bound for the required quasi-orthogonality is

$$
\boxed{\varepsilon_{4,\mathrm{safe}}=\frac{d}{\sqrt{1-\gamma}}.}
$$

In the notebook, B is diagonal, so $d^2=\sum_{i,j}|D_{ij}|^2/B_{ii}$: divide each row of `eD` by `sqrt.(diag(B))` before applying `sqrt(sum(abs2, ...))`. The denominator uses the independent Gram error `eps4_gram_error = sqrt(sum(abs2, eE))`. The checks require finite positive diagonal entries of B, a finite weighted residual, and a Gram-error upper bound below one. Interval upper bounds for d and $\gamma$ give the certified result.

### 6.4 Read the saved matrix outputs

The notebook checks $K_6^U\le0.449$, $K_6^v\le0.445$ and each projection
error against `2.8e-7`, matching the inputs to the paper's `Sec:num prev`.
The [workflow guide](../notebooks.md) lists all stage outputs and checks.

The two saved files represent different stages:

| File under `data/generated/` | Fields | Meaning and example dimensions |
|---|---|---|
| `integrate_θ_result.mat` | `result_even`, `result_odd` | Extended radial sample × two compressed angular indices × two component indices; each is `18017 × 3 × 3 × 3 × 3` for `L=6`. |
| `psi_eig_mat_AB.mat` | `A_even`, `A_odd`, `B_even`, `B_odd` | Physically scaled matrices; even `10 × 10`, odd `9 × 9` in our root-count example. In interval mode they include the quadrature allowances. |

The notebook first checks the A/B cache and loads it directly when present, skipping all assembly. On a miss, it loads the angular cache if present or computes and saves it before radial assembly. Error enlargement and physical scaling are applied only to newly assembled A/B matrices, which are then saved before the eigenpair checks. Assembly temporaries stay in a local `let` block. Actual dimensions follow
the chosen degree and root counts; the smaller sizes above are examples. Reuse
a cache with the same Q, root table, signed basis, ordering, numeric mode and
quadrature settings; see [the data guide](../data.md).

For the implementation, follow `integrate_θ` in [Trig_Poly.jl](../../src/Trig_Poly.jl), then `integrate_r` and `get_lap_matrix` with their helpers in [My_Assembly.jl](../../src/My_Assembly.jl), and finally the checking block in the notebook. In the paper, `subSec:numfr` and `lem:lap eig` give the
basis and spectral identity; `def:aq` gives the Q Gram matrix after energy
normalization. The general quadrature, product-bound, and certificate arguments remain in their linked technical notes.
