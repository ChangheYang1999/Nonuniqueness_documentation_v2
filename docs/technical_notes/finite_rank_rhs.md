

# Finite-rank RHS: reconstruction, multiplication, and rigorous bounds

This note follows Step 4B in [Part 4B of `finite_rank_construction.ipynb`](../../notebooks/finite_rank_construction.ipynb#step-4b). The matrix construction and its basis-row ordering are explained in [the assembly note](finite_rank_assembly.md#basis-ordering). Here we use those same rows to reconstruct fields from the imported coefficients and prepare the RHS for Steps 5, 6 and 7.

For a coefficient matrix V, a row indexes a basis element and a column indexes a retained field:

$$
\psi_s=\sum_pV_{ps}\Phi_p.
$$

Part 4B loads the root parameters and the external `eig.mat` independently; it does not read Part 4A's A/B caches. Basis normalization, angular signs, and coefficient-row order must nevertheless agree between the stages. Part 4A must certify the retained columns used by Part 4B before their downstream bounds can contribute to the final proof.

We use $\kappa$ for `scl_fac`, normalized radius $r(\beta)=(\kappa/R)\tan\beta$, and Q's stored argument $q(\beta)=(\pi/2)r(\beta)$. The scalar angular factors are $S_l=P_l(\cos\theta)$ and $H_l=-S_l'$, as in [the Bessel note](bessel_basis.md#angular-rows). The compressed angular index is $h(l)=\lfloor(l+1)/2\rfloor$, interpreted with the field parity and component. Examples below use the assembly note's root counts: poloidal `[2,3,1,2,2,1]` and toroidal `[1,2,2,1,1,1]` at degrees `1:6`.

The computation follows two calculations together: sum the sampled fields with their signs intact, and build conservative derivative bounds from the same coefficients. Finite differences then refine the priors using the sampled sums. The general product and coordinate-change arguments are in [comb_bound.md](comb_bound.md); the refinement method is in [matrix_func.md](matrix_func.md).

## 1. Choose the grid and evaluate the separated factors

With $a=\kappa/R$ and $\mathrm{Beta}=\arctan(R/\kappa)$, the physical part of the grid is $[0,\mathrm{Beta}]\times[0,\pi/2]$. The notebook uses `12000 × 6000` subintervals and 25 additional points at each end of each coordinate for ψ reconstruction: sample shape `(12051,6051,J,3)` for J retained fields. This supplies the order-10 refinement padding of 15 layers plus a common ten-layer domain for the exported β bounds. It passes

```julia
r_glb = tan.(β_glb) * a
eig_psi_even, eig_psi_odd = eval_eig_function(
    eig_vec_odd, eig_vec_even, Y_paras, Z_paras, r_glb, θ_glb)
```

The input order is **odd coefficient matrix, then even coefficient matrix**; the output is the two-element vector **`[ψ_even, ψ_odd]`**. Each array has shape

```text
(number of radial samples, number of angular samples, number of retained fields, 3).
```

`eval_eig_function` evaluates the three radial lists with `eval_Ur` and the four angular-code blocks with `eval_Uθ`. `_allocate_channel_field_blocks` allocates the even and odd output arrays according to the **column counts** of the corresponding V matrices.

## 2. Reconstruct each field in the common basis order

`_accumulate_poloidal_channels!` selects the field-parity group for degree $l$. If `offset` rows of that group have already been consumed and the degree has $n_l^P$ roots, it uses

```julia
rows = offset+1:offset+n_l_P
radial_Y = Ur_Y_lst[l] * V[rows, :]
radial_X = Ur_X_lst[l] * V[rows, :]
```

These explanatory variable names display the operation in the helper. Both products have shape `number_of_radial_samples × number_of_retained_fields`. **The same V rows multiply Y and X.** An outer product with the appropriate angular row then adds to output components 1 and 2:

$$
\psi^Y_s(r,\theta)\mathrel{+}=
\left(\sum_{j=1}^{n_l^P}V_{p(l,j),s}R^Y_{l,j}(r)\right)S_l(\theta),
$$

$$
\psi^X_s(r,\theta)\mathrel{+}=
\left(\sum_{j=1}^{n_l^P}V_{p(l,j),s}R^X_{l,j}(r)\right)H_l(\theta).
$$

The angular row is `h(l)`. `_angular_code_indices_by_field_parity` selects the code blocks `(Y,X,Z)`: `("11","00","01")` for even fields and `("10","01","00")` for odd fields. These code choices supply the S or H row of the correct degree, as explained in [the basis note](bessel_basis.md#angular-rows).

For the even group, the coefficient rows and angular rows are:

| Family and degree | Coefficient rows in `V_even` | Angular row $h(l)$ | Output components |
|---|---|---|---|
| poloidal, 2 | `1:3` | 1 | Y and X |
| poloidal, 4 | `4:5` | 2 | Y and X |
| poloidal, 6 | `6:6` | 3 | Y and X |
| toroidal, 1 | `7:7` | 1 | Z |
| toroidal, 3 | `8:9` | 2 | Z |
| toroidal, 5 | `10:10` | 3 | Z |

For example, degree 4 multiplies each two-column radial Y/X matrix by `V_even[4:5,:]` and uses angular row 2. Both roots share that angular row. After all three poloidal degrees, the offset is 6.

`_accumulate_toroidal_channel!` continues with that offset, rather than restarting at zero. After the degree-1 toroidal root the offset is 7; degree 3 consumes `V_even[8:9,:]`, multiplying both radial root columns by angular row 2. Degree 5 consumes the final row 10. Final assertions compare the consumed rows with `size(V,1)`.

This block multiplication is just the finite sum $\psi_s=\sum_pV_{ps}\Phi_p$, arranged so that all roots of one degree share an angular multiplication. It preserves cancellation between coefficients. There is no eigensolve, Q multiplication, or quadrature inside `eval_eig_function`.

<a id="packed-field-bounds"></a>

## 3. Pack the basis bounds: `lst_to_mat`

A bound for $R(r)$ alone is not yet a bound for a component $R(r)\Theta(\theta)$. `lst_to_mat` combines the radial tables with the angular bounds $\|S_l\|_\infty\le1$ and $\|H_l\|_\infty\le l$. It calls `Uθ_bound(L,2)` and uses its first-derivative column to attach the factor $l$ to the X and Z channels.

For each order $m$, it places the following component bounds at the packed row $p$:

$$
U[m+1,p,:]=
\begin{cases}
(b^Y_{l,j,m},\ l\,b^X_{l,j,m},\ 0),&p\leftrightarrow\Phi^P_{l,j},\\
(0,\ 0,\ l\,b^Z_{l,j,m}),&p\leftrightarrow\Phi^Z_{l,j}.
\end{cases}
$$

Here $K$ is the highest derivative order and $b^a_{l,j,m}$ is the corresponding entry in a raw radial table. The output is `(U_even, U_odd)`, each shaped `(K+1, number_of_basis_elements, 3)`. These are bounds for **radial derivatives of full components**, uniform in $\theta$; the last axis is a component axis, not a derivative-direction axis.

In the [assembly example](finite_rank_assembly.md#basis-ordering), `U_even[m+1,5,:]` contains the Y bound and four times the X bound for $\Phi^P_{4,2}$; `U_even[m+1,9,:]` has only the Z bound, multiplied by 3, for $\Phi^Z_{3,2}$. Thus the second axis matches the rows of `eig_vec_even` exactly. Section 4 below uses this alignment to bound $\psi$.

For example, `U_even[4,5,2]` bounds the third radial derivative of the X component of $\Phi^P_{4,2}$, uniformly in $\theta$: derivative order 3, packed basis row 5, component 2. It equals four times `X_bound_lst[4][4,2]`. Similarly, `U_even[4,9,3]` equals three times `Z_bound_lst[3][4,2]`: the same derivative order, now at the toroidal element's packed row 9 and component 3.

## 4. Bound the reconstructed fields and refine

The notebook starts with two more derivative orders than the first refinement needs:

```julia
U_bound_even, U_bound_odd =
    lst_to_mat(Ur_bound_lst(Y_paras, Z_paras, 12)...)
```

The packing from §3 gives tensors of shape `(13, number_of_basis_elements, 3)`: `(13,10,3)` for the even group and `(13,9,3)` for the odd group in our example. `eig_fun_bound` in [`Comb_Bound.jl`](../../src/Comb_Bound.jl) combines these basis bounds with the absolute values of the coefficients. Its triangle-inequality argument is explained in [the combinatorial-bounds note, §7](comb_bound.md). For each order it returns a `3 × number_of_retained_fields` matrix, and the result contains orders `0:12`. The basis axis must match the rows of V. Unlike sampled reconstruction, this prior discards cancellation.

`r_to_beta(..., a; β_max=beta_prior_max)` then uses the chain rule for $r=a\tan\beta$ to convert the radial bounds into $\beta$-derivative bounds. These Bessel bounds are not transformed by simply multiplying the order-$m$ row by a constant: higher derivatives of $\tan\beta$ enter. The notebook uses `beta_prior_max = 1.01 * Beta`, checks that it is below $\pi/2$, and checks that it contains the sampled $\beta$ grid. The radial polynomial envelopes must also cover the corresponding normalized radii, as specified in [the Bessel note, §6.5](bessel_basis.md#polynomial-derivative-bounds).

The exported helper [`refine_channel_bounds`](../../src/Comb_Bound.jl) processes each retained field separately. It builds finite-difference parameters from that field's component bounds, supplies angular frequency 100, and calls `norm_lst(..., "W10Inf"; padding=25)`. The explicit actual padding preserves the physical mesh spacings and domain lengths. Its result is one `11 × 2` table per field. Row `m+1` is order $m$, the columns are directions $(\beta,\theta)$, and each entry is a three-component bound vector.

Write $I_s=[-s h_\beta,\mathrm{Beta}+s h_\beta]$. Since `derivative_padding(10)=15`, the 25-layer samples give every returned β order a common domain $I_{10}$ (higher orders cover still larger nested intervals). `beta_bound_chain` exports only that β column, so these bounds cover the later RHS stencils rather than only the physical interval.

We already have a prior bound; refinement seeks a sharper bound by sampling the **assembled sum**, where its terms may cancel. The order-12 priors control the error in the order-10 estimates. Refinement is rigorous under the domain and frequency conditions of the [matrix-functions note](matrix_func.md); it need not reduce every individual bound numerically.

## 5. Apply Q and the radial weight

The Q entries must be sampled on the same grid in the physical spherical frame as $\psi$. The returned X radial channel already compensates $H_l=-S_l'$; no $DQD$ conversion is applied:

```julia
Q_pts = interpolate(Q, r_glb .* (Pi / dt(2)), θ_glb)
rhs_even, rhs_odd = apply_Q_to_channel_fields(
    Q_pts, [eig_psi_even, eig_psi_odd])
```

`apply_Q_to_channel_fields` reads the six independent entries `A11`, `A12`, `A13`, `A22`, `A23`, `A33`, using symmetry for the other three. At each grid point and for each retained field it computes

$$
\begin{pmatrix}
Q_{11}\psi^Y+Q_{12}\psi^X+Q_{13}\psi^Z\\
Q_{12}\psi^Y+Q_{22}\psi^X+Q_{23}\psi^Z\\
Q_{13}\psi^Y+Q_{23}\psi^X+Q_{33}\psi^Z
\end{pmatrix}.
$$

It returns `(Qψ_even, Qψ_odd)`, with the same four axes as its input fields. Interpolation and coordinate conversion happened in the caller. The helper only multiplies the supplied samples.

The notebook next applies the scalar weight

$$
w(\beta)=\frac{\tan\beta}{\cos\beta},\qquad
f_s(\beta,\theta)=w(\beta)\,Q(q(\beta),\theta)\psi_s(r(\beta),\theta).
$$

With physical radius $\rho=\kappa\tan\beta=Rr$, the radial volume element is:

$$
\rho^2\,d\rho=\kappa^3\tan^2\beta\,\sec^2\beta\,d\beta
=\kappa^3w(\beta)^2\,d\beta.
$$

Absorbing $w$ into the fields prepares the RHS for the later weighted representation and $L^2$ calculations. The angular weight and constant physical factors are handled separately. `apply_Q_to_channel_fields` does not insert $w$; the notebook applies it once on the full 25-layer grid, then crops 15 layers on each side of both axes before saving `rhs.mat`. Weighting must precede this crop so the sample and weight shapes agree.

## 6. Bound the weighted product and refine again

The bound calculation must follow these same operations. `read_matrix` loads the normalized-radius derivative table `Q_WkInf` from `Q_bound_radial.mat`; the notebook selects orders `0:k`, and `vec_to_mat` arranges each order as a $3\times3$ component matrix. `r_to_beta` converts this chain once to derivatives of $Q(q(\beta),\theta)$. Using a bound already in $\beta$ as though it were a radial bound would apply the wrong coordinate transformation.

`beta_bound_chain` extracts the $\beta$ column from the refined $\psi$ tables and assembles one `3 × number_of_retained_fields` matrix per order. Together with `tan_over_cos_deriv_Linf_bounds`, these are the three factors passed to

```julia
rhs_prior = multinomial_sum(10, weight_bounds, Q_beta_bounds,
                            psi_beta_bounds; last_flag=false)
```

This explanatory call gives all orders $0{:}10$. In component form, the underlying bound is

$$
\|\partial_\beta^m f_s^a\|_\infty
\le\sum_{i+j+k=m}\frac{m!}{i!j!k!}
\|w^{(i)}\|_\infty
\sum_{b=1}^3
\|\partial_\beta^j Q_{ab}\|_\infty
\|\partial_\beta^k\psi_s^b\|_\infty.
$$

The matrix product in the implementation performs the sum over $b$. The scalar weight, matrix Q, and vector field therefore retain their different roles in the product bound.

Finally, `refine_channel_bounds` uses the already cropped weighted RHS with angular frequency 410 and calls `norm_lst(..., "W8Inf"; padding=10)`, with helper `crop=0`. The ten remaining layers equal `derivative_padding(8)`; do not crop them again. The order-10 product priors cover $I_{10}$ and supply the two higher derivatives needed by this refinement. The angular frequency argument still requires the separate Bernstein domain or symmetry justification; radial padding does not establish it.

The order choices fit together as follows:

| Stage | Derivative orders available | What the next stage uses them for |
|---|---|---|
| Analytic bounds for $\psi$, converted to $\beta$ | $0{:}12$ | Control refinement through order 10. |
| Refined $\psi$ bounds, Q bounds, and weight bounds | $0{:}10$ | Form prior bounds for the weighted RHS through order 10. |
| RHS product priors | $0{:}10$ | Control refinement through order 8. |
| Refined RHS bounds | $0{:}8$ | Bound the later quadrature errors and derivative calculations. |

### How the workflow padding is chosen

Start with the last refinement and work backward. The order-8 RHS calculation
requires $P(8)=1+2+3+4=10$ layers. Its order-9 and order-10 product priors must
hold on that ten-layer interval. The product rule uses ψ derivatives of every
order from 0 through 10, so all those ψ bounds must hold there as well.
Order-10 ψ refinement needs $P(10)=1+2+3+4+5=15$ layers around its target.
Its target already includes ten layers, giving `padding_psi = 10 + 15 = 25`.

| Calculation | Refinement order | Extra target layers | Actual sample padding per side |
|---|---|---|---|
| Q in β, profile notebook | 10 | 0 | 15 |
| ψ supplying the RHS product priors | 10 | 10 | 25 |
| Weighted RHS | 8 | 0 | 10 |
| Eigenpair and finite-rank residuals | 8 | 0 | 10 |

The RHS uses the separately converted analytic Q bounds from
`Q_bound_radial.mat`; it does not reuse the profile's refined Q β table.
That is why the profile Q refinement needs only its physical target interval.

The producer therefore removes $25-10=15$ sample layers after weighting ψ
by Q and $w$, retaining the ten needed by RHS refinement and its consumers.
The [stencil derivation](matrix_func.md#why-refinement-proceeds-from-high-order-to-low-order)
explains the accumulated sums and the half-grid adjustment for odd orders.
The profile's separate order-2 calculations use one layer per side,
`derivative_padding(2)=1`. The frequency-only angular assembly also retains
its existing eight layers; it uses independent corrections rather than the
recursive sum.

## 7. Save the RHS samples and bounds

Let $J$ be the retained-field count of the relevant parity. Step 4B writes:

| File and fields | In-memory array layout passed to `save_matrix` |
|---|---|
| `rhs.mat`: `rhs_even`, `rhs_odd` | `(12021, 6021, J, 3)`: both grid axes retain all ten padding layers. |
| `rhs_bound.mat`: `rhs_bound_even`, `rhs_bound_odd` | `(3, 9, 2, J)`: component, derivative order $0{:}8$, direction $(\beta,\theta)$, retained field. |
| `rhs_bound.mat`: `rhs_prior_even`, `rhs_prior_odd` | `(3, J, 11, 1)`: component, retained field, derivative order $0{:}10$, singleton block. |

The extra singleton axis in the prior layout comes from the product-bound container; it is not another derivative direction. The refined table contains pure derivatives in each coordinate, not a full table of mixed derivatives. In interval mode, `save_matrix` splits each logical array into `_lo`/`_hi`
endpoint arrays; `read_matrix` reconstructs the intervals. See
[the data guide](../data.md) for the file layouts.

Step 5, `selfsimilar_linear_operator_finite_rank.ipynb`, consumes the even block; Step 6, `eigen_linear_operator_finite_rank.ipynb`, consumes the odd block. They retain all ten saved layers for order-8 refinement, generate matching ten-layer residual grids, and trim ten layers only when returning to quadrature. Their externally supplied approximate solutions in `phi_even/` and `phi_odd/` are distinct from the Bessel combinations $\psi_s$ reconstructed here: the latter define the forcing terms $Q\psi_s$ used to verify those solutions.

### 7.1 Normalization at the handoff to inverse verification

Let $\nu_j>0$ denote the retained generalized eigenvalues stored in `eig_val_even`/`eig_val_odd`, to distinguish them from the unstable eigenvalue. Step 4B stores the samples denoted $f_j$ in §5, now written $g_j$ to distinguish them from the unweighted column $b_j$:

$$
g_j(\beta,\theta)=w(\beta)Q\psi_j,
\qquad b_j=\frac{Q\psi_j}{\nu_j},\qquad wb_j=\frac{g_j}{\nu_j}.
$$

No $\nu_j$ division occurs in Step 4B. Steps 5–6 use $g_j$ unchanged when assembling their residual against the loaded inverse trial. The forcing Gram instead uses

$$
(G_Q)_{ij}=\langle b_i,b_j\rangle_{L^2}
=\frac{\langle g_i,g_j\rangle_{\rm weighted\ measure}}{\nu_i\nu_j}.
$$

The weighted measure here supplies $4\pi\kappa^3\sin\theta\,d\beta\,d\theta$; g already contains one w. In Step 6 the forcing/residual and forcing/trial cross-Grams divide by the row eigenvalue only. Do not divide the stored RHS a second time or infer a rescaling of the supplied inverse trials from the Gram normalization. The raw inverse-trial sign and late odd projection are specified by the consumer's residual definition and [the central notation map](../paper_code_map.md). The forcing Gram called `GQ` here is distinct from Part 4A's retained-basis Q Gram $V^TAV$.

Step 7, `verify_auxiliary_field_bound.ipynb`, also consumes `rhs_odd` and `rhs_bound_odd`. It removes the ten layers with slices `11:12011,11:6011`, retaining the full `12000 × 6000` physical grid, and forms `sum_j (c0_j/nu_j) g_j`, with derivative bounds weighted by `abs(c0_j/nu_j)`. These are fixed binary source centers obtained from inverse multipliers and matched Step 6 pairings; a separate residual-Gram allowance controls the difference from the exact coefficients. The [auxiliary-field derivation](../paper_code_map.md#direct-mass-weighted-m5)
explains how this allowance gives the paper's $M_5^v$ bound in
`eq:dual-riesz-upper-bound`. Roots, eigenvector columns, scale, physical convention, Step 6 matrices and the RHS must all correspond.

### 7.2 Cache and certification boundary

Part 4B bypasses generation when both RHS files exist. Otherwise it computes
and saves both arrays. Reuse the pair only with the same Q, roots, scale,
eigenvector ordering, grid and arithmetic mode; see [the data guide](../data.md).
The inverse-field notebooks use this pair to enclose their residuals.

The field reconstruction and Q multiplication are implemented in [My_Assembly.jl](../../src/My_Assembly.jl). The two workflow helpers `refine_channel_bounds` and `beta_bound_chain` are exported from [Comb_Bound.jl](../../src/Comb_Bound.jl). Their [input contracts and table layouts](comb_bound.md#8-per-mode-refinement-and-β-bound-chains) are caller assumptions; the lower-level norm routines retain their derivative-order, padding, and grid checks. The notebook imports the helpers and supplies the ψ and RHS inputs described above.
