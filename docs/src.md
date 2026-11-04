# Source Library (`src/`)

The `src/` directory is a Julia library loaded by the notebooks (not a standalone package).
`NS_Numerics.jl` is the module entry point: it sets the active numeric type `dtype`, includes
every sub-file, and re-exports the public API used by the notebooks. Rigorous results require
`dtype = Interval{Float64}`, the current setting (see `docs/notebooks.md`). The Bessel
routines still use Arblib internally before returning interval enclosures.

<!-- reader-guidance:source:begin -->

<h2 id="reading-the-source">How to read the source</h2>
<h3>Objects used in the notebooks</h3>
<p>Start with the meaning of these objects, then follow the short calculation below. Read the indicated definitions and functions as they become relevant.</p>
<ul>
<li><strong><code>ScalarFunc</code></strong> represents a scalar trigonometric function: <code>freq_func</code> holds its coefficients in a two-dimensional array and <code>space_func</code> holds grid samples. The boundary conditions in <code>bdry</code> determine the basis used. Either representation can be unset; inspect the one a function reads.</li>
<li><strong><code>VectorFunc</code></strong> groups named scalar components with a shared <code>scl_fac</code>. <code>UP</code> groups velocity and pressure; <code>Q</code> groups six independent symmetric-matrix entries. Start with the structures in <a href="../src/My_Scalar.jl">My_Scalar.jl</a> and <a href="../src/My_Vector.jl">My_Vector.jl</a>, and the corresponding input description in <a href="data.md">data.md</a>.</li>
<li><strong><code>Operator</code> and <code>Operator_matrix</code></strong> describe actions on coefficients. One block applies radial and angular matrices, with optional pointwise factors; a block matrix combines input components into output components. Read their definitions and <code>apply</code> in <a href="../src/My_Operator.jl">My_Operator.jl</a>.</li>
<li><strong><code>dtype</code> and <code>dt</code></strong> select the numeric representation and convert values. Interval operations enclose arithmetic results; bounds for a continuous function also need the sampling and quadrature error estimates used by the notebook.</li>
</ul>
<p>For a connected explanation of coefficients, samples, component order, and the passage from a loaded field to a norm bound, read <a href="technical_notes/scalar_vector.md">ScalarFunc and VectorFunc</a>. Return to the examples below for the notebook calculations.</p>
<h3>Example: from profile to residual</h3>
<p>This excerpt from <a href="../notebooks/selfsimilar_profile_verification.ipynb">Step 1</a> assumes the module is loaded and the notebook's <code>β_pt</code>, <code>θ_pt</code> grids are prepared:</p>
<pre><code class="language-julia">UP = from_mat_file("../data/external/UP.mat", 2; domain="frequency")
lin_mat = get_operator_matrix(UP, β_pt, θ_pt, "linear")
err = mul_A_gradB(UP, UP, β_pt, θ_pt) - apply(lin_mat, UP)
</code></pre>
<p>The loader retains the coefficient representation and applies the divergence-free projection. <code>UP</code> is a <code>VectorFunc</code> that stores the compactified unknowns for the whole physical space ℝ³. The radial map <code>r = scl_fac*tan(β)</code> sends <code>0 ≤ β &lt; π/2</code> to <code>0 ≤ r &lt; ∞</code>; the computational endpoint β = π/2 represents spatial infinity. The operators incorporate the field and integration weights. <code>lin_mat</code> is an <code>Operator_matrix</code> that maps the coefficients to weighted linear-term samples. The last line forms the weighted self-similar residual: <code>err</code> has three components, each sampled on a <code>length(β_pt) × length(θ_pt)</code> grid.</p>
<p>The notebook then builds derivative bounds with <code>norm_lst(err, "W$(k)Inf"; ...)</code> and passes them as <code>paras</code> to <code>norm_lst(err, "L2"; ...)</code>. The former supplies componentwise derivative estimates; the latter combines quadrature and its error allowance into a scalar interval enclosure of the residual norm.</p>
<!-- reader-guidance:integral-example:begin -->
<h3 id="rigorous-integral-example">Example: from samples to a rigorous integral bound</h3>
<p>Continue with the weighted residual <code>err</code> from the preceding example and the uniform, unpadded grid used in <a href="../notebooks/selfsimilar_profile_verification.ipynb">Step 1's L² residual calculation</a>: both β and θ range from zero to π/2. Its squared norm is the integral of <code>sum(err_i^2) * sin(θ)</code> over this rectangle, multiplied by <code>4π * err.scl_fac^3</code>. The other physical weights are already incorporated in <code>err</code> by the operators.</p>
<pre><code class="language-julia">@assert dtype == Interval{Float64}
M0, M1 = size(UP["u3"].freq_func)
D = [Pi / dt(2), Pi / dt(2)]
freq_lst = [dt(4 * M0 + 10), dt(4 * M1 + 10)]
W = norm_lst(err, "W8Inf"; paras=freq_lst, domain=D)
J = inner_product(err, err; W_norms_lst=[W, W], domain=D)
E = sqrt(J)
</code></pre>
<p><code>M0</code> and <code>M1</code> are coefficient-array sizes; <code>freq_lst</code> uses the notebook's corresponding frequency ceilings for this residual. <code>W</code> is a <code>9 × 2</code> table whose entries are three-component vectors: <code>W[m+1, 1][i]</code> bounds the supremum of the absolute m-th β derivative of component i, and column 2 gives θ derivatives. <code>J</code> encloses the squared norm; <code>E</code> encloses the L² norm, so <code>sup(E)</code> is its upper bound. The last two lines are equivalent to <code>norm_lst(err, "L2"; paras=W, domain=D)</code>. The notebook computes bounds through order 10; order 8 suffices for this quadrature allowance.</p>
<p>Follow these three parts of the calculation in order:</p>
<ol>
<li><strong>Obtain continuous derivative bounds.</strong> Read the frequency-vector branch of <code>norm_lst</code> in <a href="../src/My_Vector.jl">My_Vector.jl</a>. It first calls its <code>"LInf"</code> branch, then <code>get_Wk∞_norm_lst</code> in <a href="../src/My_Scalar.jl">My_Scalar.jl</a> and <code>estimate_Wk∞_norm</code> in <a href="../src/Matrix_Func.jl">Matrix_Func.jl</a>. The latter corrects the grid maximum using the frequency bounds and asserts that its denominator is positive. Bernstein's inequality then gives each derivative bound by multiplying the component's L∞ bound by the relevant frequency raised to the derivative order. This particular branch needs no derivative ghost layers.</li>
<li><strong>Compute the quadrature sum.</strong> Follow scalar <code>inner_product</code> → <code>_inner_product_space</code> in <a href="../src/My_Scalar.jl">My_Scalar.jl</a> → <code>newton_cotes_weights</code> in <a href="../src/Matrix_Func.jl">Matrix_Func.jl</a>. These form a tensor product of closed 7-point Newton–Cotes rules, including <code>sin(θ)</code> and the physical measure factor above. Interval arithmetic encloses rounding in this sum.</li>
<li><strong>Enclose the quadrature error.</strong> Return to vector <code>inner_product</code> in <a href="../src/My_Vector.jl">My_Vector.jl</a>. It calls <code>multinomial_sum</code> in <a href="../src/Comb_Bound.jl">Comb_Bound.jl</a> to bound the eighth β and θ derivatives of the complete integrand, including the sine weight and both residual factors. It multiplies these bounds by the respective mesh widths to the eighth power, the constant <code>3/2800</code>, the rectangle area, and the same physical measure factor. The resulting allowance widens the quadrature sum with <code>hull(result - err, result + err; dec=:auto)</code>; here the local <code>err</code> is the quadrature allowance. Omitting <code>W_norms_lst</code> skips this enlargement, so interval samples alone do not certify the integral.</li>
</ol>
<p>For a general inner product of two fields <code>f</code> and <code>g</code>, supply their separate bound tables as <code>W_norms_lst=[Wf, Wg]</code>. The fields must have matching components, scale, and physical grid; the bounds must apply on that domain. The frequency recipe above requires justified frequency ceilings for the actual weighted field. Both grid interval counts must be multiples of six for quadrature; this frequency-bound path also asserts multiples of 100. The current Step 1 residual calculation uses <code>18000 × 9000</code> intervals, satisfying both requirements.</p>
<p>Read <a href="technical_notes/matrix_func.md">the matrix notes</a>, “Composite Error Bound” and “Closing an estimate with frequency bounds”, at the corresponding steps. <code>newton_cotes_with_bound</code> in <a href="../src/Matrix_Func.jl">Matrix_Func.jl</a> is a short one-dimensional illustration returning the sum and error allowance separately; the vector inner-product path implements its own two-dimensional allowance. For the alternative refinement from supplied higher-derivative bounds, follow <code>Linf_to_paras</code> → <code>get_Wk∞_norm_lst</code> → <code>estimate_Wk∞_norm</code> → <code>finite_difference</code> when a notebook uses that path.</p>
<!-- reader-guidance:integral-example:end -->
<!-- reader-guidance:refinement-example:begin -->
<h3 id="derivative-refinement-example">Example: sharpen derivative bounds before integration</h3>
<p>We already have an a priori bound for the residual and its derivatives. We refine it using finite differences to obtain a sharper bound for the error in the final L² calculation.</p>
<p>Follow the residual calculation in <a href="../notebooks/eigenpair_error_analysis.ipynb">eigenpair_error_analysis.ipynb</a>. The initial pass samples the trigonometric extension over the full β-period <code>[0,2π]</code> to obtain the global norm used in Bernstein's inequality. Powers of the frequency then give conservative high-derivative bounds. Refinement sharpens these bounds on the compactified physical domain <code>[0,π/2]</code>, and integration uses the refined bounds to reduce the quadrature error. The same residual is represented throughout these three stages.</p>
<p><strong>Prepare the prior.</strong> With <code>k = 8</code>, the notebook first uses <code>32000 × 4000</code> grid intervals on <code>[0,2π] × [0,π/2]</code>. Its <code>"W10Inf"</code> call produces the initial <code>norm_dict["err"]</code>: an <code>11 × 2</code> table of three-component vectors. The two extra derivative orders supply the seeds for order-8 refinement. The candidate has mixed radial reflection parity, so a maximum on the physical radial quarter interval need not equal the full-period maximum.</p>
<p>The mixed residual uses both coefficient resolutions. In the notebook, <code>M0_UP,M1_UP</code> come from <code>size(UP["u3"].freq_func)</code>, while <code>M0,M1</code> come from <code>size(up["u3"].freq_func)</code>. The same residual ceiling is used in the coarse prior and refinement:</p>
<pre><code class="language-julia">err_freq_lst = dt.([2*(M0_UP+M0)+10, 2*(M1_UP+M1)+10])
</code></pre>
<p><strong>Rebuild the residual for refinement.</strong> Rebuild the complete weighted residual and weighted eigenvector on these padded physical-domain grids:</p>
<pre><code class="language-julia">L0, L1 = Pi/dt(2), Pi/dt(2)
N0, N1 = 36000, 18000
pad = derivative_padding(k) # 10 for k = 8
β_pt = grid(L0, N0, pad)
θ_pt = grid(L1, N1, pad)
</code></pre>
<p>Each component has <code>36021 × 18021</code> samples: <code>36001 × 18001</code> physical points plus ten ghost layers at each end of each axis. Rebuild the operators and both convection terms on these grids before the next excerpt. The prior bounds must cover the region accessed by the finite-difference stencils, including the ghost layers.</p>
<p><strong>Refine orders 0 through 8.</strong> After that residual rebuild, the notebook executes:</p>
<pre><code class="language-julia">paras_lst = Linf_to_paras(norm_dict["err"], k+2, err_freq_lst; ind=1)
norm_dict["err"] = norm_lst(err, "W$(k)Inf"; paras=paras_lst, domain=[L0,L1], padding=pad)
</code></pre>
<p><code>paras_lst</code> is a <code>3 × 2</code> matrix of parameter packages, indexed by component and derivative direction. Supplying this matrix selects finite-difference refinement in <code>norm_lst</code>; supplying a frequency vector selects the direct Bernstein estimate from the preceding example. The updated <code>norm_dict["err"]</code> is a <code>9 × 2</code> table of three-component derivative-bound vectors.</p>
<p><strong>Follow the directional seeds.</strong> The special <code>ind=1</code> branch selects the β-column seeds from the supplied prior. It refines β using explicit higher-derivative bounds and angular frequency correction. It then uses the refined β-order-0 and β-order-2 bounds to construct the θ seeds and transverse mixed-derivative corrections. This avoids applying a β Bernstein estimate on the physical quarter interval. A full-period high-order prior alone would not justify that separate local correction.</p>
<p><strong>Return to integration.</strong> The L² quadrature uses derivative bounds through order 8 to control its remainder. The notebook calls <code>trim_border!(err, pad)</code> and <code>trim_border!(I_up, pad)</code>, leaving the unpadded <code>36000 × 18000</code>-interval grid. It uses these samples and the stored bounds for the residual–eigenvector inner product, the eigenvector norm, and <code>norm_lst(err, "L2"; paras=norm_dict["err"])</code>. Trimming the ghost layers leaves the continuous derivative bounds valid on the integration domain. Step 7, <a href="../notebooks/verify_auxiliary_field_bound.ipynb">verify_auxiliary_field_bound.ipynb</a>, uses the same refinement and quadrature methods to estimate the auxiliary eigenvalue functional defined in the paper.</p>
<p>Read the functions in this order:</p>
<ol>
<li><strong><code>Linf_to_paras</code> in <a href="../src/Matrix_Func.jl">Matrix_Func.jl</a>: identify the seeds.</strong> The argument <code>k+2 = 10</code> selects rows 10 and 11, which store derivative orders 9 and 10. With <code>ind=1</code>, these supply the β seeds; the θ packages are deferred until β refinement supplies the order-0 and order-2 bounds. This function prepares parameters; it does not evaluate derivatives. Build fresh packages for each refinement call because the scalar wrapper mutates the deferred lists.</li>
<li><strong><code>norm_lst</code> in <a href="../src/My_Vector.jl">My_Vector.jl</a>, then scalar <code>get_Wk∞_norm_lst</code> in <a href="../src/My_Scalar.jl">My_Scalar.jl</a>: follow components and grids.</strong> The vector method dispatches each component to the scalar helper. The scalar helper computes the mesh widths from the physical interval counts after subtracting the padding, removes transverse ghost layers for each directional calculation, and transposes the array for θ differentiation. Keep the padding until this calculation finishes.</li>
<li><strong>Array <code>get_Wk∞_norm_lst</code> in <a href="../src/Matrix_Func.jl">Matrix_Func.jl</a>: follow the backward recursion.</strong> Order 8 uses the supplied order-10 bound; order 7 uses the supplied order-9 bound; order 6 uses the newly computed order-8 bound, and so on down to order zero. Two extra derivative orders suffice when their bounds cover the nested stencil domains; <code>derivative_padding(k)</code> supplies the required initial layers for <code>k &lt; 20</code>.</li>
<li><strong><code>estimate_Wk∞_norm</code> → <code>finite_difference</code> in <a href="../src/Matrix_Func.jl">Matrix_Func.jl</a>: inspect one refinement step.</strong> For derivative order m, enlarge the maximum absolute finite-difference estimate using the order-(m+2) bound multiplied by <code>h^2*(m+3)/24</code>, where h is the mesh width. A transverse correction then makes the bound valid between grid points.</li>
</ol>
<p>For the derivation, read <a href="technical_notes/matrix_func.md">the matrix notes</a>, “From stencil centers to the whole interval”, “Two coordinate directions”, “Why refinement proceeds from high order to low order”, and “Ordinary and special parameter preparation”, alongside the corresponding functions. Finite-rank RHS and inverse-residual calculations also use special parameter preparation with analytic or product-based β priors. Each call must satisfy its domain hypotheses. The Step 5 mixed-forcing calculation extracts explicit β priors and uses <code>ind=1</code>; see the <a href="technical_notes/symmetry_and_domains.md#mixed-forcing">mixed-forcing bounds</a>.</p>
<!-- reader-guidance:refinement-example:end -->
<h3>Follow the relevant call path</h3>
<p>For each question, read the caller, identify its inputs and return value, inspect the listed implementation, then return to the example or notebook. Consult only the corresponding technical-note section.</p>
<table>
<thead>
<tr>
<th>Question</th>
<th>Source path to follow</th>
<th>Mathematical explanation</th>
</tr>
</thead>
<tbody>
<tr>
<td>How is a field loaded?</td>
<td><code>from_mat_file</code> → <code>read_matrix</code>, component construction, <code>set_div_free!</code> in <a href="../src/My_Vector.jl">My_Vector.jl</a> and <a href="../src/Matrix_Func.jl">Matrix_Func.jl</a></td>
<td><a href="technical_notes/fourier_basis.md">Fourier notes</a>, “Divergence-Free Reconstruction”</td>
</tr>
<tr>
<td>How are coefficients evaluated?</td>
<td><code>interpolate</code> in <a href="../src/My_Vector.jl">My_Vector.jl</a> → <code>interpolate</code> in <a href="../src/My_Scalar.jl">My_Scalar.jl</a> → <code>trig_func</code> in <a href="../src/My_Fourier.jl">My_Fourier.jl</a></td>
<td><a href="technical_notes/fourier_basis.md">Fourier notes</a>, boundary codes and basis choices</td>
</tr>
<tr>
<td>How is the residual assembled?</td>
<td><code>get_operator_matrix</code> → <code>get_linear</code>, then <code>apply</code>; <code>mul_A_gradB</code> builds convection. Start in <a href="../src/NS_Terms.jl">NS_Terms.jl</a>.</td>
<td><a href="technical_notes/ns_terms.md">Operator notes</a>, physical fields, weights, and linear terms</td>
</tr>
<tr>
<td>How is its norm bounded?</td>
<td><code>norm_lst</code> → <code>inner_product</code> in <a href="../src/My_Vector.jl">My_Vector.jl</a> and <a href="../src/My_Scalar.jl">My_Scalar.jl</a>; inspect quadrature and derivative helpers in <a href="../src/Matrix_Func.jl">Matrix_Func.jl</a></td>
<td><a href="technical_notes/matrix_func.md">Matrix notes</a>, “Composite Error Bound” and derivative refinement</td>
</tr>
</tbody>
</table>

<!-- reader-guidance:source:end -->

## Dependency graph (simplified)

```text
NS_Numerics.jl          <- module entry point; includes all files below
│
├── My_Fourier.jl       <- no src/ dependencies
├── Matrix_Func.jl      <- My_Fourier.jl
├── My_Scalar.jl        <- My_Fourier.jl, Matrix_Func.jl
├── My_Vector.jl        <- My_Scalar.jl (-> My_Fourier.jl, Matrix_Func.jl)
├── My_Operator.jl      <- My_Vector.jl
├── My_Bessel.jl        <- My_Fourier.jl, Matrix_Func.jl
├── Trig_Poly.jl        <- My_Fourier.jl, Matrix_Func.jl
├── Comb_Bound.jl       <- My_Fourier.jl, Matrix_Func.jl, My_Vector.jl
├── My_Assembly.jl      <- My_Bessel.jl, Trig_Poly.jl, Comb_Bound.jl, Matrix_Func.jl
└── NS_Terms.jl         <- My_Operator.jl, My_Assembly.jl, My_Fourier.jl
```

`My_Bessel.jl` evaluates and bounds basis channels. `My_Assembly.jl` combines them into finite-rank matrices, packed bounds, and reconstructed fields.

## Include order

The actual `include` order in `NS_Numerics.jl` is:
`Matrix_Func.jl`, `My_Fourier.jl`, `My_Scalar.jl`, `My_Vector.jl`, `My_Operator.jl`,
`My_Bessel.jl`, `Trig_Poly.jl`, `Comb_Bound.jl`, `My_Assembly.jl`, `NS_Terms.jl`.

`Matrix_Func.jl` is included before `My_Fourier.jl`, where `dt` is defined.
Its function bodies may call `dt` after all files are loaded. Types used in method
signatures must already be defined; `dtype` is set in `NS_Numerics.jl` before the includes.

## Role of each `src/*.jl` file

| File | Role |
|------|------|
| `NS_Numerics.jl` | Module entry point. Sets `dtype` (the single switch between exploratory `Float64` and rigorous `Interval{Float64}` mode), includes all files, and defines the public `export` list. |
| `My_Fourier.jl` | `dtype` helpers (`dt`, `is_zero`, `Pi`, `mysin`, `mycos`); Chebyshev–Legendre conversion matrices; boundary-adapted DFT matrix `Fourier_mat`; divergence-free angular reconstruction `calc_Utheta_divfree`; the radial and angular basis-operator evaluator `trig_func`. |
| `Matrix_Func.jl` | Composite closed 7-point Newton–Cotes quadrature with a rigorous 8th-derivative error bound; finite-difference $W^{k,\infty}$ norm estimation; interval matrix I/O (interval arrays split into `_lo`/`_hi`); LDLᵀ SPD verification; the deflated generalized-eigenvalue certificate for a $K_6$ upper bound; symmetric spectral and PSD-Gram upper-matrix certificates (`verify_symmetric_spectrum`, `verify_gram_spectrum`); symmetric $3\times3$ eigenvalue enclosures. |
| `My_Scalar.jl` | `ScalarFunc` struct (space + frequency representations of one scalar component); Fourier transforms, interpolation, inner products, and boundary-condition handling. |
| `My_Vector.jl` | `VectorFunc` container grouping a variable number of named `ScalarFunc` components (e.g. `u1`, `u2`, `u3`, `p`, or the six entries of `Q`); vector-level operators, $L^2$/$L^\infty$/$W^{k,\infty}$ norms, `.mat` I/O (`from_mat_file`, `save`, `load_grad_U`), and divergence-free enforcement (`set_div_free!`). |
| `My_Operator.jl` | `Operator` and `Operator_matrix` structs encoding separated-variable linear maps on `VectorFunc`s; operator application (`apply`) and eigenvalue computation (`eigen_calc`). |
| `My_Bessel.jl` | Half-integer Bessel evaluation (`my_besselj`); rigorous second-derivative bound (`bessel_W2_bound_interval`); root equations `func_Y`/`func_Z` and local root enclosure on continuous, pole-free brackets (`rigorous_root_interval`); the poloidal Y/X and toroidal Z basis evaluations (`Ur_func`, `eval_Ur`, `eval_Uθ`); per-degree basis derivative bounds (`Ur_bound`, `Uθ_bound`, `Ur_bound_lst`). |
| `Trig_Poly.jl` | Exact trigonometric-polynomial arithmetic and θ-integration (`trig_int_0_pi2`, `tensor_int_basis`, `integrate_θ`, `contract_QT_blocks`, `split_tensor`). |
| `Comb_Bound.jl` | $W^{k,\infty}$ product/composition bounds via the multinomial Leibniz rule; $r = a\tan\beta$ change-of-variable bounds (Faà di Bruno / Bell polynomials); eigenfunction bound combination (`eig_fun_bound`); per-mode refinement and β-chain packing (`refine_channel_bounds`, `beta_bound_chain`). |
| `My_Assembly.jl` | Finite-rank packing (`lst_to_mat`), field reconstruction (`eval_eig_function`), Q multiplication (`apply_Q_to_channel_fields`), and radial Gram-matrix assembly (`integrate_r`), using the basis functions and bounds from `My_Bessel.jl`; $L^2$ inner products and η-weighted energy matrices (`eval_l2_inpd`, `eval_h1_norm`, `get_lap_matrix`; `eta=0.005` in Step 4A). |
| `NS_Terms.jl` | Builds the weighted physical Navier–Stokes operator terms: linear operator (`get_linear`), identity (`get_identity`), gradient and symmetric gradient (`get_gradient`, `get_gradient_sym`), and the nonlinear term (`mul_A_gradB`); dispatched via `get_operator_matrix`. |

## Major function groups and where they live

The `export` block of `NS_Numerics.jl` is the authoritative index; grouped by defining file:

- **`My_Fourier.jl`** — `dt`, `is_zero`, `Pi`, `mysin`, `mycos`, `double_fact`, `gamma_ratio`, `che2len`, `len2che`, `conversion_mat`, `Fourier_mat`, `div_mat`, `calc_Utheta_divfree`, `trig_func`.
- **`Matrix_Func.jl`** — `newton_cotes_weights`, `newton_cotes_with_bound`, `finite_difference`, `derivative_padding`, `Linf_to_paras`, `get_Wk∞_norm_lst`, `estimate_Wk∞_norm`, `save_matrix`, `interval_matrix`, `read_matrix`, `verify_spd_by_ldlt`, `verify_general_eig`, `verify_symmetric_spectrum`, `verify_gram_spectrum`, `eig_sym3`, `vec_to_mat`.
- **`My_Scalar.jl`** — `ScalarFunc`, `set_func!`, `set_bdry!`, `set_conv_mat!`, `Fourier_transform`, `interpolate`, `transform!`, `inner_product`, `get_Wk∞_norm_lst`, `trim_border!`.
- **`My_Vector.jl`** — `VectorFunc`, `from_mat_file`, `load_grad_U`, `set_div_free!`, `inner_product`, `norm_lst`, `save`, `BDRY_LST_SET`.
- **`My_Operator.jl`** — `Operator`, `Operator_matrix`, `set_T!`, `set_L1!`, `apply`, `change_variable!`, `add_operator!`, `apply_convmat_right!`, `eigen_calc`.
- **`My_Bessel.jl`** — `my_besselj`, `bessel_W2_bound_interval`, `func_Z`, `func_Y`, `rsign`, `sign_change`, `rigorous_root_interval`, `verify_root`, `mat_to_lst`, `preprocess_para!`, `load_besselj_zeros`, `Ur_func`, `eval_Ur`, `eval_Uθ`, `Ur_bound`, `Uθ_bound`, `Ur_bound_lst`.
- **`Trig_Poly.jl`** — `TrigPoly`, `trig_mul`, `trig_int_0_pi2`, `trig_basis_lists`, `sph_basis_lists`, `tensor_int_basis`, `contract_QT_blocks`, `split_tensor`, `integrate_θ`, `int_θ_bound`.
- **`Comb_Bound.jl`** — `multinomial_sum`, `poly_deriv`, `poly_add`, `poly_mul_x`, `poly_mul_1_plus_x2`, `poly_abs_sum_bound`, `beta_all_deriv_Linf_bounds`, `tan_over_cos_deriv_Linf_bounds`, `r_to_beta`, `eig_fun_bound`, `beta_bound_chain`, `refine_channel_bounds`.
- **`My_Assembly.jl`** — `integrate_r`, `get_lap_matrix`, `Ur_sq_Wk∞_bound_lists`, `eval_l2_inpd`, `eval_h1_norm`, `lst_to_mat`, `eval_eig_function`, `apply_Q_to_channel_fields`.
- **`NS_Terms.jl`** — `build_operator`, `get_linear`, `get_identity`, `get_gradient_sym`, `get_gradient`, `get_operator_matrix`, `mul_A_gradB_local`, `mul_A_gradB`.

The unused plotting helpers and PyPlot dependency have been removed; the
verification module does not require a plotting backend.
