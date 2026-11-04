

# ScalarFunc and VectorFunc: from coefficients to verified bounds

The notebooks pass fields between loaders, operators, and norm routines using
`ScalarFunc` and `VectorFunc`. Understanding what each object currently stores
explains why a calculation first works with coefficients and later with samples.
This note connects [My_Scalar.jl](../../src/My_Scalar.jl) and
[My_Vector.jl](../../src/My_Vector.jl) to that calculation. Start with the
[source guide](../src.md) for the overall reading path.

## 1. One scalar field, two representations

A `ScalarFunc` represents one scalar component. In the profile and eigenpair
inputs, it is a finite trigonometric function of the compactified radius $\beta$
and angle $\theta$; the angular coefficients may instead use an equivalent
spherical-harmonic basis. Once a residual or another derived field has been
evaluated, the same container can hold just its samples.

| Field | Meaning in the calculation |
|---|---|
| `name` | A component label, such as `"u1"` or `"p"`. |
| `freq_func` | Coefficient array: rows index radial modes and columns index angular modes. |
| `space_func` | Sample array: rows index radial grid points and columns index angular grid points. |
| `bdry` | The boundary codes selecting the radial and angular bases. |
| `conv_mat` | Optional angular conversion used when evaluating the stored coefficients in the trigonometric basis. |
| `scl_fac` | Scale carried into physical operators and the measure factor in inner products. |

Either array can be `nothing`. Their dimensions need not agree: a small
coefficient array can be evaluated on a much larger grid. The constructor
creates the object and its metadata; it does not evaluate the function.

**The two representations are stored independently.** `set_func!(f, "frequency", C)`
assigns `C` to `freq_func` without refreshing `space_func`; setting samples likewise
does not recompute coefficients. The setter stores the supplied array by reference.
Consequently, the reader must identify which representation a later routine reads.
Existing samples can describe the field before a coefficient update.

The object also stores no grid coordinates or tag saying “physical field” or
“weighted residual.” For profile and eigenpair coefficients, the conventions are

$$
R=\kappa\tan\beta,\qquad \kappa=\texttt{scl\_fac},\qquad
U_{\mathrm{phys}}=\cos\beta\,\widehat U,
\qquad P_{\mathrm{phys}}=\cos\beta\,\widehat P.
$$

Thus $0\leq\beta<\pi/2$ covers the whole physical radius, and the endpoint
$\beta=\pi/2$ represents infinity. Default interpolation evaluates the stored
compactified unknown, without inserting the factor $\cos\beta$. Other objects,
such as the externally supplied $Q$, have their own radial coordinate convention;
the caller determines what the grid means. See the
[data dictionary](../data.md) and
[physical-field conventions](ns_terms.md#2-compactified-coordinate-and-physical-fields).

## 2. How the coefficients become values

For `bdry = [bdry_r, bdry_θ]`, each two-character code specifies the conditions
at $0$ and $\pi/2$: `0` is Dirichlet and `1` is Neumann. The
[Fourier note, Section 2](fourier_basis.md#2-boundary-condition-codes-and-basis-choices)
gives the corresponding sine and cosine formulas and their spherical-harmonic
equivalents.

When `bdry_r` is a two-entry list, **its first entry applies to the first
coefficient column and its second entry to the remaining columns**. For the odd
velocity block, `bdry_r = ["10", "00"]` lets the $l=1$ mode use a different
radial basis from $l=3,5,\ldots$. This assignment is made before angular conversion.

To see the evaluation order, let $C$ denote `freq_func`, let $B_\beta$ and
$B_\theta$ contain trigonometric basis values at the requested points, and let
$K$ denote `conv_mat`. With a common radial basis, `interpolate` computes

$$
S=B_\beta C K B_\theta^T.
$$

Here $K$ is omitted if `conv_mat` is unset. With separate radial conditions for
the first column, the intermediate radial evaluation is instead

$$
T=\bigl[B_{\beta,1}C_{:,1}\ \ B_{\beta,2}C_{:,2:\mathrm{end}}\bigr],
\qquad S=T K B_\theta^T.
$$

The optional matrix $K$ maps spherical-harmonic coefficient rows to
trigonometric coefficient rows; `set_conv_mat!` obtains it using
`conversion_mat(..., 1)`. Attaching $K$ does not rewrite `freq_func`.
For `u1` and `u2`, it makes the stored angular modes available to trigonometric
evaluation. The basis-conversion formulas and the $-S_l'$ convention belong in
the [Fourier note](fourier_basis.md).

Read scalar `interpolate` for this matrix sequence. Its default `keys=["I","I"]`
evaluates the stored function; other keys evaluate the selected basis operators.
With `pt_flag=true`, it stops after radial evaluation and angular conversion:
the result still has angular coefficients, not angular samples.

## 3. Several components with a definite order

A `VectorFunc` groups named `ScalarFunc` objects. Its `keys` list determines
component order; `dict` provides access by name. `v[1]` means `v[v.keys[1]]`,
and both return the stored component itself.

| Object | Stored components | Interpretation |
|---|---|---|
| `UP` or `up` | `u1`, `u2`, `u3`, `p` | Three compactified velocity components and pressure. |
| `err` | `e1`, `e2`, `e3` | Three weighted residual components, usually stored as samples. |
| $\nabla U$ or $\nabla v$ (e.g. `grad_up`) | `A11`, `A12`, `A13`, `A21`, `A22`, `A23`, `A31`, `A32`, `A33` | Nine sampled components of the physical gradient in the spherical frame; the first index is the differentiation direction and the second is the output component. |
| `Q` or `grad_sym_UP` | `A11`, `A12`, `A13`, `A22`, `A23`, `A33` | Six independent entries of a symmetric matrix field: the auxiliary tensor $Q$ or the physical symmetric gradient $(\nabla U+\nabla U^T)/2$. |

The type itself does not assign a geometric meaning to these lists. In
particular, it does not distinguish a velocity vector from a tensor field.
An `Operator_matrix` consumes its input components in `keys` order, and a norm
routine returns component bounds in that same order.

`bdry_index` selects one of the four-component configurations in `BDRY_LST_SET`.
Those configurations are assigned positionally in the order `u1,u2,u3,p`.
Containers such as $Q$ instead receive an explicit boundary list; their
`bdry_index` can be `nothing` even though each component has boundary conditions.

## 4. Follow a profile into its residual

The following excerpts use the paths seen from the project's `notebooks/`
directory and assume `NS_Numerics` and the notebook's grids are already prepared.
Follow [selfsimilar_profile_verification.ipynb](../../notebooks/selfsimilar_profile_verification.ipynb)
alongside these steps.

**Load the coefficient field.**

```julia
UP = from_mat_file("../data/external/UP.mat", 2; domain="frequency")
```

The loader reads `u1/u2/u3/p`, the scale, and the saved grid shape through
`read_matrix`. The external data are precomputed floating-point values;
`read_matrix` converts them through `dt` into the active numeric type. In
interval mode, subsequent arithmetic encloses operations on those supplied
coefficients. The residual estimates then quantify how well the resulting
field satisfies the equation.

`from_mat_file` attaches the conversion matrices for `u1` and `u2`, evaluates
the loaded coefficients on the saved grid, and then calls `set_div_free!`
unless `divfree=true`. That last step reconstructs `u2.freq_func` from
`u1.freq_func` so the represented physical velocity is divergence free; it does
not refresh the previously computed `u2.space_func`. Read `set_div_free!` and
then [Fourier note, Section 5](fourier_basis.md#5-divergence-free-reconstruction)
for the modewise relation. Files containing an eigenvalue return `(up, lambda)`.

**Evaluate the current coefficients when samples are needed.**

```julia
UP_samples = interpolate(UP, β_pt, θ_pt)
```

Vector `interpolate` evaluates each component and returns a new `VectorFunc`
with `space_func` set and `freq_func` unset. It leaves `UP` unchanged. This
explicit evaluation uses the reconstructed coefficients. Alternatively,
`transform!(UP, "frequency"; shape=(Nβ, Nθ))` refreshes `UP.space_func` on a
uniform $[0,\pi/2]^2$ grid with `Nβ × Nθ` points. Here `domain="frequency"`
names the **source** representation; `domain="space"` requests the reverse
transform. Explicit `interpolate` is the path to follow for extended grids.

**Apply operators to coefficients.**

```julia
lin_mat = get_operator_matrix(UP, β_pt, θ_pt, "linear")
err = mul_A_gradB(UP, UP, β_pt, θ_pt) - apply(lin_mat, UP)
```

`lin_mat` is a $3\times4$ `Operator_matrix`: it maps four input component
coefficient arrays to three velocity-equation sample arrays. `apply` reads
`freq_func`; `mul_A_gradB` also evaluates its inputs from coefficients. Both
therefore use the reconstructed field. Their outputs share the requested grid
and component order, so subtraction forms the assembled residual before its
norm is bounded. Each `err[i].space_func` has size
`length(β_pt) × length(θ_pt)`; its coefficients are unset.

This change of representation has a purpose: differential operators act on
the finite basis, whereas quadrature and finite differences act on sampled
values. Sampling the assembled residual preserves cancellation between its
terms. Continue with [Operator and Operator_matrix](ns_terms.md#7-operator-and-operator_matrix)
and the linear and convection sections for the weights and signs.

For comparison, `load_grad_U("../data/external/approx_Q.mat")` constructs the
six-component coefficient container for $Q$ with explicit boundary conditions.
It does not evaluate samples. Read its return statement, then the notebook's
`interpolate` call, to follow how the $Q$ samples are obtained.

## 5. From samples to a bound

All three `norm_lst` modes read samples. Let $d=\texttt{length(v.keys)}$ and let
`W = norm_lst(v, "W$(k)Inf"; ...)`.

| Mode | Returned object | Meaning |
|---|---|---|
| `"LInf"` | Length-$d$ vector | One supremum estimate per stored component. |
| `"W8Inf"` (generally `"W$(k)Inf"`) | $(k+1)\times2$ table of length-$d$ vectors | Bounds for pure derivatives of orders $0,\ldots,k$ in each coordinate direction. |
| `"L2"` | One scalar | Square root of the sum of the component inner products. |

For the profile/residual grids, `W[m+1, 1][i]` bounds
$\lVert\partial_\beta^m v_i\rVert_\infty$ and `W[m+1, 2][i]` bounds
$\lVert\partial_\theta^m v_i\rVert_\infty$. These entries are derivative upper bounds,
not a single combined Sobolev norm; mixed derivatives are not additional rows.
For a three-component residual and `k=8`, the result is a `9 × 2` table of
three-component vectors.

**Choose how continuous derivative bounds are obtained.** A two-entry frequency
vector of type `Vector{dtype}` selects a corrected sample maximum followed by
Bernstein bounds. It requires justified frequency ceilings and a sampling domain
covering the norm used in that inequality. This branch and `"LInf"` use
unpadded samples with `padding=0`. A $d\times2$ matrix of packages from
`Linf_to_paras` instead selects finite-difference refinement from prior
higher-derivative bounds. For $0<k<20$, retain at least
`derivative_padding(k) = q*(q+1)÷2` ghost layers, where `q = cld(k,2)`, at
each end of each coordinate. Pass the actual layer count as `padding=pad`;
it defaults to zero. The scalar wrapper uses `size(A,j)-1-2pad` for the
physical interval count and removes `pad` transverse layers before refining
each direction. Seeds must cover every stencil, including any extra domain
being certified. The vector wrapper collects the results into `W`. Special-mode packages are updated during this calculation;
prepare fresh packages for another call.

The scalar wrapper identifies recursion from the first slot of each directional
package: `"Wk∞"` supplies explicit seeds, and `"L∞_freq"` is converted into such
seeds after β refinement. Both select `derivative_padding(k)`. A primary
`"freq"` selects the independent finite-difference correction, which needs only
`cld(k,2)` layers for stencil coverage. A frequency in the second, transverse
slot does not change this choice. For `k=8`, these requirements are ten and four
layers respectively; the unpadded vector-frequency shortcut described above is
a separate path. See [how correction tags select padding](matrix_func.md#how-correction-tags-select-padding)
for the exact check and the meaning of `primary !== nothing`.

The prior is refined because a smaller derivative bound can reduce the
quadrature error in the final integral. Read the
[source guide's worked examples](../src.md#derivative-refinement-example) for
the notebook sequence and the [matrix note](matrix_func.md#finite-differences-and-sharper-derivative-bounds)
for the justification, directional seeds, and grid requirements.

**Identify the field being integrated.** For a physical vector field $F$, the
profile and eigenpair operators prepare samples of
$f(\beta,\theta)=w(\beta)F(\kappa\tan\beta,\theta)$ with
$w(\beta)=\sin\beta/\cos^2\beta$. Under the project's axisymmetry and
half-meridian symmetry conventions,

$$
\|F\|_{L^2(\mathbb R^3)}^2
=4\pi\kappa^3\int_0^{\pi/2}\int_0^{\pi/2}
\sum_{i=1}^{3}|f_i(\beta,\theta)|^2\sin\theta\,d\theta\,d\beta.
$$

`inner_product` supplies $\sin\theta$ and $4\pi\kappa^3$; the radial weight
must already be in the input samples. Thus the norm of raw `UP` samples is
not automatically the physical velocity norm. The routine sums **all stored
components**: it includes pressure if pressure is present, and does not double
the off-diagonal contributions when given six symmetric-tensor entries.

**Include the quadrature error.** For weighted fields `f` and `g` sampled on
the same unpadded grid, with matching components and scale, use

```julia
J = inner_product(f, g; W_norms_lst=[Wf, Wg], domain=D)
E = norm_lst(f, "L2"; paras=Wf, domain=D)
```

Here `D` gives the two interval lengths, and `Wf` and `Wg` bound the sampled
fields' derivatives through order 8 on that domain. Each grid interval count
must be a multiple of six for the quadrature rule. In interval mode, the vector
inner product enlarges the quadrature sum by an allowance computed from these
bounds. The scalar inner product supplies the quadrature sum; omitting the
vector bound tables leaves out the integration error allowance. See
[the weighted-inner-product derivation](matrix_func.md#from-one-dimensional-quadrature-to-a-weighted-inner-product)
for the measure factor and the two-dimensional error argument.

**Why `_inner_product_space` is a separate helper.** The scalar wrapper extracts
and checks the sample arrays, then passes them to the quadrature kernel. Since
`space_func` is an abstractly typed field, this function boundary lets Julia
specialize the loop on the actual concrete matrix types. This is a *function
barrier*: it can reduce runtime dispatch and temporary allocations in the large
sum. The helper evaluates the same quadrature formula without copying the inputs.

## 6. Read the implementation alongside the calculation

1. Read the two type definitions and scalar `set_func!` to distinguish metadata,
   coefficient arrays, and sample arrays.
2. Read `from_mat_file` → `set_div_free!` → scalar `interpolate` to follow the
   profile field. Consult the Fourier note for the basis and divergence relation.
3. Read `apply(::Operator_matrix, ::VectorFunc)` to see how coefficient inputs
   become sampled outputs; then return to the residual construction.
4. Read vector `norm_lst` → scalar `get_Wk∞_norm_lst` for derivative bounds,
   and vector `inner_product` → scalar `inner_product` → `_inner_product_space`
   for integration.
   Consult the matrix note when an estimate needs justification.

