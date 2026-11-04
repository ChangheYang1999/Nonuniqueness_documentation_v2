# `NS_Terms.jl` Mathematical and Code Guide

> This is a detailed technical note. For the main reader path, see the [source guide](../src.md).

## 1. Overview


`NS_Terms.jl` constructs the linear and nonlinear terms used in the profile
and eigenpair residuals, together with physical gradients for pointwise bounds.
The operators map spectral coefficients of compactified unknowns to grid
samples. Residual terms carry a radial weight for physical $L^2$ integration.
The symbolic derivation in [NS spherical.nb](<NS spherical.nb>) leads to
`trig_func` factors, which are assembled into `Operator_matrix` objects.


Reader roadmap:

1. Read the stored-vs-physical field convention and the Jacobian weight first.
2. Read `Cos`, `Tan`, and `Scale`, because these keys encode the compactified
   physical field and the weighted self-similar drift.
3. Read the field-container introduction linked in Section 7,
   then `Operator`, `Operator_matrix`, and `build_operator` to follow how
   coefficient arrays become term samples.
4. Read `get_linear` and `get_linear(...; mode="transpose")` for the sign and
   transpose-branch conventions.
5. Read the gradient sections for $\nabla U_{\mathrm{phys}}$,
   $\operatorname{sym}\nabla U_{\mathrm{phys}}$, and $A\cdot\nabla B$.

## 2. Compactified Coordinate and Physical Fields


Let $\kappa=\texttt{scl\_fac}>0$, $x$ be the physical position, and $y=x/\kappa$
the scaled position. Their radii are related by

$$
\rho=|y|=\tan\beta,\qquad R=|x|=\kappa\tan\beta.
$$

The Julia grid argument named `r` is $\beta$. The interval
$0\leq\beta<\pi/2$ covers the whole physical radial domain; $\beta=\pi/2$
represents infinity. In this note, fields written as functions of $(\beta,\theta)$
are evaluated at the corresponding physical position. Unless a subscript says
otherwise, $\nabla$ and $\Delta$ denote physical derivatives. Thus

$$
\nabla=\kappa^{-1}\nabla_y,\qquad
\Delta=\kappa^{-2}\Delta_y,\qquad
R\partial_R=\rho\partial_\rho.
$$

The Julia coefficients `inv(scl_fac)` and `scl_fac^(-2)` supply these factors
when the operator keys evaluate derivatives in the scaled coordinates.


The stored velocity and pressure coefficients represent compactified unknowns
$\widehat U$ and $\widehat P$. The physical fields are recovered by multiplying
by $\cos\beta$:

$$
U_{\mathrm{phys}}=\cos\beta\,\widehat U,
\qquad
P_{\mathrm{phys}}=\cos\beta\,\widehat P.
$$

This convention is why the keys `"Cos"`, `"Tan"`, `"Scale"`, `"Scale_T"`, and
the `"_w"` gradient suffix appear throughout `NS_Terms.jl`.

## 3. The Radial Jacobian Weight

The weight

$$
w(\beta)=\frac{\sin\beta}{\cos^2\beta}
$$

is the square root of the radial Jacobian in the scaled radius $\rho$.
The additional physical scale is kept outside the integral as $\kappa^3$.

Since

$$
\rho=\tan\beta,
\qquad
d\rho=\sec^2\beta\,d\beta,
$$

the radial part of the three-dimensional volume element satisfies

$$
\rho^2\,d\rho
=
\tan^2\beta\,\sec^2\beta\,d\beta
=
\left(\frac{\sin\beta}{\cos^2\beta}\right)^2d\beta.
$$


For a physical field $F$, write
$\widetilde F(\beta,\theta)=F(\kappa\tan\beta,\theta)$.
For the axisymmetric fields with the equatorial parity used here,

$$
\|F\|_{L^2(\mathbb R^3)}^2
=4\pi\kappa^3\int_0^{\pi/2}\!\int_0^{\pi/2}
 |w(\beta)\widetilde F(\beta,\theta)|^2\sin\theta\,d\theta\,d\beta.
$$

The factor $4\pi$ combines azimuthal integration with the reflected half
meridian. `inner_product` supplies $\sin\theta$ and $4\pi\kappa^3$; its input
samples must already contain $w$ when estimating a physical $L^2$ norm.


The primary role of $w$ in this file is physical $L^2$ weighting. The residual
operators are assembled in weighted physical form because the verification
ultimately estimates physical norms using the $\beta$ grid.

## 4. Why `"Cos"` and `"Tan"` Appear

Because

$$
U_{\mathrm{phys}}=\cos\beta\,\widehat U,
$$

the key `"Cos"` maps the stored unknown $\widehat U$ to the physical field
$U_{\mathrm{phys}}$.

For weighted physical $L^2$ quantities,

$$
w(\beta)U_{\mathrm{phys}}
=
\frac{\sin\beta}{\cos^2\beta}\cdot\cos\beta\,\widehat U
=
\tan\beta\,\widehat U.
$$

Therefore `"Tan"` maps the stored unknown $\widehat U$ to the weighted physical
field $wU_{\mathrm{phys}}$.

In `get_identity`:

* `weight_flag=false` uses `"Cos"` because it represents the physical identity
  $U_{\mathrm{phys}}$;
* `weight_flag=true` uses `"Tan"` because it represents the weighted physical
  identity $wU_{\mathrm{phys}}$, suitable for physical $L^2$ integrals in the
  $\beta$ coordinate.

## 5. The `"Scale"` Key

The key `"Scale"` represents the weighted identity-plus-drift contribution for a
stored radial factor $f=\widehat U$.

Start from

$$
U_{\mathrm{phys}}=\cos\beta\,f.
$$

The self-similar identity-plus-drift part is

$$
\frac12 U_{\mathrm{phys}}
+
\frac12\rho\partial_\rho U_{\mathrm{phys}}.
$$

Using

$$
\rho=\tan\beta,
\qquad
\partial_\rho=\cos^2\beta\,\partial_\beta,
$$

we have

$$
\rho\partial_\rho
=
\tan\beta\,\cos^2\beta\,\partial_\beta
=
\sin\beta\cos\beta\,\partial_\beta.
$$

Multiplying the identity-plus-drift expression by the $L^2$ weight $w(\beta)$
gives

$$
w(\beta)
\left[
\frac12 U_{\mathrm{phys}}
+
\frac12\rho\partial_\rho U_{\mathrm{phys}}
\right]
=
\frac{\sin\beta}{2}
\left(
\cos\beta\,f+\sin\beta\,f'
\right).
$$

This is the formula implemented by the `"Scale"` key in `trig_func`.

There is no additional $\kappa$ factor because
$R\partial_R=\rho\partial_\rho$.

## 6. Reading the symbolic notebook


[NS spherical.nb](<NS spherical.nb>) derives the spherical-coordinate formulas
at unit radial scale. Its initial `r` is the radial coordinate before
compactification. Read these transformations in order:

1. `Is`, `Id`, and `Ip` contain $r$ times the drift, Laplacian, and pressure
   gradient; `Ig` contains $r$ times the vector gradient. `Igp`, `Igs`, and
   `Idv` are the unweighted gradient, symmetric gradient, and divergence.
2. `DR[U,1]` attaches $(1+r^2)^{-1/2}$, which becomes $\cos\beta$.
   `MyChange1[ArcTan,...]` reparameterizes the radial unknowns, including
   the chain rule for their derivatives.
3. `MyChange2[Tan,...]` substitutes $r=\tan\beta$ and multiplies by
   $\sqrt{dr/d\beta}=\sec\beta$. Combined with the existing $r$, this
   supplies $w$. `MyChange3` makes the substitution without this weight.

The resulting expressions are the useful comparison points:

| Notebook expression | Julia-side quantity |
| --- | --- |
| `Is3` | Weighted drift; `"Scale"` |
| `Id3` | Weighted vector Laplacian; `"Laplacian_*"` keys |
| `Ip3` | Weighted pressure gradient; `"D_p_r"` and angular `"D"` |
| `Igp3` | Physical gradient; `get_gradient` |
| `Igs3` | Physical symmetric gradient; `get_gradient_sym` |
| `Ig3` | Intended nonlinear gradient factor, $\tan\beta\,\nabla U$ |
| `Idv3` | First-mode incompressibility relation used in Section 8 |

In the saved notebook, `Ig3` still multiplies by `Cos[r]` after substitution;
its intended factor is `Cos[β]`. The final first-mode replacement rules also
use functions at `r`, whereas `Id3` already contains functions at `β`, so those
rules do not perform the intended cancellation as written. Read those cells
with the explicit relation in Section 8; their saved outputs are not the final
regularized formulas. The Julia first-mode branches implement the cancellation.


## 7. `Operator` and `Operator_matrix`


First read [ScalarFunc and VectorFunc](scalar_vector.md) for the meaning of
the input fields, coefficient versus sample arrays, and component order.
The operators below act on those coefficient arrays and produce new samples.

Both types are defined in [My_Operator.jl](../../src/My_Operator.jl).
They describe how coefficient arrays are mapped to grid values in the
calculations below.

**`Operator`: one scalar component.** Let $C$ be a coefficient array, with
radial modes in rows and angular modes in columns. With its optional fields
unset, `Operator(L, R)` acts by

$$
\mathcal O(C)=LCR.
$$

If $C$ has size $n_\beta\times n_\theta$, then $L$ has size
$N_\beta\times n_\beta$ and $R$ has size $n_\theta\times N_\theta$;
the result contains $N_\beta\times N_\theta$ grid values. This keeps the radial
and angular actions separate without assembling a large matrix acting on a
flattened coefficient array. Two optional fields extend this action:

* `T` multiplies the result pointwise: $T\odot(LCR)$, where $\odot$ means
  entrywise multiplication. It can supply a spatially varying coefficient.
* `L1` replaces the first column of `L*C` by `L1*C[:,1]` before multiplication
  by `R` and the optional pointwise multiplication by `T`. This allows the first spherical mode
  to use a different radial formula (Section 8).

**`Operator_matrix`: several components.** `Operator_matrix(M, N)` has
$M$ output rows and $N$ input columns. Each entry `op_lst[i,j]` holds a
**list of operators**, whose contributions are added. Writing this list as
$\mathcal A_{ij}$ and the input component arrays as $C_j$, its action is

$$
Y_i=\sum_{j=1}^{N}\sum_{\mathcal O\in\mathcal A_{ij}}\mathcal O(C_j).
$$

An empty list contributes zero. Lists allow several differential terms to
contribute to the same input/output pair. For example, `get_linear` builds a
$3\times4$ operator matrix: the input columns are `u1`, `u2`, `u3`, and `p`,
while the output rows are the three velocity-equation components. Each output
is itself a full grid array; $3\times4$ counts components, not grid points.

`apply(op_mat, vec)` reads each input component's `freq_func` in `vec.keys`
order and returns a `VectorFunc` whose `space_func` arrays contain the sums
$Y_i$. The output names are supplied by `new_keys`, defaulting to
`["e1", "e2", "e3"]`. Read the two type definitions and their `apply` methods
before following their construction below.


### `build_operator`: One Separated Block

Every block assembled by `NS_Terms.jl` has the separated-variable structure

```julia
op = Operator(U_left, U_right')
```

Here:

* `U_left` is the sampled compactified radial matrix. It is assembled as a
  linear combination of `trig_func(M0, r, key_r, bd_r)` with scalar coefficients
  from `key_r_lst`.
* `U_right` is the sampled angular matrix from
  `trig_func(M1, theta, key_theta, bd_theta)`, using the Greek identifiers in
  the Julia source.
* `Operator(U_left, U_right')` applies the separated block by left and right
  matrix multiplication.

If the radial boundary descriptor is `[bd_r1, bd_r]`, `build_operator` also
constructs a special first-column radial matrix `U_left1` and installs it with
`set_L1!`.


Here $L=B_\beta$ and $R=B_\theta^{\mathsf T}$ are the radial and angular
evaluation matrices in the action above. Section 15 explains how angular
basis conversion is included for the stored coefficients.


## 8. First-mode boundary conditions and cancellation


For the odd field (`bdry_index=1`), the radial descriptor of `u1` and `u2`
is `["10", "00"]`:
`"10"` applies to the first $l=1$ spherical mode and `"00"` to the remaining
odd degrees $l=3,5,\ldots$. The first mode may be nonzero at the origin.

To see why its operator formulas also differ, use the notebook's angular convention:

$$
\widehat U_r=f(\beta)\cos\theta,\qquad
\widehat U_\theta=-g(\beta)\sin\theta,\qquad
\widehat U_\phi=0.
$$

The divergence-free condition gives

$$
g=\frac{(3+\cos2\beta)f+\sin2\beta\,f'}4,
\qquad
\frac{f-g}{\sin\beta}
=\frac{\sin\beta\,f-\cos\beta\,f'}2,
$$

where primes denote $\beta$ derivatives. In particular, $g(0)=f(0)$, and the
apparently singular quotient on the left has the regular expression on the
right. Substitution into the coupled Laplacian terms gives the
`Laplacian_r_odd`, `Laplacian_θ_odd`, and `Laplacian_inter_odd` formulas.
`get_gradient` similarly combines first-mode contributions before evaluation.
These formulas use the divergence-free relation, not the boundary codes alone.

The Julia angular basis for `u2` is $-S_1'=+\sin\theta$, so its first-column
radial function is $-g$ in this notation. `L1` applies the appropriate radial
formula to that spherical column before angular conversion and evaluation;
the other columns retain the ordinary formula.


## 9. `get_linear`: Weighted Physical Linear Terms


Write $U$ and $P$ for the reconstructed physical fields from Section 2, and
let $D=\tfrac12 I+\tfrac12 x\cdot\nabla$. `get_linear` evaluates

$$
w\,\mathcal B(U,P),\qquad
\mathcal B(U,P)=DU+\Delta U+\nabla P
=DU+\kappa^{-2}\Delta_y U+\kappa^{-1}\nabla_y P.
$$

The profile and eigenpair notebooks assemble the weighted residuals

$$
E_U=w\bigl[(U\cdot\nabla)U-\mathcal B(U,P)\bigr],
$$

$$
E_v=w\bigl[(U\cdot\nabla)v+(v\cdot\nabla)U
             -\mathcal B(v,q)-\lambda v\bigr],
$$

where $v,q$ are the reconstructed physical perturbation velocity and pressure.
The corresponding core expressions are:

<pre><code>err = mul_A_gradB(UP, UP, β_pt, θ_pt) - apply(lin_mat, UP)

err = U_grad_u + u_grad_U - apply(lin_mat, up) - I_up * lbd</code></pre>

Here `lin_mat` is built for the corresponding input (`UP` or `up`), and
`I_up` is the weighted $v$. The pressure contribution in these residuals is
$-\nabla P$ or $-\nabla q$. To write the equation with $+\nabla p_{\rm NS}$,
identify $p_{\rm NS}=-P$ (and likewise for $q$); `from_mat_file` loads the
stored pressure without changing its sign.

With `mode="pressure"`, `get_linear` constructs only $w\nabla P$ and returns
before building the velocity blocks. The matrix remains `3 x 4`, with only
entries `(1,4)` and `(2,4)` nonempty; the azimuthal pressure derivative is zero.
Thus `apply(get_operator_matrix(w, β, θ, "pressure"), w)` extracts the weighted
pressure gradient directly from the combined velocity/pressure container.

After eliminating pressure, let
$\mathcal L_U v=-Dv-\Delta v+\Pi[(U\cdot\nabla)v+(v\cdot\nabla)U]$,
where $\Pi$ is the Leray projection onto divergence-free fields. The
verification uses $\mathcal L_Uv=\lambda v$ with $\lambda\approx-0.113$,
as does the paper's numerical-results section. The corresponding evolution
generator is $-\mathcal L_U$, whose eigenvalue is $-\lambda>0$.


## 10. `get_linear(...; mode="transpose")` and the projected quantity


On physical $L^2(\mathbb R^3)$, integration by parts for sufficiently decaying
fields gives $(x\cdot\nabla)^*=-x\cdot\nabla-3I$. Consequently,

$$
D^*=-D-\tfrac12 I.
$$

`Scale_T` implements $-D$ rather than $D^*$. The branch retains diffusion,
omits pressure blocks, and therefore evaluates $w(\Delta-D)v$.
For a divergence-free background $U$, combining it with the
transport terms gives

$$
H(v)=Dv-\Delta v-(U\cdot\nabla)v+v\cdot\nabla^T U.
$$

Here the last term has components $\sum_j v_jG_{ij}(U)$, with the gradient
indexing defined in Section 12. Before the Leray projection, the formal adjoint
of $\mathcal L_U$ is represented by $H(v)+\tfrac12v$; pressure gradients
vanish when paired with divergence-free test fields.

The quantity needed in the paper is projected perpendicular to $v$. For
$P_vF=F-\langle F,v\rangle v/\|v\|_2^2$,

$$
P_vH(v)=P_v\bigl(H(v)-\tfrac12v\bigr),\qquad
\|P_vH(v)\|_2\leq\|H(v)\|_2.
$$

The expression $H(v)-\tfrac12v$ is the paper's
$\tfrac12x\cdot\nabla v-\Delta v-(U\cdot\nabla)v+v\cdot\nabla^TU$.
The source $h_0=H(v)-\tfrac12v$ is the paper's
`eq:eigenvalue-functional-ibp` source. Step 7,
[verify_auxiliary_field_bound.ipynb](../../notebooks/verify_auxiliary_field_bound.ipynb),
bounds the dual weighted-energy norm of $(I+G_v^*)h_0$ on divergence-free
fields perpendicular to v, as in `eq:dual-riesz-upper-bound`. The operator
constructor supplies H without performing the projection; Step 7 includes
the finite-rank source and trial-field projection.


## 11. `get_identity`


`get_identity` constructs the $3\times4$ block matrix for the $\lambda v$
term in Section 9. Only the three diagonal velocity blocks are populated;
pressure contributes nothing. Section 4 explains why `weight_flag=true`
returns $wU$ using `"Tan"`, while `false` returns $U$ using `"Cos"`.


## 12. `get_gradient`: Full Physical Gradient

`get_gradient` constructs a `9 x 4` operator matrix representing all nine
components of $\nabla B_{\mathrm{phys}}$ acting on the stored unknown
$\widehat B$. The rows are later named

```julia
["A11", "A12", "A13", "A21", "A22", "A23", "A31", "A32", "A33"]
```

The contraction convention used by `mul_A_gradB_local` is

$$
(A\cdot\nabla B)_i=\sum_j A_j\,\mathrm{grad\_B}[\texttt{"Aji"}].
$$

Thus the first index of `Aji` is the direction/component contracted with the
velocity component `A_j`, and the second index is the output component of `B`.
For example, the first output component uses `A11`, `A21`, and `A31`.


More precisely, with $(e_1,e_2,e_3)=(e_r,e_\theta,e_\phi)$, the array labelled
`Aij` represents $G_{ij}(B)=e_j\cdot[(e_i\cdot\nabla)B]$. Differentiation
includes the changing spherical unit vectors as well as the scalar components.

**Example: the output `A21`.** Its physical expression is

$$
G_{21}(B)=\frac{\partial_\theta B_r-B_\theta}{R}
=\frac{\cos^2\beta}{\kappa\sin\beta}
  (\partial_\theta\widehat B_r-\widehat B_\theta).
$$

For ordinary radial columns, `get_gradient` implements these two terms as:

| Input component | Radial key and coefficient | Angular key | Output block |
| --- | --- | --- | --- |
| `u1` | `D12_r`, $+1/\kappa$ | `D` | `op_mat[4,1]` |
| `u2` | `D12_θ`, $-1/\kappa$ | `I` | `op_mat[4,2]` |

Both radial keys evaluate $\cos^2\beta\,f(\beta)/\sin\beta$. `build_operator`
separates this radial factor from the angular derivative or identity;
`apply` sums the two blocks into row 4, labelled `A21`. Thus the `D12` key
names refer to the component/direction order, while the output labels use
direction/component order. The first spherical column uses the cancellation
in Section 8. Multiplication by $\tan\beta$ gives the corresponding weighted
radial factor $\cos\beta\,f$, used by the `_w` keys.


### The `"_w"` Gradient Keys

An `"_w"` gradient key is $\tan\beta$ times the corresponding unweighted
physical-gradient key. The reason is the physical $L^2$ weighting in the
nonlinear term.

The physical nonlinear quantity is

$$
A_{\mathrm{phys}}\cdot\nabla B_{\mathrm{phys}}.
$$

To compute its physical $L^2$ norm in the $\beta$ coordinate, the code needs

$$
w(\beta)\,A_{\mathrm{phys}}\cdot\nabla B_{\mathrm{phys}}.
$$

Since

$$
A_{\mathrm{phys}}=\cos\beta\,\widehat A,
$$

we get

$$
w A_{\mathrm{phys}}\cdot\nabla B_{\mathrm{phys}}
=
\frac{\sin\beta}{\cos^2\beta}
\cdot
\cos\beta\,\widehat A
\cdot\nabla B_{\mathrm{phys}}
=
\widehat A\cdot
\left(\tan\beta\,\nabla B_{\mathrm{phys}}\right).
$$

Therefore:

* unweighted gradient keys represent components of
  $\nabla B_{\mathrm{phys}}$ acting on stored $\widehat B$;
* `"_w"` gradient keys represent
  $\tan\beta\,\nabla B_{\mathrm{phys}}$;
* `\widehat A · gradient_w(B)` represents
  $w A_{\mathrm{phys}}\cdot\nabla B_{\mathrm{phys}}$.

This is why `mul_A_gradB` keeps `A` in stored / compactified form and applies
`get_gradient(...; weight_flag=true)` to `B`.

## 13. `get_gradient_sym`

The mathematical target of `get_gradient_sym` is the physical symmetric gradient

$$
\operatorname{sym}\nabla U_{\mathrm{phys}}
=
\frac{\nabla U_{\mathrm{phys}}+(\nabla U_{\mathrm{phys}})^T}{2}.
$$

The input is the stored unknown $\widehat U$, and the keys compute physical
symmetric-gradient components of
$U_{\mathrm{phys}}=\cos\beta\,\widehat U$.

The function constructs a `6 x 4` operator matrix. The six rows pack the
independent components of a symmetric `3 x 3` tensor:

| Row | Component | Meaning |
| --- | --- | --- |
| 1 | `(11)` | radial-radial component |
| 2 | `(12)` | average of `(12)` and `(21)` |
| 3 | `(13)` | average of `(13)` and `(31)` |
| 4 | `(22)` | angular-angular component |
| 5 | `(23)` | average of `(23)` and `(32)` |
| 6 | `(33)` | azimuthal-azimuthal component |

The key mapping in `src/NS_Terms.jl` is:

| Row | Code keys |
| --- | --- |
| 1 | `D11_r_sym` |
| 2 | `D12_r_sym` and `D12_θ_sym` |
| 3 | `D13_ϕ_sym` |
| 4 | `D22_r_sym` and `D22_θ_sym` |
| 5 | `D23_ϕ_sym` with angular key `D23_ϕ_θ_sym` |
| 6 | `D33_r_sym` and `D33_θ_sym` |

This operator is used in the positivity compensation condition involving

$$
Q+\frac{\nabla U_{\mathrm{phys}}+\nabla^T U_{\mathrm{phys}}}{2}+\eta_1 I.
$$

## 14. Gradient Comparison

| Function | Object | Rows | Used for |
| --- | --- | --- | --- |
| `get_gradient` | full $\nabla U_{\mathrm{phys}}$ | 9 rows, `A11`...`A33` | nonlinear term $A\cdot\nabla B$ |
| `get_gradient_sym` | $(\nabla U_{\mathrm{phys}}+\nabla^T U_{\mathrm{phys}})/2$ | 6 independent symmetric rows | positivity compensation |

## 15. `get_operator_matrix`

`get_operator_matrix(sol, r, theta, key; kwargs...)` is the dispatch wrapper for
the operator families:

* `"linear"`, `"transpose"`, `"laplacian"`, `"pressure"`;
* `"identity"`;
* `"gradient_sym"`;
* `"gradient"`.

It reads the mode sizes, scaling factor, and boundary-condition index from the
supplied `VectorFunc`. After constructing the requested operator matrix, it
calls

```julia
apply_convmat_right!(op_mat, sol)
```

so that the operator acts in the internal coefficient basis of `sol`.

## 16. `mul_A_gradB_local`


`mul_A_gradB_local` contracts the sampled fields using the indexing in
Section 12. The default takes $\sum_j A_jG_{ji}(B)$; `transpose=true` takes
$\sum_j A_jG_{ij}(B)$. It adds no radial weight: that is already present when
`grad_B` comes from weighted gradient keys. Both branches return a spatial
`VectorFunc` with keys `e1`, `e2`, and `e3`.


## 17. `mul_A_gradB`


`mul_A_gradB` interpolates the stored $\widehat A$, evaluates the weighted
gradient of $B$, and calls `mul_A_gradB_local`. The identity in Section 12
explains why the result is $w(A_{\rm phys}\cdot\nabla)B_{\rm phys}$.
For background $U$ and perturbation $v$, the two calls with arguments $(U,v)$
and $(v,U)$ form the linearized convection term in Section 9.

List inputs return the pairwise matrix of interactions. With single inputs
and `transpose=true`, the function computes and returns only the transposed
gradient contraction as one `VectorFunc`; the ordinary contraction is skipped.
The three calls in `verify_auxiliary_field_bound.ipynb` use this result directly, without `[2]`.


## 18. Safety and Scope

`NS_Terms.jl` does not perform the final norm estimates by itself. It constructs
weighted physical operators and pointwise nonlinear terms consumed by the
verification notebooks. The mathematical guarantees come from combining these
assembled operators with interval arithmetic, quadrature estimates, and the
bounds documented in the companion notes.

## 19. Function-by-Function Summary

| Function | Role | Output / convention |
| --- | --- | --- |
| `build_operator` | separated radial/angular block | `Operator(U_left, U_right')`, optional `L1` |
| `get_linear` | Weighted $DU+\Delta U+\nabla P$ on stored unknowns | `3 x 4` operator matrix |
| `get_linear(...; mode="transpose")` | Weighted $(\Delta-D)v$ for the projected estimate | `3 x 4`; pressure blocks empty |
| `get_linear(...; mode="pressure")` | Weighted physical pressure gradient | `3 x 4`; only entries `(1,4)` and `(2,4)` nonempty |
| `get_identity` | physical or weighted physical identity / $\lambda I$ term | `"Cos"` or `"Tan"` diagonal operator |
| `get_gradient` | full physical gradient, optionally weighted by $\tan\beta$ | `9 x 4`, keys `A11`...`A33` |
| `get_gradient_sym` | physical symmetric gradient | `6 x 4`, independent components `(11),(12),(13),(22),(23),(33)` |
| `get_operator_matrix` | dispatch + basis conversion | calls `apply_convmat_right!` |
| `mul_A_gradB_local` | pointwise contraction | space-domain `VectorFunc` |
| `mul_A_gradB` | grid nonlinear term builder | single, pairwise, or transpose outputs |

## 20. Related Files


* [NS spherical.nb](<NS spherical.nb>) derives the transformed expressions
  and the first-mode incompressibility relation; see the reading notes in Section 6.
* [My_Fourier.jl](../../src/My_Fourier.jl) implements `trig_func` and its operator keys.
* [My_Operator.jl](../../src/My_Operator.jl) applies and sums separated blocks.
* [My_Vector.jl](../../src/My_Vector.jl) defines component storage and boundary sets.
* [selfsimilar_profile_verification.ipynb](../../notebooks/selfsimilar_profile_verification.ipynb)
  assembles the profile residual and positivity checks.
* [eigenpair_error_analysis.ipynb](../../notebooks/eigenpair_error_analysis.ipynb)
  assembles the eigenpair residual, corrects the eigenvalue, and bounds the
  normalized velocity and gradient.
* [verify_auxiliary_field_bound.ipynb](../../notebooks/verify_auxiliary_field_bound.ipynb)
  uses the adjoint source in its composed weighted-functional estimate.


