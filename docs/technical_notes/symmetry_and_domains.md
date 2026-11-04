# Symmetries and justified sampling domains

This reference explains the reflection symmetries used by the verification
notebooks and which domain reductions they justify. It describes the active
basis definitions and notebook calls, including the even finite-rank prior on
$[0,\pi]$, the eigenpair refinement with `ind=1`, and the weighted functional.

Mixed-forcing refinement uses explicit radial derivative priors and the special
`ind=1` angular path; see [the mixed-forcing bounds](#mixed-forcing).

Start with the [workflow table](#workflow-table) for a particular calculation.
Domain reduction follows from the parity of the assembled quantity. Frequency
ceilings, derivative priors and padding must cover that same quantity on the
domains listed below.

<a id="coordinates"></a>
## 1. Coordinates and the meaning of a reflection

Write $\kappa=\texttt{scl\_fac}$ for the profile scale and $R=32$ for the
support radius of Q. The coordinates used in different parts of the code are:

| Coordinate | Definition | Physical interval |
|---|---|---|
| $r_{\rm phy}$ | Physical radius | $[0,\infty)$ |
| $\beta$ | $r_{\rm phy}=\kappa\tan\beta$ | $[0,\pi/2)$ |
| $s$ | Normalized ball radius $r_{\rm phy}/R$ | $[0,1]$ |
| $x_Q$ | Q's trigonometric argument $(\pi/2)s$ | $[0,\pi/2]$ |
| $\mathrm{Beta}$ | $\arctan(R/\kappa)$ | End of Q's support in $\beta$ |
| $\theta$ | Polar angle | $[0,\pi]$ before equatorial reduction |

Variable names such as `r_pt` do not uniquely identify a coordinate. For
example, the profile notebook samples Q at arguments $x_Q\in[0,\pi/2]$
but supplies `L_Q=1` to the norm routine. Those radial derivative bounds are
with respect to $s$, and use frequencies rescaled by $\pi/2$.

In the tables, **E** and **O** mean even and odd under reflection about the
specified point. At zero the reflection is $x\mapsto-x$; at $\pi/2$ it is
$x\mapsto\pi-x$. These are statements about the analytic extension of a
scalar component. A vector's field-even or field-odd label does not mean
that all its components are even or odd.

The stored compactified velocity and pressure are $\widehat u,\widehat p$.
The physical fields and the radial integration weight are

$$
u_{\rm phy}=\cos\beta\,\widehat u,\qquad
p_{\rm phy}=\cos\beta\,\widehat p,\qquad
w(\beta)=\frac{\sin\beta}{\cos^2\beta}.
$$

Thus `get_identity(...; weight_flag=true)` returns $wu_{\rm phy}$,
while `weight_flag=false` returns $u_{\rm phy}$. `mul_A_gradB` returns the
weighted physical convection term. See the [operator conventions](ns_terms.md).

<a id="basis-rules"></a>
## 2. Basis codes and rules for combining symmetries

The definitions in [`trig_func`](../../src/My_Fourier.jl) give:

| Code | Basis | At $0$ | At $\pi/2$ |
|---|---|---|---|
| `11` | $\cos(2nx)$, $n\geq0$ | E | E |
| `00` | $\sin(2nx)$, $n\geq1$ | O | O |
| `10` | $\cos((2n-1)x)$, $n\geq1$ | E | O |
| `01` | $\sin((2n-1)x)$, $n\geq1$ | O | E |

The code enforces a whole reflection parity, stronger than a single endpoint
value or derivative condition. Differentiating in the reflected coordinate
flips E and O; differentiating in another coordinate preserves them.
Multiplication multiplies the reflection signs. Addition preserves a sign
only when the summands have that same sign, or a separate cancellation has
been proved. In particular, adding an E term to an O term generally destroys
the reflection symmetry.

Every derivative of a function with definite parity has an absolute value
invariant under that reflection. This is why derivative bounds can use the
same fundamental interval as the function. Basis conversions within the
listed angular blocks preserve these signs; see [the Fourier basis note](fourier_basis.md).

<a id="angular-symmetries"></a>
## 3. Angular symmetries of vectors and tensors

The active `BDRY_LST_SET` in [`My_Vector.jl`](../../src/My_Vector.jl) and
the channel selectors in [`My_Assembly.jl`](../../src/My_Assembly.jl) give
the following componentwise angular parities. Each pair is
**(at $\theta=0$, at $\theta=\pi/2$)**.

| Sector | Component 1 / Y | Component 2 / X | Component 3 / Z | Pressure |
|---|---|---|---|---|
| Field-even: `UP`, even `phi`, $\psi_+$ | EE (`11`) | OO (`00`) | OE (`01`) | EE (`11`), when present |
| Field-odd: `up`, odd `phi`, $\psi_-$ | EO (`10`) | OE (`01`) | OO (`00`) | EO (`10`), when present |

Define

$$
D_0=\operatorname{diag}(1,-1,-1),\qquad
D_1=\operatorname{diag}(1,-1,1).
$$

A vector in sector $\sigma\in\{+1,-1\}$ satisfies

$$
v_\sigma(-\theta)=D_0v_\sigma(\theta),\qquad
v_\sigma(\pi-\theta)=\sigma D_1v_\sigma(\theta).
$$

Radial factors and radial derivatives preserve these angular laws. The
linear operators preserve the sector; the convection of two sector vectors
has the product of their sector signs. In particular, convection with the
field-even profile preserves the other field's sector.

### The symmetric gradient and Q

For the physical symmetric gradient of `UP`, and for the current Q loader:

| Entry | Angular code | At $\theta=0$ | At $\theta=\pi/2$ |
|---|---|---|---|
| 11 | `11` | E | E |
| 12 | `00` | O | O |
| 13 | `01` | O | E |
| 22 | `11` | E | E |
| 23 | `10` | E | O |
| 33 | `11` | E | E |

For either matrix field $S$, this means

$$
S(-\theta)=D_0S(\theta)D_0,\qquad
S(\pi-\theta)=D_1S(\theta)D_1.
$$

The matrices at reflected points have identical eigenvalues, since $D_0,D_1$
are orthogonal. These are relations between different physical points, not a
change of storage frame: the Bessel X sign compensation produces physical fields
and every active stage uses the same physical Q, without an extra DQD transform.
This justifies the angular reduction for minimum-eigenvalue
and operator-norm estimates of Q, $\operatorname{sym}\nabla U$, and their sum.
It also shows that Q preserves each vector sector under multiplication.

In particular, Q23 is **even at the axis and odd at the equator**. Its basis
forces $Q_{23}(r,\pi/2)=0$, but evenness alone does not force
$Q_{23}(r,0)=0$. Reflection parity also does not impose the separate axis
identity $Q_{22}=Q_{33}$ or full Cartesian smoothness. The table records the
symmetries used for domain reduction, not every identity a physical gradient
may satisfy.

For a vector in sector $\sigma$, its full physical gradient satisfies
$G_\sigma(-\theta)=D_0G_\sigma(\theta)D_0$ and
$G_\sigma(\pi-\theta)=\sigma D_1G_\sigma(\theta)D_1$.
Its componentwise absolute bounds therefore admit angular reduction in both
sectors. For the symmetric part of an odd gradient, minimum eigenvalues
cannot be equated across the equatorial reflection by dropping the minus sign.
The active `gradient_sym` calls use
`UP` (set 2). The generic `gradient_sym` implementation does not handle the
special odd $l=1$ radial block as the full `gradient` implementation does;
this note does not assert that unsupported call is usable.

<a id="radial-symmetries"></a>
## 4. Radial symmetries depend on the represented quantity

### Compactified trigonometric fields

The following radial parities are in **$\beta$**, with the two columns referring
to zero and $\pi/2$. They follow from `BDRY_LST_SET` and the actual weighted
keys in [`NS_Terms.jl`](../../src/NS_Terms.jl) and
[`My_Fourier.jl`](../../src/My_Fourier.jl).

| Quantity | At $\beta=0$ | At $\beta=\pi/2$ |
|---|---|---|
| Stored profile velocity $\widehat U$ (set 2, radial `01`) | O | E |
| Physical profile velocity $U_{\rm phy}$ | O | O |
| Stored even $\widehat\phi$ velocity (set 3, radial `00`) | O | O |
| Physical even $\phi_{\rm phy}$ | O | E |
| Stored pressure in sets 2 and 3 (radial `10`) | E | O |
| Corresponding physical pressure | E | E |
| Weighted profile residual | E | O |
| Physical $\nabla U$ and $\operatorname{sym}\nabla U$ | E | E |
| Physical $\nabla\phi$ for even $\phi$ | E | O |
| Weighted identity $w\phi_{\rm phy}$ for even $\phi$ | E | E |
| Weighted convection $w(\phi_i\cdot\nabla)\phi_j$, both even | E | O |
| Even inverse operator-side residual `lfs`, components 1 and 2 | E | Mixed |
| Same `lfs`, component 3 | E | E |

The operator expressions in these rows are finite trigonometric polynomials
after removable quotients are cancelled. `trig_func` supplies the appropriate
limits at the singular-looking endpoints; one must use those expressions,
not separately bound singular weights such as $w$ over a full period.

The odd set 1 requires particular care. In stored `u1` and `u2`, the
$l=1$ block uses radial `10` (EO), while the higher-degree blocks use radial
`00` (OO). Stored `u3` uses `00`. Odd pressure uses the fixed descriptor
`["00","10"]` in boundary set 1: every radial column
is an even-frequency sine (OO), and every angular column is an odd-frequency
Fourier cosine. Its physical pressure is OE and its weighted pressure gradient
is OO. Pressure receives no velocity Legendre conversion.

This fixed pressure basis applies to both `up` and odd `up_phi`, independently
of MAT metadata.
Summing the unchanged velocity angular modes can still mix the origin parities.
Physical multiplication by $\cos\beta$, weighting, pressure,
and convection change the signs again. A field-odd label by itself therefore
does not justify a radial quarter-domain Bernstein estimate. The odd
notebooks use full-period radial priors instead.

For the evaluated odd fields, this gives the following more specific signs:

| Odd quantity | At $\beta=0$ | At $\beta=\pi/2$ |
|---|---|---|
| Physical `u1`, `u2` | Mixed | E |
| Physical `u3` | O | E |
| Weighted identity, components 1 and 2 | Mixed | E |
| Weighted identity, component 3 | E | E |
| Physical full gradient | Mixed in some entries | O in every entry |
| Pressure-free operator-side combination, including the eigenvalue term | Generally mixed | E |
| Complete odd residual, components 1 and 2 | Generally mixed | Generally mixed |
| Complete odd residual, component 3 | Generally mixed | E |

The full gradient's entries 13, 23, 31, and 32 are E at the origin;
the other entries can mix origin parities. Convection can transmit the
special $l=1$ contribution into residual component 3, so that component
cannot inherit a common origin parity merely from stored `u3`.

### Frequency ceilings and input compatibility

For M radial rows and N angular columns, the stored odd pressure has ceilings
`(2M,2N-1)`. With $w=\sin\beta/\cos^2\beta$, the physical and weighted
expressions have the following ceilings. The last two columns illustrate the
current 300×150 odd pressure arrays; installed metadata belongs in the
[data reference](../data.md).

| Pressure expression | β ceiling | θ ceiling | β at M=300 | θ at N=150 |
|---|---|---|---:|---:|
| $\widehat p$ and its β/θ derivatives | $2M$ | $2N-1$ | 600 | 299 |
| $p=\cos\beta\widehat p$ | $2M+1$ | $2N-1$ | 601 | 299 |
| $w\partial_r p=\kappa^{-1}\sin\beta(\cos\beta\partial_\beta\widehat p-\sin\beta\widehat p)$ | $2M+2$ | $2N-1$ | 602 | 299 |
| $wr^{-1}\partial_\theta p=\kappa^{-1}\partial_\theta\widehat p$ | $2M$ | $2N-1$ | 600 | 299 |
| $\partial_r p$ | $2M+3$ | $2N-1$ | 603 | 299 |
| $r^{-1}\partial_\theta p$ | $2M+1$ | $2N-1$ | 601 | 299 |

The last row uses the exact trigonometric cancellation of
$\sin(2j\beta)/\sin\beta$ before multiplication by $\cos^2\beta$.
Coordinate differentiation does not increase Fourier bandwidth.

The current notebooks derive frequency counts from `u3` dimensions, with
separate background, inverse and eigenfunction counts where implemented. They
assume compatible sizes for the other velocity components and pressure; they do
not automatically take maxima over all components. Larger pressure arrays or
inconsistent component shapes need an independent ceiling/compatibility check.

Under the supported velocity dimension/conversion conventions, raw velocity has
radial ceiling at most `2M+2` (allowing the extra `u2` row) and angular ceiling at
most `2N`. Physical and convection-weighted gradient keys add at most three
radial frequencies after cancellation; weighted diffusion adds at most four.
For two fields with counts $(M_a,N_a)$ and $(M_b,N_b)$, weighted convection has
ceilings at most $(2M_a+2M_b+7,2N_a+2N_b)$. The current conservative ceilings are:

| Quantity | Current beta/theta ceilings | Input counts used |
|---|---|---|
| Single inverse field | `(610,310)` | Inverse 300×150 |
| Mixed background/inverse operator | `(1810,910)` | Background 600×300 plus inverse 300×150 |
| Step 5 inverse/eigenfunction mixed forcing | `(1210,610)` | Inverse and eigenfunction each 300×150 |

These are input-derived assumptions, not permanent constants or a justification
for using Bernstein on a restricted interval. A pressure basis change alone is
not a reason to replace complete operator ceilings with the pressure-only ones.

For quadrature, products add component bandwidths and the measure's
$\sin\theta$ adds one angular frequency; the Leibniz bounds include that weight
separately. Q/Bessel RHS terms retain their independently computed radial priors.
The supplied Q coefficient arrays have angular ceiling 300. The root table
determines the largest Bessel degree in the basis. RHS priors must cover the
combined Q/basis angular bandwidth and the retained eigenvector columns.

These bandwidth formulas use the supported coefficient shapes, removable-quotient
cancellations and angular reflection laws. Mixed odd residuals require
full-period radial priors.

### Q in its native coordinate, and Bessel fields

`load_grad_U` currently assigns radial `10` to **all six** Q entries. Thus
each is E at $x_Q=0$ and O at $x_Q=\pi/2$, and Q vanishes at
$r_{\rm phy}=R$. The angular codes are the tensor table above. Equivalently,
the radial basis in normalized radius is $\cos((2n-1)\pi s/2)$.

Q and $\operatorname{sym}\nabla U$ share evenness at the radial origin.
Q's odd reflection at $x_Q=\pi/2$ concerns the finite cutoff $r_{\rm phy}=R$;
no such reflection is asserted for $\operatorname{sym}\nabla U$ there.
Composing Q with $s=(\kappa/R)\tan\beta$ preserves evenness at zero but
does **not** give an odd reflection about $\mathrm{Beta}$ or a finite
Fourier series in $\beta$.

The Bessel implementation's signed radial extension gives poloidal degree
$l$ parity $(-1)^{l-1}$ and toroidal degree $l$ parity $(-1)^l$; see
[`Ur_func`](../../src/My_Bessel.jl) and [the Bessel note](bessel_basis.md).

| Quantity in the inner region | At radial origin | Angular sector |
|---|---|---|
| $\psi_+$, all components | O | Even |
| $\psi_-$, all components | E | Odd |
| Q composed with $s(\beta)$, all entries | E | Tensor table above |
| $wQ\psi_+$, all components | E | Even |
| $wQ\psi_-$, all components | O | Odd |

No reflection about $s=1$ is asserted for $\psi$, and no reflection about
$\mathrm{Beta}$ is asserted for $wQ\psi$. Their radial bounds use Bessel,
chain-rule, and product estimates, not radial Bernstein.

<a id="domain-rules"></a>
## 5. Which reduction is justified?

### Scalar suprema and derivatives

For a $2\pi$-periodic trigonometric polynomial with definite parity at both
zero and $\pi/2$, reflections cover the full period from $[0,\pi/2]$.
For every derivative order $m$,

$$
\sup_{[0,2\pi]}|f^{(m)}|=\sup_{[0,\pi/2]}|f^{(m)}|.
$$

If only the origin parity is known, $[0,\pi]$ still covers a full-period
absolute maximum by reflection about zero and periodicity. This applies to
the even `lfs`. It is not a universal rule for mixed odd radial blocks.

These identities justify using the reduced-domain supremum in Bernstein's
inequality. For a function with maximum wavenumber $M$,

$$
\|f^{(m)}\|_{\infty,\mathrm{period}}
\le M^m\|f\|_{\infty,\mathrm{period}}.
$$

Band limitation alone does not replace either norm by an arbitrary restricted
interval norm. This restriction applies both to the direct frequency-vector
`norm_lst` branch and to frequency corrections in either direction of
finite-difference refinement. See [the estimator derivation](matrix_func.md#closing-an-estimate-with-frequency-bounds).

### Matrix bounds

Componentwise absolute derivative bounds follow from the scalar rules.
Signed eigenvalue bounds require the orthogonal conjugation proved in
[Section 3](#angular-symmetries). For example, native Q has
$Q(\pi-x_Q,\theta)=-Q(x_Q,\theta)$. This preserves its operator norm,
but exchanges minimum and negative maximum eigenvalues. A positive lower
bound on the physical interval does not extend across that radial reflection.

### Integrals and the factor $4\pi$

For same-sector vectors, $v_\sigma^Tu_\sigma$ and
$v_\sigma^TQu_\sigma$ are even about the equator. The polar measure
$\sin\theta\,d\theta$ is also preserved, so their full polar integral
is twice the integral over $[0,\pi/2]$. Together with the azimuthal factor
$2\pi$, this gives the $4\pi$ in the inner products. For opposite sectors,
the corresponding full angular integral is zero; it is not generally twice
the quarter-interval integral. This justifies the parity block split in
finite-rank assembly.

There is **no radial doubling**: $\beta\in[0,\pi/2)$ already represents all
nonnegative physical radii. Splitting it at `Beta` merely separates Q's
support from its complement. The generic `inner_product` routine includes
the factor $4\pi$ without checking sector compatibility; the caller must
establish the symmetry of the integrand.

### Priors, refinement, and padding are different requirements

The special `Linf_to_paras(...; ind=...)` path refines $\beta$ bounds first.
If $B_0$ and $B_2$ bound a component and its second $\beta$ derivative on the
target radial interval, angular symmetry and Bernstein give

$$
\|\partial_\theta^m f\|_\infty\le M_\theta^mB_0,\qquad
\|\partial_\beta^2\partial_\theta^m f\|_\infty
\le M_\theta^mB_2.
$$

These supply the angular seeds and explicit transverse corrections through
`"L∞_freq"` in [`My_Scalar.jl`](../../src/My_Scalar.jl). They do not require
radial Bernstein on the restricted interval. The ordinary path instead
retains a radial frequency correction in the angular pass and needs the
appropriate radial-domain justification too.

Padding supplies actual stencil samples and requires priors valid throughout
the nested stencil intervals. It does not establish the Bernstein domain
assumption. The recursive order-eight and order-ten padding requirements are
10 and 15 layers, respectively; the RHS workflow also reserves layers for
the next refinement stage. See [the padding derivation](matrix_func.md#why-refinement-proceeds-from-high-order-to-low-order)
and [the RHS padding choices](finite_rank_rhs.md#how-the-workflow-padding-is-chosen).

Near `Beta`, the inner-region stencils use the smooth trigonometric/Bessel
continuation of the inner formula. That continuation is distinct from
physically setting Q to zero outside its support. Do not assume that the
piecewise zero extension is smooth to every derivative order at `Beta`.

<a id="workflow-table"></a>
## 6. Reference by notebook

The table lists the prior and sampling domains used by each stage and the
reason its reduction is valid.

| Notebook / quantity | Domain used for priors or sampling | Justification |
|---|---|---|
| Profile: weighted residual | $\beta,\theta\in[0,\pi/2]$ | Valid scalar reflection reduction |
| Profile: physical symmetric gradient | $\beta,\theta\in[0,\pi/2]$ | Valid component bounds; angular eigenvalue reduction by conjugation |
| Profile: native Q bounds | $s\in[0,1]$, $\theta\in[0,\pi/2]$ | Valid, with rescaled radial frequencies |
| Profile: Q composed with $\beta$ | $\beta\in[0,\mathrm{Beta}]$ plus padding | Chain-rule radial priors; angular reduction valid |
| Eigenpair: odd fields and residual priors | $\beta\in[0,2\pi]$, $\theta\in[0,\pi/2]$ | Full radial period; angular reduction valid |
| Eigenpair: residual refinement | Physical quarter rectangle plus padding | Valid special `ind=1` path |
| Finite-rank construction: angular integrals and radial coefficient bounds | Half meridian; $s\in[0,1]$ plus padding | Sector split and native Q radial reduction valid |
| Finite-rank RHS: $\psi$ and $wQ\psi$ | $\beta\in[0,\mathrm{Beta}]$ plus padding | Explicit radial priors; angular reduction valid |
| Even finite-rank: operator-side `lfs` prior | $\beta\in[0,\pi]$, $\theta\in[0,\pi/2]$ | Valid; radial quarter reduction would not be justified |
| Even finite-rank: $\nabla\phi$, $w\phi$, weighted $\phi_i\cdot\nabla\phi_j$ | Physical quarter rectangle | Valid scalar reflection reductions |
| Even finite-rank: assembled residual refinement | Inner and outer physical intervals plus padding | Special `ind=ind` path; no restricted radial Bernstein |
| Step 5 mixed-forcing priors | Full beta period, angular quarter | Valid prior domain |
| Step 5 mixed-forcing refinement | Physical quarter rectangle plus padding | Special `ind=1` path with explicit beta priors |
| Odd finite-rank: operator and weighted-field priors | Full $\beta$ period, angular quarter | Valid without a radial parity assumption |
| Odd finite-rank: residual refinement / Gram integrals | Inner and outer physical intervals | Special refinement path; same-sector angular integration |
| Step 7: auxiliary field bound | Full-period odd priors; inner/exterior physical refinements and integrals | Explicit beta priors with special `ind=1`; same-sector whole-space pairings at mass weight eta_v |

<a id="profile-verification"></a>
### Profile verification

In [`selfsimilar_profile_verification.ipynb`](../../notebooks/selfsimilar_profile_verification.ipynb),
find `err`, `grad_norms_prior`, `Q_Wk∞`, and `Q_Wk∞_beta`.
The profile residual has radial EO parity; the physical symmetric gradient
has EE parity. Their angular components have the vector/tensor signs above.
Native Q bounds use its EO radial basis in $x_Q$, with derivatives reported
in $s$. Consequently the ordinary refinement of native Q and the symmetric
gradient has valid frequency corrections in both coordinates.

For composed Q, `r_to_beta` supplies explicit radial priors and `ind=1`
selects the special refinement path. Minimum-eigenvalue sampling covers
$[0,\mathrm{Beta}]$ and $[\mathrm{Beta},\pi/2]$ physically. Only the angular
domain is reduced by conjugation in that eigenvalue calculation.

<a id="eigenpair-verification"></a>
### Eigenpair verification

In [`eigenpair_error_analysis.ipynb`](../../notebooks/eigenpair_error_analysis.ipynb),
the current `norm_dict` uses full-beta-period priors for the residual `err`,
weighted identity `I_up`, full gradient `grad_up` and physical identity `I_phy_up`.
Their angular components have definite endpoint parities, including those of the
full odd gradient. Full-period radial bounds can then be used on a physical
subinterval.

The current residual refinement is

```julia
paras_lst = Linf_to_paras(norm_dict["err"], k+2, err_freq_lst; ind=1)
```

Here `err_freq_lst` uses the background and eigenfunction counts. `ind=1` selects
the deferred angular construction described above: explicit radial priors are
refined first, and the resulting bounds supply the angular pass. The constrained
eigenvalue functional is bounded in [Step 7](#weighted-functional).

<a id="finite-rank-construction"></a>
### Finite-rank construction

In [`finite_rank_construction.ipynb`](../../notebooks/finite_rank_construction.ipynb),
`integrate_θ` integrates the spherical weight and basis products on
$[0,\pi/2]$. Q's conjugation law makes same-sector products equator-even and
cross-sector products equator-odd, justifying the retained parity blocks.

Before radial integration, every scalar entry of the angular-contracted Q
tensor is still a linear combination of Q's native cosine-odd radial modes.
Thus `int_θ_bound` may use native Q reflection symmetry and frequency
$M\pi/2$ in normalized radius. This argument concerns those coefficient
entries, not the whole Bessel-weighted radial integrand. The latter uses
the analytic Bessel derivative bounds and product bounds during integration;
no periodic radial symmetry of a Bessel function on $[0,1]$ is assumed.

<a id="finite-rank-rhs"></a>
### Finite-rank RHS bounds

In [Part 4B of `finite_rank_construction.ipynb`](../../notebooks/finite_rank_construction.ipynb#step-4b),
`refine_channel_bounds` uses explicit radial priors and the special angular
construction for both $\psi$ and $wQ\psi$. Multiplication by the current Q
preserves the angular sector, and $\beta$ differentiation preserves its
angular signs. The angular Bernstein estimates therefore remain valid for
the mixed derivatives and angular ghost samples too.

The radial domain ends at `Beta`, not $\pi/2$. Neither origin parity nor Q's
native endpoint symmetry supplies radial Bernstein for these compositions.
See [the RHS note](finite_rank_rhs.md) for the chain-rule domains and the
25-layer to 10-layer refinement workflow.

<a id="even-finite-rank-residual"></a>
### Even finite-rank residual: why the prior needs a wider interval

In [`selfsimilar_linear_operator_finite_rank.ipynb`](../../notebooks/selfsimilar_linear_operator_finite_rank.ipynb),
write the operator-side quantity as $F=V-P$, where $V$ contains the two
convection terms minus the pressure-free linear operator, and $P$ is the
weighted pressure gradient. Velocity terms are EE in $\beta$. For a radial
pressure basis factor $p=\cos((2n-1)\beta)$, the pressure blocks have radial factors

$$
P_1:\ \sin\beta(-\sin\beta\,p+\cos\beta\,p'),
\qquad P_2:\ p,\qquad P_3=0,
$$

with the relevant angular factors and scale applied by `get_linear`.
Both nonzero pressure blocks are EO. Hence

$$
F(-\beta)=F(\beta),\qquad
F(\pi-\beta)=V(\beta)+P(\beta).
$$

Cancellation between velocity and pressure on one side need not persist on
the other side. For a simple parity-class example, take
$V=\cos^2\beta$, $P=\cos\beta$: the maximum of $|V-P|$ on $[0,\pi/2]$
is $1/4$, whereas its value at $\pi$ is $2$. Thus the full-period prior cannot
be replaced by a quarter-interval prior for this parity class.

The current prior cell therefore uses

```julia
β_prior = grid(Pi, 2*N0)
```

for `lin_mat_phi` and both convection evaluations, followed by
`norm_lst(lfs, "W$(k)Inf"; paras=freq_mixed, domain=[Pi,L1])`.
Here `freq_mixed` uses the sum of background and inverse coefficient counts.
Origin evenness and periodicity make this half-period sufficient. Doubling
the interval count preserves the original mesh spacing. `β_pt` and `L0`
still describe the physical quarter interval for the subsequent gradient,
identity, convection, and integral calculations.

After adding the RHS, refinement uses explicit radial priors and
`Linf_to_paras(...; ind=ind)` on each physical subinterval. It does not assume
that the combined Bessel/trigonometric residual is radially band limited.

<a id="mixed-forcing"></a>
### Step 5 mixed forcing

The columns are $F_j=(\Phi_j\cdot\nabla)v+(v\cdot\nabla)\Phi_j$, for raw
even inverse fields and the corrected odd eigenfunction. In the mixed-forcing
section of
[`selfsimilar_linear_operator_finite_rank.ipynb`](../../notebooks/selfsimilar_linear_operator_finite_rank.ipynb),
full-period radial priors are passed to the special refinement path:

```julia
F_β_prior = [reshape(W_F_prior[j][m, 1], :, 1) for m in 1:k+1]
paras = Linf_to_paras(F_β_prior, k, [zero(dtype), freq_forcing[2]]; ind=1)
```

For `k=10`, this supplies order-9/10 radial seeds for order-8 refinement.
After radial refinement gives $B_\beta^{(0)},B_\beta^{(2)}$, the special path
uses the valid angular Bernstein inequalities to supply theta seeds and
mixed corrections $M_\theta^qB_\beta^{(0)}$ and
$M_\theta^qB_\beta^{(2)}$. It does not infer a radial derivative bound from a
quarter-interval supremum. The physical samples retain their padding until
refinement is complete, then are trimmed for the whole-space Gram integral.

**Why use explicit radial priors?** Mixed odd blocks need not share origin
parity, so a physical quarter-domain supremum need not equal its full-period
supremum. Bernstein's frequency bound applies on the full period; it cannot
generally be applied to the smaller physical supremum.

For example,

$$
g(\beta)=3\cos\beta-2\cos(3\beta)-4\sin(2\beta)
=\cos\beta\,[1+8\sin^2\beta-8\sin\beta].
$$

On $[0,\pi/2]$, $\|g\|_\infty=1$, while $g''(0)=15>3^2\|g\|_\infty=9$,
despite maximum frequency 3. Explicit radial priors avoid this restricted-domain
Bernstein assumption. The resulting forcing Gram `GF` yields
`Theta_F_U = sqrt(sum(GQ .* GF))`, a Hilbert–Schmidt upper bound for the
paper's mixed-forcing operator norm $\Theta_F^U$. The notebook prints the
interval and asserts its upper endpoint is at most `0.0638`, matching the
paper bound $\Theta_F^U\le0.0638$.

<a id="odd-finite-rank-residual"></a>
### Odd finite-rank residual

In [`eigen_linear_operator_finite_rank.ipynb`](../../notebooks/eigen_linear_operator_finite_rank.ipynb),
the first prior pass has `L0=2*Pi` for the weighted identity fields and the
complete `lfs`, including its pressure and eigenvalue terms. It therefore
needs no radial quarter-domain symmetry argument for the mixed odd blocks.
The angular components remain in the odd sector.

The inner and outer residual refinements use `ind=ind`, explicit radial
priors, and angular frequency corrections. The Gram integrals pair fields
in the same sector and cover the physical radial intervals. The full-period
prior grid does not enlarge the physical integration domain.

<a id="weighted-functional"></a>
### Step 7 — Auxiliary field bound

[`verify_auxiliary_field_bound.ipynb`](../../notebooks/verify_auxiliary_field_bound.ipynb) obtains global odd-field
priors over a full beta period, then constructs explicit radial bounds for the
inner and exterior physical intervals. Each refinement uses a fresh special
`ind=1` parameter package; it does not infer radial Bernstein on those restricted
intervals. Angular endpoint parities continue to justify angular Bernstein.

Step 7 obtains fixed binary centers of the source coefficients
from inverse multipliers and Step 6 `t`. It bounds the difference from the exact
$c_j=\langle P_vH(v),\Phi_j\rangle$ separately using `GQ,GR,b` and the Step 2
residual bound. This expresses the source through saved pairings and residual
Grams without evaluating new source-pairing integrals.
Same-sector products still justify the $4\pi$ angular factor. Inner and exterior
integrals cover all nonnegative radii without radial doubling; the physical grids
are `12000 × 6000` and `9000 × 6000`. The inner RHS is already weighted and its
ten padding layers are removed without subsampling.

The mass weight is `eta_v=0.23187`. With the full interval trial projection,
the [auxiliary-field estimate](../paper_code_map.md#direct-mass-weighted-m5) combines
`sqrt(w1+z2/eta_v)` with the separate source allowance. The final upper endpoint
is printed and asserted against `14.9512`, giving the paper's bound $M_5^v\le15$.
The paper uses mass weight `0.232` in its eigenpair ball. The estimate at
`0.23187` also applies there because increasing the mass weight decreases
the dual energy norm. Step 7 writes no result file.

## 7. When changing a basis or a domain

Recheck the **assembled quantity**, including pressure, radial weights, and
all summands. Check each direction separately. Componentwise parity is
enough for absolute norms; signed matrix eigenvalues require conjugation,
and doubled angular integrals require an even integrand. A new Q basis must
preserve the tensor angular table for Q-multiplication and the parity block
split to retain these proofs. A changed radial coordinate requires new
derivative scaling or chain-rule bounds. Finally, validate stencil coverage
independently of every symmetry argument.
