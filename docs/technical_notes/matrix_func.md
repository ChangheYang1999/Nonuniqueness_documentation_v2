# `Matrix_Func.jl` — Mathematical Background

This note explains why the estimates in [Matrix_Func.jl](../../src/Matrix_Func.jl)
are valid and how they contribute to verification. The source docstrings describe
argument layouts, indexing, mutation, and checks.

For the field containers, component order, and the output shapes of
`norm_lst`, see [ScalarFunc and VectorFunc](scalar_vector.md#5-from-samples-to-a-bound).

There are three main uses: enclose integrals by quadrature and an error allowance;
sharpen derivative bounds from samples so that allowance is small enough; and
certify finite-dimensional matrix inequalities. A typical residual calculation
first obtains a priori derivative bounds, refines them using samples of the
assembled residual, and then uses the refined bounds in its L² quadrature error.
Sampling the assembled residual can retain cancellation lost in termwise bounds.

Throughout, a derivative-bound table contains upper bounds, not exact Sobolev
norms. Interval arithmetic encloses the numerical operations; a continuous norm
or integral also requires the smoothness, domain coverage, and auxiliary bounds
specified below. Float64 mode alone does not supply rigorous certificates.

---

## Composite Closed 7-Point Newton-Cotes Rule

`newton_cotes_weights(n, h)` and `newton_cotes_with_bound` use the composite
closed 7-point Newton-Cotes rule. Each panel has 7 nodes and 6 subintervals.
The subinterval width is $h$, so one panel has width $6h$. The composite rule on
$[0,1]$ requires `n % 6 == 0`, where `n` is the number of subintervals and
$h=1/n$.

This rule is sometimes confused with Weddle's rule. To avoid ambiguity: the code
uses the exact closed 7-point Newton-Cotes rule with weights
$(41,216,27,272,27,216,41)/140$. Weddle's rule usually refers to a perturbed
7-node rule, not this exact closed Newton-Cotes formula.

### One-Panel Formula

On one panel, let

$$
x_j=x_0+jh,\qquad f_j=f(x_j),\qquad j=0,\ldots,6.
$$

The closed 7-point Newton-Cotes approximation is

$$
Q_{\mathrm{panel}}(f)
=
\frac{h}{140}
\left(
41 f_0 + 216 f_1 + 27 f_2 + 272 f_3
+ 27 f_4 + 216 f_5 + 41 f_6
\right).
$$

These weights are obtained by integrating the degree-6 Lagrange basis
polynomials on the normalized panel $t\in[0,6]$:

$$
\ell_j(t)=\prod_{\substack{0\le m\le 6\\m\ne j}}\frac{t-m}{j-m},
\qquad
w_j=\int_0^6 \ell_j(t)\,dt.
$$

Direct integration gives

$$
(w_0,\ldots,w_6)
=
\frac{1}{140}(41,216,27,272,27,216,41).
$$

Scaling back from $t=(x-x_0)/h$ contributes the factor $h$.

### One-Panel Error Constant

Let the error functional on the normalized panel be

$$
E(g)=\int_0^6 g(t)\,dt
-
\frac{1}{140}
\left(
41g(0)+216g(1)+27g(2)+272g(3)
+27g(4)+216g(5)+41g(6)
\right).
$$

The closed 7-point rule interpolates at seven nodes, so it is exact for every
polynomial of degree at most 6. Because the nodes and weights are symmetric
about $t=3$, the degree-7 monomial error also vanishes after centering: the odd
part about the midpoint integrates to zero and is sampled symmetrically. Thus
the degree of precision is 7.

For $g\in C^8([0,6])$, the Peano representation is
$E(g)=\int_0^6 K(t)g^{(8)}(t)\,dt$, with

$$
K(t)=\frac{(6-t)^8}{8!}
-\frac{1}{7!}\sum_{j=0}^6 w_j(j-t)_+^7,
\qquad (x)_+=\max(x,0).
$$

For these weights, $K\le0$ on $[0,6]$. One exact way to check the sign is to
expand each polynomial piece on $[j,j+1]$ in the degree-8 Bernstein basis:
all coefficients are nonpositive, and the basis functions are nonnegative.
This fixed-sign property, in addition to degree-7 exactness, gives

$$
E(g)=C\,g^{(8)}(\eta)
$$

for some $\eta\in(0,6)$. To determine $C=\int_0^6K(t)\,dt$, apply the rule to
$g(t)=t^8$. Then $g^{(8)}=8!$,

$$
\int_0^6 t^8\,dt=\frac{6^9}{9},
$$

and

$$
\frac{1}{140}
\left(
216\cdot1^8+27\cdot2^8+272\cdot3^8
+27\cdot4^8+216\cdot5^8+41\cdot6^8
\right)
=\frac{5600016}{5}.
$$

Therefore

$$
E(t^8)
=
\frac{6^9}{9}-\frac{5600016}{5}
=-\frac{1296}{5}.
$$

Since $8!=40320$,

$$
C=\frac{E(t^8)}{8!}=-\frac{9}{1400}.
$$

For a physical panel $x=x_0+ht$, the eighth derivative contributes $h^8$ and
$dx=h\,dt$, so the one-panel error is

$$
E_{\mathrm{panel}}
=
-\frac{9}{1400}h^9 f^{(8)}(\xi)
$$

for some $\xi\in(x_0,x_6)$.

### Composite Weights

Every panel contributes

$$
\frac{h}{140}(41,216,27,272,27,216,41).
$$

When panels are tiled, adjacent panels share the boundary node. At an interior
panel junction, the right endpoint of one panel and the left endpoint of the
next panel both contribute $41h/140$, so the stored composite weight is

$$
\frac{h}{140}(41+41)=\frac{82h}{140}.
$$

The endpoints of the full interval are not shared, so they keep weight
$41h/140$.

### Composite Error Bound

On $[0,1]$, if `n % 6 == 0` and $h=1/n$, there are $n/6$ panels. Hence

$$
|E_{\mathrm{comp}}|
\le
\frac{n}{6}\frac{9}{1400}h^9\|f^{(8)}\|_\infty
=
\frac{3}{2800}h^8\|f^{(8)}\|_\infty,
$$

using $nh=1$. This is the bound implemented by
`newton_cotes_with_bound(f, Wk∞_bound)`:

$$
\left|\int_0^1 f(x)\,dx-S\right|
\le
\frac{3}{2800}h^8\,\|f^{(8)}\|_\infty.
$$

### From one-dimensional quadrature to a weighted inner product

On an interval of length $L=nh$, the same panel argument gives
$|I-Q|\le(3/2800)Lh^8\|f^{(8)}\|_\infty$.
`newton_cotes_weights(n,h)` supports this general length;
`newton_cotes_with_bound` specializes to $L=1$ and returns the quadrature
value and remainder allowance separately. In interval mode the value encloses
the exact weighted sample sum. It must still be enlarged by the remainder
allowance to enclose the integral.

Let $F(x,y)$ have continuous pure derivatives through order eight on
$[0,L_0]\times[0,L_1]$. Let $x_i,w_i$ and $y_j,v_j$ be the composite
quadrature nodes and weights, with mesh widths $h_0$ and $h_1$, respectively.
The double integral and its quadrature approximation are

$$
I=\int_0^{L_1}\int_0^{L_0}F(x,y)\,dx\,dy,
\qquad
Q=\sum_i\sum_j w_i v_j F(x_i,y_j).
$$

The weights have a product structure; no factorization of $F(x,y)$ into
a function of $x$ times a function of $y$ is assumed.

For each fixed $y$, define the error from quadrature in $x$:

$$
e_x(y)=\int_0^{L_0}F(x,y)\,dx-\sum_i w_iF(x_i,y).
$$

Similarly, at each $x_i$, define the error from quadrature in $y$:

$$
e_y(x_i)=\int_0^{L_1}F(x_i,y)\,dy-\sum_jv_jF(x_i,y_j).
$$

Adding and subtracting the intermediate expression
$\sum_iw_i\int_0^{L_1}F(x_i,y)\,dy$ gives the exact identity

$$
I-Q=\int_0^{L_1}e_x(y)\,dy+\sum_iw_i e_y(x_i).
$$

Apply the one-dimensional estimate to each slice. With $C=3/2800$,

$$
|e_x(y)|\le C L_0h_0^8\|\partial_x^8F\|_\infty,
\qquad
|e_y(x_i)|\le C L_1h_1^8\|\partial_y^8F\|_\infty,
$$

where both derivative norms are over the whole rectangle. Since the weights
satisfy $w_i\ge0$ and $\sum_iw_i=L_0$, integrating the first error bound
and summing the second gives

$$
|I-Q|
\le\frac{3L_0L_1}{2800}
\left(h_0^8\|\partial_x^8F\|_\infty+
      h_1^8\|\partial_y^8F\|_\infty\right).
$$

Thus the estimate applies to a general two-variable integrand.
No mixed eighth-order derivative is needed for this tensor-product estimate.
For an inner product, take $F=\sum_i u_i v_i\sin\theta$ and bound its pure
eighth derivatives with the Leibniz rule, as explained in
[comb_bound.md](comb_bound.md). The sampled fields must already carry the
radial weights appropriate to the physical inner product.

The `inner_product` methods in [My_Scalar.jl](../../src/My_Scalar.jl) and
[My_Vector.jl](../../src/My_Vector.jl) multiply by $4\pi\kappa^3$, where
$\kappa=\texttt{scl\_fac}$: $2\pi$ comes from azimuthal integration, $2$ from
extending the reflection-symmetric integrand from $\theta\in[0,\pi/2]$ to
$[0,\pi]$, and $\kappa^3$ from the radial scaling in $r=\kappa\tan\beta$.
The remaining radial Jacobian is absorbed by weighting each field with
$w(\beta)=\sin\beta/\cos^2\beta$. The prefactor multiplies both the quadrature
value and its error allowance. Interval samples without derivative-bound
tables enclose only the quadrature sum.

---

## Finite differences and sharper derivative bounds

We already have an a priori bound and want to refine it to obtain a smaller
error allowance in numerical integration. Evaluating an assembled function
before taking a maximum can retain cancellation between its terms.
A finite difference estimates its derivative from these samples; higher
derivatives control the discrepancy. Refinement is not guaranteed to decrease
a bound on every grid.

### Finite differences are averages of derivatives

Let $m\ge0$ be an integer, $h>0$, and assume f is real and $C^{m+2}$ on
the required stencil and interpolation domain. Write $g=\partial_x^m f$.
The implementation supports $0\le m<20$. From $N_X$ samples it returns all
$N_X-m$ consecutive differences, with centers $x_i+mh/2$; odd orders have
half-grid centers. At stencil center c, the normalized m-th difference is

$$
D_h^m f(c)=\frac{1}{h^m}
\sum_{j=0}^m(-1)^{m-j}\binom mj
f\bigl(c+(j-m/2)h\bigr).
$$

Repeated use of the fundamental theorem of calculus gives, for $m\ge1$,

$$
D_h^m f(c)
=\int_{[-1/2,1/2]^m}
g\left(c+h\sum_{j=1}^m t_j\right)\,dt_1\cdots dt_m.
$$

The average displacement is zero and its second moment is $mh^2/12$.
Taylor's formula about c therefore gives

$$
|D_h^m f(c)-g(c)|
\le \frac{mh^2}{24}\,\|g''\|_\infty.
$$

The bound on $g''=\partial_x^{m+2}f$ must hold on the whole stencil.
For $m=0$, the sample is already g(c), so this error is zero.

### From stencil centers to the whole interval

Suppose successive stencil centers, spaced by h, bracket the target interval.
Linear interpolation of g between two centers gives the additional error
$h^2\|g''\|_\infty/8$. Since the interpolation coefficients are nonnegative
and sum to one, replacing the derivative samples by their finite differences
adds at most the stencil error above. Consequently,

$$
\|g\|_\infty
\le \max_c|D_h^m f(c)|
+\frac{m+3}{24}h^2\|\partial_x^{m+2}f\|_\infty.
$$

The coefficient $(m+3)h^2/24$ combines the finite-difference error
$mh^2/24$ with the interpolation error $h^2/8$.
It is the primary correction used by `estimate_Wk∞_norm`. For a concave quadratic g
that is positive on the stencil domain and has its maximum midway between
neighboring centers, both contributions attain their bounds. Thus this
constant is sharp under the stated assumptions.

**Extended grids at the boundary.** The functions here are represented by
trigonometric and regular Bessel basis functions. We evaluate those same
functions on padded grids. For the trigonometric basis, boundary code `0`
corresponds to odd reflection and `1` to even reflection at the endpoint;
the basis provides these smooth extensions. The regular Bessel factors also
extend smoothly to the padded radial grid, using their parity extension and
analytic limiting values at the origin. See [fourier_basis.md](fourier_basis.md)
and [bessel_basis.md](bessel_basis.md).

For one order m, retaining $\lceil m/2\rceil$ ghost layers supplies stencil
centers that bracket the physical interval, including odd orders.
Recursive refinement needs the larger nested domains described below;
endpoint coverage alone is insufficient.

The derivative bounds used in the error terms must cover the corresponding
stencil domains. Reflection preserves absolute derivative values, so bounds
for the trigonometric functions on the fundamental interval also cover their
reflections. For radial extensions not supplied by reflection, the bounds
must cover the enlarged domain, as in the enlarged β-domain used for the
initial priors. The same domain requirement applies when refined bounds are
reused in the recursion.

### Two coordinate directions

For $g=\partial_x^m f(x,y)$, apply the primary estimate at each transverse
sample and then interpolate in y. Write S for the maximum absolute finite
difference and let

$$
B_0\ge\|\partial_x^{m+2}f\|_\infty,\qquad
B_1\ge\|\partial_y^2\partial_x^m f\|_\infty.
$$

Under the domain conditions above and transverse-grid coverage,

$$
\|g\|_\infty
\le S+\frac{m+3}{24}h_0^2 B_0+\frac{h_1^2}{8}B_1.
$$

Thus the second explicit correction is a mixed-derivative bound. It is not
another copy of the primary order-$(m+2)$ derivative bound. For a genuinely
one-dimensional calculation, the transverse term is absent.

### Closing an estimate with frequency bounds

For a trigonometric polynomial of maximum wavenumber $M_j$, Bernstein's
inequality gives $\|\partial_j^2g\|_\infty\le M_j^2\|g\|_\infty$ on a full
periodic domain. A symmetry-reduced domain may be used when its supremum equals
the full-period supremum. Band limitation alone does not justify this replacement
on an arbitrary restricted interval or between unrelated domains.

The [symmetry reference](symmetry_and_domains.md#domain-rules) proves the
quarter- and half-period reductions used here and maps them to the active
notebook calls. Its [workflow table](symmetry_and_domains.md#workflow-table)
also identifies calculations that instead require full-period priors or
explicit radial derivative bounds.

Let $a_0=(m+3)h_0^2/24$ and $a_1=h_1^2/8$. Replacing a correction bound by
$M_j^2\|g\|_\infty$ moves that term to the left. The resulting pattern is

$$
\|g\|_\infty
\le
\frac{S+\sum_{j\ \mathrm{explicit}}a_jB_j}
     {1-\sum_{j\ \mathrm{frequency}}a_jM_j^2}.
$$

Each direction uses either an explicit correction, a frequency correction,
or no correction when its omission is justified. The denominator must be
strictly positive; the code checks this with `L∞_fac > 0`. With both slots
`nothing`, the result is only a sampled finite-difference maximum.

### Why refinement proceeds from high order to low order

To bound order m using an explicit primary correction, we need order m+2.
A target of orders $0,\ldots,p$ therefore requires two seed bounds for orders
p+1 and p+2. The recursion first refines order p using the second seed, then
order p−1 using the first seed, and continues downward using the bounds
already obtained. These are two interleaved derivative chains, one for each
parity of the order.

For p=8, start with order-9 and order-10 bounds. The order-10 seed controls
the order-8 correction, the order-9 seed controls the order-7 correction, and
the refined order-8 bound then controls order 6. The reused bound must cover
the lower-order stencil, not just its target interval.

**Choosing the padding from the stencils.** Work backward from the interval
where the final bounds are needed. An even order $j=2r$ uses a centered
stencil with radius $r$ grid steps. An odd order $j=2r-1$ has radius
$r-1/2$, but its half-grid centers must extend half a step beyond both target
endpoints to bracket them. Thus either parity requires the next higher bound
on an interval enlarged by $r=\lceil j/2\rceil$ steps on each side.
These enlargements accumulate because the next bound is itself obtained by
the same procedure.

Write $I_s=[-sh,L+sh]$ and define

$$
p_0=p_1=0,\qquad p_{j+2}=p_j+\lceil j/2\rceil.
$$

The order-j estimate bounds its derivative on $I_{p_j}$: the stencils needed
to interpolate across that interval lie in $I_{p_{j+2}}$, where the
order-(j+2) bound is available. The code takes a maximum over all valid
stencils, which is at least the maximum over this subset. Descending induction
therefore preserves the required domains.
The seeds must cover $I_{P(k)}$, where

$$
P(k)=\max(p_{k+1},p_{k+2})=\frac{q(q+1)}2,\qquad q=\lceil k/2\rceil.
$$

The sum is $P(k)=1+2+\cdots+q$. For example, the even chain requires:

| Derivative bound | Required interval | Added layers per side |
|---|---|---|
| Order 2 | $I_0$ | 0 |
| Order 4 | $I_1$ | 1 |
| Order 6 | $I_3$ | 2 |
| Order 8 | $I_6$ | 3 |
| Order 10 | $I_{10}$ | 4 |
| Order 12 | $I_{15}$ | 5 |

Refinement through order 8 stops at the order-10 seed and therefore uses
$1+2+3+4=10$ layers. Refinement through order 10 continues to the order-12
seed and uses $1+2+3+4+5=15$. The corresponding odd chains give the same
counts. This derives the padding from the requested derivative order.

`derivative_padding(k)` returns this minimum for the nested scheme with $0\le k<20$: orders 2, 4, 8,
and 10 require 1, 3, 10, and 15 layers respectively. With actual padding
$S>P(k)$, the same argument shifts every domain by $S-P(k)$, giving all
returned orders on the common enlarged interval $I_{S-P(k)}$, provided the
seed bounds cover $I_S$. Consequently, if a later calculation needs every
returned derivative on an additional $H$-layer interval, choose

$$
S=H+P(k).
$$

The RHS calculation needs a ten-layer interval. To supply its product priors,
the order-10 ψ refinement therefore uses $S=10+15=25$ layers; see the
[workflow padding choices](finite_rank_rhs.md#how-the-workflow-padding-is-chosen).
Frequency-only refinement does not reuse lower-domain bounds; the angular
assembly retains its existing eight-layer grid. Angular Bernstein domain
conditions remain separate from this padding argument.

### Ordinary and special parameter preparation

[`Linf_to_paras`](../../src/Matrix_Func.jl) packages previously available bounds
for the derivative-refinement calculation. It does not sample a function or
refine any norm itself. Its purpose is to tell each directional refinement
which two high-order bounds to start from and how to control variation in the
other coordinate.

#### What one package contains

Let $p$ be the highest derivative order we want to refine, and let $l$ be the
number of function components. The call
`P = Linf_to_paras(priors, p+2, frequencies)` returns an $l\times2$ matrix.
The two columns always mean derivative directions, in both modes:

| Package | Component | Primary direction | Transverse direction |
|---|---|---|---|
| `P[i,1]` | $i$ | $\beta$ | $\theta$ |
| `P[i,2]` | $i$ | $\theta$ | $\beta$ |

Each package contains two slots. **Slot 1** supplies the order-$(p+1)$ and
order-$(p+2)$ seed bounds for the backward recursion. **Slot 2** supplies the
transverse correction: either a frequency or a list of mixed-derivative bounds
for orders $0,\ldots,p$. Thus a package such as
`[["Wk∞", [16,32]], ["freq", 2]]` means: start the primary chain with seed
bounds 16 and 32, and use transverse wavenumber 2.

The helper's argument named `k` is $p+2$, not the target order $p$.
Since prior row $q+1$ represents derivative order $q$, the seed entries are
at indices $p+2$ and $p+3$:

| Target order $p$ | Argument `k` | Prior indices read | Derivative orders supplied |
|---|---|---|---|
| 2 | 4 | 4, 5 | 3, 4 |
| 4 | 6 | 6, 7 | 5, 6 |
| 8 | 10 | 10, 11 | 9, 10 |

`frequencies = [Mβ,Mθ]` contains maximum wavenumbers in the respective
coordinates. Omit `ind` for ordinary mode; provide `ind` for special mode.
The choice is explicit, not inferred from the function being bounded.

In the examples below, `dt` converts numbers to the active numeric type.
The output comments show numerical values without interval display formatting.

**From packages to norms.** Pass the whole package matrix as
`norm_lst(field, "W$(p)Inf"; paras=P, domain=domain, padding=pad)`. This selects refinement
from samples. For each component $i$, `norm_lst` passes `P[i,:]` to the
ScalarFunc `get_Wk∞_norm_lst` wrapper, which computes the two directional
chains. The result is a $(p+1)\times2$ table of length-$l$ vectors:
`bounds[m+1,j][i]` bounds component $i$'s order-$m$ derivative in direction $j$.
Row 1 holds order-0 bounds, row 2 order-1 bounds, and so on.

The calls below use sampled arrays from the field being verified, with priors
for that same field. Arrays `A`, `A1`, `A2`, `A3`, `A_mode2`, and `F` are input
data, not outputs of `Linf_to_paras`. For packaged recursive refinement,
set `pad = derivative_padding(p)` or a larger actual padding. Each component
array has shape `(Nβ+1+2pad, Nθ+1+2pad)`, with positive `Nβ` and `Nθ` divisible
by 100. Pass `padding=pad` explicitly; the keyword defaults to zero. The same
actual padding is used for spacing and transverse crops on both axes.
Component order must match the prior vectors. The `"LInf"` and frequency-vector
paths instead use unpadded samples and `padding=0`.

For these calls, set the unpadded coordinate lengths and field scaling to

```julia
domain = [Pi/dt(2), Pi/dt(2)]
scl_fac = dt(1)
```

Use the corresponding lengths and scaling when applying the examples to a
notebook's field. `norm_lst` infers the mesh widths from these lengths and the
unpadded grid counts; it does not generate or pad the samples.

#### How correction tags select padding

The ScalarFunc wrapper checks each directional `package` separately. Its first
slot, `primary = package[1]`, describes the correction in the direction being
differentiated; the second slot describes the transverse correction. The shared
padding must satisfy both directional checks:

```julia
primary = package[1]
recursive = primary !== nothing && primary[1] in ("Wk∞", "L∞_freq")
required_padding = recursive ? derivative_padding(k) : cld(k, 2)
@assert padding >= required_padding "Insufficient padding for derivative refinement"
```

Here `k` is the requested refinement order, not the `k+2` argument supplied to
`Linf_to_paras`. `primary[1]` is the tag identifying the correction method:

| Primary tag | How the correction is obtained | Padding check |
|---|---|---|
| `"Wk∞"` | Explicit higher-derivative seeds start the downward recursion. | `derivative_padding(k)` |
| `"L∞_freq"` | Deferred angular frequency powers are scaled by refined β bounds and converted to `"Wk∞"` before recursion. | `derivative_padding(k)` |
| `"freq"` | A frequency correction is applied independently at each derivative order. | `cld(k, 2)` |

Julia's `&&` short-circuits: if `primary === nothing`, the tag lookup is skipped
and `recursive` is false. This classifies the slot; it does not justify omitting
a correction needed for a continuous bound. The current scalar wrapper requires
a tagged θ primary slot: its later θ-tag lookup does not accept `nothing`.
The local `recursive` flag selects
the padding assertion. The raw-array routine separately selects its numerical
recursion from the resolved primary tag.

For `k=8`, the recursive check requires ten layers, while an independent
frequency correction needs four layers for stencil coverage, together with its
Bernstein domain assumptions. This four-layer rule concerns finite differences:
the vector `norm_lst` shortcut with `paras::Vector{dtype}` instead estimates an
order-zero supremum and multiplies by frequency powers, so it uses unpadded
samples. The `"freq"` appearing in a package's transverse slot does not disable
recursion selected by its primary slot.

#### Ordinary mode: use frequencies in both coordinates

Here both coordinates retain their trigonometric structure, so each chain can
use the other coordinate's frequency for its transverse correction. The input
`priors` is a derivative-order-by-direction table whose entries are vectors of
component bounds: `priors[q+1,j][i]` bounds component $i$'s order-$q$ derivative
in direction $j$, where $j=1$ means $\beta$ and $j=2$ means $\theta$.

For component $i$ in direction $j$, the helper selects
`[priors[k,j][i], priors[k+1,j][i]]`. Thus the $\beta$ chain uses the
$\beta$ column and the $\theta$ chain uses the $\theta$ column. Each seed
bounds a derivative in the direction being refined; no comparison between
$\beta$ and $\theta$ derivative sizes is needed. The transverse frequency is
$M_\theta$ for the $\beta$ chain and $M_\beta$ for the $\theta$ chain.

**Ordinary mode, example 1: one component, target order 2.**
The prior table has five rows (orders 0–4) and two columns (β, θ); each entry
is a one-component vector.

```julia
priors = [dt.([b]) for b in [2 2; 4 3; 8 6; 16 12; 32 24]]
p = 2
pad = derivative_padding(p) # 1; set pad=2 if the samples actually have two layers
P = Linf_to_paras(priors, p+2, dt.([3, 2]))
# size(P) == (1, 2)
# P[1,1] = [["Wk∞", [16, 32]], ["freq", 2]]
# P[1,2] = [["Wk∞", [12, 24]], ["freq", 3]]
```

The β seeds 16 and 32 come from rows 4 and 5 of the β column: derivative
orders 3 and 4. The θ chain uses 12 and 24 from the same rows of the θ column.
The transverse frequency is 2 for the β chain and 3 for the θ chain.

Now let `A` contain the padded samples of this component:

```julia
field = VectorFunc(scl_fac, "space"; u1=A)
bounds = norm_lst(field, "W$(p)Inf"; paras=P, domain=domain, padding=pad)
# size(bounds) == (3, 2)
# length(bounds[1,1]) == 1
Bβ2 = bounds[3,1][1]
Bθ2 = bounds[3,2][1]
```

`Bβ2` and `Bθ2` bound the second β and θ derivatives. Inside the β chain,
order 2 uses the order-4 seed 32, order 1 uses the order-3 seed 16, and order 0
uses the newly refined order-2 bound. For example, its first step is the
following call to the single-order estimator:

```julia
Nβ, Nθ = size(A) .- 1 .- 2pad
h = domain ./ dt.([Nβ, Nθ])
Bβ2_direct = estimate_Wk∞_norm(A[:, pad+1:end-pad], h, p;
    paras=[["Wk∞", dt(32)], ["freq", dt(2)]])
```

This computes the same order-2 bound as `Bβ2`. The transverse padding is
removed, and the recursion's seed pair is reduced to the single order-4 bound
needed for this step. The θ chain transposes the samples and reverses the
mesh widths. It starts order 2 with seed 24, then order 1 with seed 12,
and uses transverse frequency 3.

**Ordinary mode, example 2: three components, target order 8.**
Here each table entry is a three-component vector; both columns contain the
same bounds. The helper reads rows 10 and 11, which represent orders 9 and 10.

```julia
priors = [dt.([q+1, 2*(q+1), 3*(q+1)]) for q in 0:10, j in 1:2]
p = 8
pad = derivative_padding(p) # 10
P = Linf_to_paras(priors, p+2, dt.([4, 3]))
# size(P) == (3, 2)
# P[2,1] = [["Wk∞", [20, 22]], ["freq", 3]]
# P[2,2] = [["Wk∞", [20, 22]], ["freq", 4]]
```

These are the packages for component 2. Components 1 and 3 instead use
seed pairs `[10,11]` and `[30,33]`, with the same transverse frequencies.

Supply the three padded component arrays in that same order:

```julia
field = VectorFunc(scl_fac, "space"; u1=A1, u2=A2, u3=A3)
bounds = norm_lst(field, "W$(p)Inf"; paras=P, domain=domain, padding=pad)
# size(bounds) == (9, 2)
# length(bounds[1,1]) == 3
component2_beta_order8 = bounds[9,1][2]
component2_theta_order8 = bounds[9,2][2]
```

For component 2, `norm_lst` selects `P[2,:]`. The β recursion starts at order 8
with seed 22 (order 10), then uses seed 20 for order 7. The θ recursion reads
its own column, which contains the same seeds in this example. Each chain
continues down to order 0. This illustrates ordinary-mode preparation when the
required Bernstein domains are justified in both directions. The current
[eigenpair residual](../../notebooks/eigenpair_error_analysis.ipynb) uses the
special `ind=1` construction below: explicit β priors are refined first, and
the θ corrections use the refined β-order-0 and β-order-2 bounds.

#### Special mode: refine β first, then construct the θ bounds

This mode is used when a coordinate transformation changes the radial
representation so that the ordinary $\beta$ frequency estimate is unavailable,
or when a mixed-parity trigonometric expression is refined on a restricted
radial interval without a valid local Bernstein estimate. The function must
still satisfy the angular Bernstein hypotheses in $\theta$.

The input layout changes: `priors[q+1]` is now a component-by-mode matrix of
order-$q$ $\beta$-derivative bounds. **`ind` selects a mode column inside each
of these matrices.** It does not select a derivative direction or a component.
The output still has components as rows and $(\beta,\theta)$ as columns.

For the selected mode, refinement proceeds in this order:

1. Refine the $\beta$ chain using its two explicit high-order seeds and the
   transverse $\theta$ frequency. Let $B_\beta^{(0)}$ and $B_\beta^{(2)}$ be the
   resulting order-0 and order-2 bounds for one component $f$.
2. Use angular Bernstein estimates to obtain the $\theta$ chain's seeds and
   mixed-derivative corrections:

$$
\|\partial_\theta^q f\|_\infty
\le M_\theta^q B_\beta^{(0)},\qquad
\|\partial_\beta^2\partial_\theta^q f\|_\infty
\le M_\theta^q B_\beta^{(2)}.
$$

The first inequality supplies the two seed orders $q=p+1,p+2$; the second
supplies the transverse correction for each $q=0,\ldots,p$.
Only then can the $\theta$ refinement run.

When `Linf_to_paras` is called, the refined $\beta$ bounds do not yet exist.
Its $\theta$ package therefore contains **frequency powers awaiting a scale
factor**, tagged `"L∞_freq"`:

| Slot | Stored powers | Multiplier after β refinement |
|---|---|---|
| 1: seeds | $M_\theta^{p+1},M_\theta^{p+2}$ | $B_\beta^{(0)}$ |
| 2: mixed corrections | $M_\theta^0,\ldots,M_\theta^p$ | $B_\beta^{(2)}$ |

The ScalarFunc `get_Wk∞_norm_lst` wrapper performs these multiplications and
changes both tags to `"Wk∞"`. This is why the special path requires $p\ge2$:
it needs the refined order-2 $\beta$ bound. The raw-array refinement methods
cannot consume `"L∞_freq"` packages directly.

**Special mode, example 1: one component, two modes, target order 2.**
Now `priors` is a list of five matrices. Entry `q+1` is a `1 × 2` matrix
of order-q β bounds: its columns are modes, not coordinate directions.

```julia
priors = [dt.([q+1 10*(q+1)]) for q in 0:4]
p = 2
pad = derivative_padding(p) # 1; set pad=2 if the samples actually have two layers
P = Linf_to_paras(priors, p+2, dt.([7, 2]); ind=2)
# size(P) == (1, 2)
# P[1,1] = [["Wk∞", [40, 50]], ["freq", 2]]
# P[1,2] = [["L∞_freq", [8, 16]], ["L∞_freq", [1, 2, 4]]]
```

`ind=2` selects the second mode's β seeds, 40 and 50. The θ lists contain
`2 .^ (3:4)` for the two seed orders and `2 .^ (0:2)` for mixed corrections.
Suppose β refinement gives order-0 bound 3 and order-2 bound 5. The ScalarFunc
wrapper then replaces the θ package by
`[["Wk∞", [24,48]], ["Wk∞", [5,10,20]]]`: multiply the first list by 3 and
the second by 5. The β frequency 7 is unused in this special mode.

To perform the actual refinement, use the padded samples of mode 2:

```julia
field_mode2 = VectorFunc(scl_fac, "space"; u1=A_mode2)
bounds_mode2 = norm_lst(field_mode2, "W$(p)Inf"; paras=P, domain=domain, padding=pad)
# size(bounds_mode2) == (3, 2)
# length(bounds_mode2[1,1]) == 1
B0 = bounds_mode2[1,1][1]
B2 = bounds_mode2[3,1][1]
theta_order2 = bounds_mode2[3,2][1]
```

The numbers 3 and 5 above illustrate the scaling; the actual call computes
`B0` and `B2` from `A_mode2`. It then changes `P[1,2]` in place to the package
with seeds `[8*B0,16*B0]` and mixed corrections `[B2,2*B2,4*B2]`, both tagged
`"Wk∞"`. The θ recursion starts at order 2 using primary bound `16*B0` and
transverse bound `4*B2`; it samples θ differences and returns the refined
`theta_order2`. Thus the frequency products supply error allowances for a
second refinement, rather than being the final θ bounds.

**Special mode, example 2: three components, two modes, target order 4.**
Each prior entry is now a `3 × 2` matrix: rows are components and columns
are modes. Changing `ind` selects a different mode for all three components.

```julia
priors = [dt.([q+1       10*(q+1);
              2*(q+1)   20*(q+1);
              3*(q+1)   30*(q+1)]) for q in 0:6]
p = 4
pad = derivative_padding(p) # 3
P = Linf_to_paras(priors, p+2, dt.([7, 3]); ind=1)
# size(P) == (3, 2)
# P[2,1] = [["Wk∞", [12, 14]], ["freq", 3]]
# P[2,2] = [["L∞_freq", [243, 729]], ["L∞_freq", [1, 3, 9, 27, 81]]]
P_mode2 = Linf_to_paras(priors, p+2, dt.([7, 3]); ind=2)
# P_mode2[2,1] = [["Wk∞", [120, 140]], ["freq", 3]]
# P_mode2[2,2] has the same frequency-power lists as P[2,2].
```

The second row of P is component 2; `ind` chooses a column inside each prior
matrix. The β seeds now represent orders 5 and 6. Before refinement, every
component has the same θ frequency powers; the wrapper subsequently scales
them by that component's own refined β bounds.

Let `F` have axes `(β grid, θ grid, mode, component)`, including
`pad = derivative_padding(4) = 3` layers on each side of both axes. Pair each mode's sampled field with its own package matrix:

```julia
fields = [VectorFunc(scl_fac, "space";
    e1=F[:, :, ind, 1], e2=F[:, :, ind, 2], e3=F[:, :, ind, 3])
    for ind in 1:2]
P_modes = [P, P_mode2]
bounds_modes = [norm_lst(fields[ind], "W$(p)Inf";
    paras=P_modes[ind], domain=domain, padding=pad) for ind in 1:2]
# length(bounds_modes) == 2
# size(bounds_modes[2]) == (5, 2)
# length(bounds_modes[2][1,1]) == 3
mode2_component2_theta_order4 = bounds_modes[2][5,2][2]
```

`P_modes[2]` is the package matrix for mode 2, so its component-2 β chain starts
with seeds 120 and 140. After that chain has produced its own order-0 and
order-2 bounds, the wrapper uses those values to scale `P_modes[2][2,2]` and
then refines θ. The corresponding values from mode 1 are not reused.

There are three indexing levels in `bounds_modes[ind][m+1,j][i]`: the mode
selects a table, derivative order and direction select a cell, and component
selects an entry of the cell's vector. Each mode gets its own $(p+1)\times2$
table. The exported helper
[`refine_channel_bounds`](../../src/Comb_Bound.jl)
uses this same per-mode construction, creating each package just before its
`norm_lst` call.

The helper allocates new packages without modifying `priors`. The ScalarFunc
wrapper subsequently modifies the deferred packages, so create fresh packages
for each refinement call. The resulting derivative bounds then enter the
integration error calculation described next.

### Returning to integration

The refinement stage produces derivative bounds, not an integral. The product
rule combines bounds for the factors in the integrand; the pure eighth-order
bounds then enter the quadrature remainder described above. Sharper derivative
bounds can make a small residual's L² enclosure useful for the verification.

The roles are: `finite_difference` forms sampled differences;
`estimate_Wk∞_norm` adds one order's allowances;
`get_Wk∞_norm_lst` performs the descending recursion; and `Linf_to_paras`
prepares seeds and transverse information. Their array contracts belong in
[Matrix_Func.jl](../../src/Matrix_Func.jl), with the field wrappers in
[My_Scalar.jl](../../src/My_Scalar.jl) and [My_Vector.jl](../../src/My_Vector.jl).

---

## $LDL^T$ Certificate — `verify_spd_by_ldlt`

`verify_spd_by_ldlt(A)` certifies positive definiteness of a symmetric interval
matrix by an interval LDL$^T$ factorization without pivoting. Symmetry of the
represented matrix is a hypothesis: the code reads only the diagonal and lower
triangle and checks squareness, not symmetry. The recurrence is

$$
D_k
=
A_{kk}
-
\sum_{j<k}L_{kj}^2D_j,
$$

and, for $i>k$,

$$
L_{ik}
=
\frac{
A_{ik}-\sum_{j<k}L_{ij}L_{kj}D_j
}{
D_k
}.
$$

All arithmetic is interval arithmetic. If every pivot interval has positive
lower bound,

$$
\inf D_k>0\qquad\text{for every }k,
$$

then every symmetric point matrix represented by the interval matrix has a factorization
with positive diagonal pivots enclosed by the computed intervals. For any
nonzero vector $x$,

$$
x^TAx
=
(L^Tx)^TD(L^Tx)
>0,
$$

because $L$ is unit lower triangular, so $L^Tx\ne0$, and $D$ is positive
diagonal. Hence the matrix is SPD.

The implementation requires finite pivot endpoints as well as a positive lower
bound, rejecting NaN/NaI, empty, and unbounded pivots even if they arise during factorization.

If the check fails, it does not prove the matrix is non-SPD. It only means this
particular interval LDL$^T$ certificate did not succeed.

---

## Generalized Eigenvalue Certificate — `verify_general_eig`

Let A and B be the symmetric Q and energy matrices on the finite-dimensional
trial space $V_N$, let X contain the retained trial coefficients, and write
$\Lambda=\operatorname{diag}(\lambda_j)$. The manuscript's constant (label
`constant small` in the paper) is the constrained quotient

$$
K_6=\sup_{\substack{u\ne0\\X^TAu=0}}\frac{u^TAu}{u^TBu}.
$$

Here u denotes a coefficient vector in $V_N$. The constraint expresses
Q-orthogonality to the span of the retained columns. A full-space bound also
needs the complementary-space argument from the application.

Let $\tau$ be the scalar threshold to be certified as an upper bound. The
helper constructs

$$
S
=
\tau\,B
-A
+AX\Lambda^{-1}X^TA,
$$

Assume B is positive definite and every $\lambda_j$ is nonzero; interval
eigenvalue entries must exclude zero before inversion. The application uses
positive retained eigenvalues. These are input hypotheses, not conclusions
of `verify_general_eig`.

The constrained quotient is bounded by the unrestricted deflated quotient:

$$
K_6\le\sup_{u\ne0}
\frac{u^TAu-(X^TAu)^T\Lambda^{-1}(X^TAu)}{u^TBu}.
$$

The subtraction is zero when $X^TAu=0$, and taking the supremum over all
nonzero vectors can only enlarge the constrained supremum. Equality is not
assumed. This inequality holds for the supplied X and $\Lambda$ even when
they are approximate; separate checks are needed for claims about their
eigenpair residuals or normalization.

For any nonzero vector $u$,

$$
u^TSu
=
\tau\,u^TBu
-u^TAu
+(X^TAu)^T\Lambda^{-1}(X^TAu).
$$

Therefore $u^TSu>0$ is equivalent to

$$
u^TAu
-
(X^TAu)^T\Lambda^{-1}(X^TAu)
<
\tau\,u^TBu.
$$

Thus $S\succ0$ is a sufficient deflated matrix certificate. Equivalently,
with $\psi_j$ denoting the retained fields represented by X, it proves

$$
\tau\|u\|_E^2
+
\sum_{j=1}^m\lambda_j^{-1}\langle u,Q\psi_j\rangle_\Omega^2
-
\langle u,Qu\rangle_\Omega
>0\qquad(u\ne0).
$$

Since B is positive definite and the trial space is finite dimensional,
the Rayleigh quotient attains its maximum on the B-unit sphere. Consequently,

$$
K_6\le
\sup_{u\ne0}\frac{u^TAu-(X^TAu)^T\Lambda^{-1}(X^TAu)}{u^TBu}
<\tau.
$$

This is not a lower bound on $\lambda_{\min}(A,B)$. The retained directions have
been deflated, so the certificate bounds the quotient after subtracting the
captured modes. If retained eigenvectors are discussed as a sanity check, the
statement must be qualified: under exact generalized eigenpairs and the chosen
normalization, the retained block behaves as expected. The code should not be
read as unconditionally proving $x^TSx=\tau$ for approximate retained
vectors.

The implementation constructs S and then calls `verify_spd_by_ldlt(S)`.
It does not separately certify B, eigenpair accuracy, or orthonormality.
In interval mode the matrix arithmetic encloses S before the pivot test;
a Float64 calculation of the same formula is not that rigorous certificate.

---

## Symmetric and Gram spectral certificates

`verify_symmetric_spectrum(A,t)` calls the interval LDL$^T$ test on $tI-A$.
For a symmetric A, success proves $\lambda_{\max}(A)<t$. This is one-sided:
it bounds $\|A\|_{\mathrm{op}}$ when A is positive semidefinite, or when a
corresponding lower spectral bound is also supplied.

`verify_gram_spectrum(W,K,t)` uses **upper matrices in the
positive-semidefinite order**. Its inputs enclose exact symmetric matrices
$W\succeq0$ and K; the exact W must be a Gram matrix. Symmetry, the Gram
interpretation and enclosure provenance are caller hypotheses, not established
by this function. The code returns false unless `inf(t)>0`.

For a symmetric center C and symmetric entrywise deviation bound E, with
$E_{ij}\ge0$ and $|A_{ij}-C_{ij}|\le E_{ij}$, define for any positive vector p

$$
\mathcal U(C,E,p)=C+\operatorname{diag}((Ep)_i/p_i).
$$

This is an upper matrix for A. Indeed, using
$2|x_i x_j|\le(p_j/p_i)x_i^2+(p_i/p_j)x_j^2$ and symmetry,

$$
x^T(A-C)x\le\sum_{ij}E_{ij}|x_i x_j|
\le\sum_i\frac{(Ep)_i}{p_i}x_i^2.
$$

The implementation chooses point midpoint matrices and computes outward-rounded
entrywise deviations. For W, these are

```julia
c = mid.(W)
e = sup.(abs.(W .- interval_matrix(c,c)))
```

The same construction is applied to K. Thus E encloses the distance from the
chosen center to every matrix in the input interval; a rounded radius is not
substituted for this deviation. The implementation constructs V for K with
unit weights and U for W with positive weights chosen by a floating-point search. It tries 241 values
$a=\exp(s)$ for $s\in[-6,6]$ and weights
$p=(a,\ldots,a,a^{-1},\ldots,a^{-1})$, split after `floor(n/2)` entries.
A Float64 Cholesky/eigenvalue score only chooses p; it is not a certificate.
The final upper matrices are formed with interval arithmetic and tested by

$$
U\succ0,\qquad tU-UVU\succ0.
$$

For the exact matrices enclosed by the inputs and constructed intervals, these
checks imply $K\preceq V\prec tU^{-1}$ and $W\preceq U$. Indeed, congruence
by $U^{-1}$ gives

$$
tU^{-1}-V=U^{-1}(tU-UVU)U^{-1}\succ0.
$$

To use $W\preceq U$, set $R=U^{-1/2}W^{1/2}$. Then

$$
RR^*=U^{-1/2}WU^{-1/2}\preceq I,
$$

so $\|R\|_{\mathrm{op}}\le1$ and hence

$$
W^{1/2}U^{-1}W^{1/2}=R^*R\preceq I.
$$

This does not require W to be invertible or to commute with U. Consequently,
using $t>0$,

$$
W^{1/2}KW^{1/2}\preceq W^{1/2}VW^{1/2}
\preceq tW^{1/2}U^{-1}W^{1/2}
\preceq tI,
$$

with largest eigenvalue strictly below the positive threshold t. To see
strictness, use the strict V inequality on vectors not killed by $W^{1/2}$;
on its kernel the eigenvalue is zero, below t. Equivalently, for a column map F
with $F^*F=W$, the ambient operator $FKF^*$ has the same nonzero spectrum and
zero on the complementary space. The exact W may be singular: the current
helper does not require the interval box for W to pass a positive-definite test.

The upper-matrix construction can add slack, so a failed test is inconclusive.
The symmetric enclosures and exact Gram interpretation are the mathematical
input assumptions.

Testing K and $-K$ separately controls both spectral signs. For a composition
$T=\Phi\Psi^*$ with $W=\Psi^*\Psi$ and $K=\Phi^*\Phi$, the squared operator
norm is $\lambda_{\max}(W^{1/2}KW^{1/2})$, so callers pass squared norm targets.
For $G_v=-\Phi_\perp\mathcal B^*$, the signed block
$K_{\rm quad}=\tfrac12\left(\begin{smallmatrix}0&-I\\-I&0\end{smallmatrix}\right)$
represents $\operatorname{sym}G_v$. Step 6 checks both signs at `32.52`,
then adds the identity outside the norm:

$$
\Theta_{\rm s}^v
=1+\|\operatorname{sym}G_v\|\le33.52\le33.6.
$$

This is the definition used in the paper. For the separate full operator norm
of $I+G_v$, the squared-norm block is tested at `B0_v^2-1`. See
[the quantity map](../paper_code_map.md#late-odd-projection).

The final tests inherit the interval LDL$^T$ lower-triangle convention. The
caller supplies symmetric Gram enclosures computed with the required
quadrature and normalization. Use interval mode for certificates; Float64
mode supplies numerical estimates.


---

## Symmetric $3\times3$ Eigenvalues — `eig_sym3`

`eig_sym3(a,b,c,d,e,f)` computes the eigenvalues of the symmetric matrix

$$
A=
\begin{pmatrix}
a&b&c\\
b&d&e\\
c&e&f
\end{pmatrix}.
$$

The main branch uses the trigonometric cubic formula, with outputs in largest,
smallest, middle order. Interval outputs enclose those eigenvalues and may
overlap. If the interval scale contains zero, the routine returns three copies
of a Gershgorin enclosure instead of dividing by that scale. The derivation
below explains the cubic formula first, then the interval square calculation
and fallback.

In Float64 mode, a computed scale `p == 0` returns `[q,q,q]`; this is an exact
floating-point zero test, not the tolerance-based `is_zero` test. Otherwise the
code uses the cubic formula and clips its acos argument to `[-1,1]`.
Float64 results are approximations rather than interval certificates.

Define the trace center

$$
q=\frac{\operatorname{tr}A}{3},
$$

and the centered matrix

$$
B=A-qI.
$$

Then $B$ is real symmetric and trace-free. Its eigenvalues are real, and if they
are denoted by

$$
\mu_1,\mu_2,\mu_3,
\qquad
\mu_1+\mu_2+\mu_3=0,
$$

then the eigenvalues of $A$ are $\lambda_i=q+\mu_i$.

The code defines the scale $p$ by

$$
p^2
=
\frac{1}{6}\operatorname{tr}(B^2)
=
\frac{1}{6}(\mu_1^2+\mu_2^2+\mu_3^2).
$$

If $p=0$, then $\mu_1^2+\mu_2^2+\mu_3^2=0$, hence all $\mu_i=0$. Since a real
symmetric matrix is diagonalizable, $B=0$, so $A=qI$ and all eigenvalues are
$q$. The interval implementation covers this case with the fallback described below.

Assume from now on that $p>0$. Normalize the centered eigenvalues by

$$
\nu_i=\frac{\mu_i}{2p}.
$$

Then

$$
\nu_1+\nu_2+\nu_3=0,
\qquad
\nu_1^2+\nu_2^2+\nu_3^2=\frac32.
$$

The numbers $\nu_i$ are roots of a depressed cubic. Since their sum is zero, the
monic cubic is

$$
(z-\nu_1)(z-\nu_2)(z-\nu_3)
=
z^3+sz-r_0,
$$

where

$$
s
=
\nu_1\nu_2+\nu_1\nu_3+\nu_2\nu_3
=
-\frac12(\nu_1^2+\nu_2^2+\nu_3^2)
=
-\frac34
$$

and

$$
r_0=\nu_1\nu_2\nu_3.
$$

Thus each normalized eigenvalue satisfies

$$
z^3-\frac34 z-r_0=0.
$$

Now use

$$
\cos(3\phi)=4\cos^3\phi-3\cos\phi.
$$

If $z=\cos\phi$, then

$$
z^3-\frac34z
=
\frac14\cos(3\phi).
$$

So the depressed cubic becomes

$$
\frac14\cos(3\phi)=r_0.
$$

The code computes

$$
r=\frac{\det B}{2p^3}.
$$

Because

$$
\det B
=
\mu_1\mu_2\mu_3
=
(2p)^3\nu_1\nu_2\nu_3
=
8p^3r_0,
$$

we have

$$
r=4r_0.
$$

Hence

$$
\cos(3\phi)=r.
$$

Taking

$$
\phi=\frac13\arccos(r),
$$

the three normalized roots are

$$
\nu_j
=
\cos\left(\phi+\frac{2\pi j}{3}\right),
\qquad j=0,1,2.
$$

Finally, undoing the normalization gives

$$
\lambda_j
=
q+2p\cos\left(\phi+\frac{2\pi j}{3}\right),
\qquad j=0,1,2.
$$

This is the formula implemented by `eig_sym3`. The code uses `cospi`, so it
stores the angle divided by $\pi$.

### Why the `acos` Argument Lies in $[-1,1]$

For an exact real symmetric matrix with $p>0$, the normalized eigenvalues
$\nu_i=\mu_i/(2p)$ are real and satisfy

$$
\nu_1+\nu_2+\nu_3=0,\qquad
\nu_1^2+\nu_2^2+\nu_3^2=\frac32,\qquad
r=4\nu_1\nu_2\nu_3.
$$

Squaring the first identity gives
$\nu_1\nu_2+\nu_1\nu_3+\nu_2\nu_3=-3/4$. Hence the polynomial with these
three real roots is

$$
P(z)=(z-\nu_1)(z-\nu_2)(z-\nu_3)
=z^3-\frac34z-\frac r4.
$$

Its derivative is $P'(z)=3z^2-3/4$, so its local maximum occurs at $z=-1/2$
and its local minimum at $z=1/2$. Because $P$ has three real roots, counting
multiplicity, the local maximum must be nonnegative and the local minimum
nonpositive. If the maximum were negative or the minimum positive, the
monotonicity on the three intervals separated by these critical points would
allow only one real root.

Evaluating at the critical points therefore gives

$$
P(-1/2)=\frac{1-r}{4}\ge0,\qquad
P(1/2)=-\frac{1+r}{4}\le0.
$$

The first inequality gives $r\le1$, and the second gives $r\ge-1$. Thus

$$
\boxed{-1\le r\le1.}
$$

Equality occurs for permutations of $(1,-1/2,-1/2)$ or its negative as the
normalized eigenvalues, so the bound is sharp. This proof establishes the
domain of $\arccos(r)$ independently of the cosine parametrization.

### Interval implementation and Gershgorin fallback

The formula for $p$ uses squares of the centered diagonal entries and the three
off-diagonal entries. If an interval $X$ crosses zero, `X*X` can have a negative
lower endpoint because interval multiplication treats its factors independently.
The code instead uses `abs(X)*abs(X)`. Writing $|X|=[u,v]$ with $0\le u\le v$,
this encloses the square range $[u^2,v^2]$ with outward rounding. The radicand
therefore stays nonnegative.

Let $P$ be the computed interval enclosing $p$. When $0\notin P$, the code uses
the cubic formula and intersects its interval $r$ with `ACOS_INT = [-1,1]`,
using `dec=:auto`, before `acos`. The preceding proof shows that every represented
matrix's exact normalized determinant survives this intersection. The removed
width may come from interval dependency as well as rounding.

When $0\in P$, division by $p$ is unsafe. This does not mean that every
represented matrix is a scalar multiple of the identity. Instead, form the
Gershgorin row bounds using interval arithmetic:

$$
\begin{aligned}
R_1&=|b|+|c|,& \ell_1&=a-R_1,& u_1&=a+R_1,\\
R_2&=|b|+|e|,& \ell_2&=d-R_2,& u_2&=d+R_2,\\
R_3&=|c|+|e|,& \ell_3&=f-R_3,& u_3&=f+R_3.
\end{aligned}
$$

The code returns three copies of

$$
G=\left[\min_{1\le i\le3}\inf(\ell_i),\;
        \max_{1\le i\le3}\sup(u_i)\right].
$$

To prove that $G$ contains every eigenvalue, fix any real symmetric matrix in
the input intervals and an eigenvector $v\ne0$. Choose $i$ with $|v_i|$ maximal.
The $i$-th row of $Av=\lambda v$ gives

$$
|\lambda-A_{ii}|
=\left|\sum_{j\ne i}A_{ij}\frac{v_j}{v_i}\right|
\le\sum_{j\ne i}|A_{ij}|.
$$

Thus the real eigenvalue lies in one of the row intervals, each enclosed by $G$.
Repeating $G$ preserves the three-output interface; it does not assert that the
eigenvalues coincide. For example, if $a=d=f=1$, $c=e=0$, and $b\in[-\delta,\delta]$
with $\delta\ge0$, the fallback returns three copies of an enclosure of
$[1-\delta,1+\delta]$.

These guarantees concern each sampled matrix. Bounds between spatial grid
points still require the separate derivative or perturbation estimates used
by the notebooks.

---

## Paper Reference

The quadrature weights and error constant correspond to the composite closed
7-point Newton-Cotes rule used in the paper's
`subsec:Quadrature` discussion. The $W^{k,\infty}$ norm estimation strategy follows the
interpolation/Bernstein enclosures in `subsec:Linf-envelope`, especially
`eq:f Iij est` and `eq:linf-envelope`. The LDL$^T$ and generalized eigenvalue
checks supply the finite-rank bounds in `Sec:num prev`. The $K_6$ check implements
the sufficient deflated matrix inequality following the manuscript definition
labelled `constant small`; it does not identify $K_6$ with the unrestricted
deflated quotient.
