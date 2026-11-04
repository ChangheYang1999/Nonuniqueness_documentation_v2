

# Bessel basis: from spectral roots to separated vector fields

This note explains the individual basis elements implemented in [`My_Bessel.jl`](../../src/My_Bessel.jl). The companion [finite-rank assembly note](finite_rank_assembly.md) explains how `My_Assembly.jl` combines them into matrices; the [RHS note](finite_rank_rhs.md) covers reconstructed fields and their bounds. The [source guide](../src.md) gives the broader reader path.

## 1. The objects produced by this file

The finite-rank calculation needs a divergence-free spectral basis on a ball. Separation of variables reduces each basis element to radial Bessel functions and angular Legendre functions. The root equations select the admissible radial spectral parameters; the same parameters determine both the sampled basis and its analytic derivative bounds.

Use $r=\rho/R$ for normalized radius, where $\rho$ is physical radius and $R=32$ in the finite-rank notebooks. The radial formulas below live on the unit interval, with smooth extensions used for padded grids. Degree $l\ge1$ labels the angular harmonic, and $j$ labels a root within that degree.

The notebook's preparation is:

```julia
Y_paras, Z_paras = load_besselj_zeros(root_file)
Ur_Y_lst = eval_Ur(r, Y_paras, "Y")
Ur_X_lst = eval_Ur(r, Y_paras, "X")
Ur_Z_lst = eval_Ur(r, Z_paras, "Z")
Uθ_lst = eval_Uθ(θ, max(length(Y_paras), length(Z_paras)))
Y_bounds, X_bounds, Z_bounds = Ur_bound_lst(Y_paras, Z_paras, K)
```

Here `root_file` is `data/generated/bessel_zeros.mat`; the path is relative to the caller's working directory. These calls produce **per-degree** samples and bounds, before finite-rank packing or summation. The calculation is

```text
root matrices → checked root parameters → radial coefficients
                                           ├→ radial samples
                                           └→ radial derivative bounds
angular degree + boundary code → angular samples and bounds
```

Sections 2–6 define the basis and explain the Bessel evaluations, root parameters, and radial derivative bounds. Sections 7–9 show how to read the resulting arrays, padding, and angular conventions and bounds. The final section identifies the handoff to assembly. Numeric enclosures refer to interval mode; Float64 evaluation is for numerical exploration.

## 2. Basis elements vs. component labels

The strings `"Y"`, `"X"`, `"Z"` select **components**, also called channels. There are only two basis families:

| Family | Parameter source | Nonzero channels | One basis element uses |
|---|---|---|---|
| Poloidal | a root of `func_Y` | Y and X | one shared parameter triple $(\alpha,A,B)$ |
| Toroidal | a root of `func_Z` | Z | one parameter pair $(\alpha,A)$, with standard $A=1$ |

Let $S_l(\theta)=P_l(\cos\theta)$ and $H_l(\theta)=-S_l'(\theta)$. Using the stored scalar factors of §9, its physical spherical components are

$$
\Phi^P_{l,j}(r,\theta)
=\bigl(R^Y_{l,j}(r)S_l(\theta),\ R^X_{l,j}(r)H_l(\theta),\ 0\bigr),
$$

or

$$
\Phi^Z_{l,j}(r,\theta)
=\bigl(0,\ 0,\ R^Z_{l,j}(r)H_l(\theta)\bigr).
$$

The poloidal Y and X channels are coupled by the divergence-free construction. They are not two independently adjustable basis elements. A coefficient multiplying $\Phi^P_{l,j}$ multiplies **both** channels; a coefficient multiplying $\Phi^Z_{l,j}$ multiplies only Z. This is why `eval_Ur(r, Y_paras, "X")` uses the same `Y_paras` as the Y call.

The retained finite-rank fields are later formed as $\psi_s=\sum_pV_{ps}\Phi_p$. Rows of V index complete basis elements, while columns index retained fields. `My_Bessel.jl` supplies the factors of $\Phi_p$; the summation is performed in `My_Assembly.jl`.

## 3. Half-integer Bessel functions and rigorous evaluation

In three dimensions the radial separated equation produces the factor

$$
\frac{J_{l+1/2}(\alpha r)}{\sqrt r}.
$$

Thus only positive half-integer orders are needed here. The apparent singular factors at $r=0$ in this expression and in the poloidal formulas have analytic limits, handled in §6.

### What `my_besselj` encloses

The Float64 method calls `SpecialFunctions.besselj`. The interval method accepts an exact-zero argument or a strictly positive interval. It evaluates the endpoints internally with Arb, bounds the variation between them, and returns an `Interval{Float64}`. The separate Arb method supports those internal evaluations; this does not require running the full verification in an Arb numeric mode.

For an exact-zero argument the result is zero, as required for the positive half-integer orders. For a positive interval $[a,b]$, enclosing only the two endpoint values would be insufficient: the Bessel function may attain a larger or smaller value inside the interval.

### Why the chord remainder covers the interior

Let $L(x)$ be the affine interpolant through the endpoint values of $J_\nu$. The interpolation remainder satisfies

$$
J_\nu(x)-L(x)=\frac{(x-a)(x-b)}2J_\nu''(\xi),\qquad \xi\in[a,b].
$$

Since $|(x-a)(x-b)|\le(b-a)^2/4$,

$$
|J_\nu(x)-L(x)|\le
E:=\frac{(b-a)^2}{8}\sup_{[a,b]}|J_\nu''|.
$$

Every chord value lies between the endpoint values. The interval method therefore enlarges their enclosures by E and takes their hull. This encloses all interior values as well. `bessel_W2_bound_interval` supplies the second-derivative bound used here.

<a id="bessel-derivative-enclosure"></a>

## 4. Bessel derivative enclosure — `bessel_W2_bound_interval`

The useful starting point is the regular function

$$
g_l(x)=\frac{J_{l+1/2}(x)}{\sqrt x}
=\frac{i^{-l}}{\sqrt{2\pi}}\int_{-1}^{1}e^{ixt}P_l(t)\,dt.
$$

The integral gives its smooth continuation to all real x. Differentiating under the integral and applying Cauchy–Schwarz gives

$$
|g_l^{(k)}(x)|
\le\frac1{\sqrt{2\pi}}
\left(\int_{-1}^1t^{2k}\,dt\right)^{1/2}
\left(\int_{-1}^1P_l(t)^2\,dt\right)^{1/2}
=\sqrt{\frac2{\pi(2k+1)(2l+1)}}.
$$

For the endpoint enclosure, differentiate $J_{l+1/2}(x)=\sqrt x\,g_l(x)$ twice:

$$
J_{l+1/2}''(x)
=-\tfrac14x^{-3/2}g_l(x)+x^{-1/2}g_l'(x)+x^{1/2}g_l''(x).
$$

Using the preceding bounds for orders $0,1,2$, on $0<a\le x\le b$ we obtain

$$
\sup_{[a,b]}|J_{l+1/2}''|
\le\sqrt{\frac2{\pi(2l+1)}}
\left(\frac1{4a^{3/2}}+\frac1{\sqrt3\,a^{1/2}}+\frac{b^{1/2}}{\sqrt5}\right).
$$

The two decreasing powers take their worst values at a; the increasing power takes its worst value at b. This is the expression evaluated by `bessel_W2_bound_interval`. The same bound for $g_l^{(k)}$ supplies the radial derivative estimates derived in [§6.4 below](#radial-derivative-bounds).

## 5. Root equations and root completeness

### 5.1 What a root selects

For each degree l, the boundary conditions of the divergence-free spectral problem determine the admissible positive values of $\alpha$. After rescaling to physical radius R, the corresponding spectral value in the energy identity is $\alpha^2/R^2$. These basis roots are distinct from the generalized eigenvalues in `eig.mat`, which are obtained from the Q and energy matrices built on this basis.

The Bessel recurrence

$$
J_{l+1/2}'(x)=J_{l-1/2}(x)-\frac{l+1/2}{x}J_{l+1/2}(x)
$$

removes explicit Bessel derivatives from the boundary equations.

For the toroidal condition $xJ_{l+1/2}'(x)-\tfrac12J_{l+1/2}(x)=0$, division by the nonzero denominator gives `func_Z`:

$$
f_Z(l,x)=x\frac{J_{l-1/2}(x)}{J_{l+1/2}(x)}-(l+1).
$$

For the poloidal condition, `func_Y` uses, for $l\ge2$,

$$
f_Y(l,x)=x\frac{J_{l-1/2}(x)}{J_{l+1/2}(x)}-(l+2)
+\frac{(x^2-(l-1)l)(x^2-(l-1)(l+2))}{x^2-(l-1)l(l+2)}.
$$

At $l=1$ the last term simplifies to $x^2$, so the implementation uses

$$
f_Y(1,x)=x\frac{J_{1/2}(x)}{J_{3/2}(x)}-3+x^2.
$$

Zeros of the Bessel denominator are poles of these divided equations. The poloidal equation also has the extra pole $\sqrt{(l-1)l(l+2)}$ for $l\ge2$. Root brackets must account for both kinds of pole; a sign change across a pole is not evidence of a zero.

### 5.2 What loading and local refinement establish

`load_besselj_zeros` reads `Y_zeros` and `Z_zeros`, whose rows correspond to degrees. `mat_to_lst` converts each row into a list of entries `[α]`. It stops at the first padding zero after column 1; the first column is retained so that the known poloidal zero mode at degree 1 is preserved. `preprocess_para!` then appends the coefficients from §6.

By default, interval mode uses `rigorous_root_interval` and Float64 mode uses `verify_root`. The latter checks a floating-point sign change around a candidate and returns that candidate. It does not produce an interval root certificate.

`rigorous_root_interval` expands a supplied candidate interval by an absolute margin $\delta$ and checks strictly opposite endpoint signs with interval arithmetic. On a **continuous, pole-free bracket**, the intermediate value theorem gives a root enclosure. Bisection retains such an enclosure; if a midpoint sign is undecidable, the function returns its current bracket. Accordingly, `tol` is a stopping threshold, not a guarantee of the final width. The `prec` keyword is unused by this routine. An exact-zero candidate is returned directly for the known zero mode.

The routine does not test whether the bracket contains a pole. That is a caller precondition. Nor does local enclosure establish uniqueness or completeness of the root list. Uniqueness uses the one-root result on the relevant pole interval; completeness also requires checking the searched intervals, terminal cutoff, and degree range. Those arguments and the notebook's checks belong to [the root-certificate note](bessel_zeros_certificate.md).

## 6. Y/X/Z radial formulas and coefficients

### 6.1 Store a consistent normalization

For a nonzero poloidal root $\alpha$, `preprocess_para!` stores

$$
A=-\frac{(\alpha^2-(l+2)(l-1))J_{l+1/2}(\alpha)}{\alpha^{3/2}},\qquad
B=\frac{\alpha^2-l(l-1)(l+2)}{\alpha^{3/2}}.
$$

These are the coefficients $A_{ml},B_{ml}$ in the paper's `lem:lap eig`. The
paper writes the Bessel term as $\alpha^{3/2}B\tilde j_l(\alpha r)$; since
$\tilde j_l(x)=J_{l+1/2}(x)/x^{3/2}$, this equals
$B J_{l+1/2}(\alpha r)/r^{3/2}$, the expression evaluated by the code. The
root equation ensures compatible radial boundary conditions. The stored entry
is `Y_paras[l][j] = [α,A,B]`; A and B fix the basis normalization.

For Z the entry is `Z_paras[l][j] = [α,1]`. This normalization matters: the external eigenvector coefficients must refer to these same basis elements, and the standard Z derivative bound assumes $A=1$.

### 6.2 Evaluate each channel

For $r>0$, set $\mu_l=l(l+1)$. Distinguish the physical poloidal radial
coefficients $Y_{l,j},X_{l,j}$ from the returned channels $R^Y_{l,j},R^X_{l,j}$.
`Ur_func` evaluates

$$
R^Y_{l,j}(r)=Y_{l,j}(r)=lAr^{l-1}+B\frac{J_{l+1/2}(\alpha r)}{r^{3/2}},
$$

$$
R^X_{l,j}(r)=-X_{l,j}(r)=-Ar^{l-1}
-\frac{B}{\mu_l}\,
\frac{\alpha J_{l-1/2}(\alpha r)-lJ_{l+1/2}(\alpha r)/r}{\sqrt r},
$$

$$
R^Z_{l,j}(r)=A\frac{J_{l+1/2}(\alpha r)}{\sqrt r}.
$$

Y and X use the same $(\alpha,A,B)$. With $H_l=-S_l'$, their physical
poloidal element is

$$
\Phi^P_{l,j}=(Y_{l,j}S_l,-X_{l,j}H_l,0)
=(Y_{l,j}S_l,X_{l,j}S_l',0),\qquad
rY_{l,j}'+2Y_{l,j}=\mu_lX_{l,j}.
$$

The last identity proves physical divergence freedom for every positive mode.
`Ur_func` applies the X minus sign through `sign_r` before every return path,
including the origin and zero mode. The toroidal field $(0,0,R^ZH_l)$ is
the negative of the paper's toroidal field, a whole-mode normalization choice
that must also be used by the corresponding coefficient rows.

### 6.3 The origin and the known zero mode

For $l=1$, the Y and X limits at the origin have opposite signs:

$$
R^Y(0)=A+\frac{\sqrt{2/\pi}}3B\alpha^{3/2},\qquad R^X(0)=-R^Y(0).
$$

This follows from $J_{3/2}(x)\sim\tfrac13\sqrt{2/\pi}\,x^{3/2}$. For $l\ge2$, Y/X vanish at the origin, as does Z for every $l\ge1$. `Ur_func` inserts these analytic values instead of dividing by zero.

The separate $l=1$, $\alpha=0$ poloidal mode is stored with $A=B=1$ and
evaluated with constant radial channels $R^Y=1$, $R^X=-1$. It is supplied
analytically, rather than by substituting zero into the nonzero-root coefficient
formulas. Since $S_1=\cos\theta$ and $H_1=\sin\theta$, its physical components
are $(\cos\theta,-\sin\theta,0)=e_z$.

<a id="radial-derivative-bounds"></a>

### 6.4 Bound one radial derivative: `Ur_bound`

`Ur_bound(paras, vari, l, k)` bounds the single derivative $\|\partial_r^k R^{\mathrm{vari}}\|_\infty$. For Y and X, it adds bounds for the Bessel and polynomial portions of the radial formulas above. For Z, only the Bessel portion is present.

The [Bessel integral representation](#bessel-derivative-enclosure) gives, for $g_l(x)=J_{l+1/2}(x)/\sqrt{x}$,

$$
\|g_l^{(k)}\|_\infty
\le C_{l,k}:=\sqrt{\frac{2}{\pi(2k+1)(2l+1)}}.
$$

The regular continuation makes this a bound on the signed radial extension as well. For the standard Z normalization,

$$
R^Z(r)=\sqrt{\alpha}\,g_l(\alpha r),\qquad
\|\partial_r^k R^Z\|_\infty\le C_{l,k}\alpha^{k+1/2}.
$$

The factor $\sqrt\alpha$ comes from changing $\sqrt{\alpha r}$ to $\sqrt r$; the remaining $\alpha^k$ is the chain rule.

For Y and X, let A and B denote the scalar radial coefficients stored in `Y_paras[l][j]`. The recurrence at argument $\alpha r$ gives

$$
\frac{J_{l+1/2}(\alpha r)}{r^{3/2}}
=\frac{\alpha}{2l+1}
\frac{J_{l-1/2}(\alpha r)+J_{l+3/2}(\alpha r)}{\sqrt r}.
$$

Hence the Bessel portions of the radial channels can be written as

$$
R^Y_{\mathrm{Bes}}
=\frac{B\alpha^{3/2}}{2l+1}
\bigl(g_{l-1}(\alpha r)+g_{l+1}(\alpha r)\bigr),
$$

$$
R^X_{\mathrm{Bes}}
=-\frac{B\alpha^{3/2}}{l(l+1)(2l+1)}
\bigl((l+1)g_{l-1}(\alpha r)-l\,g_{l+1}(\alpha r)\bigr).
$$

Taking absolute values and differentiating gives exactly the Bessel contributions used in `Ur_bound`:

$$
\|\partial_r^k R^Y_{\mathrm{Bes}}\|_\infty
\le\frac{\sqrt2\,|B|\alpha^{k+3/2}}{\sqrt{\pi(2k+1)}(2l+1)}
\left(\frac1{\sqrt{2l-1}}+\frac1{\sqrt{2l+3}}\right),
$$

$$
\|\partial_r^k R^X_{\mathrm{Bes}}\|_\infty
\le\frac{\sqrt2\,|B|\alpha^{k+3/2}}{\sqrt{\pi(2k+1)}l(l+1)(2l+1)}
\left(\frac{l+1}{\sqrt{2l-1}}+\frac{l}{\sqrt{2l+3}}\right).
$$

<a id="polynomial-derivative-bounds"></a>

### 6.5 Bound the polynomial portion: `interval_prod`

The polynomial portions are $lAr^{l-1}$ for Y and $-Ar^{l-1}$ for stored X.
The minus sign does not change their absolute derivative bounds. For $k<l$,
their derivative coefficients in absolute value are respectively

$$
|A|\prod_{s=l-k}^{l}s,\qquad
|A|\prod_{s=l-k}^{l-1}s.
$$

`interval_prod` computes these products in `dtype`, with an empty product equal to 1. For $k\ge l$, the product contains zero, matching the vanishing derivative. The factor `1.1^(l-k)` bounds the remaining radial power on $|r|\le1.1$ with a margin. These polynomial bounds are added to the Bessel bounds. The known $\alpha=0$ mode is handled separately; returning 1 for every order is conservative for its constant radial channels.

<a id="radial-bound-tables"></a>

### 6.6 Collect all channels and orders: `Ur_bound_lst`

`Ur_bound_lst(Y_paras, Z_paras, K)` evaluates `Ur_bound` for every stored root and every derivative order $0{:}K$. It returns `(Y_bound_lst, X_bound_lst, Z_bound_lst)`, with one matrix per degree in each list. The Y and X matrices have shape `(K+1, n_l_P)` and share their root columns; the Z matrices have shape `(K+1, n_l_Z)`. Here `n_l_P` and `n_l_Z` are the poloidal and toroidal root counts for degree l.

Row `m+1` bounds the order-$m$ radial derivative. For example, `X_bound_lst[l][m+1,j]` bounds the X radial factor of poloidal element $(l,j)$. These tables contain radial bounds only; angular magnitudes are attached later by `lst_to_mat`. Their sample-array counterparts and indexing are explained in §7.1, and their two uses are described in [assembly §2](finite_rank_assembly.md#assembly-inputs) and [RHS §3](finite_rank_rhs.md#packed-field-bounds).

## 7. Indexing and packing conventions

### 7.1 Read the per-degree arrays first

`eval_Ur` calls `Ur_func` for every stored root and places each resulting sample vector in one column. For degree l with $n_l^P$ poloidal roots and $n_r$ evaluation points:

```text
Y_paras[l][j]        [α, A, B] for one poloidal element
Ur_Y_lst[l]         n_r × n_l^P radial samples
Ur_X_lst[l]         n_r × n_l^P radial samples, using the same root columns
Y_bounds[l]         (K+1) × n_l^P, rows for derivative orders 0:K
X_bounds[l]         (K+1) × n_l^P, rows for derivative orders 0:K
```

The Z lists have the same layouts with their own root count $n_l^Z$. For example, `Ur_X_lst[2][q,3]` is the X radial value at sample q for the third poloidal root of degree 2. `X_bounds[2][5,3]` bounds its fourth radial derivative. Neither index 3 denotes a vector component: the channel is already specified by the list name.

### 7.2 Then assign complete elements to finite-rank rows

`My_Assembly.jl` partitions the elements by field parity:

| Field block | Poloidal degrees, in order | Toroidal degrees, in order |
|---|---|---|
| even, index `1` | $2,4,6,\ldots$ | $1,3,5,\ldots$ |
| odd, index `2` | $1,3,5,\ldots$ | $2,4,6,\ldots$ |

Within each block, all poloidal elements precede all toroidal elements, and roots remain in their stored column order within each degree. Thus the indexing chain is

```text
(degree l, root j, family) → one full basis element → packed row p → row p of V.
```

The same p indexes matrix rows/columns and the basis axis of packed bound tensors. The [worked example in the assembly note](finite_rank_assembly.md#basis-ordering) shows how different root counts produce the offsets. The Bessel routines themselves keep the simpler per-degree layout.

## 8. Sign / parity extension, and field parity

Padded finite-difference grids extend past the endpoints. The regular Bessel and trigonometric functions can be evaluated smoothly on these extensions; endpoint boundary conditions determine their reflection behavior.

### Radial extension in `Ur_func`

The integral representation of $g_l$ gives $g_l(-x)=(-1)^lg_l(x)$. The neighboring-degree representations of Y/X, together with their $r^{l-1}$ polynomial terms, therefore give

$$
R^Z(-r)=(-1)^lR^Z(r),\qquad
R^Y(-r)=(-1)^{l-1}R^Y(r),\qquad
R^X(-r)=(-1)^{l-1}R^X(r).
$$

`Ur_func` evaluates the positive-radius formula at `abs(r)` and restores this
parity sign. Odd profiles receive `sign(r + EPS)`; even profiles receive 1.
The X channel multiplies this factor by $-1$ as described in §6.2.
`EPS = 1e-16` shifts only the parity sign factor. It does not decide when to use
the origin limit.

The origin test is `is_zero(r)`: exact `[0,0]` in interval mode and `abs(r) < 1e-14` in Float64 mode. The verification grids separate nonzero nodes from that small interval, while endpoint nodes use the analytic limit. Negative nonzero nodes also remain negative after adding EPS. For interval samples this evaluation assumes exact zero or an interval strictly on one side of zero; it is not an evaluation formula for an arbitrary interval straddling the origin.

### Angular reflection and field grouping

At the equator,

$$
S_l(\pi-\theta)=(-1)^lS_l(\theta),\qquad
H_l(\pi-\theta)=(-1)^{l+1}H_l(\theta).
$$

Combined with the component conventions, these give the field-parity groups in §7. Signed-radius parity, degree parity, and vector-field parity should not be used interchangeably. Angular evaluation also extends through the endpoints using the same finite trigonometric functions; it requires no separately patched endpoint values.

<a id="angular-rows"></a>

## 9. Angular rows and the boundary-code lookup (`eval_Uθ`)

### 9.1 Convert the trigonometric basis, then evaluate it

`eval_Uθ(θ,L)` uses `conversion_mat` to express S/H in the appropriate trigonometric basis and `trig_func` to evaluate that basis. For each code it computes

```julia
conversion_mat(lh, bdry, 1) * trig_func(lh+(bdry=="11"), θ, "I", bdry)'
```

with `lh = (L+1)÷2`. Its result is a four-element list, in the fixed code order `["00", "01", "10", "11"]`. Each matrix has shape `lh × length(θ)`. The angular row for degree l is `(l+1)÷2` **within the selected code block**.

The code blocks distinguish the following rows:

| List index | Boundary code | Angular rows, starting at row 1 |
|---|---|---|
| 1 | `"00"` | $H_2,H_4,H_6,\ldots$ |
| 2 | `"01"` | $H_1,H_3,H_5,\ldots$ |
| 3 | `"10"` | $S_1,S_3,S_5,\ldots$ |
| 4 | `"11"` | $S_2,S_4,S_6,\ldots$ |

The extra trigonometric constant mode in the `"11"` calculation is needed to represent even Legendre polynomials such as $S_2$. It does not insert a degree-zero vector basis element into the packed basis.

Consequently, the even field block selects `(Y,X,Z)` from list indices `(4,1,2)` and the odd field block from `(3,2,1)`. For example, degree 2 poloidal Y/X uses row 1 of blocks 4 and 1; degree 1 toroidal Z uses row 1 of block 2. The [Fourier basis note](fourier_basis.md) explains the boundary codes and conversion matrices.

<a id="92-keep-the-internal-angular-signs-consistent-with-the-data"></a>

### 9.2 Scalar angular signs and physical components

The angular tables use $H_l=-S_l'$ for X and Z. The poloidal radial X
coefficient returned by `Ur_func` already contains the compensating minus
sign, so reconstruction returns physical $(e_r,e_\theta,e_\phi)$ components.
The toroidal whole-mode minus sign is part of the basis normalization and is
carried by its coefficient rows.

Use the same physical tensor Q in Step 1, Step 4A, Step 4B, Steps 5–6 and the
weighted-functional calculation. No additional $DQD$ transformation is applied:
do not flip $Q_{12}$ or $Q_{23}$ on loading. Spatial reflections relate distinct
points of the physical field, as explained in
[the symmetry note](symmetry_and_domains.md).

Coefficient ordering, whole-mode signs and physical scaling must agree
across the imported and generated data; see [the data guide](../data.md).

<a id="angular-derivative-bounds"></a>

### 9.3 Bound angular derivatives: `Uθ_bound`

Bernstein's inequality for the degree-$l$ trigonometric polynomial $S_l$ gives

$$
\|\partial_\theta^m S_l\|_\infty\le l^m,\qquad
\|S_l\|_\infty\le1,\qquad \|H_l\|_\infty\le l.
$$

`Uθ_bound(L,k)` records $l^m$ in row $l$, column $m+1$, for degrees $1{:}L$ and derivative orders $0{:}k+1$. It returns an `L × (k+2)` integer matrix. These are bounds for derivatives of **S**; H already contains one derivative, so its order-$m$ bound is $l^{m+1}$.

For example, row 2 of `Uθ_bound(L,2)` is `[1,2,4,8]`: it bounds S₂ and its first three derivatives. The entry 2 also bounds H₂ itself. The function requires `k < 5`, and the integer powers must fit in `Int`. The small angular table used by assembly is separate from the radial derivative order K in `Ur_bound_lst`.

## 10. From an individual basis element to assembly

`My_Bessel.jl` provides the following ingredients. The next stages are explained where each ingredient is consumed, in [the assembly note](finite_rank_assembly.md) and [the RHS note](finite_rank_rhs.md).

| Ingredient | Functions | Where to continue |
|---|---|---|
| Bessel values with interval enclosures | `my_besselj`, `bessel_W2_bound_interval` | The radial evaluations in §6 above. |
| Root parameters and radial normalization | `func_Y`, `func_Z`, `verify_root`, `rigorous_root_interval`, `mat_to_lst`, `preprocess_para!`, `load_besselj_zeros` | The sample/bound inputs in [assembly §2](finite_rank_assembly.md#assembly-inputs). |
| Per-degree radial sample matrices | `Ur_func`, `eval_Ur` | Matrix integrands in [assembly §4](finite_rank_assembly.md#radial-assembly); field reconstruction in [RHS §1–2](finite_rank_rhs.md). |
| Angular sample rows | `eval_Uθ` | The outer products in [RHS §2](finite_rank_rhs.md). |
| Single-channel radial derivative bounds | `Ur_bound`, `interval_prod`, `Ur_bound_lst` | Derivation and tables in §6.4–§6.6; their use in [assembly §4–5](finite_rank_assembly.md#radial-assembly). |
| Bounds on S and its angular derivatives | `Uθ_bound` | Angular bounds in §9.3; their use by `lst_to_mat` in [RHS §3](finite_rank_rhs.md#packed-field-bounds). |

For instance, `Ur_bound_lst(Y_paras,Z_paras,8)` supplies orders $0{:}8$ for matrix quadrature. `lst_to_mat` subsequently packs radial bounds and attaches angular magnitudes when bounding a finite-rank field. `eval_eig_function` reconstructs that field from samples and coefficients; `apply_Q_to_channel_fields` multiplies it by sampled Q. Those last three functions belong to `My_Assembly.jl`, and their full process is covered in [the RHS note](finite_rank_rhs.md).

In the paper, `lem:lap eig` and `subSec:numfr` give the
spectral basis and `eq:alphaY`/`eq:alphaZ` give its root equations. The Bessel
integral representation `int rep` and the derivative estimates in
`sec:bes ln` justify §4 and §6.4–6.6. The separate
[root-certificate note](bessel_zeros_certificate.md) explains the complete search.

