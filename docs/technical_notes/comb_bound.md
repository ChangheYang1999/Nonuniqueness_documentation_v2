# Comb_Bound.jl Mathematical Notes


These routines propagate supplied derivative bounds through products, coordinate
changes, eigenfunction combinations, and per-mode sample refinement. Start with the [source guide](../src.md)
for their place in the verification; [Comb_Bound.jl](../../src/Comb_Bound.jl)
specifies the array contracts.

The main example is the finite-rank RHS: combine basis bounds with eigenvector
coefficients, convert normalized-radius bounds to β, and bound the product of
a scalar weight, Q, and an eigenfunction. Those prior bounds are then used in
finite-difference refinement and quadrature-error estimates.

Read Sections 1–2 for product bounds, Sections 3–5 for the coordinate change
and weight, Sections 6–7 for the array and eigenfunction wrappers, and Section 8
for per-mode refinement and β-chain packing. Interval
arithmetic encloses the formulas; the supplied derivative bounds must already
be valid on the target domain. Ordinary floating-point evaluation alone does
not provide a rigorous enclosure.


## 1. Derivative directions and the Leibniz bound


Each table column describes one derivative direction. Two-column tables in
the profile calculations use β and θ; the radial assembly uses one radial
column, and the RHS bound uses one β column. A vector of order entries is
also accepted for a single direction.

For factor j, `a[j][m+1,l]` bounds the absolute derivative of order m in
direction l. Fix a direction q and write its bound as $A_{j,q}(m)$.
The bounds are taken over the common target domain, including any padding
needed by the subsequent refinement. Each column is processed independently;
the routine does not construct mixed β–θ derivative bounds.


Let

$$
F=\prod_{j=1}^k f_j.
$$


For the fixed direction $q$, the one-dimensional Leibniz rule gives


$$
\partial_q^n F
=
\sum_{m_1+\cdots+m_k=n}
\frac{n!}{m_1!\cdots m_k!}
\prod_{j=1}^k \partial_q^{m_j}f_j.
$$

If

$$
\|\partial_q^m f_j\|_\infty\le A_{j,q}(m),
$$

then

$$
\|\partial_q^n F\|_\infty
\le
\sum_{m_1+\cdots+m_k=n}
\frac{n!}{m_1!\cdots m_k!}
\prod_{j=1}^k A_{j,q}(m_j).
$$


The same formula is applied independently to each direction column.


## 2. Factorial convolution and ordered matrix products

The code evaluates the bound above by factorial-scaled convolution. Fix one
direction and suppress the direction index. If $A_j(m)$ bounds the $m$-th
derivative of factor $j$, define

$$
\widetilde A_j(m)=\frac{A_j(m)}{m!}.
$$

Then

$$
\sum_{m_1+\cdots+m_k=n}
\frac{n!}{m_1!\cdots m_k!}
\prod_{j=1}^k A_j(m_j)
=
n!
\sum_{m_1+\cdots+m_k=n}
\prod_{j=1}^k \widetilde A_j(m_j).
$$

The inner sum is an ordinary convolution in derivative order. Equivalently, if

$$
P_j(t)=\sum_{m=0}^n \frac{A_j(m)}{m!}t^m,
$$

then the coefficient of $t^n$ in $P_1(t)\cdots P_k(t)$ is

$$
\sum_{m_1+\cdots+m_k=n}
\prod_{j=1}^k \frac{A_j(m_j)}{m_j!}.
$$


Dividing the order-m bounds by m! turns the Leibniz sum into an ordinary
convolution. The code accumulates it in the supplied factor order and multiplies
back by n! at the end. `last_flag=true` returns only order n;
`last_flag=false` retains all orders 0:n.


### Entrywise matrix bounds

For matrix or vector factors, bounds are entrywise absolute bounds, not matrix
operator norms. If $|A|\le B_A$ and $|B|\le B_B$ entrywise, with nonnegative
bound matrices, then

$$
|AB|\le |A|\,|B|\le B_A B_B.
$$

Applying this inequality to each Leibniz term justifies the same convolution
for a matrix-product chain. Matrix multiplication sums over the contracted
components; factor order must be preserved. The implementation requires a
scalar first factor followed by compatible scalar, vector, or matrix blocks.

### Example: weight × Q × ψ

In [Part 4B of finite_rank_construction.ipynb](../../notebooks/finite_rank_construction.ipynb#step-4b),
let $w(\beta)=\tan\beta/\cos\beta$. Denote the bounds for its derivatives by
$B_w(i)$, for Q by the nonnegative $3\times3$ matrices $B_Q(j)$, and for J
eigenfunctions by the component-by-mode matrices $B_\psi(\ell)$ of size $3\times J$.
The β-derivatives of the weighted RHS satisfy, entrywise,

$$
|\partial_\beta^n(wQ\psi)|
\le \sum_{i+j+\ell=n}\frac{n!}{i!\,j!\,\ell!}
B_w(i)B_Q(j)B_\psi(\ell).
$$

The notebook implements this as

```julia
B = tan_over_cos_deriv_Linf_bounds(beta_prior_max, k)
rhs_β_norms_even_prior = multinomial_sum(
    k, B, Q_beta_bounds, eig_bound_even; last_flag=false)
```

The result is a `(k+1) × 1` table of `3 × J` matrices. Entry `[m+1,1]`
bounds order m. This is an initial bound for the complete product; the notebook
uses samples of that product to refine its derivative bounds with finite differences.


## 3. Coordinates, padded domains, and derivative polynomials


### Which radius is being differentiated?

Write $\kappa=\texttt{scl\_fac}$ and let R be the finite physical radius used
in the finite-rank calculation. Distinguish the physical radius from the
normalized radius $r$:

$$
r_{\mathrm{physical}}=\kappa\tan\beta,\qquad
r=\frac{r_{\mathrm{physical}}}{R}=a\tan\beta,\qquad
a=\frac{\kappa}{R}.
$$

The argument r in `beta_all_deriv_Linf_bounds` and `r_to_beta` is
this normalized radius in the current callers. Thus
$\mathrm{Beta}=\arctan(1/a)$ maps $[0,\mathrm{Beta}]$ to $[0,1]$ in r.
The whole physical space still corresponds to $0\le\beta<\pi/2$; Beta marks
the finite region used by these bounds.

For Q, the normalized-radius field is $Q((\pi/2)r,\theta)$ in the stored
trigonometric coordinate. Its exported radial derivatives already include this
scaling. The β transform uses $a=\kappa/R$ without another π/2 factor.

### The domain of a composition bound

By default, the transform uses $|\beta|\le\arctan(1/a)$, hence
$|\tan\beta|\le1/a$ and $|r|\le1$.
With `β_max=b`, it instead uses

$$
0<b<\pi/2,\qquad |\beta|\le b,\qquad
|\tan\beta|\le T=\tan b,\qquad |r|\le aT.
$$

The notebook chooses `beta_prior_max = Beta * dt(1.01)` to cover its padded
β grid. This is a bound on the coordinate map, not an extension of the input
derivative estimates: `M[j+1]` must already bound the derivatives of f on the
image of every target or stencil point. Changing β_max leaves a unchanged.
The source asserts $0<a<1$ but leaves the endpoint condition on β_max to the caller.

The trigonometric and regular Bessel formulas provide smooth extensions to
the padded grids. The remaining requirement is that their supplied numerical
derivative bounds hold there. For Q, the radial bounds use its trigonometric
reflection/periodicity. For basis bounds, the domain must be checked against
the estimates supplied by `Ur_bound`; this transform does not perform that check.

### Derivative polynomials


The change-of-variable routine uses derivative polynomials $P_n$ defined by

$$
\frac{d^n}{d\beta^n}\tan\beta=P_n(\tan\beta).
$$

The base case is

$$
P_0(x)=x.
$$

Since

$$
\frac{d}{d\beta}\tan\beta=1+\tan^2\beta,
$$

differentiating $P_n(\tan\beta)$ gives

$$
P_{n+1}(x)=(1+x^2)P_n'(x).
$$

This recurrence is implemented by

```julia
P = poly_mul_1_plus_x2(poly_deriv(P))
```

starting from `P = BigInt[0, 1]`, which represents $P_0(x)=x$.


For $r(\beta)=a\tan\beta$, the derivative of order n is
$r^{(n)}=aP_n(\tan\beta)$. Therefore an input bound T on $|\tan\beta|$ gives

$$
x_n=a\sum_i |p_i^{(n)}|T^i\ \ge\ \|r^{(n)}\|_\infty,
\qquad P_n(x)=\sum_i p_i^{(n)}x^i.
$$

The recurrence stores low-to-high coefficients as `BigInt`: `[c0,c1,...]`
represents $c_0+c_1x+\cdots$. The helpers differentiate, add, or multiply these
polynomials exactly. `poly_abs_sum_bound` converts their coefficients to
`dtype` when evaluating $\sum_i|c_i|T^i$; in interval mode, this evaluation
includes arithmetic enclosure.


## 4. Faà di Bruno and the Bell-polynomial bound


For partial Bell polynomials and the composition formula, see L. Comtet,
*Advanced Combinatorics: The Art of Finite and Infinite Expansions*, revised
and enlarged edition, D. Reidel, 1974,
[Chapter III](https://doi.org/10.1007/978-94-010-2196-8_3): §3.3, “Bell Polynomials”
(starting on p. 133), and §3.4, on substitution of formal series and Faà di Bruno's
formula (starting on p. 137).

Let

$$
g(\beta)=f(r(\beta)).
$$

Faà di Bruno's formula gives, for $n\ge1$,

$$
g^{(n)}(\beta)
=
\sum_{j=1}^n
f^{(j)}(r(\beta))
B_{n,j}\bigl(r'(\beta),r''(\beta),\ldots,r^{(n-j+1)}(\beta)\bigr),
$$

where $B_{n,j}$ are partial Bell polynomials.

The input vector `M` is indexed so that `M[j+1]` bounds
$\|f^{(j)}\|_\infty$ on the mapped domain from Section 3. The output vector `out` is indexed so that `out[k+1]`
bounds the $k$-th $\beta$ derivative of $f(a\tan\beta)$. For $k=0$, the code
sets

```julia
out[1] = M[1]
```

which is the bound $\|g\|_\infty\le\|f\|_\infty$.

For $k\ge1$, if $x_i\ge\|r^{(i)}\|_\infty$, then

$$
\|g^{(k)}\|_\infty
\le
\sum_{j=1}^k
M_{j}\,
B_{k,j}(x_1,\ldots,x_{k-j+1}).
$$

Here $M_j$ denotes the numerical bound stored in `M[j+1]`.
This is the formula evaluated by `beta_all_deriv_Linf_bounds`.


### Bell-polynomial recurrence

The partial Bell polynomials are evaluated by the recurrence

$$
B_{0,0}=1,
$$


and, for $n\ge1$ and $1\le j\le n$,


$$
B_{n,j}
=
\sum_{i=1}^{n-j+1}
\binom{n-1}{i-1}
x_i B_{n-i,j-1}.
$$


All other boundary values are zero: $B_{n,0}=0$ for $n>0$ and
$B_{0,j}=0$ for $j>0$. Because the recurrence has nonnegative coefficients,
substituting nonnegative derivative bounds for its inputs preserves the
upper-bound property. In the source, `Bbell[n+1,j+1]` represents $B_{n,j}$.


### Low-order example

For $g(\beta)=f(r(\beta))$, the first two derivatives are

$$
g'=f'(r)r',\qquad
g''=f''(r)(r')^2+f'(r)r''.
$$

If $M_j$ bounds $f^{(j)}$ and $x_j$ bounds $r^{(j)}$ on the corresponding
domains, the first three returned entries are

$$
M_0,\qquad M_1x_1,\qquad M_2x_1^2+M_1x_2.
$$

The Bell recurrence generates these coefficients and their higher-order
counterparts without enumerating every chain-rule term.


## 5. The weight tan(β)/cos(β)

`tan_over_cos_deriv_Linf_bounds` bounds derivatives of

$$
h(\beta)=\frac{\tan\beta}{\cos\beta}=\sec\beta\tan\beta.
$$

The representation used by the code is

$$
h^{(n)}(\beta)=\sec\beta\,R_n(\tan\beta),
$$

where $R_n$ is a polynomial. The base case is $R_0(x)=x$. If

$$
h^{(n-1)}(\beta)=\sec\beta\,R_{n-1}(\tan\beta),
$$

then using

$$
\frac{d}{d\beta}\sec\beta=\sec\beta\tan\beta,
\qquad
\frac{d}{d\beta}\tan\beta=1+\tan^2\beta
$$

gives

$$
R_n(x)=xR_{n-1}(x)+(1+x^2)R_{n-1}'(x).
$$

The routine assumes

$$
0<\beta_0<\frac{\pi}{2}.
$$

On $[0,\beta_0]$, both $\tan\beta$ and $\sec\beta$ are nonnegative and
increasing. Therefore

$$
|\tan\beta|\le T=\tan\beta_0,
\qquad
\sec\beta\le S=\sec\beta_0=\frac{1}{\cos\beta_0}.
$$

Since $h^{(n)}(\beta)=\sec\beta\,R_n(\tan\beta)$,

$$
\|h^{(n)}\|_\infty
\le
S\sup_{|x|\le T}|R_n(x)|.
$$

The final polynomial supremum is bounded coefficientwise:

$$
\sup_{|x|\le T}|R_n(x)|
\le
\sum_i |r_i^{(n)}|T^i,
\qquad
R_n(x)=\sum_i r_i^{(n)}x^i.
$$

The returned vector `B` satisfies `B[n+1]` $\ge \|h^{(n)}\|_\infty$ for
$n=0,\ldots,k$. For $n=0$, the code directly sets `B[1] = S * T`.


The same absolute derivative bounds apply to $[-\beta_0,\beta_0]$: h is odd,
and its successive derivatives alternate parity. This covers negative ghost
points near β=0 as well as the positive part of the padded grid.


## 6. Entrywise coordinate conversion: r_to_beta


The function `r_to_beta(A, a0; β_max=nothing)` applies the scalar transform from
`beta_all_deriv_Linf_bounds` entrywise to a vector of matrices. The substitution
is

$$
r=a_0\tan\beta.
$$

For matrix inputs, the sequence is `A[1], ..., A[k]`. Its entry `A[j][p,q]` is
an upper bound for

$$
\left\|
\frac{\partial^{j-1}}{\partial r^{j-1}}f_{p,q}
\right\|_\infty.
$$

For each matrix entry $(p,q)$, the code collects the derivative-order vector
`A[1][p,q], A[2][p,q], ..., A[k][p,q]`, passes it to
`beta_all_deriv_Linf_bounds`, and restores the vector-of-matrices layout. The code returns `out`; its entry `out[j][p,q]` is an upper bound for

$$
\left\|
\frac{\partial^{j-1}}{\partial\beta^{j-1}}
f_{p,q}(a_0\tan\beta)
\right\|_\infty.
$$

The conversion is purely entrywise. It does not use matrix multiplication or
matrix calculus.


The sequence can also contain length-m vectors, as in the profile notebook's
six Q-component bounds. These are treated as m×1 matrices, and the returned
entries are column matrices. The wrapper passes β_max through unchanged;
the same mapped-domain requirement from Section 3 applies to every entry.


## 7. Eigenfunction bounds: eig_fun_bound

The function `eig_fun_bound(bd_mat, eig_vec)` combines basis-function bounds
with eigenvector coefficients. If `bd_mat` has shape `(K, N, 3)`, then
`bd_mat[m+1, i, c]` stores the bound for derivative order $m$, basis element
$i$, and component/channel $c$, for $m=0,\ldots,K-1$.

Suppose

$$
\phi_s=\sum_{i=1}^N v_{i,s}\psi_i.
$$

If

$$
\|\partial^m \psi_i^{(c)}\|_\infty\le B_{m,i,c},
$$

then

$$
\|\partial^m \phi_s^{(c)}\|_\infty
\le
\sum_{i=1}^N |v_{i,s}|B_{m,i,c}.
$$

The code computes, for each derivative-order index `i`,

```julia
bd_mat[i, :, :]' * abs.(eig_vec)
```

where `bd_mat[i, :, :]'` has shape `3 × N` and `abs.(eig_vec)` has shape
`N × M`. The result has shape `3 × M`, giving a component-by-eigenfunction
bound table for that derivative order. The use of `abs.(eig_vec)` is the
safety step: it intentionally loses cancellation between basis functions in
exchange for a valid upper bound.


The basis axis must follow exactly the same ordering as the eigenvector rows.
The finite-rank caller obtains it from `lst_to_mat`: poloidal basis elements
first, then toroidal elements, within each field-parity group. These tables
bound normalized-radius derivatives, with angular factors already bounded.
`eig_fun_bound` preserves that derivative coordinate; the subsequent
`r_to_beta` call converts it. See [the RHS note](finite_rank_rhs.md#packed-field-bounds)
for packing, [the assembly note](finite_rank_assembly.md#basis-ordering) for the common basis-row order,
and [the Bessel note](bessel_basis.md) for the basis functions.


## 8. Per-mode refinement and β bound chains

Both helpers are defined in [Comb_Bound.jl](../../src/Comb_Bound.jl) and exported
by `NS_Numerics`; the [RHS section](../../notebooks/finite_rank_construction.ipynb#step-4b)
calls them for the even and odd blocks.

`refine_channel_bounds(fields, priors, scl_fac, domain, order;
angular_frequency, padding, crop=0, label="mode")` takes samples with axes
`(β grid, θ grid, mode, component)`, with three components. Prior entry `m+1`
is a `3 × modes` matrix of absolute β-derivative bounds of order m, through
`order+2`. The domain is `[Beta, L1]` and `2 <= order < 20`.

For each mode it removes `crop` layers from each edge, constructs a
`VectorFunc`, prepares fresh `Linf_to_paras(...; ind=mode)` parameters, and
calls `norm_lst(..., "W$(order)Inf"; padding=padding)`. Fresh parameters are
needed because the scalar refinement fills the deferred angular bounds in
place. The supplied fields and priors are unchanged.

`padding` is the number of ghost layers remaining after the nonnegative crop.
It must be at least `derivative_padding(order)`; the remaining physical grid
counts must be positive multiples of 100. Priors must cover the full stencil
domain, and the angular frequency must have its Bernstein justification.
The helpers assume compatible input layouts; they do not repeat the checks
in the lower-level norm routines. Those routines still check the derivative
order, padding, and physical grid counts.

The result is one `(order+1) × 2` table W per mode; `W[m+1,j][c]` bounds
component c's pure derivative of order m, with j=1 for β and j=2 for θ.
`beta_bound_chain(bounds)` takes a nonempty list of consistently sized tables
and returns one component-by-mode matrix per derivative order:

```julia
result[m+1][:, j] = bounds[j][m+1, 1]
```

This extracts the β column without changing the bounds or inputs, so it can
be passed directly to the ordered product rule in `multinomial_sum`.
See the [RHS padding workflow](finite_rank_rhs.md#how-the-workflow-padding-is-chosen)
for the order-10 ψ pass with 25 layers and the order-8 RHS pass with 10 layers.


## 9. Implementation map


| Mathematical operation | Function |
|---|---|
| Pure-direction Leibniz bound, with ordered matrix factors | `multinomial_sum` |
| Exact integer polynomial updates | `poly_deriv`, `poly_add`, `poly_mul_x`, `poly_mul_1_plus_x2` |
| Coefficient-sum bound | `poly_abs_sum_bound` |
| Composition with normalized radius `a*tan(β)` | `beta_all_deriv_Linf_bounds` |
| Derivatives of the geometric weight | `tan_over_cos_deriv_Linf_bounds` |
| Entrywise coordinate conversion of bound tables | `r_to_beta` |
| Linear combinations of basis bounds | `eig_fun_bound` |
| Sample refinement for each retained mode | `refine_channel_bounds` |
| β-derivative matrices for the product rule | `beta_bound_chain` |

The [source docstrings](../../src/Comb_Bound.jl) specify shapes, indices, and
return conventions. The [matrix-function note](matrix_func.md) explains how
these prior bounds enter finite differences and quadrature error estimates.


