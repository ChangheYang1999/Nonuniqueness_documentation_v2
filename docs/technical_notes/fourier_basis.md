# My_Fourier.jl Math Notes

> This is a detailed technical note. For the main reader path, see the [source guide](../src.md).

This note records the formulas and conventions behind `src/My_Fourier.jl`. The
source file keeps only concise code-facing comments and docstrings; this file
carries the longer derivations.

For how these bases are stored and evaluated in a field object, see
[ScalarFunc and VectorFunc](scalar_vector.md#2-how-the-coefficients-become-values).

The main references in the paper are `sec:div-free` for
the vector spherical harmonic basis, `sec:res cg` for interpolation,
`subsec:analytic-L2` for trigonometric residuals and removable quotients,
and `sec:ops` for spherical-coordinate operators.

## 1. Overview

Trigonometric coefficients support evaluation, differentiation, and frequency
bounds. Spherical-harmonic coefficients separate the angular degrees and make
the divergence-free constraint algebraic. The conversion matrices connect these
two descriptions of the same field; the radial β basis remains trigonometric.

`My_Fourier.jl` provides the angular and one-dimensional spectral infrastructure
used by the rest of the verification code:

- active numeric conversion through `dt`;
- Chebyshev-Legendre conversion matrices;
- boundary-adapted Fourier matrices;
- the divergence-free reconstruction of the angular velocity component from the
  radial component;
- sampled radial and angular operator factors used by `NS_Terms.jl`.

The file is mostly about representation. It does not prove new estimates by
itself; instead, it implements the basis conventions used in the paper's
computer-assisted verification pipeline.

## 2. Boundary-Condition Codes and Basis Choices

The code uses two-character boundary-condition strings on the interval
$[0,\pi/2]$. The first character is the condition at $0$; the second character
is the condition at $\pi/2$. In both positions, `0` means Dirichlet and `1`
means Neumann.

| Code | Endpoint meaning | Typical trigonometric block | Basis formula | Why the boundary conditions hold |
|---|---|---|---|---|
| `"11"` | Neumann at both endpoints | cosine modes with a constant mode | $f_k(x)=\cos(2kx)$, $k\geq0$ | $f_k'(0)=0$; $f_k'(\pi/2)=-2k\sin(k\pi)=0$ |
| `"00"` | Dirichlet at both endpoints | sine modes | $f_k(x)=\sin(2kx)$, $k\geq1$ | $f_k(0)=0$; $f_k(\pi/2)=\sin(k\pi)=0$ |
| `"10"` | Neumann at $0$, Dirichlet at $\pi/2$ | shifted cosine modes | $f_k(x)=\cos((2k+1)x)$, $k\geq0$ | $f_k'(0)=0$; $f_k(\pi/2)=\cos((k+\tfrac12)\pi)=0$ |
| `"01"` | Dirichlet at $0$, Neumann at $\pi/2$ | shifted sine modes | $f_k(x)=\sin((2k+1)x)$, $k\geq0$ | $f_k(0)=0$; $f_k'(\pi/2)=(2k+1)\cos((k+\tfrac12)\pi)=0$ |

Here $k$ is an integer. The constant mode occurs only in `"11"`;
the mixed codes select odd frequencies.

For the reflection signs of the assembled fields and the resulting valid
domain reductions, see [Symmetries and justified sampling domains](symmetry_and_domains.md#basis-rules).
Combining terms from different basis families can destroy an endpoint
reflection even when each term separately has a definite parity.

**Spherical-harmonic basis.** For the angular variable $\theta$, axisymmetry
means there is no dependence on the azimuthal angle $\phi$. The scalar spherical
harmonics used here are

$$
S_l(\theta)=P_l(\cos\theta),\qquad
-\frac{1}{\sin\theta}\frac{d}{d\theta}
 \left(\sin\theta\,S_l'(\theta)\right)=l(l+1)S_l(\theta),
\qquad l=0,1,2,\ldots,
$$

where $P_l$ is the Legendre polynomial of degree $l$. Thus they are
eigenfunctions of the negative angular Laplacian on the sphere. They separate
angular degrees in the spherical differential operators. For vector fields,
the paper uses

$$
Y_l=S_l e_r,\qquad X_l=S_l' e_\theta,\qquad Z_l=S_l' e_\phi,
$$

where primes denote $\theta$ derivatives and $e_r,e_\theta,e_\phi$ are the
spherical unit vectors. The code's sine conversion branches use the angular
factor $-S_l'$; Section 4 explains this sign convention. These bases concern
$\theta$; the radial $\beta$ basis remains trigonometric.

**Why the bases are equivalent.** Set $t=\cos\theta$. Both $P_l(t)$ and the
Chebyshev polynomial $T_l(t)$ have degree $l$, the same parity, and a nonzero
leading coefficient. At a fixed degree cutoff, each therefore gives a basis
of the same even or odd polynomial space. Since $T_l(\cos\theta)=\cos(l\theta)$,
this gives an invertible conversion between the Legendre and cosine bases.
Differentiating the cosine expansions gives the sine bases: the constant
vanishes, and every nonzero frequency is multiplied by its nonzero frequency
factor. For any integer $K\geq1$, the two sets in each row span the same space:

| Code | Spherical-harmonic angular factors | Trigonometric factors |
|---|---|---|
| `"11"` (full even space) | $S_0,S_2,\ldots,S_{2K}$ | $1,\cos(2\theta),\ldots,\cos(2K\theta)$ |
| `"00"` | $-S_2',-S_4',\ldots,-S_{2K}'$ | $\sin(2\theta),\sin(4\theta),\ldots,\sin(2K\theta)$ |
| `"10"` | $S_1,S_3,\ldots,S_{2K-1}$ | $\cos\theta,\cos(3\theta),\ldots,\cos((2K-1)\theta)$ |
| `"01"` | $-S_1',-S_3',\ldots,-S_{2K-1}'$ | $\sin\theta,\sin(3\theta),\ldots,\sin((2K-1)\theta)$ |

The first row includes $S_0=1$; the code's additional constraint on this block
is described below.

For the sine families (`"00"` and `"01"`),
`conversion_mat` takes an antiderivative to obtain a cosine series before
Legendre conversion, because the angular vector basis uses $S_l'$.

For `conversion_mat(N, bdry, direction)`, `direction = 0` maps trigonometric
coefficients into the spherical-harmonic basis, while `direction = 1` maps back.
The code uses the boundary code to choose the correct parity block, decide
whether a constant mode is present, and decide whether an antiderivative is
needed before applying `che2len` or `len2che`.

Coefficient rows multiply the conversion matrix on the right. For `"11"`, the
spherical block excludes $P_0$, while its trigonometric expansion still needs
a constant term, as in $P_2(\cos\theta)=\tfrac14+\tfrac34\cos(2\theta)$.
This explains the rectangular matrix and why inversion applies to the subspace
with zero $P_0$ coefficient.

For `Fourier_mat(N, bdry)`, the same boundary code selects the corresponding
discrete sine/cosine transform and endpoint weights. Rows are spectral modes and
columns are samples on the endpoint grid. The zero columns and half weights are
not arbitrary numerical choices; they encode the endpoint identities verified
above.

These boundary codes match the paper's parity expansions in `sec:div-free`,
including `eq:Utld exp`, `eq:U1 ansatz 1` and `eq:v ansatz 2`.

## 3. Chebyshev-Legendre Conversion Formulas

`che2len` and `len2che` convert between parity-separated Chebyshev and Legendre
coefficient vectors. The implementation splits even and odd blocks because the
axisymmetric solutions are expanded in parity-adapted bases.

The matrix convention is:

- `che2len` returns rows that expand Legendre modes in the Chebyshev basis;
- `len2che` returns rows that expand Chebyshev modes in the Legendre basis.

These are standard polynomial identities, not new estimates from the paper.
They convert the trigonometric coefficients used in `sec:res cg` into the
Legendre and vector-spherical-harmonic basis of `sec:div-free`.

A quick orientation check is:

$$
P_2(x) = \frac14 T_0(x) + \frac34 T_2(x).
$$

This is a row of the matrix returned by `che2len`: it expands a Legendre mode in
Chebyshev modes, and the $T_0$ coefficient is not doubled.

The inverse direction is:

$$
T_2(x) = \frac43 P_2(x) - \frac13 P_0(x).
$$

This is a row of `len2che`: it expands a Chebyshev mode in Legendre modes.

### 3.1 Legendre Modes Expanded in Chebyshev Modes: `che2len`

DLMF 18.5.11 gives the Gegenbauer trigonometric expansion

$$
C_n^{(\lambda)}(\cos\theta)
= \sum_{\ell=0}^{n}
  \frac{(\lambda)_\ell(\lambda)_{n-\ell}}
       {\ell!(n-\ell)!}
  \cos((n-2\ell)\theta).
$$

DLMF 18.7.9 gives

$$
P_n(x)=C_n^{(1/2)}(x).
$$

Setting $x=\cos\theta$ and using $T_m(\cos\theta)=\cos(m\theta)$ gives

$$
P_n(x)
= \sum_{\ell=0}^{n}
  \frac{(1/2)_\ell(1/2)_{n-\ell}}
       {\ell!(n-\ell)!}
  T_{|n-2\ell|}(x).
$$

The two terms with indices $\ell$ and $n-\ell$ give the same Chebyshev
polynomial, because $T_{-m}=T_m$. Therefore, for
$0 \le k \le \lfloor n/2\rfloor$,

$$
P_n(x)
= \sum_{k=0}^{\lfloor n/2\rfloor}
  (2-\delta_{n,2k})
  \frac{(1/2)_k(1/2)_{n-k}}
       {k!(n-k)!}
  T_{n-2k}(x).
$$

The Kronecker factor records the special constant term: when $n=2k$, the term
$T_0$ occurs only once, so it is not doubled.

Define

$$
q_m := \frac{(1/2)_m}{m!}
     = \frac{(2m)!}{4^m(m!)^2}
     = \frac{(2m-1)!!}{(2m)!!},
\qquad q_0=1.
$$

The code's `double_fact` builds exactly these products by the recurrence

$$
q_m=q_{m-1}\left(1-\frac{1}{2m}\right),
\qquad m\ge 1.
$$

The identity above immediately implies the even block. For $n=2i$ and
$T_{2j}$ set $k=i-j$. For $j>0$,

$$
[T_{2j}]P_{2i}
= 2 q_{i-j} q_{i+j}.
$$

For $j=0$, the constant mode is not doubled:

$$
[T_0]P_{2i}
= q_i^2.
$$

This is exactly the code:

$$
A[i+1,j+1] =
\begin{cases}
q_i^2, & j=0,\\
2q_{i-j}q_{i+j}, & 1\le j\le i.
\end{cases}
$$

The division by $2$ in the first column of the even branch is precisely the
$T_0$ exception.

For the odd block, take $n=2i+1$ and $T_{2j+1}$. Then $k=i-j$, and there is no
constant mode. Thus

$$
[T_{2j+1}]P_{2i+1}
= 2q_{i-j}q_{i+j+1},
\qquad 0\le j\le i,
$$

which is the `flag=true` branch of `che2len`.

### 3.2 Chebyshev Modes Expanded in Legendre Modes: `len2che`

For the inverse direction, start from the Gegenbauer connection formula
DLMF 18.18.16:

$$
C_n^{(\mu)}(x)
= \sum_{\ell=0}^{\lfloor n/2\rfloor}
  \frac{\lambda+n-2\ell}{\lambda}
  \frac{(\mu)_{n-\ell}}{(\lambda+1)_{n-\ell}}
  \frac{(\mu-\lambda)_\ell}{\ell!}
  C_{n-2\ell}^{(\lambda)}(x).
$$

Set $\lambda=1/2$, so $C_m^{(1/2)}=P_m$. DLMF 18.7.25 gives, for $n \ge 1$,

$$
T_n(x)
= \lim_{\mu\to 0} \frac{n+\mu}{2\mu} C_n^{(\mu)}(x),
$$

which is equivalent to the common shorthand
$T_n(x)=\lim_{\mu\to0} n C_n^{(\mu)}(x)/(2\mu)$ for $n \ge 1$.

Since $n-\ell \ge 1$ in the connection sum for
$0 \le \ell \le \lfloor n/2\rfloor$,

$$
\lim_{\mu\to0}\frac{(\mu)_{n-\ell}}{\mu}=(n-\ell-1)!,
\qquad
\lim_{\mu\to0}(\mu-1/2)_\ell=(-1/2)_\ell.
$$

Therefore, for $n \ge 1$,

$$
T_n(x)
= \sum_{\ell=0}^{\lfloor n/2\rfloor} c_{n,\ell} P_{n-2\ell}(x),
$$

with

$$
c_{n,\ell}
= \frac{n}{2}(2n-4\ell+1)
   \frac{(n-\ell-1)!(-1/2)_\ell}
        {\ell!(3/2)_{n-\ell}}.
$$

The case $n=0$ is outside the Chebyshev limit above and is simply

$$
T_0(x)=P_0(x),
$$

which is why `len2che` explicitly sets `A[1,1]=1` in the even block.

Now specialize the coefficient formula to the parity-separated matrices.

For the even block, set $n=2i$ and $\ell=i-j$. Then

$$
T_{2i}(x)
= \sum_{j=0}^{i} A[i+1,j+1]P_{2j}(x),
$$

where, for $i\ge 1$,

$$
A[i+1,j+1]
= i(4j+1)
   \frac{(i+j-1)!(-1/2)_{i-j}}
        {(i-j)!(3/2)_{i+j}}.
$$

For the odd block, set $n=2i+1$ and $\ell=i-j$. Then

$$
T_{2i+1}(x)
= \sum_{j=0}^{i} A[i+1,j+1]P_{2j+1}(x),
$$

with

$$
A[i+1,j+1]
= \frac{2i+1}{2}(4j+3)
   \frac{(i+j)!(-1/2)_{i-j}}
        {(i-j)!(3/2)_{i+j+1}}.
$$

The code evaluates these coefficients through Gamma-function ratios. Using

$$
\frac{(-1/2)_m}{m!}
= \frac{\Gamma(m-1/2)}{\Gamma(-1/2)\Gamma(m+1)}
$$

and

$$
\frac{(r-1)!}{(3/2)_r}
= \frac{\Gamma(r)\Gamma(3/2)}{\Gamma(r+3/2)},
$$

the product of the two normalizing constants is

$$
\frac{\Gamma(3/2)}{\Gamma(-1/2)}=-\frac14.
$$

Thus the even coefficient can be written as

$$
A[i+1,j+1]
= -\frac{i(4j+1)}{4}
   \frac{\Gamma(i+j)}{\Gamma(i+j+3/2)}
   \frac{\Gamma(i-j-1/2)}{\Gamma(i-j+1)}.
$$

The odd coefficient becomes

$$
A[i+1,j+1]
= -\frac{(i+1/2)(4j+3)}{4}
   \frac{\Gamma(i+j+1)}{\Gamma(i+j+5/2)}
   \frac{\Gamma(i-j-1/2)}{\Gamma(i-j+1)}.
$$

This is the implementation in `len2che`. The first `gamma_ratio` call uses a
half-integer cutoff and supplies factors of the form
$\Gamma(r)/\Gamma(r+3/2)$; the second uses an integer cutoff and supplies
$\Gamma(m-1/2)/\Gamma(m+1)$. Multiplying them by the branch-dependent factor and
by $-1/4$ gives exactly the formulas above.

The derivation in this section uses standard DLMF identities:
[DLMF 18.5.11](https://dlmf.nist.gov/18.5.E11),
[DLMF 18.7.9](https://dlmf.nist.gov/18.7.E9),
[DLMF 18.7.25](https://dlmf.nist.gov/18.7.E25), and
[DLMF 18.18.16](https://dlmf.nist.gov/18.18.E16). The paper-specific role of
these identities is to implement the basis conversion in `sec:div-free` and
`sec:res cg`.

<a id="4-internal-reflected-angular-basis-convention"></a>

## 4. Angular coefficient signs and the physical vector frame

The paper uses $Y_l=S_le_r$, $X_l=S_l'e_\theta$, and $Z_l=S_l'e_\phi$. `conversion_mat` represents tangential coefficients using $H_l=-S_l'$. For example, $H_2=\tfrac32\sin(2\theta)$ and $H_1=\sin\theta$; the displayed conversion matrices $[3/2]$, $[2/3]$, and $[1]$ therefore remain correct.

These are scalar-basis signs, not a physical coordinate transformation. `calc_Utheta_divfree` pairs H with its negative $1/[l(l+1)]$ coefficient map, and the Bessel evaluator pairs H with the negative physical poloidal X coefficient. Both reconstruct physical spherical components. The Bessel toroidal convention differs from the paper by one whole-mode sign. Use matching coefficient rows for that convention.

All stages use one physical Q. No additional $DQD$ conjugation or tangential component flip is required. See [the Bessel basis note](bessel_basis.md#92-scalar-angular-signs-and-physical-components), §9.2, for the common frame and [§6](bessel_basis.md#6-yxz-radial-formulas-and-coefficients) for the signed radial formulas.

## 5. Divergence-Free Reconstruction

For an axisymmetric vector field

$$
U = U_r(r,\theta)e_r + U_\theta(r,\theta)e_\theta,
$$

the spherical divergence identity is

$$
\operatorname{div} U
= \frac{1}{r^2}\partial_r(r^2 U_r)
  + \frac{1}{r\sin\theta}\partial_\theta(\sin\theta\,U_\theta).
$$

The paper's `sec:ops` appendix gives the spherical-coordinate operators.
In the vector spherical harmonic expansion, `eq:div u` gives the divergence
in modewise form:

$$
\operatorname{div}_{sph} u
= \sum_l
\left(
  \partial_r u_l^Y
  + \frac{2}{r}u_l^Y
  - \frac{\mu_l}{r}u_l^X
\right)S_l,
\qquad \mu_l = l(l+1).
$$

The transformed divergence-free condition used for the self-similar profile is
`eq:div sph cv 1`:

$$
\frac12\sin(2\beta)\,\partial_\beta \widehat U_l^Y
+ \frac12(3+\cos(2\beta))\,\widehat U_l^Y
- \mu_{2l}\widehat U_l^X = 0.
$$

The paper's `eq:U1 ansatz 1` expands the profile coefficients in sine series;
`eq:div sph cv 2` determines the angular block from the radial block.
`eq:v ansatz 2.1`, `eq:v1 ansatz 2` and `eq:v ansatz 2` give the analogous
eigenfunction parity blocks.

In code, `div_mat(M, bdry)` builds the left coefficient recurrence in the
radial trigonometric index. `calc_Utheta_divfree(U, bdry)` then multiplies by

$$
-\frac{1}{l(l+1)}
$$

mode by mode. The negative sign compensates the scalar factor $H_l=-S_l'$
in Section 4, giving the physical angular component. The function returns the intermediate
matrices as well as the reconstructed coefficient matrix so later verification
steps can reuse the exact same linear map.

Multiplication by the double-angle factors shifts a β frequency by two.
Consequently, $M$ radial input modes require $M+1$ output modes to retain the
exact divergence-free reconstruction; this accounts for the extra row returned
by `div_mat`.

## 6. Fourier Matrices and Endpoint Weights

`Fourier_mat(N, bdry)` constructs the boundary-adapted discrete Fourier transform
matrix that maps endpoint-grid samples to modal coefficients. The rows index the
trigonometric mode, and the columns index the grid values. The endpoint weights
are exactly the interpolation weights dictated by the four boundary families in
Section 2.

Here $N$ counts grid intervals: $x_j=j\pi/(2N)$, $j=0,\ldots,N$.
For `"11"`, discrete orthogonality gives

$$
\sum_{j=0}^{N} w_j\cos(2kx_j)\cos(2mx_j)
=\begin{cases}
N,&k=m\in\{0,N\},\\
N/2,&k=m,\quad 0<k<N,\\
0,&k\ne m,
\end{cases}
\qquad w_0=w_N=\tfrac12,\quad w_j=1\ (0<j<N).
$$

The endpoint **columns** therefore have half weights, and the constant and
highest-frequency **rows** need another factor $1/2$ after the common $2/N$
scaling. For the other families, Dirichlet endpoint columns are zero and a
remaining Neumann endpoint has half weight; no row half weights are needed.

`Fourier_mat` is used by the spatial-to-frequency path in
[My_Scalar.jl](../../src/My_Scalar.jl). The current verification notebooks
load precomputed coefficients and evaluate them with `trig_func`; their
`from_mat_file(...; domain="frequency")` calls do not use `Fourier_mat`.

## 7. `trig_func` Key Groups and Their Role in `NS_Terms.jl`

`trig_func(N, x, key, bdry)` evaluates one-dimensional basis functions or
operator expressions at radial β points or angular θ points. Here `N` counts
modes. The return shape is

$$
\text{length}(x) \times N.
$$

The basic keys are:

| Key | Meaning |
|---|---|
| `"I"` | basis function |
| `"D"` | first derivative |
| `"D2"` | second derivative |
| `"Cot"`, `"Cos"`, `"Tan"` | $f\cot x$, $f\cos x$, $f\tan x$ for compatible bases |
| `"Sec"` | $f/\cos x$ for odd-cosine `"10"` modes; direct division and analytic endpoint values |
| `"Scale"`, `"Scale_T"` | scaled expressions from the coordinate transform |
| `"zero"` | zero block with the correct shape |

The Laplacian and `D11`, `D12`, `D21`, `D22`, `D23`, `D31`, `D32`, `D33`
keys supply radial or angular factors of the transformed operators.
`NS_Terms.jl` combines these into blocks acting as
$B_\beta C B_\theta^{\mathsf T}$ on trigonometric coefficient arrays $C$.
Derivatives are evaluated from the basis formulas. The code groups these strings so that shared quantities such as
$f$, $f'$, $f''$, $f/\sin x$, and $f'/\sin x$ are computed only when needed.

Representative keys are:

$$
\operatorname{Scale}(f)
= \frac{\sin x}{2}\left(\cos x\,f+\sin x\,f'\right),
$$

$$
\operatorname{Scale\_T}(f)
= -\frac{\sin x}{2}\left(\cos x\,f+\sin x\,f'\right),
$$

$$
\operatorname{Laplacian\_r\_\mathnormal{\theta}}(f)
= \cos x\,\frac{f'}{\sin x}+f'',
$$

$$
\operatorname{D12\_r\_sym}(f)
= \frac{\cos^2 x}{2}\frac{f}{\sin x},
$$

and

$$
\operatorname{D\_p\_r}(f)
= \sin x\left(-\sin x\,f+\cos x\,f'\right).
$$

The remaining keys follow the same pattern: they are code-level expansions of
the spherical-coordinate operators in `sec:ops` after the trigonometric
coordinate changes used in the paper. `subsec:analytic-L2` explains why these terms
remain finite trigonometric sums or finite combinations with removable quotient
singularities.

## 8. Endpoint Fixes and Removable Singularities

Several `trig_func` keys require quotients such as

$$
\frac{\sin(n x)}{\sin x}
\quad\text{or}\quad
\frac{\partial_x\cos(n x)}{\sin x}.
$$

These quotients are evaluated by direct division away from endpoints and then
patched at endpoint rows using their limiting values. This matches the analytic
principle in the paper's `subsec:analytic-L2`: quotient expressions such as
$\sin(n\beta)/\sin\beta$ have exact finite trigonometric representations, so
the apparent singularities are removable. The code-level endpoint replacements
therefore preserve the intended analytic expression rather than adding a
numerical approximation.

The `"Sec"` key evaluates $\cos((2m-1)x)/\cos x$ for the odd-cosine (`"10"`)
family. It divides directly at ordinary grid nodes. At rows identified as
$\pi/2$ or $3\pi/2$, it fills in the exact limits $1,-3,5,\ldots$ using the
endpoint masks described below. Step 1 uses this key to evaluate
$H=Q/\cos(x_Q)$ in the positivity check.

For `"Tan"`, the compatible families are $f(x)=\sin(2kx)$ (`"00"`) and
$f(x)=\cos((2k-1)x)$ (`"10"`), $k\ge1$. At either zero $x_0$ of $\cos x$,

$$
\lim_{x\to x_0} f(x)\tan x
=\frac{f'(x_0)\sin x_0}{-\sin x_0}=-f'(x_0).
$$

Hence the endpoint values are:

| Basis | At $\pi/2$ | At $3\pi/2$ |
|---|---|---|
| $\sin(2kx)$ | $(-1)^{k+1}2k$ | $(-1)^{k+1}2k$ |
| $\cos((2k-1)x)$ | $(-1)^{k-1}(2k-1)$ | $(-1)^k(2k-1)$ |

On the production grids, no non-endpoint node lies within $10^{-14}$ of a
singular endpoint. Nodes selected by these masks are intended endpoints,
where the code substitutes the exact analytic limit; the resulting interval
values remain rigorous. The target is the value at the known mathematical
node, whose interval representation accounts for rounding. This justification
depends on the grid construction; it does not cover arbitrary nearby inputs.

For example, `eigenpair_error_analysis.ipynb` constructs a prior grid with
`L0 = 2*Pi`, `N0 = 32000` and a refinement grid with `L0 = Pi/2`,
`N0 = 36000`. Their radial spacings are $\pi/16000$ and $\pi/72000$,
respectively; the singular endpoints are grid nodes and every other node
is much farther than $10^{-14}$ away. For a changed grid, check the mask
against its intended endpoint indices before using the same interpretation.

## 9. Interval Arithmetic Conventions

`dtype` is selected in `NS_Numerics.jl`. When `dtype == Float64`, `dt(x)` returns
a floating-point approximation and the results are exploratory. When
`dtype == Interval{Float64}`, `dt(x)` returns an interval enclosure through
`@interval`, and the resulting arithmetic tracks outward-rounded enclosures.

The wrappers `mysin` and `mycos` call `sinpi(x / Pi)` and `cospi(x / Pi)`,
with `Pi = dt(π)`. Interval division can widen the argument even at an intended
multiple of π. The result encloses the trigonometric value; exact values for
removable quotients are supplied by the endpoint limits in Section 8.

The zero test `is_zero` is also type-dependent. In floating point, small values
below a fixed threshold are treated as zero for sparse coefficient handling. In
interval mode, only the exact zero interval is treated as zero.

## 10. Cross-Reference Table

| Function or group | Role | Paper reference |
|---|---|---|
| `dt`, `is_zero`, `Pi`, `mysin`, `mycos` | Active numeric-type and endpoint-trigonometry conventions | Implementation convention; supports rigorous interval arithmetic throughout the verification |
| `double_fact`, `gamma_ratio` | Recurrence-generated constants for Chebyshev-Legendre conversion | Orthogonal-polynomial conversion in `sec:res cg` |
| `che2len`, `len2che` | Convert parity-separated Chebyshev and Legendre coefficient blocks | Legendre basis in `sec:div-free`; interpolation in `sec:res cg` |
| `conversion_mat` | Convert trigonometric coefficients to/from vector-spherical-harmonic coefficient blocks | `xyz`, `sec:div-free`, `sec:res cg` |
| `Fourier_mat` | Boundary-adapted discrete Fourier transform from samples to modes | `sec:res cg` |
| `div_mat` | Build the radial-to-angular coefficient recurrence for divergence-free fields | `eq:div u`, `eq:div sph cv 1`, `eq:div sph cv 2`, `sec:ops` |
| `calc_Utheta_divfree` | Reconstruct angular coefficients from radial coefficients using the recurrence | `eq:div sph cv 1`, `eq:div sph cv 2`, `eq:v ansatz 2` |
| `trig_func` basic keys | Evaluate basis functions, derivatives, and elementary multipliers | `subsec:analytic-L2`, `sec:ops` |
| `trig_func` Laplacian and `D*` keys | Evaluate radial or angular factors of transformed differential operators for `NS_Terms.jl` | `sec:ops`, `subsec:analytic-L2` |
| Endpoint quotient fixes | Known-node analytic limits, including `"Sec"` | `subsec:analytic-L2` |
