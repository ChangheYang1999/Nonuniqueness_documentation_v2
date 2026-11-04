# Rigorous Bessel-Zero Certificate — Theory Note

For the main reader path, see the [source guide](../src.md). This note follows
the root search in [get_bessel_zeros.ipynb](../../notebooks/get_bessel_zeros.ipynb):

**Choose the degrees → locate the poles → bracket the family roots → refine
their enclosures → save the root tables.**

The notebook organizes the search; [My_Bessel.jl](../../src/My_Bessel.jl)
provides the root equations, interval Bessel evaluations, and local refinement.
The notebook requires `dtype == Interval{Float64}`. The mathematical argument
for finding every root is brought together in §5. The paper's basis lemma
(`lem:lap eig`) and root lemma (`lem:zeros`) are restated at the end of this
note as [Lemma 4](#paper-lemma-4) and [Lemma 5](#paper-lemma-5).

## 1. What roots do we need, and what is the strategy?

Start with one of the basis functions from [Lemma 4](#paper-lemma-4) of the paper:

$$
\psi^Z_{l,j}(r,\theta)
=\frac{J_{l+1/2}(\alpha^Z_{l,j}r)}{\sqrt r}\,Z_l(\theta).
$$

Here $J_\nu$ is the Bessel function of the first kind, $0<r\le1$ is the distance
from the origin in the unit ball, and $\theta$ is the polar angle.
$Z_l(\theta)$ is the angular vector for degree $l\ge1$.
For a fixed $l$, we can evaluate this function once we know
$\alpha^Z_{l,j}$. The index $j$ counts the positive roots in increasing order.
[Lemma 4](#paper-lemma-4) tells us how to find them: solve

$$
\alpha J_{l+1/2}'(\alpha)-\tfrac12J_{l+1/2}(\alpha)=0,
\qquad \alpha>0.
$$

Each solution supplies one Z basis function. The Y basis functions use roots
of a second equation given below. This notebook finds the positive roots of
both equations below $100$ (`R_CUT`), storing them for the later basis
construction. The [basis note](bessel_basis.md), §§6–9, explains how the
radial and angular factors are evaluated.

### 1.1 The two equations evaluated by the code

For Z, the notebook evaluates `func_Z(l, x)` and seeks its zeros:

$$
f_Z(l,x)=x\frac{J_{l-1/2}(x)}{J_{l+1/2}(x)}-(l+1)=0.
$$

This is the preceding equation divided by $J_{l+1/2}$ and rewritten to
avoid evaluating a Bessel derivative; the derivation is in §1.3.

For Y, [Lemma 4](#paper-lemma-4) instead requires

$$
\bigl(\alpha^2-l(l-1)(l+2)\bigr)\alpha\tilde j_l'(\alpha)
+\bigl(\alpha^2-l(l-1)\bigr)\bigl(\alpha^2-(l-1)(l+2)\bigr)
\tilde j_l(\alpha)=0,
\qquad \tilde j_l(x)=\frac{J_{l+1/2}(x)}{x^{3/2}}.
$$

The code rewrites this as `func_Y(l, x)`. For $l\ge2$, it is

$$
f_Y(l,x)=x\frac{J_{l-1/2}(x)}{J_{l+1/2}(x)}-(l+2)
+\frac{(x^2-l(l-1))(x^2-(l-1)(l+2))}{x^2-l(l-1)(l+2)}=0.
$$

For $l=1$, the rational term reduces to $x^2$:

$$
f_Y(1,x)=x\frac{J_{1/2}(x)}{J_{3/2}(x)}-3+x^2=0.
$$

**Why locate poles before roots?** These quotients are undefined where their
denominators vanish. A sign change across such a point does not establish a
root: $1/(x-p)$ changes sign across $p$ but is never zero. We therefore locate
the poles first and search between them. [Lemma 5(1)–(2)](#paper-lemma-5) gives exactly one
family root in each full interval between consecutive poles, and one between
zero and the first pole. The current search includes the first family pole
strictly above $R=100$, encloses the root in every required full pole interval,
and then classifies each root relative to the cutoff. It does not use a
cutoff-truncated terminal interval.

### 1.2 What a root tells us about the eigenvalue

A root $\alpha$ gives the eigenvalue

$$
\lambda=\frac{\alpha^2}{32^2}
$$

on the radius-$32$ ball used in the finite-rank construction. Thus finding all
positive roots below $R=100$ identifies the corresponding positive eigenvalues
below $(100/32)^2$ for that ball.

To justify the scale factor, [Lemma 4](#paper-lemma-4) gives $\alpha^2$ on the unit ball. Write
$X=R_0y$ with $R_0=32$, where $X$ is the physical position and $y$ is the
unit-ball coordinate. The volume element gains $R_0^3$ and each derivative
gains $R_0^{-1}$. The squared gradient norm therefore gains $R_0$, while
the squared $L^2$ norm gains $R_0^3$. Their quotient gains $R_0^{-2}$, giving
$\lambda=(\alpha/R_0)^2$. Since squaring is increasing for positive numbers,
$\alpha<R$ is equivalent to $\lambda<(R/R_0)^2$.

### 1.3 Why the code's equations have the same roots

The defining Bessel series is

$$
J_\nu(x)=\sum_{n=0}^{\infty}
\frac{(-1)^n(x/2)^{2n+\nu}}{n!\,\Gamma(n+\nu+1)}.
$$

Termwise differentiation gives $J_\nu'=J_{\nu-1}-\nu J_\nu/x$.
For the Z equation, this gives

$$
\frac{xJ_{l+1/2}'(x)-\tfrac12J_{l+1/2}(x)}{J_{l+1/2}(x)}
=x\frac{J_{l-1/2}(x)}{J_{l+1/2}(x)}-(l+1)=f_Z(l,x).
$$

For Y, first use

$$
\frac{x\tilde j_l'(x)}{\tilde j_l(x)}
=\frac{xJ_{l+1/2}'(x)}{J_{l+1/2}(x)}-\frac32
=x\frac{J_{l-1/2}(x)}{J_{l+1/2}(x)}-(l+2).
$$

Dividing the Y equation in §1.1 by
$\bigl(x^2-l(l-1)(l+2)\bigr)\tilde j_l(x)$ now gives $f_Y(l,x)=0$.
These divisions are valid away from the poles. [Lemma 5(1)–(2)](#paper-lemma-5) places the
positive family roots between the poles, where the rewritten equations apply.

## 2. Which degrees and starting points are needed?

If the first positive root for a degree is at least $100$, all its later roots
are also outside the search. Otherwise we need a starting point `lb` below
which there is no positive root. The notebook computes that point using
`y_lower_bound` or `z_lower_bound`.

### 2.1 Use the bounds to choose the search range

| Family | Degree | `lb` returned | Bound on the first positive root |
|---|---|---|---|
| Y | $l\ge4$ | $l$ | $\alpha^Y_{l,1}>l$. |
| Y | $l=2,3$ | $\sqrt{l(l-1)}$ | $\alpha^Y_{l,1}>\sqrt{l(l-1)}$. |
| Y | $l=1$ | $1$ | $\alpha^Y_{1,1}\ge2$. |
| Z | all $l$ | $\sqrt{l(l+1)}$ | $\alpha^Z_{l,1}\ge\sqrt{l(l+1)}$. |

The [proof below](#first-root-bounds-proof) derives the Y bounds from
[Lemma 5](#paper-lemma-5); the Z bound is [Lemma 5(3)](#paper-lemma-5).

For Y, $l\ge100$ implies $\alpha^Y_{l,1}>l\ge R$. For Z,
$l(l+1)\ge100\cdot101>100^2$ when $l\ge100$, so its lower bound exceeds $R$.
Thus only degrees $1,\ldots,99$ can contribute positive roots below the cutoff.

The notebook searches Y degrees `1:100` (`L_CAP_Y = 101`) and Z degrees
`1:99` (`L_CAP_Z = 100`). The extra Y degree is excluded by its lower bound:
`search_family_roots` returns early when it certifies `lb ≥ R`.
For the remaining degrees, `lb` is the left boundary of the family-root search.

<a id="first-root-bounds-proof"></a>

### 2.2 Why these starting points are valid

Let $s_l$ be the unique positive root of

$$
Q_l(s)=s^3-4l^2s^2+l(l-1)(6l^2+11l+7)s
-3l(l-1)^2(l+1)^2(l+2),
$$

and let $t_l=\sqrt{l(l+1)}$. [Lemma 5(3)](#paper-lemma-5) gives

$$
\alpha^Y_{l,1}\ge\sqrt{s_l},\qquad
\alpha^Z_{l,1}\ge t_l.
$$

The cubic sign pattern and the threshold non-root conditions are justified
analytically in [§2.3](#threshold-contacts-and-uniqueness) below. The notebook
does not perform a numerical threshold non-root test.

The notebook does not solve for $s_l$. Since $Q_l$ has a unique positive root
and is negative before it, any $v>0$ with $Q_l(v)<0$ gives

$$
\alpha^Y_{l,1}\ge\sqrt{s_l}>\sqrt v.
$$

Taking $v=l^2$, direct expansion yields

$$
Q_l(l^2)=-l\bigl(l^4-2l^3-5l^2+3l+6\bigr).
$$

For $l=t+4$, $t\ge0$, the quartic factor is

$$
l^4-2l^3-5l^2+3l+6=t^4+14t^3+67t^2+123t+66>0.
$$

Consequently $\alpha^Y_{l,1}>l$ for $l\ge4$, which explains the main branch
of `y_lower_bound`. For the smaller degrees, direct substitution gives

$$
Q_l\bigl(l(l-1)\bigr)=-2l(l-1)^2(l+1)(l+3)<0\quad(l\ge2),
\qquad Q_1(s)=s^2(s-4).
$$

Thus $s_l>l(l-1)$ for $l=2,3$, while $s_1=4$. These calculations justify
all the Y entries in the table above.

<a id="threshold-contacts-and-uniqueness"></a>

### 2.3 Threshold contacts and uniqueness

The threshold non-root clauses can be discharged analytically; the notebook does not test them numerically. Put $\nu=l+1/2$, $C=l(l-1)(l+2)$, $R_l(x)=xJ_\nu'(x)/J_\nu(x)$, and

$$
T_l(x)=\frac{(x^2-l(l-1))(x^2-(l-1)(l+2))}{x^2-C},\qquad F_l=R_l-\tfrac32+T_l,
$$

using $T_1=x^2$. On a pole-free interval the Bessel equation gives

$$
F_l'=-\frac{F_l^2+(3-2T_l)F_l}{x}-\frac{xQ_l(x^2)}{(x^2-C)^2}.
$$

At a first zero approached from positive values, $F_l'\le0$, so $x^2\ge s_l$. Equality would imply $F_l=F_l'=0$ and

$$
F_l''=-\frac{2x^2Q_l'(x^2)}{(x^2-C)^2}<0.
$$

This contradicts positivity immediately to its left. For $l\ge2$, $Q_l'>0$ follows from its negative quadratic discriminant and positive leading coefficient; for $l=1$, $s_1=4$ and $Q_1'(4)=16>0$. Hence the first zero lies strictly above $\sqrt{s_l}$. Every subsequent zero in the same pole interval would also have strictly negative derivative; after the first downward crossing another zero would require a nonnegative crossing or contact, a contradiction. The pole limits and positive small-x expansions derived in [§4.3](#43-how-the-sign-of-fx-locates-x-relative-to-the-root) give existence in each interval.

For Z set $G_l=R_l-\tfrac12$. Then

$$
G_l'=\frac{l(l+1)-x^2-G_l-G_l^2}{x}.
$$

At $x=\sqrt{l(l+1)}$ a zero would satisfy $G_l'=0$ and $G_l''=-2$, again impossible at the first contact from positive values. This proves the strict first-root bound and the same one-root conclusion. All statements are used only on intervals free of Bessel poles and, for Y, the extra rational pole; the code checks separation of the poles needed for its finite partition.

## 3. How do we locate all the poles?

Fix a degree $l$ that passed the lower-bound test. Both equations have
denominator $J_{l+1/2}$. Its positive zeros
$\beta_{l,1}<\beta_{l,2}<\cdots$ are their shared Bessel poles. These are
different from the roots $\alpha^Y_{l,j}$ and $\alpha^Z_{l,j}$ that we want
to save. The numerator is nonzero at a denominator zero; §4.3 proves this.

### 3.1 Scan one cell at a time

`bessel_pole_ladder` scans $J_{l+1/2}$ from
$x_{\mathrm{start}}=\max(\mathrm{lb},3)$ to $R$ in cells of width at most
`SCAN_STEP = 3`, including a node at $R$. It then continues in width-3 cells
until the first sign change beyond $R$. After refining the Bessel-pole
enclosures, it requires `inf(poles[end]) > sup(R)`. Thus the complete ladder
includes a pole certified strictly above the cutoff; failure of this last
comparison stops the search.

For each cell $[a,b]$, evaluate $J_{l+1/2}$ at both endpoints using interval
arithmetic. A positive sign means the entire value enclosure is positive;
an enclosure touching zero has undecidable sign. The notebook then makes
the following decision:

| Rigorous endpoint signs | Action |
|---|---|
| Opposite | Enclose the denominator zero with `rigorous_root_interval` and add it to the pole list. |
| Equal and nonzero | Discard the cell: it contains no denominator zero. |
| Either sign undecidable | Stop: these evaluations do not classify the cell. |

Why is that decision sufficient? Each cell is shorter than $\pi$, while
consecutive denominator zeros are more than $\pi$ apart. Thus a cell contains
at most one zero. Such a zero is simple: if $J_\nu$ and $J_\nu'$ both vanished
at a positive point, uniqueness for the Bessel differential equation in §3.3
would force $J_\nu$ to vanish identically. A simple zero reverses the sign.
Consequently, opposite endpoint signs give exactly one zero by the intermediate
value theorem, and equal strict signs exclude a zero.

The notebook checks the cell-width condition with
`sup(SCAN_STEP) < inf(Pi)`. The [spacing proof](#bessel-zero-spacing-proof)
also explains why the scan loses no poles before its starting point.
The [interval-evaluation argument](#bessel-interval-evaluation-proof) explains
how the Bessel value enclosures used in these sign tests are obtained.

### 3.2 Split once more at the extra Y pole

For $l\ge2$, the rational term in $f_Y$ also has a pole at

$$
\gamma=\sqrt{l(l-1)(l+2)}.
$$

Suppose $p_i<\gamma<p_{i+1}$ for two neighboring Bessel poles. The Y search
must replace the single interval $(p_i,p_{i+1})$ by

$$
(p_i,\gamma),\qquad(\gamma,p_{i+1}).
$$

Otherwise a bracket could cross $\gamma$, where $f_Y$ is undefined. The same
split applies if $\gamma$ lies in the first interval or in the full interval
that crosses $R$. There is no extra positive rational pole for $l=1$.

`add_y_extra_pole!` checks that the enclosure of $\gamma$ is disjoint from
every computed Bessel-pole enclosure. Its caller passes the terminal Bessel
pole as the helper's argument `R`, so insertion requires
`sup(gamma) < inf(poles[end])` with respect to that original Bessel list.
After insertion, `search_family_roots` truncates the sorted family list at
its first pole with `inf(pole) > sup(R)` for the search cutoff. If $\gamma$ is that pole, it
becomes the terminal boundary. The disjointness checks supply the separation
needed by the finite partition; an overlap aborts rather than being interpreted
as a root or skipped interval.

<a id="bessel-zero-spacing-proof"></a>

### 3.3 Proof of the zero spacing and the scan's starting point

Write $j_{\nu,n}$ for the $n$th positive zero of $J_\nu$. For our denominator,
$\nu=l+1/2$ and $j_{\nu,n}=\beta_{l,n}$. We will show that, for $\nu>1/2$,

$$
j_{\nu,1}>\pi,\qquad j_{\nu,n+1}-j_{\nu,n}>\pi.
$$

Every denominator order used in the pole scan is at least $3/2$, so the
result applies to every scanned degree.

Substituting the series in §1.3 and
matching powers gives $x^2J_\nu''+xJ_\nu'+(x^2-\nu^2)J_\nu=0$.
Writing $u(x)=\sqrt{x}J_\nu(x)$ and differentiating then gives

$$
u''+q_\nu u=0,\qquad
q_\nu(x)=1-\frac{\nu^2-1/4}{x^2}<1.
$$

Take either $a=0$, $b=j_{\nu,1}$, or two consecutive positive zeros $a,b$.
Integration by parts and the Dirichlet–Poincaré inequality give

$$
\left(\frac{\pi}{b-a}\right)^2\int_a^b |u|^2\,dx
\le\int_a^b |u'|^2\,dx
=\int_a^b q_\nu|u|^2\,dx
<\int_a^b |u|^2\,dx.
$$

The Poincaré step follows by expanding $u$ in Dirichlet sine modes on $[a,b]$:
mode $n$ contributes $(n\pi/(b-a))^2$ times its squared $L^2$ norm to
$\int_a^b |u'|^2$, and every $n\ge1$ has this factor at least $(\pi/(b-a))^2$.

At the origin, the first term of the series in §1.3 gives
$u(x)\sim Cx^{\nu+1/2}$, with $C=2^{-\nu}/\Gamma(\nu+1)>0$. Thus $uu'=O(x^{2\nu})\to0$,
and $|u'|^2$ and $|q_\nu u^2|$ are $O(x^{2\nu-1})$, which is integrable.
This justifies integration by parts at zero. Since $u$ is nonzero, the
inequality gives $b-a>\pi$ in both cases.

**Why nothing is lost before the scan starts.** The first denominator zero
exceeds $\pi>3$. If `lb > 3`, the first-root bound in §2 and the fact that
the first family root precedes the first pole ([Lemma 5(1)–(2)](#paper-lemma-5))
place that pole beyond `lb` as well. For Y, that first pole is the smaller of the
first Bessel pole and $\gamma$, so this also excludes a Bessel pole below `lb`.
Starting at $\max(\mathrm{lb},3)$ therefore discards no denominator zeros.

<a id="bessel-interval-evaluation-proof"></a>

### 3.4 Why the Bessel value enclosure is valid

The sign tests need an enclosure of $J_\nu(x)$ for the entire interval
argument supplied by the computation. Endpoint values alone would not bound
the interior, so the evaluation adds an interpolation error allowance.

For $0<a<b$, let $L$
be the affine interpolant of the endpoint Bessel values. The interpolation
remainder gives, for some $\xi\in[a,b]$,

$$
J_\nu(x)-L(x)=\tfrac12(x-a)(x-b)J_\nu''(\xi),\qquad
|J_\nu(x)-L(x)|\le\frac{(b-a)^2}{8}M_2,
$$

where $M_2\ge\sup_{[a,b]}|J_\nu''|$. The factor $1/8$ follows from
$|(x-a)(x-b)|\le(b-a)^2/4$. Since every chord value lies between the endpoint
values, enlarging their hull by this error encloses every interior value.
For $a=b$, the endpoint enclosure alone suffices.

To obtain $M_2$ for $\nu=l+1/2$, $l\ge0$ an integer, let $P_l$ be the
Legendre polynomial and put
$g_l(x)=J_{l+1/2}(x)/\sqrt{x}$. Rodrigues' formula
$P_l(t)=(2^l l!)^{-1}(d/dt)^l(t^2-1)^l$, followed by $l$ integrations by
parts and expansion of the exponential, gives

$$
g_l(x)=\frac{i^{-l}}{\sqrt{2\pi}}\int_{-1}^1e^{ixt}P_l(t)\,dt,
\qquad \int_{-1}^1P_l(t)^2\,dt=\frac{2}{2l+1}.
$$

For the first identity, compare the resulting power series with the Bessel
series in §1.3. For the second, the same integrations by parts reduce the
integral to $(2l)!/(2^{2l}(l!)^2)$ times $\int_{-1}^1(1-t^2)^l\,dt$;
the latter equals $2^{2l+1}(l!)^2/(2l+1)!$ by integration by parts.
Differentiation under the integral and Cauchy–Schwarz now give

$$
|g_l^{(k)}(x)|\le\frac{1}{\sqrt{2\pi}}
\left(\int_{-1}^1t^{2k}\,dt\int_{-1}^1P_l(t)^2\,dt\right)^{1/2}
=\sqrt{\frac{2}{\pi(2k+1)(2l+1)}}.
$$

Differentiating $J_{l+1/2}=\sqrt{x}\,g_l$ twice gives
$J_{l+1/2}''=-g_l/(4x^{3/2})+g_l'/\sqrt{x}+\sqrt{x}\,g_l''$.
The bounds for $k=0,1,2$, with decreasing powers evaluated at $a$ and the
increasing power at $b$, therefore give the uniform bound

$$
M_2=\sqrt{\frac{2}{\pi(2l+1)}}
\left(\frac{1}{4a^{3/2}}+\frac{1}{\sqrt{3}\,a^{1/2}}+
\frac{b^{1/2}}{\sqrt{5}}\right).
$$

The implementation details are in [bessel_basis.md](bessel_basis.md), §§3–4.

## 4. How does the pole partition produce root brackets?

Between neighboring poles $p_i$ and $p_{i+1}$, we seek two points $a,b$ such
that

$$
p_i<a<\alpha<b<p_{i+1},\qquad f(a)>0>f(b),
$$

where $f$ is `func_Y(l, ·)` or `func_Z(l, ·)` and $\alpha$ is its root.
We do not yet know $\alpha$: the two strict signs certify that it lies
between $a$ and $b$. Continuity between poles gives existence by the
intermediate value theorem; [Lemma 5(1)–(2)](#paper-lemma-5) gives uniqueness.

### 4.1 Use the pole enclosures to choose safe boundaries

Let $p_1<\cdots<p_m$ be the family poles through the first pole strictly
above $R$, including $\gamma$ for Y when applicable. The exact search partition
is

$$
(\mathrm{lb},p_1),\quad(p_1,p_2),\quad\ldots,\quad(p_{m-1},p_m).
$$

The lower bound from §2 replaces zero in the first interval. The last right
boundary is a pole above the cutoff. Even if there is no pole below $R$, the
single interval $(\mathrm{lb},p_1)$ extends to the first pole above it.
Every interval therefore has one family root by Lemma 5; its position relative
to $R$ is determined after enclosure.

Each pole is known through an interval $P_i=[p_i^-,p_i^+]$. The usable interval
between $P_i$ and $P_{i+1}$ is $(p_i^+,p_{i+1}^-)$: every point in it lies
strictly between the true poles. The notebook uses

```julia
lefts = [lb; dt.(sup.(poles[1:end-1]))]
rights = dt.(inf.(poles))
```

For example, the interval after the second recorded pole starts at `sup(poles[2])`
and ends at `inf(poles[3])`. These are enclosure endpoints; the actual poles
remain inside their respective enclosures.

### 4.2 Move inside those boundaries and test the signs

For a usable interval with boundaries $A,B$, `bracket_in_interval` places
its probes at

$$
a=A+\varepsilon,\qquad b=B-\varepsilon,
\qquad \varepsilon=\texttt{POLE\_MARGIN}=10^{-5}.
$$

For example, between the second and third recorded poles,
$A=\sup(\texttt{poles[2]})$ and $B=\inf(\texttt{poles[3]})$.
The margin moves both probes farther into the region between the true poles.
The notebook checks that the entire left probe enclosure lies before the right
one, then evaluates the two signs:

| Probe values | Result |
|---|---|
| Strictly positive at $a$, strictly negative at $b$ | Return `:root` with a bracket enclosing the unique root. |
| Either sign undecidable | Stop: the bracket has not been certified. |
| Other pairs of strict signs | Return `:none`. |

The sign pattern proved next explains the positive-left, negative-right test.
A `:none` result only says that these probes failed to bracket the root;
it does not exclude a root in the omitted margins. The caller immediately
raises an error for this result, so no full interval can be silently skipped
(§5.2).

### 4.3 How the sign of $f(x)$ locates $x$ relative to the root

At a Bessel pole $p$, the recurrence derived in §1.3 gives
$J_{l+1/2}'(p)=J_{l-1/2}(p)\ne0$, using simplicity from §3.1.
Taylor expansion gives $J_{l+1/2}(x)=J_{l+1/2}'(p)(x-p)+O((x-p)^2)$;
substitution yields

$$
x\frac{J_{l-1/2}(x)}{J_{l+1/2}(x)}\sim\frac{p}{x-p}.
$$

The family function tends to $+\infty$ just right of the pole and to
$-\infty$ just left of it. The extra Y pole has the same signs: writing
$A=l(l-1)$, $B=(l-1)(l+2)$ and $C=l(l-1)(l+2)$, we have $A<B<C$ for
$l\ge2$: indeed $B-A=2(l-1)>0$ and $C-B=(l-1)^2(l+2)>0$.
Its rational term is asymptotic to
$(C-A)(C-B)/(2\gamma(x-\gamma))$, whose coefficient is positive.
The other term stays finite at each pole by the separation in §3.2,
so it does not change these limits.

For the first interval, retain the first two terms of the series in §1.3
and divide the numerator and denominator expansions to obtain

$$
x\frac{J_{l-1/2}(x)}{J_{l+1/2}(x)}
=2l+1-\frac{x^2}{2l+3}+O(x^4),
$$

which gives

$$
f_Z(l,x)=l+O(x^2),\qquad
f_Y(l,x)=\frac{3(l+1)^2}{l(l+2)(2l+3)}x^2+O(x^4).
$$

For Y with $l\ge2$, the coefficient is
$(2l+1)/(l(l+2))-1/(2l+3)=3(l+1)^2/(l(l+2)(2l+3))$;
for $l=1$ it is $1-1/5=4/5$.
Both leading coefficients are positive. By continuity, a function cannot
change sign on either side of its unique root $\alpha$ without introducing
another zero. The positive left-end and negative right-end signs therefore give,
within the full pole interval,

$$
f(x)>0\quad\text{for }x<\alpha,\qquad
f(x)<0\quad\text{for }x>\alpha.
$$

## 5. What does refinement establish, and when is the search complete?

`search_family_roots` combines the preceding steps for one degree and one
family. We first narrow each accepted bracket, then ask whether the list
accounts for every root below $R$.

### 5.1 Refine an accepted bracket

Start with a bracket whose probe signs are certified positive on the left and
negative on the right. The notebook passes this Y/Z bracket to
`rigorous_root_interval` using its defaults on the first call. The routine expands each endpoint
outward by `δ=1e-10` and checks the endpoint signs again. With the current
`POLE_MARGIN=1e-5` and Float64 scales, this expansion leaves the endpoints
strictly inside the same pole-free partition, with a margin much larger than
rounding errors. Changes to the margin or library defaults require rechecking
this condition. Write $[a,b]$ for the resulting bracket. At its midpoint
$m=(a+b)/2$, the routine proceeds as follows:

| Interval evaluation at $m$ | Next bracket |
|---|---|
| Strictly positive | $[m,b]$: retain the negative right endpoint. |
| Strictly negative | $[a,m]$: retain the positive left endpoint. |
| Contains zero | Stop and return the current bracket $[a,b]$. |

Here the points are represented by interval enclosures in the computation;
the returned interval encloses both endpoints of the final bracket.
Each accepted step keeps opposite signs inside the original region without
poles, so the intermediate value theorem keeps a root inside. [Lemma 5](#paper-lemma-5) supplies
uniqueness in the surrounding pole interval. An undecidable midpoint sign
leaves the existing enclosure valid, even though it prevents further narrowing.

The default `tol=1e-20` controls when the loop stops; it is not a guaranteed final
diameter. The routine checks signs but does not check for poles across the
bracket: the partition in §4 supplies that requirement. See
[how the loader refines saved brackets](bessel_basis.md#52-what-loading-and-local-refinement-establish)
for the library details and its endpoint expansion. A root enclosure that
straddles the cutoff is retried with `δ=zero(dtype)` as described next.

<a id="52-check-that-no-interval-has-been-missed"></a>

### 5.2 Complete-interval coverage and cutoff classification

Every complete family-pole interval through the first family pole strictly above R must produce a certified bracket. `bracket_in_interval` may return `:none`, but its caller immediately raises an error; it never silently skips that interval. Lemma 5 supplies uniqueness in each interval. The initial segment before lb and all unsearched degrees are excluded by the lower bounds in §2.

After local refinement, an enclosure z is retained only when `sup(z) < inf(R)`. If it straddles the cutoff or touches it at its upper endpoint, the code retries with `δ=zero(dtype)`. It then retains z when strictly below R, discards it when `inf(z) >= sup(R)`, or raises an error if classification remains undecidable. No evaluation at `R-POLE_MARGIN` is used as a root exclusion.

<a id="53-interpret-the-returned-status-and-checks"></a>

### 5.3 Meaning of success and the exported table

`:complete` means that every required full interval has contributed a certified unique root and every such root has been classified relative to the cutoff. `:excluded_by_lemma5` means the degree was excluded before scanning. Under the analytic pole-spacing, uniqueness, lower-bound and pole-separation premises, successful execution over all required degrees therefore includes every positive root below R. The degree bounds and complete pole intervals together establish coverage below the cutoff.

The known Y zero mode is added separately. The MAT file stores endpoint arrays;
the search statuses and pole checks belong to the producing notebook. Loading
and locally refining an arbitrary root list does not replace the complete
search described here.

## 6. How are the results stored and used?

### 6.1 Pack one row per degree

After the degree loops, `roots_to_padded_matrix` constructs `Y_zeros` and
`Z_zeros` as `Matrix{Interval{Float64}}`:

- Row `l` contains the roots for degree $l$ in increasing order.
- Rows are padded on the right with the interval `[0,0]`.
- The Y row at $l=1$ starts with the zero eigenpair
  $\psi_0=(\cos\theta,-\sin\theta,0)$ from [Lemma 4](#paper-lemma-4), which also excludes
  a Z zero mode. Current reconstruction gives this physical vector directly;
  see [the basis note, §6.3](bessel_basis.md#63-the-origin-and-the-known-zero-mode).

The matrix has `L_last` rows, ending at the last nonempty degree, and `ncol`
columns, the largest row length. `checked_root_table_shape` checks contiguous
nonempty rows, empty higher rows with an allowed status, and maximal row length
at degree 1. These are storage-layout checks. Dimensions and root counts
are determined by the search. With cutoff $100$, the parity-restricted raw
bases in the paper have 1257 even and 1260 odd elements, including the separate
zero mode in the odd sector. These dimensions follow from the root counts and
parity packing; completeness is established by the pole-interval argument above.

### 6.2 Save the endpoints, then recover the basis parameters

`save_matrix` writes `data/generated/bessel_zeros.mat` with four `Float64`
arrays: `Y_zeros_lo`, `Y_zeros_hi`, `Z_zeros_lo`, and `Z_zeros_hi`.
For example, the $j$th positive Z root for degree $l$ is stored as

$$
\alpha^Z_{l,j}\in
[\texttt{Z\_zeros\_lo[l,j]},\ \texttt{Z\_zeros\_hi[l,j]}].
$$

These two entries store the lower and upper endpoints of the same root
enclosure. Padding entries and the conventional Y
zero are stored with both endpoints equal to zero. Because the first Y row
begins with that zero, its $j$th positive root occupies column $j+1$;
the Z indexing in the example has no such shift.

When `load_besselj_zeros` requests `Z_zeros`, `read_matrix` pairs
`Z_zeros_lo` with `Z_zeros_hi` to reconstruct the interval matrix in interval
mode; it does the same for Y. The loader refines the nonzero root intervals,
assuming each expanded bracket contains no pole, then converts the rows into
parameter lists and adds the radial basis coefficients.

For Z, the resulting interval for $\alpha^Z_{l,j}$ supplies the unknown in
the opening formula: evaluate $J_{l+1/2}(\alpha^Z_{l,j}r)/\sqrt r$ and combine
it with $Z_l(\theta)$ to construct the basis function. Float64 mode instead
reads midpoint approximations and applies the floating-point sign check.

For those loading and basis-construction details, see [bessel_basis.md](bessel_basis.md),
§§5–6. The endpoint-array convention is also described in
[data.md](../data.md). Parts 4A and 4B directly load these root tables when
their generation branches run. Steps 5–7 instead read
the derived Step 4B RHS samples and bounds; those consumers do not independently
reconstruct the Bessel roots or certify their completeness.

## Cited lemmas from the paper

The two statements below follow `lem:lap eig` and `lem:zeros` in
the paper. The root index is written as $j$ and the
coefficient indices as $(l,j)$ throughout. The proofs are in the paper's
`eig cons` appendix.

<a id="paper-lemma-4"></a>

### Lemma 4 — Eigenvalues and basis functions on the unit ball

Let $\Omega$ be the unit ball in three dimensions. Consider the eigenvalue
problem associated with the Rayleigh quotient

$$
\frac{\|\nabla u\|_{L^2(\Omega)}^2}{\|u\|_{L^2(\Omega)}^2}
$$

on the space of divergence-free axisymmetric $H^1$ vector fields.
Write $(r,\theta,\phi)$ for spherical coordinates and
$(e_r,e_\theta,e_\phi)$ for their unit vectors. The angular factors are

$$
S_l(\theta)=P_l(\cos\theta),\qquad
Y_l(\theta)=S_l(\theta)e_r,\qquad
X_l(\theta)=S_l'(\theta)e_\theta,\qquad
Z_l(\theta)=S_l'(\theta)e_\phi,
$$

where $P_l$ is the Legendre polynomial, and set $\mu_l=l(l+1)$.
These are physical spherical harmonics. The code pairs its scalar table
$H_l=-S_l'$ with the negative physical poloidal X coefficient, so the
poloidal fields agree with these formulas. Its toroidal field has one whole-mode
minus sign, absorbed by matching coefficient rows. No physical Q-frame
conversion is involved; see [the basis note](bessel_basis.md#92-scalar-angular-signs-and-physical-components).

The first eigenpair is

$$
\lambda_0=0,\qquad \psi_0=(\cos\theta,-\sin\theta,0),
$$

with components in the spherical frame. All remaining eigenvalues are
$\lambda^i_{l,j}=(\alpha^i_{l,j})^2$, with $l\ge1$, $j\ge1$ and
$i\in\{Y,Z\}$, as follows.

**Z family.** Let $\alpha^Z_{l,j}$ be the $j$th positive root of

$$
xJ_{l+1/2}'(x)-\frac12J_{l+1/2}(x)=0.
$$

Its eigenfunction is

$$
\psi^Z_{l,j}(r,\theta)
=\frac{J_{l+1/2}(\alpha^Z_{l,j}r)}{\sqrt r}\,Z_l(\theta).
$$

**Y family.** Put $\tilde j_l(x)=J_{l+1/2}(x)/x^{3/2}$, and let
$\alpha^Y_{l,j}$ be the $j$th positive root of

$$
\bigl(x^2-l(l-1)(l+2)\bigr)x\tilde j_l'(x)
+\bigl(x^2-l(l-1)\bigr)\bigl(x^2-(l-1)(l+2)\bigr)\tilde j_l(x)=0.
$$

For $\alpha=\alpha^Y_{l,j}$, define

$$
A_{l,j}=-\bigl(\alpha^2-(l-1)(l+2)\bigr)\tilde j_l(\alpha),
\qquad B_{l,j}=\frac{\alpha^2-l(l-1)(l+2)}{\alpha^{3/2}}.
$$

The code stores these same coefficients. The identity

$$
\alpha^{3/2}B_{l,j}\tilde j_l(\alpha r)
=B_{l,j}J_{l+1/2}(\alpha r)/r^{3/2}
$$

explains the equivalent evaluation formula in
[the basis note](bessel_basis.md#6-yxz-radial-formulas-and-coefficients).

The eigenfunction is the sum of its two angular components,

$$
\psi^Y_{l,j}(r,\theta)
=u^Y_{l,j}(r)Y_l(\theta)+u^X_{l,j}(r)X_l(\theta),
$$

where

$$
u^Y_{l,j}(r)=\alpha^{3/2}B_{l,j}\tilde j_l(\alpha r)+lA_{l,j}r^{l-1},
$$

$$
u^X_{l,j}(r)
=\frac{\alpha^{3/2}B_{l,j}}{\mu_l}
\bigl(\alpha r\tilde j_l'(\alpha r)+2\tilde j_l(\alpha r)\bigr)
+A_{l,j}r^{l-1}.
$$

<a id="paper-lemma-5"></a>

### Lemma 5 — One root between consecutive poles and first-root bounds

Fix $l\ge1$. Let $\beta_{l,1}<\beta_{l,2}<\cdots$ be the positive zeros of
$J_{l+1/2}$, and put $\beta_{l,0}=0$ and

$$
\gamma=\sqrt{l(l-1)(l+2)}.
$$

For $l=1$, set $\tilde\beta_{1,j}=\beta_{1,j}$. For $l\ge2$, assume
$\gamma\ne\beta_{l,j}$ for every $j\ge1$; insert $\gamma$ into the Bessel
zero sequence and sort it to obtain

$$
0=\tilde\beta_{l,0}<\tilde\beta_{l,1}<\tilde\beta_{l,2}<\cdots.
$$

Let $s_l$ be the unique positive root of

$$
Q_l(s)=s^3-4l^2s^2+l(l-1)(6l^2+11l+7)s
-3l(l-1)^2(l+1)^2(l+2),
$$

and put $t_l=\sqrt{l(l+1)}$. Assume that $\sqrt{s_l}$ is not a root of
the Y equation in Lemma 4 and that $t_l$ is not a root of its Z equation.
The threshold-contact argument in [§2.3](#threshold-contacts-and-uniqueness)
discharges these hypotheses analytically and strengthens the first-root
inequalities below to strict inequalities. Then:

1. The Y equation has exactly one root in every interval
   $(\tilde\beta_{l,j},\tilde\beta_{l,j+1})$, for $j\ge0$.
2. The Z equation has exactly one root in every interval
   $(\beta_{l,j},\beta_{l,j+1})$, for $j\ge0$.
3. The first positive roots satisfy

$$
\alpha^Y_{l,1}\ge\sqrt{s_l},\qquad
\alpha^Z_{l,1}\ge t_l.
$$

