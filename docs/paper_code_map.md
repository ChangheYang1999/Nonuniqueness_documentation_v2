# Paper ↔ Code Map

This reference connects the notation in the paper to the Julia
implementation and the seven verification notebooks. Paper labels below are searchable
LaTeX labels, so they remain useful when equation numbering changes.

The paper uses $\overline U$, $\overline v$ and $\overline\lambda$ for the approximate
profile and normalized eigenpair. In the formulas below these are abbreviated to U, v
and $\lambda$ when describing code. The paper's unbarred U and v are the **corrections**
to those approximate fields. Retained generalized eigenpairs are written
$(\nu_j,\psi_j)$ below; they correspond to the paper's
$(\overline\lambda_j,\overline\psi_j)$ and are distinct from the unstable eigenpair.

The [notebook guide](notebooks.md#current-acceptance-and-reporting) lists the numerical
checks and paper bounds. The [data reference](data.md) describes arrays, file layouts
and producer/consumer relationships.

## 1. Core objects and sign conventions

| Object | Meaning | Code / data / location |
|---|---|---|
| Numeric mode | `Interval{Float64}` is the verification mode; `Float64` is exploratory. | `dtype` in [NS_Numerics.jl](../src/NS_Numerics.jl) |
| U, P | Divergence-free reconstructed trial profile and pressure. | Step 1 reads `external/UP.mat`, writes `generated/UP_divfree.mat` before its assertions; subsequent profile-dependent stages use that export. |
| Raw unstable candidate | Input to normalization and eigenvalue correction. | `external/up_eig.mat`; current raw lambda metadata is approximately `-0.11316095052362504`. |
| v, lambda | One distinguished exactly unit candidate and its corrected elliptic eigenvalue; negative lambda corresponds to positive growth in the evolution convention. | Step 2's `generated/up_eig_corrected.mat`, consumed by Steps 5, 6 and 7; export is only written if absent. The upper endpoint is checked against `-0.1131623`, stronger than `eq:45-input-lambda`. |
| $Q_s$ | Physical symmetric gradient $\operatorname{sym}\nabla U$. | `get_gradient_sym`, Step 1. |
| Q | One physical symmetric compact compensator, used consistently in every stage. | Six `A11`…`A33` components of `external/approx_Q.mat`, loaded by `load_grad_U`; no additional DQD transformation. |
| $D=(I+x\cdot\nabla)/2$ | Self-similar first-order operator. | Base terms supplied by `get_linear` / `get_operator_matrix`; notebooks add convection, eigenvalue and constraint terms explicitly. See [operator conventions](technical_notes/ns_terms.md). |
| $\kappa$, R | $r_{\rm phys}=\kappa\tan\beta$ and Q support radius R. | `scl_fac` (currently 22), R=32; these are different scales. |
| $\eta_1$, $\eta_2$, $\eta_v$ | Compensation shift, finite-basis energy mass weight, and functional mass weight. | Step 1 uses `0.19084`; the paper relaxes this to `0.1909`. Part 4A uses `0.005`. Step 7 uses `0.23187`, and the paper eigenpair ball uses `0.232`. |

<a id="raw-inverse-fields-and-residuals"></a>
### Raw inverse fields and residuals

Let $\Phi$ denote the column map of **raw loaded** trial inverse velocities,
and let $P_v=I-v\otimes v$ for the distinguished unit v. The manuscript inverse
fields have signs

$$
\phi_{\rm math}=-\Phi\quad\hbox{(even)},\qquad
\phi_{\rm math}=-P_v\Phi\quad\hbox{(odd)}.
$$

Define the forcing column map $\mathcal B$ by
$\mathcal B_j=Q\psi_j/\nu_j$. This calligraphic map is distinct from the
finite-basis energy matrix B below. Then

$$
G_U=-\Phi\mathcal B^*,\qquad G_v=-P_v\Phi\mathcal B^*.
$$

Thus $I+G$ has the implemented $I-\Phi\mathcal B^*$ sign. To read the code,
keep raw $\Phi$ rather than substituting the manuscript inverse silently.
Write $\mathscr C_U(\Phi)=(U\cdot\nabla)\Phi+(\Phi\cdot\nabla)U$ and
$\mathscr A_U(\Phi,p)=D\Phi+\Delta\Phi+\nabla p-\mathscr C_U(\Phi)$.
The unweighted raw residual columns are

$$
\mathcal E_{U,j}=Q\psi_j-\mathscr A_U(\Phi_j,p_j),
$$
$$
\mathcal E_{v,j}=Q\psi_j-\mathscr A_U(\Phi_j,p_j)
                 -\lambda\Phi_j-\mu_jv.
$$

Here $\mu_j$ is the supplied trial constraint multiplier `lbd_lst[j]`, distinct
from both corrected `lbd` and spectral `eig_val[j]`. Code samples these residuals
with the weight $w=\sin\beta/\cos^2\beta$. For same-sector physical vector fields,

$$
\langle F,H\rangle_{\mathbb R^3}
=4\pi\kappa^3\int_0^{\pi/2}\!\int_0^{\pi/2}
 (wF)\cdot(wH)\sin\theta\,d\theta\,d\beta.
$$

The angular symmetry supplies the factor $4\pi$; there is no radial doubling.
The stored RHS already contains w, so do not apply it again. It does not contain
$1/\nu_j$; that division belongs to $\mathcal B$. See the
[domain and measure proof](technical_notes/symmetry_and_domains.md#domain-rules).

## 2. Paper quantities and notebook locations

The numerical inputs are collected in paper Section `Sec:num prev`. Their definitions
and code names are:

| Paper quantity / definition label | Code quantity | Where it is bounded |
|---|---|---|
| $\varepsilon^U$, $\varepsilon^v$; `asm:num` | Profile residual, normalized `epsilon_v` | Steps 1 and 2 |
| $Q\succeq0$, $Q_s+\eta_1I+Q\succeq0$; `eq:Q cond` | `min_eig_H`, `min_eig_parts` | Step 1 |
| $\lVert \lambda_{\max}(Q)\rVert_\infty$ | `max_eig_Q` | Step 1 |
| $K_5$; `eq:K5-definition` | Q norm times spectral-tail ratio | Step 3 complete roots and the tail formula below |
| $K_6$; `constant small` | `verify_general_eig` threshold | Step 4A |
| $\varepsilon_3$, $\varepsilon_4$; `def-eps3`, `lem:quasi ort1` | `eigen_diagnostics` | Step 4A |
| $\Theta_r^U$, $\Theta_r^v$; `lem:fe est` | `K7_U`, `K7_v` | Steps 5 and 6 |
| $\Theta_\phi^i=\lVert G^i\rVert$; `eq:phi gram U` | `A0_U`, `A0_v` | Steps 5 and 6 |
| $\Theta_{\nabla\phi}^i=\lVert \nabla G^i\rVert$; `eq:phi gram U` | `A1_U`, `A1_v` | Steps 5 and 6 |
| $\Theta_Q^U=\lVert \mathcal B\rVert^2$; `eq:qusi-or1` | `C_U` | Step 5, forcing-Gram spectral bound |
| $\Theta_F^U$; `eq:odd-forcing-Gram` | `Theta_F_U` | Step 5, mixed-forcing Hilbert–Schmidt bound |
| $\Theta_{\mathrm s}^v=1+\lVert \operatorname{sym}G^v\rVert$ | `Bquad_v` | Step 6, two signed Gram tests |
| $M_5^v$; `eq:dual-riesz-upper-bound` | `M5_v_upper` | Step 7, auxiliary-field estimate |
| $M_1^i,M_2^i$; `eq:finite-rank-constants` | Scalar combinations of spectral and residual bounds | Paper Section `Sec:num prev` |
| $M_3^U,M_4^U$; `eq:U-M3-definition`, `eq:U-M4-definition` | Scalar profile-closure quantities | Paper Section `Sec:num prev` |

The paper's numerical choices and their tighter notebook thresholds are listed together
in the [check table](notebooks.md#current-acceptance-and-reporting).

## 3. Finite-rank objects and Gram matrices

<a id="finite-rank-matrices-and-certificates"></a>
### Finite-rank matrices and certificates

Part 4A uses raw physical natural-traction Stokes modes $b_p$, not the retained
columns $\psi_j$ and not assumed unit-normalized modes:

$$
A_{pq}=\langle b_p,Qb_q\rangle,\qquad
B_{pq}=\eta_2\langle b_p,b_q\rangle+\langle\nabla b_p,\nabla b_q\rangle,
$$
$$
B_{pp}=\left[\eta_2+(\alpha_p/R)^2\right]\lVert b_p\rVert_2^2,
\qquad \psi_j=\sum_pV_{pj}b_p,\qquad \Lambda=\operatorname{diag}(\nu_j).
$$

The physical $4\pi R^3$ scaling is applied once. The external `eig.mat` supplies
V and $\Lambda$; Part 4A checks their normalization and residual, and Part 4B
reconstructs the same retained columns. The manuscript spectral error and its
computed Frobenius upper bound are

$$
\varepsilon_3=\lVert I-\Lambda^{-1/2}V^*AV\Lambda^{-1/2}\rVert_2
\le \widehat\varepsilon_3:=\lVert I-\Lambda^{-1/2}V^*AV\Lambda^{-1/2}\rVert_F.
$$

The energy conditioning and weak-residual bound are

$$
\gamma=\lVert I-V^*BV\rVert_F<1,
$$
$$
\varepsilon_4=
\frac{\lVert B^{-1/2}(AV\Lambda^{-1}-BV)\rVert_F}{\sqrt{1-\gamma}}.
$$

K6 bounds the Q Rayleigh quotient on the **Q-orthogonal complement** of the
retained fields, measured in the energy inner product. The current sufficient
certificate checks positive definiteness of

$$
\tau B-A+(AV)\Lambda^{-1}(AV)^*.
$$

On $V^*Ax=0$ the deflation term vanishes, proving the constrained bound
$K_6\le\tau$. The full-space deflated supremum is a sufficient upper bound,
not identically the constrained supremum. The notebook tests 0.449 in the even
sector and 0.445 in the odd sector.
For K5, the complete root cutoff $\alpha_{\rm cut}=100$, $R=32$ and
$\eta_2=0.005$ give

$$
T_{\rm tail}\le[\eta_2+(\alpha_{\rm cut}/R)^2]^{-1}\le0.1024,
\qquad K_5\le4.47\times0.1024.
$$

The factor 4.47 is the Q norm bound; 0.1024 alone is the spectral-tail ratio.
See [finite-rank assembly](technical_notes/finite_rank_assembly.md).

### Operator, symmetric-part and trace bounds

Set $H=\Phi^*\Phi$, $D_\phi=(\langle\nabla\Phi_i,\nabla\Phi_j\rangle)$,
and $GR=\mathcal E^*\mathcal E$ for the complete raw residual, including its
exterior part. The `GQ` below is the forcing Gram, distinct from both Q and A.

| Quantity | Exact definition or sufficient bound | Implementation |
|---|---|---|
| GQ | $\mathcal B^*\mathcal B$, entries $\langle Q\psi_i,Q\psi_j\rangle/(\nu_i\nu_j)$ | Both eigenvalue denominators are included. |
| C_U | $\lambda_{\max}(GQ)=\lVert\mathcal B\rVert^2$ | Step 5 spectral candidate; distinct from $\lVert GQ\rVert_F$. |
| A0 | Upper bound for $\lVert G\rVert_{L^2\to L^2}$ | Uses the inverse mass Gram; distinct from the HS quantity $\sqrt{\operatorname{tr}(GQH)}$. |
| A1 | Upper bound for $\lVert\nabla G\rVert_{L^2\to L^2}$ | Uses the inverse gradient Gram; distinct from its HS trace. |
| B0 | Upper bound for $\lVert I+G\rVert_{L^2\to L^2}$ | Even: triangle bound $1+A0_U$. Odd: cross-Gram spectral certificate. |
| Bquad_v | $\Theta_{\mathrm s}^v=1+\lVert\operatorname{sym}G_v\rVert$ | Also bounds $\lvert\langle(I+G_v)f,f\rangle\rvert$ for unit f. |
| rho_U | Absolute residual quadratic-form bound | Current stronger HS bound $\sqrt{\operatorname{tr}(GQGR)}$, called `K7_U`. |
| rho_v | Absolute residual quadratic-form bound | Two signed raw block checks plus one projection correction, stored as `K7_v`. |
| Mixed forcing | $\mathcal Ff=(G_Uf\cdot\nabla)v+(v\cdot\nabla)G_Uf$ | `Theta_F_U` uses the sufficient HS bound $\sqrt{\operatorname{tr}(GQGF)}$. |
| M5($\eta_v$) | Bound for the dual norm of $(I+G_v)^*h_0$ on divergence-free odd $v^\perp$, with energy $\eta_v\lVert u\rVert_2^2+\lVert\nabla u\rVert_2^2$ | Step 7; mass weight 0.23187, fixed source centers plus a separate allowance, upper endpoint asserted against 14.9512. |

Here `GF` is the Gram of the raw mixed-forcing columns
$(\Phi_j\cdot\nabla)v+(v\cdot\nabla)\Phi_j$. An HS bound also bounds the operator
norm used in the paper. The ambient-sector matrix tests apply on the required
axisymmetric divergence-free spaces and their constrained subspaces.

<a id="late-odd-projection"></a>
### Late odd projection

Let $b=\Phi^*v$, $d_i=\langle\nabla v,\nabla\Phi_i\rangle$,
$d_0=\lVert\nabla v\rVert_2^2$, $X=\mathcal B^*\Phi$ and
$t=\mathcal B^*v$. Then

$$
H_\perp=H-bb^*,\qquad
D_\perp=D_\phi-bd^*-db^*+d_0bb^*,\qquad
X_\perp=X-tb^*.
$$

These identities use the same exactly unit v as the Step 2 normalization.
Step 6 applies them to the raw-field Grams after integration.

For the raw residual column map $\mathcal E$, define

$$
W_r=\begin{pmatrix}GQ&\mathcal B^*\mathcal E\\
\mathcal E^*\mathcal B&GR\end{pmatrix},\qquad
K_r=\begin{pmatrix}0&-I/2\\-I/2&0\end{pmatrix},
\qquad \delta_{\rm proj}\le\epsilon_v\sqrt{b^*GQb}.
$$

Certify both signs at the remaining budget $\rho_v-\delta_{\rm proj}$. Current
code uses `sqrt(abs(b' * GQ * b))` to enclose the nonnegative Gram square root
when interval overestimation crosses zero. This is one correction from the same
corrected v, pressure, lambda and background as Step 2, with
$\epsilon_v=4.808\times10^{-6}$.

For the projected inverse block set

$$
W_\phi=\begin{pmatrix}GQ&X_\perp\\X_\perp^*&H_\perp\end{pmatrix},\qquad
K_{\rm norm}=\begin{pmatrix}H_\perp&-I\\-I&0\end{pmatrix},\qquad
K_{\rm quad}=\begin{pmatrix}0&-I/2\\-I/2&0\end{pmatrix}.
$$

The operator-norm certificate tests $K_{\rm norm}$ at $B0^2-1$. For the paper
constant $\Theta_{\mathrm s}^v=1+\lVert\operatorname{sym}G^v\rVert$, Step 6 tests
**both** $K_{\rm quad}$ and $-K_{\rm quad}$ at `32.52`. It therefore obtains
$\lVert\operatorname{sym}G^v\rVert\le32.52$ and stores the bound
$\Theta_{\mathrm s}^v\le33.52$ in `Bquad_v`. In particular,

$$
|\langle(I+G^v)f,f\rangle|\le33.52\lVert f\rVert_2^2.
$$

`verify_gram_spectrum(W,K,t)` uses the structural identity $W=T^*T\succeq0$.
It forms upper matrices $U\succeq W$ and $V\succeq K$, chooses positive weights
with a floating-point heuristic, and uses interval LDL to verify
$U\succ0$ and $tU-UVU\succ0$. For positive t this bounds the spectrum of
$TKT^*$, including the ambient zero complement. The exact W may be singular;
W itself need not be positive definite.
The [matrix note](technical_notes/matrix_func.md#symmetric-and-gram-spectral-certificates)
derives the upper-matrix inequality.

<a id="direct-mass-weighted-m5"></a>
### Direct mass-weighted M5

Step 7, [`verify_auxiliary_field_bound.ipynb`](../notebooks/verify_auxiliary_field_bound.ipynb),
uses the auxiliary trial w and matched Step 4/6 data. It bounds the same functional
on divergence-free odd tests in
$v^\perp$, for the distinguished exactly unit corrected eigenfunction v. Write

$$
H(v)=Dv-\Delta v-(U\cdot\nabla)v+(\nabla U)^Tv=h_0+\tfrac12v,
\qquad c_j=\langle P_vH(v),\Phi_j\rangle,
\qquad \mathcal B_j=Q\psi_j/\nu_j.
$$

Let $\mu_j$ be the `lambda` stored in each raw odd inverse file. With the
[raw residual convention](#raw-inverse-fields-and-residuals),

$$
Lv=\lambda v+e,\qquad
L\Phi_j=\lambda\Phi_j+\mu_jv-Q\psi_j+E_j.
$$

Pairing with v and removing its component gives

$$
c_j=\mu_j-\nu_jt_j+\langle v,E_j\rangle-b_j\langle v,e\rangle,
\quad t_j=\langle\mathcal B_j,v\rangle,\quad b_j=\langle v,\Phi_j\rangle.
$$

The code reads `GQ,GR,t,b` from the complete Step 6 output and reads only
$\mu_j$ from the inverse files. It selects exact binary centers
$c_j^0=\operatorname{mid}(\mu_j-\nu_jt_j)$. The source estimate requires
coefficient-error bounds $a_j\ge|\mu_j-\nu_jt_j-c_j^0|$.
The notebook forms `cs_radius=dt.(radius.(cs_base))` for a. These bounds must
cover both endpoint distances from the selected binary centers; outward-rounded
interval subtraction provides a direct check of that enclosure. With
$GQ=\mathcal B^*\mathcal B$ and $GR=\mathcal E^*\mathcal E$ including both
regions, the source error obeys

$$
\|\mathcal B(c-c^0)\|_2\le\delta_{L^2}
=\sqrt{\operatorname{tr}(GQ\,GR)}
 +\sqrt{a^T|GQ|a}
 +\epsilon_v\sqrt{b^TGQb},\qquad
\delta M_5=\delta_{L^2}/\sqrt{\eta_v}.
$$

The first term bounds $\mathcal B\mathcal E^*v$ by its Hilbert–Schmidt norm,
using $\|v\|_2=1$; the last uses the matching Step 2 bound
`epsilon_v=4.808e-6`. The absolute value in the middle term is entrywise.
Code also applies `abs` to the enclosed scalar norm squares before `sqrt`,
which preserves upper bounds for their known nonnegative exact values. This
separate source allowance is essential: the centers alone are not the exact c.
No direct source-pairing integrals or inverse-field samples are recomputed here.

For an arbitrary trial $\overline w$ with pressure p and fixed multiplier
$\lambda_w$, retain the full interval projection coefficient and set

$$
\gamma=\langle\overline w,v\rangle,\qquad w_\perp=\overline w-\gamma v,
\qquad Y_0=H(v)-\mathcal Bc^0-\lambda_wv,
\qquad Z=Y_0+\Delta w_\perp-\nabla p.
$$

For divergence-free odd test fields $f\perp v$, integration by parts gives

$$
\langle Y_0,f\rangle
=\langle\nabla w_\perp,\nabla f\rangle+\langle Z,f\rangle.
$$

Weighted Cauchy–Schwarz, with
$\|f\|_{H^1,\eta}^2=\eta\|f\|_2^2+\|\nabla f\|_2^2$, gives

$$
|\langle Y_0,f\rangle|
\le\left(\|\nabla w_\perp\|_2^2+\frac{\|Z\|_2^2}{\eta}\right)^{1/2}
\|f\|_{H^1,\eta}.
$$

This is the estimate in paper label `eq:dual-riesz-upper-bound`. The source H(v)
differs from the paper's $h_0$ by $v/2$, which vanishes against the constrained
tests. Pressure and the trial multiplier also vanish in these pairings. Native
NC8 quadrature computes `w1` and `z2` over the inner and exterior regions with
derivative remainders. The final bound is

$$
M_5^v\le\sqrt{w_1+z_2/\eta}+\delta M_5,
\qquad\eta=0.23187.
$$

The notebook prints `M5_v_upper=sup(M5_center+delta_M5)`, where
`M5_center=sqrt(w1+z2/eta_v)`, and asserts an upper bound of `14.9512`.
It writes no result file. The paper uses the relaxed bound $M_5^v\le15$
in `eq:45-input-M5v`. This estimate also applies at the paper's eigenpair-ball
weight $\eta_v=0.232$: increasing the mass weight increases the primal energy
norm and decreases its dual norm.

## 4. Bessel / basis / angular-frame conventions

| Object | Physical convention / implementation |
|---|---|
| $\alpha$, $J_{l+1/2}$ | Boundary-condition roots of `func_Y` / `func_Z`; `my_besselj` supplies the half-integer Bessel functions. See [the Bessel note](technical_notes/bessel_basis.md). |
| Y / X / Z | $Y_l=S_le_r$, $X_l=S_l'e_\theta$, $Z_l=S_l'e_\varphi$; Y-root modes couple the poloidal Y/X channels and Z-root modes are toroidal. |
| Angular X table | The evaluation table stores $-S_l'$; `Ur_func` compensates by negating the complete poloidal X radial coefficient. The resulting field uses physical $X_l=S_l'e_\theta$. |
| Degree-one zero mode | $(\cos\theta,-\sin\theta,0)=e_z$. |
| Q frame | Every active stage uses the same physical Q. Do not add a DQD storage transformation or another X sign change. |
| Geometric reflections | D0 and D1 in [the symmetry note](technical_notes/symmetry_and_domains.md#angular-symmetries) relate different physical points; they are not changes of storage frame. |

## 5. Constants and thresholds

The Sobolev product constant in manuscript label `eq:sobolev-product` is

$$
K_4=\frac{2}{3^{3/4}\pi},\qquad
\lVert fg\rVert_2\le K_4\lVert\nabla f\rVert_2
\lVert g\rVert_2^{1/2}\lVert\nabla g\rVert_2^{1/2}.
$$

In particular,
$\lVert f\rVert_4^2\le K_4\lVert f\rVert_2^{1/2}\lVert\nabla f\rVert_2^{3/2}$.
The finite-rank errors enter the scalar closure through

$$
M_1^i=K_5^i+\frac{K_6^i}{(1-\varepsilon_4^i)^2},
\qquad
M_2^i=\eta_2M_1^i+\varepsilon_3^i\|\lambda_{\max}(Q)\|_\infty+\Theta_r^i,
\qquad i=U,v,
$$

as in `eq:finite-rank-constants`. The paper then constructs the profile ball and
its refined joint norm estimate (`eq:U-joint-bound`), uses those bounds in the
eigenpair forcing and scalar feedback, and verifies the eigenpair ball through
`eq:45-v-radius-budget`. The resulting eigenvalue correction is less than
0.00243, preserving the negative sign of the exact elliptic eigenvalue.

## 6. Data-file fields

The [data reference](data.md) gives the file layouts. Principal links are:

| File / field | Mathematical object |
|---|---|
| `UP.mat`, `UP_divfree.mat`: `u1,u2,u3,p` | Raw / divergence-free reconstructed U and P |
| `up_eig.mat`, `up_eig_corrected.mat`: velocity/pressure, `lambda` | Raw / distinguished normalized corrected unstable eigenpair |
| External `approx_Q.mat`: `A11`…`A33` | Physical Q coefficients; distinct from the generated sample cache |
| `eig.mat`: `eig_val_*`, `eig_vec_*` | $\nu_j$, V for retained $\psi_j$ |
| `bessel_zeros.mat`: `Y_zeros_lo/hi`, `Z_zeros_lo/hi` | Complete root enclosures, zero-padded by degree |
| `Q_bound_radial.mat`: `Q_WkInf_lo/hi` | Native normalized-radius Q derivative bounds, converted to beta by Step 4B |
| `rhs.mat`: `rhs_even`, `rhs_odd` | Weighted $wQ\psi_j$, without eigenvalue division |
| `psi_eig_mat_AB.mat`: `A_*`, `B_*` | Raw-basis physical matrices A and B |
| `step5_even_matrices.mat`, `step6_odd_matrices_clean.mat` | Raw/projected Gram outputs and selected scalars; see exact schemas in the data reference |
| `up_w.mat` | Trial velocity, pressure and multiplier for the auxiliary-field estimate |

## 7. Notebook truncation / grid variables

| Code variable | Meaning | Caution |
|---|---|---|
| `M0`, `M1` and related counts | Input-derived truncation sizes | Not the manuscript constants $M_1,\ldots,M_4$; current code generally derives frequency counts from `u3` and assumes compatible other components. |
| `N0`, `N1` | Grid interval counts in beta/radial and theta directions | Keep physical intervals, padded samples and prior domains distinct. |
| `BDRY_LST_SET` | Boundary-condition code sets `00`, `01`, `10`, `11` | Componentwise parity is determined by the selected field set and special angular blocks. |
