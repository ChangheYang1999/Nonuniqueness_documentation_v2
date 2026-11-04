# Rigorous Computer-Assisted Proof of Non-Uniqueness of Leray–Hopf Solutions to the 3D Navier–Stokes Equation

## Abstract

This repository contains the Julia verification code, notebooks and candidate data for
the paper on nonuniqueness of Leray–Hopf solutions to the
unforced incompressible three-dimensional Navier–Stokes equations. The verification
constructs a self-similar profile and an unstable eigenpair, using interval arithmetic
to bound PDE residuals, finite-rank approximation errors and the auxiliary eigenvalue
functional. The paper combines these bounds with its fixed-point and regularity
arguments to obtain infinitely many solutions.

The code uses the stationary elliptic eigenvalue convention: a negative eigenvalue
corresponds to positive growth for the self-similar evolution operator.

Start with this README. For the step-by-step verification pipeline, see **[docs/notebooks.md](docs/notebooks.md)**; the remaining reference docs are listed below.

---

## Citation

```
@misc{hou_nonuniqueness,
  title  = {Nonuniqueness of Leray--Hopf solutions to the unforced incompressible 3D Navier--Stokes Equation},
  author = {Thomas Hou and Yixuan Wang and Changhe Yang},
  eprint = {2509.25116},
  archivePrefix = {arXiv},
  primaryClass  = {math.AP},
  url    = {https://arxiv.org/abs/2509.25116}
}
```

---

## System Requirements

| Requirement | Guidance |
|-------------|---------|
| Julia | 1.12.6 (version recorded in `Manifest.toml`) |
| RAM | Plan for a large-memory machine; the largest stages use grids with hundreds of millions of interval samples |
| OS | Linux / macOS recommended |
| Disk | Allow hundreds of GB for sampled fields, matrices and temporary files |

Memory and storage depend on the stage, grid, coefficient counts and thread count.
Run large stages sequentially unless the machine has enough capacity for concurrent work.

---

## Repository Structure

```
3d-navier-stokes-nonuniqueness/
├── notebooks/        ← seven notebooks: Steps 1–7
├── src/              ← Julia library loaded by the notebooks (not a standalone package)
├── data/
│   ├── external/     ← imported inputs
│   └── generated/    ← notebook-produced outputs / cache
├── docs/             ← documentation (start with docs/notebooks.md)
├── Project.toml      ← Julia environment specification
└── Manifest.toml     ← pinned Julia dependency resolution
```

Per-file details: source files are described in [docs/src.md](docs/src.md); `.mat`
file formats are described in [docs/data.md](docs/data.md).

`Project.toml` and `Manifest.toml` define the Julia environment. The manifest pins
Julia 1.12.6, IntervalArithmetic 1.0.9, Arblib 1.7.0, ArbExtras 1.2.0 and
OpenBLASConsistentFPCSR_jll 0.3.33+0.

---

## Setup

1. Clone the repository and start Julia from the project root.
2. Activate and instantiate the environment:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.precompile()
```

3. Verify that `src/NS_Numerics.jl` defines `const dtype = Interval{Float64}`
   before loading that module in a fresh kernel; this is already the current setting.
   The following is source configuration, not an assignment to run in the notebook:

```julia
# Configuration inside src/NS_Numerics.jl
const dtype = Interval{Float64}
# const dtype = Float64
```

   Using `dtype = Float64` produces fast but **non-rigorous** floating-point estimates.
   Use interval mode for the verification bounds.

> **Notebook setup:** Use an installed Julia kernel with `notebooks/` as its working directory
> (`Pkg.activate("..")` relies on this). Restart the kernel after environment setup.

Use generated files from the same input coefficients, basis ordering, scale and grid
settings. Some notebook cells reuse a cache when its file exists; consult the
[cache dependency table](docs/notebooks.md#cache-dependencies) before changing inputs.

---

## Running the Verification

Run the **seven notebooks, Steps 1–7**, using the dependency order in the
[pipeline guide](docs/notebooks.md):

1. Verify the self-similar profile, the compensating tensor Q and its derivatives.
2. Normalize the approximate eigenfunction and bound its corrected eigenvalue and residual.
3. Enclose the Bessel roots used by the finite spectral basis.
4. Assemble the finite-rank matrices (Part 4A) and the RHS samples and derivative bounds (Part 4B).
5. Bound the even inverse fields and the mixed forcing in the eigenpair problem.
6. Bound the odd inverse fields, their projected Grams and the symmetric part of the finite-rank operator.
7. Bound the auxiliary eigenvalue functional with
   [`verify_auxiliary_field_bound.ipynb`](notebooks/verify_auxiliary_field_bound.ipynb).

Steps 5 and 6 are independent of one another's result files. Step 7 uses the complete
Step 6 output as well as the Step 2 and Step 4B data. It estimates the functional in
the dual weighted energy norm at mass weight `0.23187` and asserts an upper bound
of `14.9512`. The paper uses the relaxed bound $M_5^v\le15$ and transfers it to the
larger mass weight $\eta_v=0.232$.

The [paper–code map](docs/paper_code_map.md) connects these calculations to the
paper's numerical inputs and the profile/eigenpair closure in its
Numerical results section. The [notebook guide](docs/notebooks.md#current-acceptance-and-reporting)
lists the executable checks and the corresponding paper bounds.

---

## Documentation

- **[docs/notebooks.md](docs/notebooks.md)** — notebook pipeline with separate matrix and RHS stages: reads/writes, paper
  definitions, numerical checks and cache dependencies.
- **[docs/data.md](docs/data.md)** — data files: fields, shapes and producer/consumer relationships.
- **[docs/src.md](docs/src.md)** — source-file structure: dependency map, include order, and
  major function locations.
- **[docs/paper_code_map.md](docs/paper_code_map.md)** — paper symbols, equations, and constants
  mapped to code variables, notebook variables, source functions, and data fields.

---

## License / Author

**Author:** Changhe Yang<br>
California Institute of Technology

The code is released for academic use in connection with the accompanying paper. If you use
this code, please cite the paper above.
