# ============================================================
# My_Fourier.jl
#
# Provides the numeric-type helper `dt`, the Boolean zero-test `is_zero`,
# the constant `Pi`, and the trigonometric/basis-function utilities used
# throughout the verification pipeline. Key routines include the
# Chebyshev–Legendre conversion matrices (`che2len`, `len2che`), the
# Discrete Fourier Transform matrix (`Fourier_mat`), the divergence-free
# reconstruction helper (`calc_Utheta_divfree`), and the master basis-function
# evaluator `trig_func`, which dispatches on a string key to produce sampled
# radial or angular factors of the operators used in NS_Terms.jl.
#
# Dependencies: dtype and the LinearAlgebra, SpecialFunctions, and
# IntervalArithmetic imports supplied by NS_Numerics.jl.
# See: docs/technical_notes/fourier_basis.md for formulas and conventions.
# ============================================================

# Active numeric conversion helper.
if dtype == Float64
    """
    Convert `x` to the active exploratory scalar type (`Float64`).

    Decimal strings are parsed explicitly; numeric inputs are converted by `Float64(x)`.
    This branch is not rigorous. See `docs/technical_notes/fourier_basis.md` for the numeric-type convention.
    """
    @inline dt(x) = x isa String ? parse(Float64, x) : Float64(x)
elseif dtype == Interval{Float64}
    """
    Convert `x` to the active rigorous scalar type (`Interval{Float64}`).

    The result is an interval enclosure produced by `@interval`. See `docs/technical_notes/fourier_basis.md`.
    """
    @inline dt(x) = @interval(x)
else
    error("Unsupported dtype: $dtype")
end

# Zero test used to skip negligible coefficients.
"""
Return whether `x` is treated as zero under the active coefficient convention.

`Float64` uses a small absolute threshold; `Interval{Float64}` requires the exact zero interval.
"""
@inline is_zero(x::Float64)::Bool = abs(x) < 1e-14
@inline is_zero(x::Interval{Float64})::Bool = isequal_interval(x, @interval(0))

# Active-`dtype` value of π.
const Pi = dt(π)
"""
Evaluate `sin(x)` through `sinpi(x / Pi)` in the active numeric type.
"""
@inline mysin(x) = sinpi(x / Pi)
"""
Evaluate `cos(x)` through `cospi(x / Pi)` in the active numeric type.
"""
@inline mycos(x) = cospi(x / Pi)

"""
Return the partial products used in Chebyshev-Legendre conversion coefficients.

`n` must be a non-negative even integer. See Section 3 of `docs/technical_notes/fourier_basis.md`.
"""
function double_fact(n::Int)
    @assert n ≥ 0 && iseven(n) "n must be a non-negative even integer"
    result = [one(dtype)]
    b = one(dtype)
    for i in 2:2:(n-2)
        val = one(dtype) - inv(dt(i))
        b *= val
        push!(result, b)
    end
    return result
end

"""
Return the recurrence-generated ratios `Gamma(k - 3/2) / Gamma(k)` used by `len2che`.

`n` is an integer or half-integer cutoff. See Section 3 of `docs/technical_notes/fourier_basis.md`.
"""
function gamma_ratio(n::Real)
    frac = n - floor(n)
    @assert n ≥ 1 && frac in (0.0,0.5) "n must be a non-negative integer or half-integer"
    if n == 1
        return [gamma(dt(n - 1.5))/gamma(dt(n))]
    elseif n == 1.5
        return [Inf]
    end
    result = dtype[]
    if frac == 0.0
        push!(result, gamma(dt(-0.5))/gamma(dt(1.0)))
        b = gamma(dt(0.5))/gamma(dt(2.0))
    else
        push!(result, Inf)
        b = gamma(dt(1.0))/gamma(dt(2.5))
    end
    push!(result, b)
    for i in (2+frac):1:(n-1)
        b *= dt(i - 1.5)/dt(i)
        push!(result, b)
    end
    return result
end

"""
Build the lower-triangular matrix expanding Legendre modes in Chebyshev modes.

Returns an `(n+1) × (n+1)` matrix. For `0 ≤ i,j ≤ n`, `A[i+1,j+1]`
is the coefficient of `T_{2j}` in `P_{2i}` when `flag=false`, or of
`T_{2j+1}` in `P_{2i+1}` when `flag=true`. Rows index the expanded Legendre
mode; columns index Chebyshev modes. See Section 3 of
`docs/technical_notes/fourier_basis.md`.
"""
function che2len(n::Int; flag::Bool=false)
    A = zeros(dtype, n+1, n+1)
    b_lst = double_fact(4*(n+1))
    for i in 0:n, j in 0:i
        if flag
            A[i+1,j+1] = dt(2)*b_lst[i+j+2]*b_lst[i-j+1]
        else
            A[i+1,j+1] = dt(2)*b_lst[i+j+1]*b_lst[i-j+1]
            if j == 0
                A[i+1,j+1] /= dt(2)
            end
        end
    end
    return A
end

"""
Build the lower-triangular matrix expanding Chebyshev modes in Legendre modes.

Returns an `(n+1) × (n+1)` matrix. For `0 ≤ i,j ≤ n`, `A[i+1,j+1]`
is the coefficient of `P_{2j}` in `T_{2i}` when `flag=false`, or of
`P_{2j+1}` in `T_{2i+1}` when `flag=true`. Rows index the expanded Chebyshev
mode; columns index Legendre modes. See Section 3 of
`docs/technical_notes/fourier_basis.md`.
"""
function len2che(n::Int; flag::Bool=false)
    A = zeros(dtype, n+1, n+1)
    offset = flag ? 2.5 : 1.5
    ratio_lst1 = gamma_ratio(2 * n + offset)
    ratio_lst2 = gamma_ratio(n + 1)
    for i in 0:n, j in 0:i
        if flag
            factor = dt((4j + 3)*(i + 0.5))
            arg1 = i + j + 1
        else
            factor = dt((4j + 1)*i)
            arg1 = i + j
        end
        arg2 = i - j
        A[i+1,j+1] = - (factor * ratio_lst1[Int(arg1)+1] * ratio_lst2[Int(arg2)+1]) / dt(4)
    end
    if !flag
        A[1,1] = one(dtype) # The closed formula has a removable singularity at T_0 = P_0.
    end
    return A
end

"""
Return the angular basis-conversion matrix for a boundary-condition code.

`bdry[1]` is the condition at `0`, `bdry[2]` is the condition at `π/2`; `0`
means Dirichlet and `1` means Neumann. `direction=0` maps trigonometric
coefficients to spherical-harmonic coefficients, and `direction=1` maps back.
`N` counts input coefficients; coefficient rows multiply the matrix on the right.
For `"00"`, `"01"`, and `"10"` the matrix is `N × N`. For `"11"` it is
`N × (N-1)` in direction 0 and `N × (N+1)` in direction 1: the spherical
block excludes `P_0`, although its trigonometric expansion includes a constant.
With `S_l(θ) = P_l(cos θ)`, the stored angular factors are `S_2,S_4,...`
for `"11"`, `S_1,S_3,...` for `"10"`, `-S_2',-S_4',...` for `"00"`, and
`-S_1',-S_3',...` for `"01"`; primes denote θ derivatives. The sine cases
use an antiderivative before conversion. For `"11"`, converting from trigonometric
to spherical coefficients and back preserves the function only if its `S_0`
coefficient is zero.
See Sections 2-4 of `docs/technical_notes/fourier_basis.md`.
"""
function conversion_mat(N::Int, bdry::String, direction::Int)
    if bdry == "11" && direction == 0
        # This constrained block excludes the constant spherical harmonic mode.
        A = len2che(N-1)
        return A[:,2:end]
    elseif bdry == "11" && direction == 1
        # Keep the inverse map on the same constrained subspace.
        A = che2len(N)
        return A[2:end,:]
    elseif bdry == "00" && direction == 0
        # Sine modes represent derivatives of the Legendre block, so integrate before conversion.
        n = dt.(1:N)
        A = transpose(len2che(N))
        B = Diagonal(dt(0.5) ./ n) # Stores coefficients in the scalar angular basis -S'_l.
        A = A[2:end,2:end]
        return transpose(A * B)
    elseif bdry == "00" && direction == 1
        n = dt.(1:N)
        A = transpose(che2len(N))
        B = Diagonal(dt(2) .* n)
        A = A[2:end,2:end]
        return transpose(B * A)
    elseif bdry == "10" && direction == 0
        return len2che(N-1, flag=true)
    elseif bdry == "10" && direction == 1
        return che2len(N-1, flag=true)
    elseif bdry == "01" && direction == 0
        # The shifted sine antiderivative has no constant cosine mode to remove.
        n = dt.(1:N)
        A = transpose(len2che(N-1, flag=true))
        B = Diagonal(inv.(2 .* n .- 1))
        return transpose(A * B)
    elseif bdry == "01" && direction == 1
        n = dt.(1:N)
        A = transpose(che2len(N-1, flag=true))
        B = Diagonal(dt(2) .* n .- dt(1))
        return transpose(B * A)
    else
        error("Wrong flag: \$bdry and direction: \$direction")
    end
end

"""
Return the boundary-adapted discrete Fourier transform matrix.

`N` counts grid intervals: samples are at `x_j = j*π/(2N)`, `j = 0:N`.
For a sample column `v`, `A*v` gives modal coefficients. There are `N+1`
columns and `N+1`, `N-1`, or `N` rows for `"11"`, `"00"`, or either mixed
code, respectively. Dirichlet endpoint columns are zero. In `"11"`, both
endpoint columns and both the constant and highest-frequency rows are
half-weighted. See Section 6 of `docs/technical_notes/fourier_basis.md`.
"""
function Fourier_mat(N::Int, bdry::String)
    if bdry == "11"
        # Half weights for endpoint samples and for the constant and highest modes.
        n = reshape(dt.(0:N), :, 1)
        A = (dt(2)/dt(N)) .* mycos.(Pi .* (n * transpose(n)) ./ dt(N))
        A[[1,N+1],:] ./= dt(2)
        A[:,[1,N+1]] ./= dt(2)
    elseif bdry == "00"
        # Sine modes vanish at both endpoints, so boundary sample columns are zero.
        n = reshape(dt.(1:N-1), :, 1)
        A = (dt(2)/dt(N)) .* mysin.(Pi .* (n * transpose(n)) ./ dt(N))
        zeros_col = zeros(dtype, N-1, 1)
        A = hcat(zeros_col, A, zeros_col)
    elseif bdry == "01"
        # Mixed interpolation has one zero endpoint and one half-weighted endpoint sample.
        n = reshape(dt.(1:N), :, 1)
        A = (dt(2)/dt(N)) .* mysin.(Pi .* (((n .- dt(0.5)) * transpose(n)) ./ dt(N)))
        A[:,end] ./= dt(2)
        zeros_col = zeros(dtype, N, 1)
        A = hcat(zeros_col, A)
    elseif bdry == "10"
        # Mixed interpolation has one half-weighted endpoint sample and one zero endpoint.
        n = reshape(dt.(0:N-1), :, 1)
        A = (dt(2)/dt(N)) .* mycos.(Pi .* (((n .+ dt(0.5)) * transpose(n)) ./ dt(N)))
        A[:,1] ./= dt(2)
        zeros_col = zeros(dtype, N, 1)
        A = hcat(A, zeros_col)
    else
        error("Wrong flag")
    end
    return A
end

"""
Return the coefficient matrix used to reconstruct the angular component from the radial one.

Returns an `(M+1) × M` matrix: columns are input β modes, rows are output
β modes. Implements the modewise divergence-free relation in the paper's
Section `sec:div-free`. The angular table uses `-S'_l`; the corresponding
coefficient sign reconstructs physical spherical components. See Section 4 of
`docs/technical_notes/fourier_basis.md`.
"""
function div_mat(M::Int, bdry::String)
    @assert bdry in ("00","01","10") "Invalid boundary condition"
    I_mat = vcat(diagm(0 => ones(dtype, M)), zeros(dtype, 1, M))
    diag = (dt.(1:M) .+ dt(0.5)) ./ dt(2)
    if bdry != "00" 
        diag .-= dt(0.25)
    end
    L = diagm(-1 => diag) - diagm(1 => diag)
    L = L[:,1:end-1]
    return dt(1.5) .* I_mat .+ L
end

"""
Compute the angular coefficients `Utheta` from radial coefficients `U` by the divergence-free map.

Rows of `U` index radial β trigonometric modes; columns index angular `S_l`
modes, where `S_l(θ) = P_l(cos θ)`. The returned `Utheta` uses angular factors
`-S_l'`, with primes denoting θ derivatives.

Returns `(DL, DR, Utheta)` or `(DL, DL2, DR, Utheta)` when the first column uses
the special `l=1` boundary block. For `U` of size `M0 × M1`, `DL` and `DL2`
are `(M0+1) × M0`, `DR` is `M1 × M1`, and `Utheta` is `(M0+1) × M1`.
One boundary entry selects degrees `2,4,...`; two select `1,3,...`, with
`bdry[1]` the default radial code and `bdry[2]` the first-column code.
`U` is not mutated. See Sections 4-5 of `docs/technical_notes/fourier_basis.md`.
"""
function calc_Utheta_divfree(U::AbstractMatrix{dtype}, bdry::Vector{String})
    M0, M1 = size(U)
    DL = div_mat(M0, bdry[1]) # Coefficient map for the default block.
    lbd = dt.(2 .* (1:M1)) # Spherical harmonic degrees for the even block.
    if length(bdry) > 1
        @assert length(bdry) == 2 "At most two boundary conditions allowed"
        DL2 = div_mat(M0, bdry[2]) # Special l=1 column uses its own boundary block.
        lbd .-= one(dtype) # Switch from even degrees 2l to odd degrees 2l-1.
    end

    lbd .*= (lbd .+ one(dtype)) # Eigenvalues l(l+1).
    DR = -Diagonal(inv.(lbd)) # Spherical-harmonic factor -1/(l(l+1)).

    # The coefficient sign pairs with the scalar angular table -S'_l.
    Uθ = DL * U 
    if length(bdry) > 1
        Uθ[:,1] = DL2 * U[:,1]  # First column uses the l=1 block.
    end
    Uθ = Uθ * DR
    
    if length(bdry) == 1
        return DL, DR, Uθ
    else
        return DL, DL2, DR, Uθ
    end
end


"""
Evaluate a one-dimensional basis or operator expression in β or θ.

`N` counts modes, not grid intervals. The returned matrix has shape
`(length(x), N)`: rows are sample points, columns are modes. `key` selects
an expression evaluated from the basis formulas, not from sampled differences.
Keys `"I"`, `"D"`, and `"D2"` return the basis function and its first two
derivatives with respect to `x`, which is β or θ.
Callers must choose compatible boundary families; `"Tan"` requires `"00"` or
`"10"`. `"Sec"` requires `"10"` and evaluates `cos((2m-1)x)/cos(x)` by
direct division at ordinary nodes and exact limits `1,-3,5,...` at the known
grid endpoints π/2 and 3π/2. Endpoint masks follow the grid condition below.
See Sections 7-8 of `docs/technical_notes/fourier_basis.md`.
"""
function trig_func(N::Int, x::Vector{dtype}, key::String, bdry::String)
    @assert size(x,2)==1
    # In the production grids, no non-endpoint node lies within 1e-14 of a
    # singular endpoint. Nodes in this window are intended endpoints, where the
    # code substitutes the exact analytic limit; the resulting interval values
    # therefore remain rigorous.
    # See docs/technical_notes/fourier_basis.md, Section 8, for the grid interpretation.
    zero_lines = (abs.(x .- 0.0) .< 1e-14) .| (abs.(x .- 2*Pi) .< 1e-14)
    pi_lines = abs.(x .- Pi) .< 1e-14
    nrow = dt.(2 .* (1:N))'
    if bdry == "11"
        nrow .-= dt(2)
    elseif bdry in ("01","10")
        nrow .-= dt(1)
    elseif bdry == "00"
    else
        error("Wrong boundary condition: $bdry")
    end

    Lap_r_lst = ("Laplacian_θ_r","Laplacian_ϕ_r")
    Lap_θ_lst = ("Laplacian_θ_θ","Laplacian_ϕ_θ")
    D_lst1 = ("D11_r_w","D21_θ_w","D31_ϕ_w")
    D_lst2 = ("D11_r_sym","D11_r","D21_θ","D31_ϕ")
    D_lst3 = ("D12_r_sym", "D23_ϕ_sym")
    D_lst4 = ("D12_θ_sym","D13_ϕ_sym")
    D_lst5 = ("D22_r_sym","D22_θ_sym","D33_r_sym","D33_θ_sym",
                "D12_r","D22_θ","D32_ϕ","D22_r","D33_r",
                "D33_θ","D12_θ","D13_ϕ","D23_ϕ")
    D_lst6 = ("D12_r_w","D22_θ_w","D32_ϕ_w","D22_r_w","D33_r_w",
                "D33_θ_w","D12_θ_w","D13_ϕ_w","D23_ϕ_w")

    # Avoid computing shared trigonometric factors unused by the selected key.
    need_f_sin_x = key in ("Cot", "Laplacian_r_r", "Laplacian_θ_r_r", "D23_ϕ_θ_sym") || 
                key in D_lst3 || key in D_lst4 || key in D_lst5

    need_df_sin_x = (key == "Laplacian_r_θ")

    need_d2f = key in ("D2", "Laplacian_r_r", "Laplacian_r_θ", "Laplacian_θ_r", 
                    "Laplacian_ϕ_r", "Laplacian_θ_θ", "Laplacian_ϕ_θ", 
                    "Laplacian_r_odd", "Laplacian_θ_odd")

    need_f = key in ("I", "Cos", "Sec", "Tan", "Scale", "Scale_T", "Laplacian_θ_r", "Laplacian_ϕ_r", 
                    "Laplacian_θ_θ", "Laplacian_ϕ_θ", "Laplacian_r_odd", "Laplacian_θ_odd", 
                    "Laplacian_inter_odd", "D_p_r") || 
            key in D_lst1 || key in D_lst2 || key in D_lst6 || need_f_sin_x || need_d2f

    need_df = key in ("D", "Scale", "Scale_T", "Laplacian_r_r", "Laplacian_θ_r_r", 
                    "Laplacian_θ_r", "Laplacian_ϕ_r", "Laplacian_θ_θ", "Laplacian_ϕ_θ", 
                    "Laplacian_r_odd", "Laplacian_θ_odd", "Laplacian_inter_odd", "D_p_r", 
                    "D23_ϕ_θ_sym") || key in D_lst1 || key in D_lst2 || key in D_lst4 || need_df_sin_x

    need_sin_x = key in ("Tan", "Scale", "Scale_T", "Laplacian_θ_θ", "Laplacian_ϕ_θ", "D_p_r") || 
                        key in D_lst1 || key in D_lst2 || need_f_sin_x || need_df_sin_x

    need_cos_x = key in ("Cot", "Cos", "Sec", "Tan", "Scale", "Scale_T", "Laplacian_r_r", 
                        "Laplacian_r_θ", "Laplacian_θ_r_r", "Laplacian_θ_r", "Laplacian_ϕ_r", 
                        "Laplacian_θ_θ", "Laplacian_ϕ_θ", "D_p_r", "D23_ϕ_θ_sym") || key in D_lst1 ||
                        key in D_lst2 || key in D_lst3 || key in D_lst4 || key in D_lst5 || key in D_lst6

    need_sin_2x = key in ("Laplacian_r_r", "Laplacian_θ_r", "Laplacian_ϕ_r", "Laplacian_r_odd", 
                        "Laplacian_θ_odd", "Laplacian_inter_odd") || key in D_lst1

    need_cos_2x = key in ("Laplacian_r_r", "Laplacian_θ_r", "Laplacian_ϕ_r", "Laplacian_r_odd", 
                        "Laplacian_θ_odd", "Laplacian_inter_odd") || key in D_lst1 || key in D_lst4

    need_sin_4x = key in ("Laplacian_r_r", "Laplacian_r_odd", "Laplacian_θ_odd")

    need_cos_4x = key in ("Laplacian_r_r", "Laplacian_r_odd", "Laplacian_θ_odd")

    # Evaluate only the trigonometric factors used by this key.
    sin_x = need_sin_x ? mysin.(x) : nothing
    cos_x = need_cos_x ? mycos.(x) : nothing

    sin_2x = need_sin_2x ? mysin.(dt(2) .* x) : nothing
    cos_2x = need_cos_2x ? mycos.(dt(2) .* x) : nothing

    sin_4x = need_sin_4x ? mysin.(dt(4) .* x) : nothing
    cos_4x = need_cos_4x ? mycos.(dt(4) .* x) : nothing

    # Boundary code selects the cosine or sine family for f and its derivatives.
    use_cos_basis = (bdry[1] == '1')

    f  = nothing
    df = nothing
    d2f = nothing

    if need_f || need_df
        if use_cos_basis
            # f = cos(n x)
            if need_f;  f  = mycos.(x .* nrow); end
            if need_df; df = .-(nrow) .* mysin.(x .* nrow); end
        else
            # f = sin(n x)
            if need_f;  f  = mysin.(x .* nrow); end
            if need_df; df = (nrow) .* mycos.(x .* nrow); end
        end
        if need_d2f; d2f = .-(nrow .^ 2) .* f; end
    end

    # Removable quotients such as sin(nx)/sin(x) and derivative analogues.
    f_sin_x  = nothing
    df_sin_x = nothing

    if need_f_sin_x
        @assert !use_cos_basis "$key only defined for bdry[1]=='0'"
        f_sin_x = f ./ sin_x
        f_sin_x[zero_lines, :] .= nrow # Analytic limit of sin(nx)/sin(x) at 0 and 2π.
        f_sin_x[pi_lines, :] .= (bdry == "00" ? -nrow : nrow) # Analytic limit at π, with boundary-dependent sign.
    end

    if need_df_sin_x
        @assert use_cos_basis "$key only defined for bdry[1]=='1'"
        df_sin_x = df ./ sin_x
        df_sin_x[zero_lines, :] .= -nrow.^2 # Analytic derivative-quotient limit at 0 and 2π.
        df_sin_x[pi_lines, :]   .= (bdry=="11" ?  nrow.^2 : -nrow.^2) # Analytic limit at π, with boundary-dependent sign.
    end
    
    # Dispatch to the requested one-dimensional operator expression.
    result = nothing
    if key == "I"
        result = f

    elseif key == "D"
        result = df

    elseif key == "D2"
        result = d2f

    elseif key == "Cot"
        result = cos_x .* f_sin_x

    elseif key == "Cos"
        result = cos_x .* f

    elseif key == "Sec"
        @assert bdry == "10" "Sec requires the odd-cosine basis."
        endpoint_lines = (abs.(x .- Pi/dt(2)) .< 1e-14) .|
                         (abs.(x .- dt(3)*Pi/dt(2)) .< 1e-14)
        result = similar(f)
        result[.!endpoint_lines,:] .= f[.!endpoint_lines,:] ./ cos_x[.!endpoint_lines]
        # At the known grid endpoints π/2 and 3π/2, the limits are 1, -3, 5, ... .
        result[endpoint_lines,:] .= nrow
        result[endpoint_lines,2:2:end] .*= -one(dtype)

    elseif key == "Tan"
        result = f .* sin_x ./ cos_x
        # Use analytic limits where tan(x) has removable singularities against the basis.
        pihalf_lines = abs.(x .- Pi/dt(2)) .< 1e-14
        threepihalf_lines = abs.(x .- 3*Pi/dt(2)) .< 1e-14
        result[pihalf_lines,:] .= nrow
        result[pihalf_lines,2:2:end] .*= -one(dtype) # Sign flip for even-numbered columns.
        result[threepihalf_lines, :] .= (bdry == "00" ? nrow : -nrow)
        result[threepihalf_lines,2:2:end] .*= -one(dtype) # Sign flip for even-numbered columns.

    elseif key == "Scale"
        result = (sin_x/dt(2)) .* (cos_x .* f .+ sin_x .* df)
    
    elseif key == "Scale_T"
        result = -(sin_x/dt(2)) .* (cos_x .* f .+ sin_x .* df)

    elseif key == "Laplacian_r_r"
        term1 = (-dt(19) .+ dt(3)*cos_4x) .* f_sin_x
        term2 = dt(4) .* cos_x .* (dt(4)*cos_2x .* df .+ sin_2x .* d2f)
        result = (cos_x .* (term1 .+ term2)) ./ dt(8)

    elseif key == "Laplacian_r_θ"
        result = df_sin_x .* cos_x .+ d2f

    elseif key == "Laplacian_θ_r_r"
        result = -dt(2) .* (f_sin_x .* cos_x .+ df)

    elseif key in Lap_r_lst
        term1 = dt(3) .* sin_2x .* f
        term2 = dt(4) .* cos_2x .* df
        term3 = sin_2x .* d2f
        result = (cos_x.^2) .* (.-term1 .+ term2 .+ term3) ./ dt(2)

    elseif key in Lap_θ_lst
        result = df .* (cos_x ./ sin_x) .- f ./ sin_x.^2 .+ d2f
        result[zero_lines,:] .= dt(0)
        result[pi_lines,:] .= dt(0)

    elseif key == "Laplacian_r_odd"
        result = ((dt(2)*sin_2x .+ sin_4x).*d2f)./dt(8) .+ ((dt(3) .+ dt(4)*cos_2x .+ cos_4x).*df)./dt(2) .- ((dt(14)*sin_2x .+ dt(3)*sin_4x).*f)./dt(8)

    elseif key == "Laplacian_θ_odd"
        result = ((dt(2)*sin_2x .+ sin_4x).*d2f)./dt(8) .+ ((dt(1) .+ dt(2)*cos_2x .+ cos_4x).*df)./dt(2) .- ((dt(6)*sin_2x .+ dt(3)*sin_4x).*f)./dt(8)

    elseif key == "Laplacian_inter_odd"
        result = (sin_2x./dt(4)) .* f .- ((dt(1) .+ cos_2x)./dt(4)) .* df

    elseif key == "D_p_r"
        result = sin_x .* (.-sin_x .* f .+ cos_x .* df)
    
    elseif key in D_lst1
        result = (sin_2x ./ dt(2)) .* (.-sin_x .* f .+ cos_x .* df)

    elseif key in D_lst2
        result = cos_x.^2 .* (.-sin_x .* f .+ cos_x .* df)

    elseif key in D_lst3
        result = (cos_x.^2 ./ dt(2)) .* f_sin_x

    elseif key in D_lst4
        result = (cos_x.^2 ./ dt(4)) .* ((-dt(3) .+ cos_2x).*f_sin_x .+ dt(2)*cos_x.*df)

    elseif key in D_lst5
        result = cos_x.^2 .* f_sin_x
    
    elseif key in D_lst6
        result = cos_x .* f

    elseif key == "D23_ϕ_θ_sym"
        result = .-f_sin_x .* cos_x .+ df

    elseif key == "zero"
        result = zeros(dtype, size(x,1), N)

    else
        error("Invalid key: $key")
    end
    return result
end
