# ============================================================
# NS_Numerics.jl
#
# Module entry point for the computer-assisted proof library.
# Selects dtype when this module is loaded, imports the packages below,
# then includes source files in dependency order. The grouped exports identify
# each defining file; includes resolve relative to this source file.
#
# Loading also creates gen_dir if absent and prints dtype and the thread count.
# gen_dir is relative to the process working directory: from notebooks/ it
# resolves to the project data/generated directory. Loading does not change
# the working directory. Select the numeric mode before loading the module.
# See docs/src.md and docs/notebooks.md for the reading and execution order.
# ============================================================
module NS_Numerics

using LinearAlgebra
using SpecialFunctions
using IntervalArithmetic
using Combinatorics
using SparseArrays
using MAT
using Base:tail, front
using Base.Threads
using Arblib
import LinearAlgebra: norm # need to reload norm
import Base: +, -, *, /, copy, getindex, setindex! # reload arithmetic operators
import ArbExtras: besselj
import FunctionZeros: besselj_zero
import HypergeometricFunctions:pFq

# Load-time numeric mode: interval enclosures; Float64 is the exploratory alternative.
const dtype = Interval{Float64}
# const dtype = Float64

# Path is relative to the current working directory, not @__DIR__.
const gen_dir = "../data/generated"
if !isdir(gen_dir)
    mkpath(gen_dir)
end


println("Using dtype = ", dtype)
println("Number of threads = ", nthreads())

include("Matrix_Func.jl")
include("My_Fourier.jl")
include("My_Scalar.jl")
include("My_Vector.jl")
include("My_Operator.jl")
include("My_Bessel.jl")
include("Trig_Poly.jl")
include("Comb_Bound.jl")
include("My_Assembly.jl")
include("NS_Terms.jl")

export
    dtype, norm, inf, sup,
    Diagonal, diagm, diag, isdiag, Interval, hull, in_interval,

    # My_Fourier
    dt, is_zero, Pi, mysin, mycos, double_fact, gamma_ratio, che2len, len2che, 
    conversion_mat, Fourier_mat, div_mat, calc_Utheta_divfree, trig_func,

    # Matrix_Func
    newton_cotes_weights, newton_cotes_with_bound, derivative_padding, finite_difference, Linf_to_paras,
    get_Wk∞_norm_lst, estimate_Wk∞_norm, add_error, save_matrix, 
    interval_matrix, read_matrix, verify_spd_by_ldlt, verify_general_eig, eig_sym3, 
    verify_symmetric_spectrum, verify_gram_spectrum,
    vec_to_mat,

    # My_Scalar
    grid, BDRY_LIST, check_bdry, ScalarFunc, set_func!, set_bdry!, set_conv_mat!,
    Fourier_transform, interpolate, transform!, inner_product, get_Wk∞_norm_lst, 
    trim_border!, add_equal!, multiply_equal!, minus_equal!, divide_equal!,

    # My_Vector
    BDRY_LST_SET, VectorFunc, from_mat_file, load_grad_U, transform!, interpolate, 
    set_div_free!, inner_product, norm_lst, trim_border!, save, add_equal!,
    multiply_equal!, minus_equal!, divide_equal!,

    # My_Operator
    Operator, set_T!, set_L1!, apply, change_variable!, Operator_matrix, 
    add_operator!, apply_convmat_right!, eigen_calc,

    # My_Bessel
    bessel_W2_bound_interval, my_besselj, func_Z, func_Y, rsign, sign_change, 
    rigorous_root_interval, verify_root, mat_to_lst, preprocess_para!, load_besselj_zeros,
    Ur_func, eval_Ur, eval_Uθ, Ur_bound, Uθ_bound, Ur_bound_lst,

    # Trig_Poly
    TrigPoly, trig_mul, trig_int_0_pi2, trig_basis_lists, sph_basis_lists, 
    tensor_int_basis, contract_QT_blocks, split_tensor, integrate_θ, int_θ_bound,

    # Comb_Bound
    multinomial_sum, poly_deriv, poly_add, poly_mul_x, poly_mul_1_plus_x2,
    poly_abs_sum_bound, beta_all_deriv_Linf_bounds, tan_over_cos_deriv_Linf_bounds,
    r_to_beta, eig_fun_bound, beta_bound_chain, refine_channel_bounds,

    # My_Assembly
    integrate_r, eval_l2_inpd, eval_h1_norm, Ur_sq_Wk∞_bound_lists, get_lap_matrix,
    lst_to_mat, eval_eig_function, apply_Q_to_channel_fields,

    # NS_Terms
    build_operator, get_linear, get_identity, get_gradient_sym, get_gradient, 
    get_operator_matrix, mul_A_gradB_local, mul_A_gradB
end
