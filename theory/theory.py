import numpy as np
import numba as nb
import scipy as sp
from scipy.integrate import quad, dblquad, nquad
from scipy.special import i1
from scipy.integrate import odeint
import math
import os


def MP_alpha_minus(alpha):
    return (1 - np.sqrt(alpha)) ** 2


def MP_alpha_plus(alpha):
    return (1 + np.sqrt(alpha)) ** 2


def _MP_pdf_expression(x, alpha, alpha_minus, alpha_plus):

    return 1 / (2 * np.pi) * np.sqrt(
        np.abs((alpha_plus - x) * (x - alpha_minus))) / (alpha * x)

    
def MP_expectation(f, alpha):
    """
    Expectation of E_{x \\sim MP(\\alpha)} [f(x)]
    """
    if alpha == 0:
        return f(1)
    
    alpha_minus = MP_alpha_minus(alpha)
    alpha_plus = MP_alpha_plus(alpha)

    exp_val = quad(lambda x: f(x) * _MP_pdf_expression(x, alpha, alpha_minus, alpha_plus),
                   alpha_minus, alpha_plus)[0]
    return exp_val


def expectation_MP_exp(alpha, t, eps=1e-2, max_i1_arg=100):
    res = 1 / np.sqrt(alpha) * odeint(
        lambda y, s:  np.where(
            s == 0,
            2 * np.sqrt(alpha),
            np.where(
                4 * np.sqrt(alpha) * s < max_i1_arg,
                np.exp(- 2 * (1 + alpha) * s) * i1( 4 * np.sqrt(alpha) * s) / s,
                np.exp(- 2 * (1 - np.sqrt(alpha)) ** 2 * s) / 
                (2 * np.sqrt(2 * np.pi *  np.sqrt(alpha) * s ** 3))
            )),
        0, np.hstack(([0], t)))
    return res[1:, 0]


##### GF TRAIN ERROR OLD ######

def GF_train_error_inf_time(alpha, psi, mu2, beta2=1.): 
    # no dependency on betadiff2
    res = ((1 - alpha / psi) * beta2 + mu2) * (1 - np.minimum(alpha, 1))
    res = res / 2  # factor 1/2
    return res
    
    
def GF_train_error_inf_time_precise(p, n, d, mu2, beta2=1.): 
    # no dependency on betadiff2
    res = ((1 - p / d) * beta2 + mu2) * (1 - np.minimum(p, n) / n)
    res = res / 2  # factor 1/2
    return res


def _GF_train_error_alpha_leq_1(alpha, psi, mu2, t, betadiff2, beta2=1.):
    f1 = lambda x: x * math.exp(-2 * t * x)
    f2 = lambda x: math.exp(-2 * t * x)
    res = alpha * betadiff2 / psi * MP_expectation(f1, alpha) + \
          ((1 - alpha / psi) * beta2 + mu2) * (1 - alpha + alpha * MP_expectation(f2, alpha))
    res = res / 2  # factor 1/2
    return res


def _GF_train_error_alpha_g_1(alpha, psi, mu2, t, betadiff2, beta2=1.):
    f1 = lambda x: x * math.exp(-2 * t * alpha * x)
    f2 = lambda x: math.exp(-2 * t * alpha * x)
    res =  betadiff2 / psi * alpha * MP_expectation(f1, 1 / alpha) + \
            ((1 - alpha / psi) * beta2 + mu2) * MP_expectation(f2, 1 / alpha)
    res = res / 2  # factor 1/2
    return res


_GF_train_error_alpha_leq_1 = np.vectorize(_GF_train_error_alpha_leq_1)
_GF_train_error_alpha_g_1 = np.vectorize(_GF_train_error_alpha_g_1)


def GF_train_error(alpha, psi, mu2, t, betadiff2, beta2=1.):
    if alpha <= 1:
        return _GF_train_error_alpha_leq_1(alpha, psi, mu2, t, betadiff2, beta2=beta2)
    else:
        return _GF_train_error_alpha_g_1(alpha, psi, mu2, t, betadiff2, beta2=beta2)
    
GF_train_error = np.vectorize(GF_train_error)


##### GF TEST ERROR LEGACY ######


def _legacy_GF_test_error_inf_time_alpha_leq_1(alpha, psi, mu2, betadiff2, beta2=1.):
    res = ((1 - alpha / psi) * beta2 + mu2) * 1 / (1 - alpha)
    res = res / 2  # factor 1/2
    return res


def _legacy_GF_test_error_inf_time_alpha_g_1(alpha, psi, mu2, betadiff2, beta2=1.):
    res = alpha / psi * betadiff2 * (1 - 1 / alpha) + ((1 - alpha / psi) * beta2 + mu2) * (1 + 1 / (alpha - 1))
    res = res / 2  # factor 1/2
    return res
    
    
def legacy_GF_test_error_inf_time(alpha, psi, mu2, betadiff2, beta2=1.):
    alpha = np.asarray(alpha)
    res = np.zeros(alpha.shape)
    
    is_alpha_leq_1 = alpha <= 1 
    res[is_alpha_leq_1] = _legacy_GF_test_error_inf_time_alpha_leq_1(alpha[is_alpha_leq_1], psi, mu2, betadiff2, beta2=beta2)
    res[~is_alpha_leq_1] = _legacy_GF_test_error_inf_time_alpha_g_1(alpha[~is_alpha_leq_1], psi, mu2, betadiff2, beta2=beta2)
    return res


def GF_test_error_inf_time_precise(p, n, d, mu2, betadiff2, beta2=1.):
    p = np.asarray(p)
    res = np.full(p.shape, np.inf)
    
    is_p_l_nm1 = p < (n - 1)
    p_l_nm1 = p[is_p_l_nm1]
    res[is_p_l_nm1] = ((1 - p_l_nm1 / d) * beta2 + mu2) * (1 + p_l_nm1 / (n - p_l_nm1 - 1))
    
    is_p_g_np1 = p > (n + 1)
    p_g_np1 = p[is_p_g_np1]
    res[is_p_g_np1] = p_g_np1 / d * betadiff2 * (1 - n / p_g_np1) + ((1 - p_g_np1 / d) * beta2 + mu2) * (1 + n / (p_g_np1 - n - 1))

    res = res / 2  # factor 1/2
    return res


def _legacy_GF_test_error_alpha_leq_1(alpha, psi, mu2, t, betadiff2, beta2=1.):
    f1 = lambda x: math.exp(-2 * t * x)
    f2 = lambda x: (1 - math.exp(- t * x)) ** 2 / x
    res = alpha * betadiff2 / psi * MP_expectation(f1, alpha) + \
          ((1 - alpha / psi) * beta2 + mu2) * (1 + alpha * MP_expectation(f2, alpha))
    res = res / 2  # factor 1/2
    return res


def _legacy_GF_test_error_alpha_g_1(alpha, psi, mu2, t, betadiff2, beta2=1.):
    f1 = lambda x: math.exp(-2 * alpha * t * x)
    f2 = lambda x: (1 - math.exp(- alpha * t * x)) ** 2 / x
    res = betadiff2 * (alpha - 1) / psi + betadiff2 / psi * MP_expectation(f1, 1 / alpha) + \
          ((1 - alpha / psi) * beta2 + mu2) * (1 + 1 / alpha * MP_expectation(f2, 1 / alpha))
    res = res / 2  # factor 1/2
    return res


_legacy_GF_test_error_alpha_leq_1 = np.vectorize(_legacy_GF_test_error_alpha_leq_1)
_legacy_GF_test_error_alpha_g_1 = np.vectorize(_legacy_GF_test_error_alpha_g_1)
 

def legacy_GF_test_error(alpha, psi, mu2, t, betadiff2, beta2=1.):
    """ 
    with factor 1/2
    """
    if alpha <= 1:
        return _legacy_GF_test_error_alpha_leq_1(alpha, psi, mu2, t, betadiff2, beta2=beta2)
    else:
        return _legacy_GF_test_error_alpha_g_1(alpha, psi, mu2, t, betadiff2, beta2=beta2)

legacy_GF_test_error = np.vectorize(legacy_GF_test_error)


##### GF TEST ERROR NEW ######

@nb.cfunc('float64(intc, CPointer(float64))')
def gf_test_int1_numba(n, args):
    """
    args = (s, t, aminus, aplus)
    """
    return np.exp(-2 * args[0] * args[1]) * np.sqrt((args[3] - args[0]) * (args[0] - args[2])) / args[0]

gf_test_int1_c = sp.LowLevelCallable(gf_test_int1_numba.ctypes)

def gf_test_int1(t, aminus, aplus):
    return quad(gf_test_int1_c, aminus, aplus, args=(t, aminus, aplus))[0]

gf_test_int1 = np.vectorize(gf_test_int1)


@nb.cfunc('float64(intc, CPointer(float64))')
def gf_test_int2_numba(n, args):
    """
    args = (s, t, aminus, aplus)
    """
    return ((1 - np.exp(-args[0] * args[1])) / args[0]) ** 2 * np.sqrt((args[3] - args[0]) * (args[0] - args[2])) 

gf_test_int2_c = sp.LowLevelCallable(gf_test_int2_numba.ctypes)

def gf_test_int2(t, aminus, aplus):
    return quad(gf_test_int2_c, aminus, aplus, args=(t, aminus, aplus))[0]

gf_test_int2 = np.vectorize(gf_test_int2)


def GF_test_error(alphas, ts, psi=2.5, betadiff2=2, beta2=1, mu2=0.04):
    ts = np.asarray(ts)[np.newaxis, :]
    alphas = np.asarray(alphas)[:, np.newaxis]
    aminus = MP_alpha_minus(alphas)
    aplus = MP_alpha_plus(alphas)
    # axis 0: alpha, axis1: t 
    int1 = np.maximum(alphas - 1, 0) + 1 / (2 * np.pi) * gf_test_int1(ts, aminus, aplus)
    int2 = 1 + 1 / (2 * np.pi) * gf_test_int2(ts, aminus, aplus)
    
    res = 1 / 2 * (betadiff2 / psi * int1 + ((1 - alphas / psi) * beta2 + mu2) * int2)
    return res


def GF_test_error_inf_time(alphas, psi=2.5, betadiff2=2, beta2=1, mu2=0.04):
    # 0.5 below is placeholder value, can be anything as long as
    # it evaluates on double descent expression
    safe = np.where((alphas == 0) | (alphas == 1), 0.5, alphas) 
    double_descent = np.where(alphas == 0, 1.0,
         np.where(alphas == 1, np.inf, 1 / (1 - np.minimum(safe, 1 / safe))))
    res = 1 / 2 * (betadiff2 / psi * np.maximum(alphas - 1, 0) + \
                   ((1 - alphas / psi) * beta2 + mu2) * double_descent)
    return res


##### TODO SGF need to check #####

def get_z_covariation(alpha, psi, mu2, t, betadiff2, eps=1e-15):
#     alpha = np.asarray(alpha)

    t = np.asarray(t)
    
    coef1 = 8 * np.sqrt(alpha) / psi * betadiff2
    coef2 = 1 - alpha / psi + mu2
    
    aminus = MP_alpha_minus(alpha)
    aplus = MP_alpha_plus(alpha)
    denom_term = aminus / (4 * np.sqrt(alpha))
    
    res = 0
    if alpha < 1.:
        res += coef2 * (1 - alpha) / 2 * expectation_MP_exp(alpha, t, eps=1e-15)
        
#     f1 = 16 * np.exp(-2 * aminus * t) / (np.pi ** 2 * np.sqrt(alpha)) * int1
#     f2 = 2 * np.exp(-2 * aminus * t) / (np.pi ** 2) * int2
    
#     res += coef1 * f1 + coef2 * f2 

    exp_coef = - 8 * np.sqrt(alpha) * t
    
    def get_int_term(exp_coef):
        f = lambda sigma, rs: (coef1 + coef2 * (1 / (denom_term + rs) + 1 / (denom_term + sigma))) * \
                (np.exp(exp_coef * sigma) - np.exp(exp_coef * rs))/(rs - sigma) * \
                                np.sqrt(sigma * (1 - sigma) * (1 - rs) * rs) 
        int3 = dblquad(f, 0, 1, lambda sigma: sigma, lambda sigma: 1)[0] 
        return int3
    get_int_term = np.vectorize(get_int_term)
    
    
    res += 2 * np.exp(-2 * aminus * t) / (np.pi ** 2) * get_int_term(exp_coef)
    
    res *= alpha
    return res/2.


@nb.cfunc('float64(intc, CPointer(float64))')
def ld_integral_numba(n, args):
    """
    args = (s, t, aminus, aplus)
    """
    return (1 - np.exp(-args[0] * args[1])) * np.sqrt((args[3] - args[0]) * (args[0] - args[2])) / args[0] ** 2

ld_integral_c = sp.LowLevelCallable(ld_integral_numba.ctypes)


def ld_integral(arg_t, aminus, aplus):
    return quad(ld_integral_c, aminus, aplus, args=(arg_t, aminus, aplus))[0]

ld_integral = np.vectorize(ld_integral)


def get_ld_z_covariance_exp(alpha, t):
    """
    TODO: check factor 1/2
    """
    arg_t = 2 * np.where(alpha > 1, alpha * t, t)
    
    aminus = MP_alpha_minus(alpha)
    aplus = MP_alpha_plus(alpha)
    
    res = ld_integral(arg_t, aminus, aplus) / (2 * np.pi)
    
    return res


def get_ld_z_covariance(alphas, ts, only_data_space=False):
    """
    TODO: check factor 1/2
    """
    alphas = np.asarray(alphas)[:, np.newaxis]
    ts = np.asarray(ts)[np.newaxis, :]
    res = get_ld_z_covariance_exp(alphas, ts)
    if not only_data_space:
        res += 2 * np.maximum(0, alphas - 1) * ts
    return res


def get_ld_z_covariance_infty(alphas, only_data_space=False):
    """
    TODO: check factor 1/2
    """
    alphas = np.asarray(alphas)
    res = np.empty_like(alphas, dtype=float)

    mask_lt = alphas < 1
    mask_eq = alphas == 1
    mask_gt = alphas > 1

    res[mask_lt] = alphas[mask_lt] / (1 - alphas[mask_lt])

    if not only_data_space:
        res[mask_gt] = np.inf
    else:
        res[mask_gt] = 1 / (alphas[mask_gt] - 1)  # idk alpha or 1?
    
    res[mask_eq] = np.inf

    return res
