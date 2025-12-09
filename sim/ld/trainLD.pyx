# cython: boundscheck = False
# cython: cdivision = True
# cython: wraparound = False
import cython
cimport cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.stdint cimport uint64_t
from rng cimport XorShift64Star


cpdef uint64_t make_numpy_seed():
    """
    Generate a uniform 64-bit seed using old NumPy RNG (32-bit based).
    """
    high = np.random.randint(0, 1 << 32, dtype=np.uint32)
    low  = np.random.randint(0, 1 << 32, dtype=np.uint32)

    return (<uint64_t> int(high) << 32) | <uint64_t> int(low)


cpdef trainLD(int p, int nS, double lr, double[:,:] X, double[:] y, double[:] beta0, np.int64_t[:] plot_list,
int n, int d, int n_it_max, int n_runs, double epsilon):
    cdef XorShift64Star rng
    cdef int k, r, s, N
    cdef double gd_aux0, gd_aux1, diff_coeff
    cdef double inv_n_runs = 1.0 / n_runs
    #####
    cdef double *arr = <double*>malloc(p * sizeof(double))
    if arr == NULL:
        raise MemoryError()
    cdef double[:] betaS = <double[:p]>arr
    #####
    cdef double *arr3 = <double*>malloc(n * sizeof(double))
    if arr3 == NULL:
        raise MemoryError()
    cdef double[:] D = <double[:n]>arr3
    #####
    cdef double *arr4 = <double*>malloc(p * sizeof(double))
    if arr4 == NULL:
        raise MemoryError()
    cdef double[:] GRAD = <double[:p]>arr4
    #####
    cdef double *arr5 = <double*>malloc(p * sizeof(double))
    if arr5 == NULL:
        raise MemoryError()
    cdef double[:] betaS0 = <double[:p]>arr5
    #####
    cdef double *arr6 = <double*>malloc(n * p * sizeof(double))
    if arr6 == NULL:
        raise MemoryError()
    cdef double[:,:] XS = <double[:n, :p]>arr6
    #####
    cdef double *arr_noise = <double*>malloc(p * sizeof(double))
    if arr_noise == NULL:
        raise MemoryError()
    cdef double[:] noise = <double[:p]>arr_noise
    #####
    cdef np.int64_t[::1] S_view
    #####
    cdef np.ndarray[np.float64_t, ndim=2] betaS_av
    cdef double[:, ::1] betaS_mv
    cdef Py_ssize_t n_plots
    cdef Py_ssize_t count_aux = 1
    cdef long next_plot_k

    try:
        diff_coeff = np.sqrt(2 * epsilon)
        ###### Select randomly p integers from [d]
        S = np.random.choice(d, size=p, replace=False)
        S_view = S

        ###### Initializing the estimator
        betaS0[:] = 0.
        for s in range(p):
            betaS0[s] = beta0[S_view[s]]
        del beta0

        ###### Constructing XS
        for r in range(n):
            for s in range(p):
                XS[r, s] = X[r, S_view[s]]
        del X
        
        ######## Preparing array for beta evolution
        n_plots = plot_list.shape[0]
        betaS_av = np.zeros((n_plots + 1, p))
        betaS_mv = betaS_av
        for s in range(p):
            betaS_mv[0, s] = betaS0[s]
        if n_plots > 0:
            next_plot_k = plot_list[0]
        else:
            next_plot_k = -1


        rng = XorShift64Star(make_numpy_seed())
        for N in range(n_runs):

            ### N-th stochastic run
            betaS[:] = betaS0[:]
            #######
            for k in range(1, n_it_max + 1):
                #### GD ####
                ## Inner product: XS and beta_k
                for r in range(n):
                    D[r] = 0. 
                for s in range(p):
                    GRAD[p] = 0.
                
                for r in range(n):
                    gd_aux0 = 0.
                    for s in range(p):
                        gd_aux0 += XS[r, s] * betaS[s]
                    D[r] = y[r] - gd_aux0
                ## Inner product: y - XS*betaS times xs
                for s in range(p):
                    gd_aux1 = 0.
                    for r in range(n):
                        gd_aux1 += XS[r, s] * D[r]
                    GRAD[s] = gd_aux1
                rng.fill_gaussian_c(noise)
                ## GD update
                for s in range(p):
                    betaS[s] += lr * (GRAD[s] / n + diff_coeff * noise[s])

                ## Retrieving dynamics 
                if k == next_plot_k:
                    for s in range(p):
                        betaS_mv[count_aux, s] += betaS[s] * inv_n_runs
                    count_aux += 1
                    if count_aux <= n_plots:
                        next_plot_k = plot_list[count_aux - 1]
                    else:
                        next_plot_k = -1

        return betaS_av

    finally:
        free(arr3)
        free(arr4)
        free(arr6)
        free(arr5)
        free(arr)
        free(arr_noise)

