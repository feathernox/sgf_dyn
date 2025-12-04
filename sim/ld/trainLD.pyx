# cython: boundscheck = False
# cython: cdivision = True
# cython: wraparound = False
import cython
cimport cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cpdef trainLD(int p, int nS, double lr, double[:,:] X, double[:] y, double[:] beta0, np.int64_t[:] plot_list,
int n, int d, int n_it_max, int n_runs, double epsilon):
    cdef int k, r, s, q, N
    cdef double gd_aux0, gd_aux1, diff_coeff
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
    cdef np.int64_t[::1] S_view

    try:
        diff_coeff = np.sqrt(2 * epsilon)
        ###### Select randomly p integers from [d]
        S = np.random.choice(d, size=p, replace=False)
        S_view = S

        ###### Initializing the estimator
        betaS0[:] = 0.
        for q in range(p):
            betaS0[q] = beta0[S_view[q]]
        del beta0

        ###### Constructing XS
        for r in range(n):
            for q in range(p):
                XS[r, q] = X[r, S_view[q]]
        del X
        
        ######## LANGEVIN DIFFUSION

        betaS_av = np.zeros((len(plot_list) + 1, p))
        betaS_av[0] = np.array(betaS0)

        noise = np.empty(p, dtype=np.float64)

        for N in range(n_runs):

            ### N-th stochastic run
            betaS[:] = betaS0[:]
            #######
            count_aux = 1
            for k in range(1, n_it_max+1):
                #### GD ####
                ## Inner product: XS and beta_k 
                D[:] = 0. 
                GRAD[:] = 0.
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
                ## GD update
                noise[:] = np.random.randn(p)
                for s in range(p):
                    betaS[s] += lr * (GRAD[s] / n + diff_coeff * noise[s])

                ## Retrieving dynamics
                save_k = k in plot_list
                if save_k:
                    betaS_av[count_aux] += np.array(betaS) / n_runs
                    count_aux += 1
            # print(np.asarray(betaS))

        return betaS_av

    finally:
        free(arr3)
        free(arr4)
        free(arr6)
        free(arr5)
        free(arr)

