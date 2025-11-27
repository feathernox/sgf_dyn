# cython: boundscheck = False
# cython: cdivision = True
# cython: wraparound = False
import cython
cimport cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
import time

cpdef trainLD(int p, int nS, double lr, double[:,:] X, double[:] y, double[:] beta0, np.int64_t[:] plot_list,
int n, int d, int n_it_max, int n_runs, double epsilon):
    cdef unsigned int k, r, s, q
    #####
    cdef double *arr = <double*>malloc(p * sizeof(double))
    cdef double[:] betaS = <double[:p]>arr
    #####
    cdef double *arr3 = <double*>malloc(n * sizeof(double))
    cdef double[:] D = <double[:n]>arr3
    #####
    cdef double *arr4 = <double*>malloc(p * sizeof(double))
    cdef double[:] GRAD = <double[:p]>arr4
    #####
    cdef double *arr5 = <double*>malloc(p * sizeof(double))
    cdef double[:] betaS0 = <double[:p]>arr5
    #####
    cdef double gd_aux0, gd_aux1, diff_coeff
    #####
    cdef long *arr2 = <long*>malloc(p * sizeof(double))
    cdef long[:] S = <long[:p]>arr2
    #####
    cdef double *arr6 = <double*>malloc(n * p * sizeof(double))
    cdef double[:,:] XS = <double[:n, :p]>arr6

    diff_coeff = np.sqrt(2 * epsilon)

    print('Beginning LD training ----- p= %d --- nS= %d\n' % (p,nS), end='')
    ###### Select randomly p integers from [d]
    S = np.random.choice(d, size=p, replace=False)
         
    ###### Initializing the estimator
    betaS0[:] = 0.
    for q in range(0, p):
        betaS0[q] = beta0[S[q]]
    del beta0

    ###### Constructing XS
    for r in range(0, n):
        for q in range(0, p):
            XS[r, q] = X[r, S[q]]
    del X
    free(arr2)
       
    ######## LANGEVIN DIFFUSION
    ##############
    start = time.time()
    ##############

    betaS_av = np.zeros((len(plot_list) + 1, p))
    betaS_av[0] = np.array(betaS0)

    for N in range(n_runs):
        ### N-th stochastic run
        betaS = np.copy(betaS0)
        #######
        count_aux = 1
        for k in range(1, n_it_max+1):
            #### GD ####
            ## Inner product: XS and beta_k 
            D[:] = 0. 
            GRAD[:] = 0.
            for r in range(0,n):
                gd_aux0 = 0.
                for s in range(0,p):
                    gd_aux0 += XS[r,s] * betaS[s]
                D[r] = y[r] - gd_aux0
            ## Inner product: y - XS*betaS times xs
            for r in range(0,p):
                gd_aux1 = 0.
                for s in range(0, n):
                    gd_aux1 += XS[s,r] * D[s]
                GRAD[r] = gd_aux1   
            ## GD update
            for u in range(0,p):
                b_ku = np.random.randn()
                betaS[u] += lr * (GRAD[u] / n + diff_coeff * b_ku)
            
            # print(np.asarray(betaS))

            ## Retrieving dynamics
            save_k = k in plot_list
            if save_k:
                betaS_av[count_aux] += np.array(betaS)/n_runs
                count_aux += 1
    ##############
    end = time.time()
    ##############
    free(arr3)
    free(arr4)
    free(arr6)
    free(arr5)
    free(arr)

    print('Time elapsed LD: %f' % (end-start))
    print('End ----- p= %d --- nS= %d' % (p,nS))
    print(' ')

    return betaS_av

