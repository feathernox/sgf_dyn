import numpy as np


def xlog_scale(log_x_max, scale, log_base= 10): 
    '''Logaritmic scale up to log_alpha_max'''
    bd_block = np.arange(0, log_base**2, log_base) + log_base
    bd_block = bd_block[0:-1]
    xlog = np.tile(bd_block, log_x_max)
    xlog[(log_base-1) : 2*(log_base-1)] = log_base*xlog[(log_base-1) : 2*(log_base-1)]
    for j in range(1, log_x_max - 1):
        xlog[(j+1)*(log_base-1) : (j+2)*(log_base-1)] = log_base*xlog[  j*(log_base-1) :  (j+1)*(log_base-1)  ]
    xlog = np.insert(xlog, 0,  np.arange(1,log_base), axis=0)
    xlog = np.insert(xlog, len(xlog),log_base**(log_x_max+1), axis=0)
    jlog = (xlog*scale).astype(int)
    return jlog
