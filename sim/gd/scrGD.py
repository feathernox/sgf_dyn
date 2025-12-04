import numpy as np
import itertools
import sys
import logging
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Pool, Manager
import os
import pickle
import argparse
from trainGD import trainGD
import time
##### Fix a seed for reproducibility
np.random.seed(0)


def xlog_scale(log_x_max, scale, log_base=10): 
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


def worker_init(log_queue):
    """
    Called once in each worker process.
    Routes all logging from that process into the shared queue.
    """
    queue_handler = QueueHandler(log_queue)
    root = logging.getLogger()  # process-local root logger
    root.setLevel(logging.INFO)

    # Avoid duplicated handlers if Pool reuses processes
    root.handlers.clear()
    root.addHandler(queue_handler)


def setup_main_logging(log_queue):
    # This handler will be used for BOTH:
    # - records coming from workers via QueueListener
    # - records logged directly in the main process
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(processName)s] %(levelname)s: %(message)s"
    )
    handler.setFormatter(formatter)

    # Attach handler to root logger in the main process
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)

    # Listener will forward records from the queue to the same handler
    listener = QueueListener(log_queue, handler)
    listener.start()
    return listener


def training(p, nS):
    global X, y, beta0, plot_list, n, d, n_it_max, lr

    logger = logging.getLogger(__name__)
    logger.info('Beginning GD training ----- p = %d --- nS = %d' % (p,nS))
    start = time.time()
    np.random.seed(nS)
    res = trainGD(p, nS, lr=lr, X=X, y=y, beta0=beta0, plot_list=plot_list, n=n, d=d, n_it_max=n_it_max)
    end = time.time()
    logger.info('End ----- p = %d --- nS = %d | Time elapsed GD: %f' % (p, nS, end-start))
    return res


if __name__ == '__main__':
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    # p-range
    parser.add_argument("--np_points", type=int, default=50, help="Number of p-grid points")
    parser.add_argument("--intI", type=int, default=0, help="Initial index of p-range")
    parser.add_argument("--intF", type=int, default=7, help="Final index of p-range")

    # parameters
    parser.add_argument("-d", "--d", type=int, default=1000, help="Data dimension")
    parser.add_argument("-n", "--n", type=int, default=400, help="Number of data points")
    parser.add_argument("--nS", type=int, default=1000, help="Number of S random instances")
    parser.add_argument("--snr", type=float, default=1./5, help="Signal-to-noise ratio")
    parser.add_argument("--n_it_max", type=lambda x: int(float(x)), default=int(1e6),
                                            help="Max number of (S)GD iterations")
    parser.add_argument("--only_end", action="store_true", help="Store only the terminal point in the dynamics")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 1/d)")

    args = parser.parse_args()
    if args.lr is None:
        args.lr = 1.0 / args.d

    np_points = args.np_points
    intI = args.intI
    intF = args.intF
    d = args.d
    n = args.n
    nS = args.nS
    snr = args.snr
    n_it_max = args.n_it_max
    only_end = args.only_end
    lr = args.lr

    ##### Data matrix
    X = np.random.randn(n, d) 
    ##### Beta star d-dimension (ground true)
    beta_star_ = np.random.randn(d)
    beta_star = np.copy(beta_star_) / np.linalg.norm(beta_star_)
    ###### Initializing the estimator
    beta0_ = np.random.randn(d)
    beta0  = np.copy(beta0_)/np.linalg.norm(beta0_)
    ##### Targets
    y = X @ beta_star + snr * np.random.randn(n)
    #### Plot
    log_x_max = (np.log10(n_it_max)-1).astype(int)  
    plot_list = xlog_scale(log_x_max, scale=1., log_base= 10)
    if only_end:
        plot_list = plot_list[-1:]
    #### List of p's
    if np.abs(d-1000) > 0:
        d_AUX = 1000
        np_points_AUX = 50
        n_AUX = 400
        p__AUX = np.arange(0, d_AUX , int(d_AUX / np_points_AUX))
        p__AUX[0] = 1
        p_AUX = p__AUX
        alpha_AUX = p_AUX / n_AUX 
        p_ = (alpha_AUX * n).astype(int)
        p_[0] = 1
        p_ = p_[intI:intF]
    else:
        p__ = np.arange(0, d, int(d / np_points))
        p__[0] = 1
        p_ = p__[intI:intF]
    pID = '{:02d}'.format(intI) + '{:02d}'.format(intF) 
    ####################################################################
    #### List of nS
    nS_ = np.arange(0,nS)
    #### Folder results
    subfolder = 'lr%.0e_d%d_n%d_snr%.0e_nS%d_Nmax%.0e_np%d' % (lr, d, n, snr, nS, n_it_max, np_points)
    folder = 'results/GD/' + subfolder
    #### If the folder does not exist, create
    isExist = os.path.exists(folder)
    if not isExist:
        os.makedirs(folder)
    path = folder + '/'


    # --- set up logging via queue ---
    manager = Manager()
    log_queue = manager.Queue()
    listener = setup_main_logging(log_queue)

    log = logging.getLogger(__name__)
    log.info('##### Parameters #####')
    log.info('lr = %.0e' % lr)
    log.info('d = %d' % d)
    log.info('n = %d' % n)
    log.info('snr = %.0e' % snr)
    log.info('nS = %d' % nS)
    log.info('n_it_max = %.0e' % n_it_max)
    log.info('np_points = %d' % np_points)
    log.info('p = %s' % p_)
    log.info('pID = %s' % pID)
    log.info('Subfolder name: %s' % subfolder)

    try:
        with Pool(
            initializer=worker_init,
            initargs=(log_queue,),
        ) as pool:
            # Create an iterator p x nS
            iter = list(itertools.product(p_, nS_))
            # Run in parallel
            result = pool.starmap(training, iter)
    finally:
        listener.stop()

    log.info('   ')
    log.info('##### Saving #####')
    log.info('   ')

    plen = len(p_)
    nSlen = len(nS_)

    count_aux = 0
    for k in np.arange(0,nSlen*plen, nSlen):
        with open(path+'p%d_DYN_betaS.npy' % p_[count_aux], 'wb') as f:
            np.save(f, np.array(result[k:k+nSlen]))
        Sp_ = []
        for nS0 in nS_:
            np.random.seed(nS0)
            Sp_.append(np.random.choice(d, size=p_[count_aux], replace=False))
        with open(path+'p%d_S.npy' % p_[count_aux], 'wb') as f:
            np.save(f, np.array(Sp_))
        count_aux += 1


    plot_list0 = np.insert(plot_list, 0, 0)
    ################
    folder2 = path + 'data'
    isExist = os.path.exists(folder2)
    if not isExist:
        os.makedirs(folder2 )
    path2 = folder2 + '/'
    ################
    with open(path2 + 'beta0.npy', 'wb') as f:
        np.save(f, beta0)   
    with open(path2 + 'betastar.npy', 'wb') as f:
        np.save(f, beta_star)
    with open(path2 + 'plotlist.npy', 'wb') as f:
        np.save(f, plot_list0)
    with open(path2 + 'X.npy', 'wb') as f:
        np.save(f, X)
    with open(path2 + 'y.npy', 'wb') as f:
        np.save(f, y)
    with open(path2 + 'plist.npy', 'wb') as f:
        np.save(f, p_)


    with open(path2 + '__parameters.txt', 'w') as f:
        f.write('lr = %.0e \n' % lr)
        f.write('d = %d \n' % d)
        f.write('n = %d \n' % n)
        f.write('snr = %.0e \n' % snr)
        f.write('nS = %d \n' % nS)
        f.write('n_it_max = %.0e \n' % n_it_max)
        f.write('np_points = %d \n' % np_points)
        f.write('p = %s \n' % p_)
        f.write('pID = %s \n' % pID)

    log.info('   ')
    log.info('##### END #####')
    log.info('   ')
