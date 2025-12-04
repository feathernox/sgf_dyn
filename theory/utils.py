import ast
import logging
import sys
from logging.handlers import QueueHandler, QueueListener
import numpy as np


def xlog_scale(log_x_max, scale, log_base=10): 
    '''Logaritmic scale up to log_alpha_max'''
    bd_block = np.arange(0, log_base ** 2, log_base) + log_base
    bd_block = bd_block[0:-1]
    xlog = np.tile(bd_block, log_x_max)
    xlog[(log_base-1) : 2*(log_base-1)] = log_base*xlog[(log_base-1) : 2*(log_base-1)]
    for j in range(1, log_x_max - 1):
        xlog[(j+1)*(log_base-1) : (j+2)*(log_base-1)] = log_base*xlog[  j*(log_base-1) :  (j+1)*(log_base-1)  ]
    xlog = np.insert(xlog, 0,  np.arange(1,log_base), axis=0)
    xlog = np.insert(xlog, len(xlog),log_base ** (log_x_max + 1), axis=0)
    jlog = (xlog * scale).astype(int)
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


def parse_config(path):
    config = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Try evaluating numbers, lists, scientific notation, etc.
            try:
                value = ast.literal_eval(value)
            except Exception:
                # fallback to string
                pass

            config[key] = value

    return config

