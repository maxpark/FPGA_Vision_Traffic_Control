import numpy as np
from multiprocessing import Process, Lock
import multiprocessing
import logging
import os
from scipy.signal import convolve2d
from time import time


def abc(b, i, lock):
    if lock != None:
        lock.acquire()
    b[i] = 15
    if lock != None:
        lock.release()
    print("process_id:", os.getpid(), "\ti:", i)


def bcd(b, i, return_list):
    return_list.append(b[i]**2)
    print("process_id:", os.getpid(), "\ti:", i)


def with_multi():
    procs = []
    lock = Lock()

    a = [1, 2, 3]

    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    for i in range(3):
        proc = Process(target=abc, args=(a, i, lock,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
    print("Len procs: ", len(procs), "\ta:", a)


def with_multi_2():
    a = [1, 2, 3]
    procs = []
    manager = multiprocessing.Manager()
    return_list = manager.list()
    for i in range(3):
        proc = Process(target=bcd, args=(a, i, return_list))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
    print(return_list)


def without_multi():
    a = [1, 2, 3]
    for i in range(3):
        abc(a, i, None)
    print(a)


im_full = np.random.randn(228, 228, 32)
kernel_full = np.random.randn(3, 3, 32, 64)

times = []


def full_conv(im_full, kernel_full):
    out = np.empty((228, 228, 64))

    for out_ch_i in range(64):
        out_ch = np.empty((228, 228))

        t1 = time()
        for in_ch_i in range(32):
            out_ch += convolve2d(im_full[:, :, in_ch_i],
                                 kernel_full[:, :, in_ch_i, out_ch_i], 'same')
        times.append(time()-t1)
        out[:, :, out_ch_i] = out_ch

    return out


def in_conv(im_full, kernel_full, in_ch_i, out_ch_i, return_list):
    return_list.append(convolve2d(im_full[:, :, in_ch_i],
                                  kernel_full[:, :, in_ch_i, out_ch_i], 'same'))


def multi_conv(im_full, kernel_full):
    out = np.empty((228, 228, 64))

    for out_ch_i in range(64):
        procs = []
        manager = multiprocessing.Manager()
        return_list = manager.list()

        t1 = time()
        for in_ch_i in range(32):
            proc = Process(target=in_conv, args=(im_full, kernel_full,
                                                 in_ch_i, out_ch_i, return_list))
            procs.append(proc)
            proc.start()
            # out_ch += convolve2d(im_full[:, :, in_ch_i],
            #                      kernel_full[:, :, in_ch_i, out_ch_i], 'same')
        for proc in procs:
            proc.join()

        times.append(time()-t1)
        out[:, :, out_ch_i] = np.sum(return_list, 0)

    return out


if __name__ == "__main__":
    # with_multi_2()
    t1 = time()
    multi_conv(im_full, kernel_full)
    # full_conv(im_full, kernel_full)
    print(time()-t1)
    print(np.mean(times))
