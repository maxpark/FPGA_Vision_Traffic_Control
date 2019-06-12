import os
from multiprocessing import Process

import numpy as np
from scipy.signal import convolve2d

im_full = np.random.randn(228, 228, 32)
kernel_full = np.random.randn(3, 3, 32, 64)


def conv_out(im_full, kernel_full, out, out_ch_i):
    out_ch = np.empty((228, 228))

    for in_ch_i in range(32):
        out_ch += convolve2d(im_full[:, :, in_ch_i],
                             kernel_full[:, :, in_ch_i, out_ch_i], 'same')

    out[:, :, out_ch_i] = out_ch


def loop_conv(im_full, kernel_full):
    out = np.empty((228, 228, 64))

    for out_ch_i in range(64):
        conv_out(im_full, kernel_full, out, out_ch_i)

    return out


# if __name__ == '__main__':
#     procs = []

#     for i in range(64):
#         proc = Process(target=)


# def doubler(number):
#     """
#     A doubling function that can be used by a process
#     """
#     result = number * 2
#     proc = os.getpid()
#     print('{0} doubled to {1} by process id: {2}'.format(number, result, proc))


# if __name__ == '__main__':
#     numbers = [5, 10, 15, 20, 25]
#     procs = []

#     for index, number in enumerate(numbers):
#         proc = Process(target=doubler, args=(number,))
#         procs.append(proc)
#         proc.start()

#     for proc in procs:
#         proc.join()

# from multiprocessing import Process, Lock


# def printer(item, lock):
#     """
#     Prints out the item that was passed in
#     """
#     lock.acquire()
#     try:
#         print(item)
#     finally:
#         lock.release()

# if __name__ == '__main__':
#     lock = Lock()
#     items = ['tango', 'foxtrot', 10]
#     for item in items:
#         p = Process(target=printer, args=(item, lock))
#         p.start()
