import numpy as np
from scipy.signal import convolve2d
from time import time

im_full = np.random.randn(228, 228, 32)
kernel_full = np.random.randn(3, 3, 32, 64)
flipped_kernel = np.flip(kernel_full, [0, 1])


def full_conv(im_full, kernel_full):
    out = np.empty((228, 228, 64))

    for out_ch_i in range(64):
        out_ch = np.empty((228, 228))

        for in_ch_i in range(32):
            out_ch += convolve2d(im_full[:, :, in_ch_i],
                                 kernel_full[:, :, in_ch_i, out_ch_i], 'same')
        out[:, :, out_ch_i] = out_ch

    return out


def convolve3d(img, kernel):
    # calc the size of the array of submatracies
    sub_shape = tuple(np.subtract(img.shape, kernel.shape) + 1)

    # alias for the function
    strd = np.lib.stride_tricks.as_strided

    # make an array of submatracies
    submatrices = strd(img, kernel.shape + sub_shape, img.strides * 2)

    # sum the submatraces and kernel
    convolved_matrix = np.einsum('hij,hijklm->klm', kernel, submatrices)

    return convolved_matrix


t1 = time()
out_ch = np.zeros((226, 226))
for in_ch_i in range(32):
    out_ch += convolve2d(im_full[:, :, in_ch_i],
                         flipped_kernel[:, :, in_ch_i, 0], 'valid')
tx = time()-t1

t1 = time()
b = convolve3d(im_full[:, :, :], kernel_full[:, :, :, 0])
tx2 = time()-t1

print(tx/tx2 * 100)

print(np.allclose(out_ch, b))
