import numpy as np
import matplotlib.pyplot as plt


def show_hist(a, color):
    n, bins, patches = plt.hist(
        a.flatten(),
        100, facecolor=color, alpha=0.5)
    plt.show()


def e(a1, a2):
    '''
    Returns the L1 error (sum of absolute error between two ndarrays)
    '''
    print(np.sum(np.abs(np.asarray(a1)-np.asarray(a2))))


def eval(keras_t):
    '''
    keras_layer.output is a tensor. It should be evaluated through a sess.run()

    Args:
        keras_t : tensor

    Return: ndarray
    '''
    return keras.backend.get_session().run(keras_t, feed_dict={'input_1:0': input_image})
