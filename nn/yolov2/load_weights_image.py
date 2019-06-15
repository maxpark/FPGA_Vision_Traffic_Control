import h5py
import cv2
import numpy as np

# --------------------------- LOAD WEIGHTS -------------------------


'''
Open the weights file
'''
try:
    f.close()
except:
    pass

f = h5py.File('../../large/weights/yolov2.h5')
model_weights = f['model_weights']


def get_weights(name):
    '''
    Return weights based on layer name
    '''
    if 'conv' in name:
        w = [model_weights[name][name]['kernel:0'][()]]
        if len(list(model_weights[name][name])) == 2:
            w += [model_weights[name][name]['bias:0'][()]]
        return w

    elif 'batch_normalization' in name:
        w = [model_weights[name][name]['gamma:0'][()],
             model_weights[name][name]['beta:0'][()],
             model_weights[name][name]['moving_mean:0'][()],
             model_weights[name][name]['moving_variance:0'][()]]
        return w

# -------------------------LOAD IMAGE------------------------------


image_path = '5.png'
image = cv2.imread(image_path)

input_image = cv2.resize(image, (416, 416))
input_image = input_image / 255.
input_image = input_image[:, :, ::-1]
input_image = np.expand_dims(input_image, 0)
