try:
    import sys
    import os
    sys.path.append("../framework")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    from MyLayers import MyLayer, MyInput, MyConv, MyMaxPool, MyConcat, MySpaceToDepth, MyModel, MyLeakyRelu
    from MyUtils import e, eval
    from load_weights_image import get_weights, image, input_image
    from preprocessing import parse_annotation, BatchGenerator
    from utils import decode_netout, draw_boxes

except:
    pass


m = MyModel()
dtype_str = 'float16'

layer_name = 'input'
m.d[layer_name] = MyInput(input_image, name=layer_name)


for i in range(1, 21):
    prev_name = layer_name
    layer_name = 'conv2d_' + str(i)
    bn_name = 'batch_normalization_' + str(i)
    m.d[layer_name] = MyConv(prev_layer=m.d[prev_name],
                             weights_biases=get_weights(layer_name),
                             bn_weights=get_weights(bn_name),
                             name=layer_name,
                             dtype_str=dtype_str)

    prev_name = layer_name
    layer_name = 'lrelu_' + str(i)
    m.d[layer_name] = MyLeakyRelu(prev_layer=m.d[prev_name],
                                  name=layer_name,
                                  dtype_str=dtype_str)

    if i in [1, 2, 5, 8, 13]:
        prev_name = layer_name
        layer_name = 'maxpool_' + str(i)
        m.d[layer_name] = MyMaxPool(prev_layer=m.d[prev_name],
                                    name=layer_name,
                                    pool_size=(2, 2),
                                    dtype_str=dtype_str)
i = 21
prev_name = 'lrelu_13'
layer_name = 'conv2d_' + str(i)
bn_name = 'batch_normalization_' + str(i)
m.d[layer_name] = MyConv(prev_layer=m.d[prev_name],
                         weights_biases=get_weights(layer_name),
                         bn_weights=get_weights(bn_name),
                         name=layer_name,
                         dtype_str=dtype_str)

prev_name = layer_name
layer_name = 'lrelu_' + str(i)
m.d[layer_name] = MyLeakyRelu(prev_layer=m.d[prev_name],
                              name=layer_name,
                              dtype_str=dtype_str)

prev_name = layer_name
layer_name = 'space2depth'
m.d[layer_name] = MySpaceToDepth(prev_layer=m.d[prev_name],
                                 name=layer_name,
                                 dtype_str=dtype_str)

layer_name = 'concat'
m.d[layer_name] = MyConcat(prev_layers=[m.d['space2depth'], m.d['lrelu_20']],
                           name=layer_name,
                           dtype_str=dtype_str)

i = 22
prev_name = layer_name
layer_name = 'conv2d_' + str(i)
bn_name = 'batch_normalization_22'
m.d[layer_name] = MyConv(prev_layer=m.d[prev_name],
                         weights_biases=get_weights(layer_name),
                         bn_weights=get_weights(bn_name),
                         name=layer_name,
                         dtype_str=dtype_str)

prev_name = layer_name
layer_name = 'lrelu_' + str(i)
m.d[layer_name] = MyLeakyRelu(prev_layer=m.d[prev_name],
                              name=layer_name,
                              dtype_str=dtype_str)

i = 23
prev_name = layer_name
layer_name = 'conv2d_' + str(i)
m.d[layer_name] = MyConv(prev_layer=m.d[prev_name],
                         weights_biases=get_weights(layer_name),
                         name=layer_name,
                         dtype_str=dtype_str)

m.output_name = 'conv2d_23'


def encode(in_data):
    return in_data


def decode(in_data):
    return in_data


m.set_encode_decode(encode, decode)

m_out = m.get_output_data()
