if True:    # To prevent reordering of imports
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import sys
    sys.path.append("../framework")
    import os
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    from MyLayers import MyLayer, MyInput, MyConv, MyMaxPool, MyConcat, MySpaceToDepth, MyModel, MyLeakyRelu
    from MyUtils import e, eval
    from load_weights_image import get_weights, image, input_image

    from preprocessing import parse_annotation, BatchGenerator
    from utils import decode_netout, draw_boxes


class YOLO_v2_Numpy():
    def __init__(self):

        self.model = MyModel()
        self.output = None
        self.dtype_str = 'float16'
        self.image = image

        self.compile()

    def compile(self):
        layer_name = 'input'
        dtype_str = self.dtype_str
        self.model.d[layer_name] = MyInput(input_image, name=layer_name)

        for i in range(1, 21):
            prev_name = layer_name
            layer_name = 'conv2d_' + str(i)
            bn_name = 'batch_normalization_' + str(i)
            self.model.d[layer_name] = MyConv(prev_layer=self.model.d[prev_name],
                                              weights_biases=get_weights(
                                                  layer_name),
                                              bn_weights=get_weights(bn_name),
                                              name=layer_name,
                                              dtype_str=dtype_str)

            prev_name = layer_name
            layer_name = 'lrelu_' + str(i)
            self.model.d[layer_name] = MyLeakyRelu(prev_layer=self.model.d[prev_name],
                                                   name=layer_name,
                                                   dtype_str=dtype_str)

            if i in [1, 2, 5, 8, 13]:
                prev_name = layer_name
                layer_name = 'maxpool_' + str(i)
                self.model.d[layer_name] = MyMaxPool(prev_layer=self.model.d[prev_name],
                                                     name=layer_name,
                                                     pool_size=(2, 2),
                                                     dtype_str=dtype_str)
        i = 21
        prev_name = 'lrelu_13'
        layer_name = 'conv2d_' + str(i)
        bn_name = 'batch_normalization_' + str(i)
        self.model.d[layer_name] = MyConv(prev_layer=self.model.d[prev_name],
                                          weights_biases=get_weights(
                                              layer_name),
                                          bn_weights=get_weights(bn_name),
                                          name=layer_name,
                                          dtype_str=dtype_str)

        prev_name = layer_name
        layer_name = 'lrelu_' + str(i)
        self.model.d[layer_name] = MyLeakyRelu(prev_layer=self.model.d[prev_name],
                                               name=layer_name,
                                               dtype_str=dtype_str)

        prev_name = layer_name
        layer_name = 'space2depth'
        self.model.d[layer_name] = MySpaceToDepth(prev_layer=self.model.d[prev_name],
                                                  name=layer_name,
                                                  dtype_str=dtype_str)

        layer_name = 'concat'
        self.model.d[layer_name] = MyConcat(prev_layers=[self.model.d['space2depth'], self.model.d['lrelu_20']],
                                            name=layer_name,
                                            dtype_str=dtype_str)

        i = 22
        prev_name = layer_name
        layer_name = 'conv2d_' + str(i)
        bn_name = 'batch_normalization_22'
        self.model.d[layer_name] = MyConv(prev_layer=self.model.d[prev_name],
                                          weights_biases=get_weights(
                                              layer_name),
                                          bn_weights=get_weights(bn_name),
                                          name=layer_name,
                                          dtype_str=dtype_str)

        prev_name = layer_name
        layer_name = 'lrelu_' + str(i)
        self.model.d[layer_name] = MyLeakyRelu(prev_layer=self.model.d[prev_name],
                                               name=layer_name,
                                               dtype_str=dtype_str)

        i = 23
        prev_name = layer_name
        layer_name = 'conv2d_' + str(i)
        self.model.d[layer_name] = MyConv(prev_layer=self.model.d[prev_name],
                                          weights_biases=get_weights(
                                              layer_name),
                                          name=layer_name,
                                          dtype_str=dtype_str)

        self.model.output_name = 'conv2d_23'

    def fwd_pass(self):
        self.output = self.model.get_np_output_data()

    def show_on_image(self):
        LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
                  'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                  'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                  'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                  'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                  'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                  'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                  'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                  'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                  'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        CLASS = len(LABELS)
        OBJ_THRESHOLD = 0.3  # 0.5
        NMS_THRESHOLD = 0.3  # 0.45
        ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
                   5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
        dummy_array = np.zeros((1, 1, 1, 1, 50, 4))

        m_out_r = np.reshape(self.output, (1, 13, 13, 5, 85))
        boxes = None
        boxes = decode_netout(m_out_r[0],
                              obj_threshold=OBJ_THRESHOLD,
                              nms_threshold=NMS_THRESHOLD,
                              anchors=ANCHORS,
                              nb_class=CLASS)
        image = draw_boxes(self.image, boxes, labels=LABELS)
        cv2.imwrite('detected_2.png', image)
        fig = plt.figure(figsize=(20, 20))
        plt.imshow(image[:, :, ::-1])
        plt.show()

    def show_hist(self):
        for name in self.model.d.keys():
            print('\n\n\nLAYER: ', name)
            self.model.d[name].show_hist()
