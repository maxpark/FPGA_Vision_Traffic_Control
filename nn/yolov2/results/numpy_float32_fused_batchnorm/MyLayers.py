from scipy.signal import convolve2d
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


#--------------------------- MODEL --------------------------------

class MyModel:
    '''
    Attributes:
        d : dict
            - Dictionary of {'layer_name': MyLayerObject} pairs
        
        output_name : str
            - Name of the output layer. Used to start the recursive call
    
    Methods:
        get_np_output_data()
        get_keras_output_data() : ndarray
            - Call the get_np/keras_output_recursively() fucntion of the output layer
            - Which will call the same fucntion of its prev_layer(s)
            - This call will go like a chain
            - When the fucntion of input layer is called, it simply returns the image,
                without calling anyone else
            - The results are passed through the np_out() of each layer
                as each recursive call returns
        np/keras_reset()
            - The output_lock of each layer is changed to True when the 
                layer computes an output. This is to prevent wasting time
                in recomputing if we want the values again.
            - If we want to recompute the results (say for a different image),
                this function will turn off (False) output_lock of all layers
            - So when we call output again, the output will be freshly calculated
    '''
    def __init__(self):
        self.d = {}
        self.input_name = None
        self.output_name = None
        
    def get_np_output_data(self):
        return self.d[self.output_name].get_np_output_recursively()
    def get_keras_output_data(self):
        return self.d[self.output_name].get_keras_output_recursively()
    def np_reset(self):
        for name in self.d.keys():
            self.d[name].np_output_lock = False
    def keras_reset(self):
        for name in self.d.keys():
            self.d[name].keras_output_lock = False


class MyLayer:
    '''
    * Base class for all Layers: Conv, BatchNorm
    * Can show histogram of outputs & weights
    * Can compare different implementations
    
    Args:
        name (str)
        dtype_str (str)
        
    Attributes:
        name (str)
        np_dtype, tf_dtype

        prev_layer : MyLayer
            - Helps to chain the layers and get
                results from previous layers

        np_output, keras_output : ndarray
            -initally None, stores the value of 
                calculation result from np_out()/keras_out()

        np_output_lock, keras_ouput_lock : bool
            - False by default
            - When output is calculated, set to True
            - When true, get_output_recursively()
                doesnt recalculate results. it returns
                previously calculated value from
                np_output or keras_output

    Methods:
        set_dtype():
            - Sets the default types for np, tf, keras

        compare (in_data: ndarray 4 dims):
            - compares implementations in
                numpy, keras, tf (if avail)
                
        show_hist(in_data: ndarray 4 dims):
            - shows histograms of layer
                weights
                biases (if not zero)
                outputs (if in_data is not None or if np_out is available)

        get_np_output_recursively()
        get_np_output_recursively()
            - Recursively call the get_np/keras_output_recursively
                method(s) of the prev_layer(s),
            - Feed the output(s) to np/keras_out() of current layer
            - Return the output
            - If output_lock is placed (True), 
                return np/keras_output without 
                calculation or recursive call
            - Overriden under MyConcat child class to allow multiple inputs

        np_out(in_data: nd_array) : ndarray
        keras_out(in_data: nd_array) : ndarray
        tf_out(in_data: nd_array) : ndarray
            - 3 methods defined inside each child class
            - Apply the current layer to input array and
                return the result
                
    '''
    def __init__(self, prev_layer, name, dtype_str):
        self.set_dtype(dtype_str)
        self.name = name
        self.prev_layer = prev_layer
        
        self.np_output_lock = False
        self.keras_output_lock = False
        
    def set_dtype(self, dtype_str):
        dtype_dict = {
            'float32': [np.float32, tf.float32],
            'float64': [np.float64, tf.float64],
            'float16': [np.float64, tf.float64]
        }
        self.np_dtype = dtype_dict[dtype_str][0]
        self.tf_dtype = dtype_dict[dtype_str][1]
        
        keras.backend.set_floatx(dtype_str)
        
    def show_hist(self, in_data = None):
        if isinstance(self, MyConv):
            print('Weights')
            n, bins, patches = plt.hist(self.weights.flatten(), 100, facecolor='blue', alpha=0.5)
            plt.show()

            if not np.array_equal(self.biases, np.zeros(self.biases.shape)):
                print('\nBiases')
                n, bins, patches = plt.hist(self.biases.flatten(), 100, facecolor='green', alpha=0.5)
                plt.show()
        if self.np_output_lock:
            print('\nOutputs')
            n, bins, patches = plt.hist(self.np_out_data.flatten(), 100, facecolor='red', alpha=0.5)
            plt.show()

        if in_data is not None:
            out = self.np_out(in_data)
            print('\nOutputs')
            n, bins, patches = plt.hist(out.flatten(), 100, facecolor='red', alpha=0.5)
            plt.show()
            
    def compare(self, in_data):
        np_out = self.np_out(in_data)
        keras_out = self.keras_out(in_data)
        
        print('\nnp vs keras ALLCLOSE: ', np.allclose(np_out, keras_out))
        print('np vs keras abs error: ', np.sum(np.abs(np_out - keras_out)))
        
        if hasattr(self, 'tf_out'):
            tf_out = self.tf_out(in_data)
            print('np vs tf ALLCLOSE: ', np.allclose(np_out, tf_out))
            print('np vs tf abs error: ', np.sum(np.abs(np_out - tf_out)))
        
            print('\ntf vs keras ALLCLOSE: ', np.allclose(tf_out, keras_out))
            print('tf vs keras abs error: ', np.sum(np.abs(tf_out - keras_out)))
    
    def get_np_output_recursively(self):
        if self.np_output_lock:
            return self.np_out_data
        else:
            self.np_output_lock = True
            
            out = self.np_out(self.prev_layer.get_np_output_recursively())
            print('Evaluated Layer: ' + self.name)
            return out
    
    def get_keras_output_recursively(self):
        if self.keras_output_lock:
            return self.keras_out_data
        else:
            self.keras_output_lock = True
            out = self.keras_out(self.prev_layer.get_keras_output_recursively())
            print('Evaluated Layer: ' + self.name)
            return out

#---------------------- CONVOLUTION ---------------------------------------

class MyConv(MyLayer):
    '''
    * Numpy, TF, Keras Based Implementations of Convolution Layer
    * Support Multiple dtypes (Defined in Class:Layer)
    * Optionally allows fused Batch Normalization

    Fused Batch Norm:
        Batch norm is: output = beta + gamma(input - mean)/sqrt(var + epsilon)
            beta, gamma - learnt paramters
            mean, var   - of all batches from training data
            epsilon     - small constant to prevent division by zero

            During inference time, all 5 parameters are known

        Fusing Batch Norm:
            since input  = weights * x + bias,
                  output = beta + gamma(weights * x + bias - mean)/sqrt(var + epsilon)
            Let sigma = sqrt(var + epsilon), then:
                  output = [gamma*weights/sigma]*x + [beta+(bias-mean)*gamma/sigma]
            Therefore, by setting:
                  weights <- [gamma*weights/sigma]
                  bias    <- [beta+(bias-mean)*gamma/sigma]
            Convolution + BatchNorm layers can be fused
                into (are equivalent to) a single convoluion
                layer with above modified weights and biases

            Only possible during inference (when all 5 are fixed constants)
            Tremendously reduces number of calculations
    
    Args:
        weights_biases : list [weights: ndarray, biases: ndarray]
            - weights, biases are numpy arrays of 4 dims
            - unflipped (tf format)
            
        bn_weights :list [gamma, beta, mean, variance]):
            - each 1 dim, shape = (out_ch_n,)
            - if None, batch_norm not applied (not fused)

    Attributes:
        is_fused_bn : bool
        weights, biases : ndarray
        weights_fipped, biases_flipped : ndarray
            - Scipy's convolve2d flips weights before convolving
                (mathematical definition for convolution)
            - Tensorflow / Keras doesnt flip the weights
                (Machine learning convention of convolving)
            - To create similar results, we pre-flip the kernel and store here

        pure_weights, pure_biases : ndarray 
            if is_fused_bn, true_weights are stored here
            and fused weights are stored in weights, biases
            
        gamma, beta, mean, variance : ndarray
        epsilon : float
        
        kernel : tuple , eg (3,3)
        in_ch_n : int
        out_ch_n : int
            Number of input and output channels (filters)
        
        np_out_data : ndarray
        tf_out_data : ndarray
        keras_out_data : ndarray
        
    Methods:
        np_out(in_data : ndarray 4 dims): ndarray
        tf_out(in_data : ndarray 4 dims): ndarray
        keras_out(in_data : ndarray 4 dims): ndarray
        
        fuse_bn(bn_weights: list, epsilon: int)
            Performs the BN fusions
    
    NOTE:
        - Have not generalized for multiple images yet
        - Keras BatchNorm accepts only float32
    '''
    def __init__(self, 
                 weights_biases, 
                 prev_layer = None,
                 bn_weights = None, 
                 name = '',
                 dtype_str = 'float32'):
        
        MyLayer.__init__(self,
                       name = name, 
                       prev_layer = prev_layer,
                       dtype_str = dtype_str)
        
        assert len(weights_biases[0].shape) == 4
        # Set Weights and Biases
        
        self.weights = weights_biases[0].astype(self.np_dtype)
        self.weights_flipped = np.flip(self.weights, [0,1])
        
        self.kernel  = self.weights.shape[0:2]
        self.in_ch_n  = self.weights.shape[2]
        self.out_ch_n = self.weights.shape[3]


        if len(weights_biases) > 1:
            self.biases = weights_biases[1].astype(self.np_dtype)
        else:
            self.biases = np.zeros((self.out_ch_n), dtype = self.np_dtype)
        
        # Fusing Batch Normalization
        
        if bn_weights is None:
            self.is_fused_bn = False
        else:
            self.fuse_bn(bn_weights)
            
    def np_out(self,in_data):
        
        assert len(in_data.shape) == 4
        n_samples, in_h, in_w, in_data_ch_n = in_data.shape
        in_data = in_data.astype(self.np_dtype)
        
        assert in_data_ch_n == self.in_ch_n
        
        out = np.empty((n_samples,in_h, in_w, self.out_ch_n), dtype = self.np_dtype)
        
        for out_ch_i in range(self.out_ch_n):
            out_ch = np.zeros((in_h, in_w))
            for in_ch_i in range(self.in_ch_n):                
                out_ch += convolve2d(in_data[0,:,:,in_ch_i], 
                                     self.weights_flipped[:,:,
                                            in_ch_i,
                                            out_ch_i], 
                                     mode ='same')
                
            out_ch += self.biases[out_ch_i]
            out[0,:,:,out_ch_i] = out_ch
            
        self.np_out_data = out
        
        return self.np_out_data
    
    def fuse_bn(self, bn_weights, epsilon = 0.001):
        gamma, beta, mean, variance = bn_weights
        assert gamma.shape == beta.shape == mean.shape == variance.shape == (self.out_ch_n,)
        
        self.gamma = gamma.astype(self.np_dtype)
        self.beta  = beta.astype(self.np_dtype)
        self.mean  = mean.astype(self.np_dtype)
        self.variance   = variance.astype(self.np_dtype)
        self.epsilon = epsilon
        
        scale = self.gamma / np.sqrt(self.variance + self.epsilon)
        
        self.pure_weights = self.weights.copy()
        self.pure_biases = self.biases.copy()
        
        self.weights = self.weights * scale
        self.weights_flipped = np.flip(self.weights, [0,1])
        self.biases = beta + scale * (self.biases - self.mean)
        self.is_fused_bn = True
    
    def tf_out(self,in_data):
        
        if self.is_fused_bn:
            kernel_t = tf.convert_to_tensor(self.pure_weights, dtype=self.tf_dtype)
            bias_t = tf.convert_to_tensor(self.pure_biases, dtype=self.tf_dtype)
        else:
            kernel_t = tf.convert_to_tensor(self.weights, dtype=self.tf_dtype)
            bias_t = tf.convert_to_tensor(self.biases, dtype=self.tf_dtype)
            
        in_t = tf.convert_to_tensor(in_data, dtype=self.tf_dtype)
        out_t = tf.nn.conv2d(in_t,kernel_t,[1,1,1,1],"SAME")
        out_t = tf.nn.bias_add(out_t, bias_t)
        
        if self.is_fused_bn:
            out_t = tf.nn.batch_normalization(out_t,
                                              mean = self.mean,
                                              variance = self.variance,
                                              offset= self.beta,
                                              scale = self.gamma,
                                              variance_epsilon = self.epsilon,
                                              name=None)
        
        sess = keras.backend.get_session()
        
        self.tf_out_data = sess.run(out_t)
        
        return self.tf_out_data
    
    def keras_out(self,in_data):         
        
        input_image = keras.layers.Input(shape=in_data.shape[1:4], name='input_image')
        x = keras.layers.Conv2D(self.out_ch_n, 
                                self.kernel, 
                                strides=(1,1), 
                                padding='same', 
                                name='conv_keras', 
                                use_bias=True)(input_image)
        
        if self.is_fused_bn:
            x = keras.layers.BatchNormalization(name='norm_keras')(x)
        
        model = keras.models.Model(input_image, x)
        conv_keras_layer = model.get_layer('conv_keras')
        
        if self.is_fused_bn:
            conv_keras_layer.set_weights([self.pure_weights, self.pure_biases])
            norm_keras_layer = model.get_layer('norm_keras')
            norm_keras_layer.set_weights([self.gamma, 
                                          self.beta,
                                          self.mean,
                                          self.variance])
            out_layer = norm_keras_layer
        else:
            conv_keras_layer.set_weights([self.weights, self.biases])
            out_layer = conv_keras_layer
        
        sess = keras.backend.get_session()
        
        self.keras_out_data = sess.run(out_layer.output, 
                                       feed_dict = {
                                           model.inputs[0].op.name+':0': in_data})
        
        return self.keras_out_data
        
#----------------------- OTHER LAYERS ---------------------------------------

class MyInput(MyLayer):
    '''
    The first layer for any custom Model.
    prev_layer is always None

    get_np/keras_output_recursively() 
        - Overidden (from parent class) here
        - Simply returns the image, ending the recursive call
    '''

    def __init__(self, input_image, name = 'input', dtype_str = 'float32'):
        MyLayer.__init__(self, prev_layer = None, 
                       name = name, dtype_str = dtype_str)
        self.input_image = input_image
        
    def get_np_output_recursively(self):
        return self.input_image
    def get_keras_output_recursively(self):
        return self.input_image
    
class MyLeakyRelu(MyLayer):
    def __init__(self, prev_layer = None, alpha=0.1, name = 'input', dtype_str = 'float32'):
        
        MyLayer.__init__(self, prev_layer = prev_layer, 
                       name = name, dtype_str = dtype_str)
        self.alpha = alpha
        
    def np_out(self, in_data):
        x = in_data
        self.np_out_data = x* ((x > 0) + (x < 0) * self.alpha)
        
        return self.np_out_data
    
    def keras_out(self, in_data):
        in_data_t = keras.layers.Input(shape=in_data.shape[1:4], name='in_data')
        x = keras.layers.LeakyReLU(alpha=self.alpha, name = 'leaky_relu_keras')(in_data_t)
        model = keras.models.Model(in_data_t, x)
        
        leaky_relu_keras_layer = model.get_layer('leaky_relu_keras')
        
        sess = keras.backend.get_session()
        
        self.keras_out_data = sess.run(leaky_relu_keras_layer.output, 
                                       feed_dict = {model.inputs[0].op.name+':0': in_data})
        return self.keras_out_data
        
class MySpaceToDepth(MyLayer):
    '''
    Tensorflow's tf.space_to_depth behavior
    Reduces the size of spacial dimensions and puts elements in the channel dimension
    Don't worry about it
    '''
    def __init__(self, prev_layer = None, block_size = 2, name = '', dtype_str = 'float32'):
        MyLayer.__init__(self, prev_layer = prev_layer, 
                       name = name, dtype_str = dtype_str)
        
        self.block_size = block_size
        
        self.keras_out = self.tf_out # Cannot implement in keras
    
    def np_out(self,in_data):
        
        batch, height, width, depth = in_data.shape
        reduced_height = height // self.block_size
        reduced_width = width // self.block_size
        
        y = in_data.reshape(batch, reduced_height, self.block_size,
                             reduced_width, self.block_size, depth)
        self.np_out_data = np.swapaxes(y, 2, 3).reshape(batch, reduced_height, reduced_width, -1)
        return self.np_out_data
    
    def tf_out(self,in_data):
        in_data_t = tf.convert_to_tensor(in_data, dtype=self.tf_dtype)
        x = tf.space_to_depth(in_data_t, self.block_size)
        self.tf_out_data = keras.backend.get_session().run(x)
        return self.tf_out_data
    
class MyConcat(MyLayer):
    '''
    Concats a list of input layers (their outputs) along the channel dimension
    get_np/keras_output_recursively() are overidden to work with a list of prev_layers
    '''
    def __init__(self, prev_layers = None,
                 name = '', dtype_str = 'float32'):
        
        MyLayer.__init__(self, prev_layer = None, 
                       name = name, dtype_str = dtype_str)
        
        self.prev_layers = prev_layers
        
    def np_out(self,in_data_list):
        self.np_out_data = np.concatenate(in_data_list, axis=-1)
        return self.np_out_data
        
    def keras_out(self,in_data_list):
        in_data_t_list = [keras.layers.Input(shape=in_data.shape[1:4])
                          for in_data in in_data_list]
        
        x = keras.layers.merge.concatenate(in_data_t_list, name = 'concat_keras')
        model = keras.models.Model(in_data_t_list, x)
        
        feed_dict = {}
        for i in range(len(model.inputs)):
            feed_dict[model.inputs[i].op.name+':0'] = in_data_list[i]
        
        sess = keras.backend.get_session()
        concat_keras_layer = model.get_layer('concat_keras')
        self.keras_out_data = sess.run(concat_keras_layer.output, 
                                       feed_dict = feed_dict)
        return self.keras_out_data
    
    def get_np_output_recursively(self):
        if self.np_output_lock:
            return self.np_out_data
        else:
            self.np_output_lock = True
            in_data_list = [prev_layer.get_np_output_recursively()
                           for prev_layer in self.prev_layers]
            return self.np_out(in_data_list)
    
    def get_keras_output_recursively(self):
        if self.keras_output_lock:
            return self.keras.out_data
        else:
            self.np_output_lock = True
            in_data_list = [prev_layer.get_keras_output_recursively()
                           for prev_layer in self.prev_layers]
            return self.keras_out(in_data_list)
    
class MyMaxPool(MyLayer):
    def __init__(self, prev_layer = None,
                 pool_size = (2,2),
                 name = '', dtype_str = 'float32'):
        
        MyLayer.__init__(self, prev_layer = prev_layer, 
                       name = name, dtype_str = dtype_str)
        self.pool_size = pool_size
        
    def np_out(self,in_data):
        batch, height, width, depth = in_data.shape
        reduced_height = height // self.pool_size[0]
        reduced_width = width // self.pool_size[1]
        
        self.np_out_data = in_data.reshape(batch, reduced_height, self.pool_size[0],
                             reduced_width, self.pool_size[1], depth)
        self.np_out_data = self.np_out_data.max(axis=2).max(axis=3)
        
        return self.np_out_data
        
    def keras_out(self,in_data):
        in_data_t = keras.layers.Input(shape=in_data.shape[1:4])
        
        x = keras.layers.MaxPooling2D(pool_size = self.pool_size, name = 'out_keras')(in_data_t)
        model = keras.models.Model(in_data_t, x)
        
        sess = keras.backend.get_session()
        out_layer = model.get_layer('out_keras')
        self.keras_out_data = sess.run(out_layer.output, 
                                       feed_dict = {model.inputs[0].op.name+':0': in_data})
        return self.keras_out_data

            
            
#---------------------------- PURE BATCH NORM----------------------

class MyBatchNorm(MyLayer):
    '''
    Dont worry about this

    Batch norm is: output = beta + gamma(input - mean)/sqrt(var + epsilon)
            beta, gamma - learnt paramters
            mean, var   - of all batches from training data
            epsilon     - small constant to prevent division by zero

            During inference time, all 5 parameters are known
    '''
    def __init__(self, weights, prev_layer = None, epsilon = 0.001, name = '', dtype_str = 'float32'):
        MyLayer.__init__(self, name = name, prev_layer = prev_layer, dtype_str = dtype_str)
        
        gamma, beta, mean, variance = weights
        assert gamma.shape == beta.shape == mean.shape == variance.shape
        
        self.gamma = gamma.astype(self.np_dtype)
        self.beta  = beta.astype(self.np_dtype)
        self.mean  = mean.astype(self.np_dtype)
        self.variance   = variance.astype(self.np_dtype)
        self.epsilon = epsilon
        
    def np_out(self, in_data):
        in_data = in_data.astype(self.np_dtype)
        
        self.sigma = np.sqrt(self.variance + self.epsilon)
        
        out = self.gamma * (in_data - self.mean)/self.sigma + self.beta
        assert out.dtype == self.np_dtype
        
        return out
    
    def np_out2(self,in_data):
        self.sigma = np.sqrt(self.variance + self.epsilon)
        A = self.gamma / self.sigma
        B = self.beta - A * self.mean
        
        out = A * in_data + B
        assert out.dtype == self.np_dtype
        return out
    
    def keras_out(self,in_data):         
        
        input_data = Input(shape=in_data.shape[1:4], name='input_data')
        bn = keras.layers.BatchNormalization(name='bn_keras')(input_data)
        model = keras.models.Model(input_data, bn)
        bn_keras_layer = model.get_layer('bn_keras')
        bn_keras_layer.set_weights([self.gamma,
                                    self.beta,
                                    self.mean,
                                    self.variance
                                   ])
        sess = keras.backend.get_session()
        
        out = sess.run(bn_keras_layer.output, feed_dict = {model.inputs[0].op.name+':0': in_data})
        
        return out
    