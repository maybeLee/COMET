from keras.layers.core import Lambda
from tensorflow.python.ops.gen_linalg_ops import Lu


class LayerStack:
    def __init__(self):
        self._layer_stack = []

    def add(self, layer):
        import keras
        if layer is not None:
            if int(keras.__version__.split(".")[1]) >= 7:
                layer._name += "_insert"
            else:
                layer.name += "_insert"
            self._layer_stack.append(layer)

    def get_layers(self, ):
        return self._layer_stack


class LayerUtils:
    parameter_size_limit = 1e6

    # class for atom layers
    # all atom layers are stored in `self.available_model_level_layers`
    def __init__(self):
        # these layers take effect both for training and testing
        self.available_model_level_layers = {}
        # these layers only take effect for training
        self.available_source_level_layers = {}
        self.is_input_legal = {}
        #
        # self.available_model_level_layers["attention-1"] = LayerUtils.attention
        # self.available_model_level_layers["alpha_dropout-n"] = LayerUtils.alpha_dropout
        self.available_model_level_layers['average_pooling_1d-1'] = LayerUtils.average_pooling_1d
        self.is_input_legal['average_pooling_1d-1'] = LayerUtils.average_pooling_1d_input_legal
        self.available_model_level_layers['average_pooling_2-2'] = LayerUtils.average_pooling_2d
        self.is_input_legal['average_pooling_2d-2'] = LayerUtils.average_pooling_2d_input_legal
        # self.available_model_level_layers['average_pooling_3d'] = LayerUtils.average_pooling_3d
        # self.is_input_legal['average_pooling_3d-3'] = LayerUtils.average_pooling_3d_input_legal

        self.available_model_level_layers['batch_normalization-n'] = LayerUtils.batch_normalization
        self.is_input_legal['batch_normalization-n'] = LayerUtils.batch_normalization_input_legal

        self.available_model_level_layers['conv_1d-1'] = LayerUtils.conv1d
        self.is_input_legal['conv_1d-1'] = LayerUtils.conv1d_input_legal

        self.available_model_level_layers['conv_2d-2'] = LayerUtils.conv2d
        self.is_input_legal['conv_2d-2'] = LayerUtils.conv2d_input_legal

        self.available_model_level_layers['conv_2d_transpose-2'] = LayerUtils.conv_2d_transpose
        self.is_input_legal['conv_2d_transpose-2'] = LayerUtils.conv_2d_transpose_input_legal

        # self.available_model_level_layers['conv_3d'] = LayerUtils.conv_3d
        # self.is_input_legal['conv_3d-3'] = LayerUtils.conv_3d_input_legal

        # self.available_model_level_layers['conv_3d_transpose'] = LayerUtils.conv_3d_transpose
        # self.is_input_legal['conv_3d_transpose'] = LayerUtils.conv_3d_transpose_input_legal

        # self.available_model_level_layers["conv_lstm_2d-2"] = LayerUtils.conv_lstm_2d

        self.available_model_level_layers["cropping1d-1"] = LayerUtils.cropping1d

        self.available_model_level_layers["cropping2d-2"] = LayerUtils.cropping2d

        # TODO: https://www.tensorflow.org/versions/r2.2/api_docs/python/tf/keras/layers/Cropping3D
        # self.available_model_level_layers["cropping3d"] = LayerUtils.cropping3d

        self.available_model_level_layers['dense-n'] = LayerUtils.dense
        self.is_input_legal['dense-n'] = LayerUtils.dense_input_legal

        self.available_model_level_layers['depthwise_conv_2d-2'] = LayerUtils.depthwise_conv_2d
        self.is_input_legal['depthwise_conv_2d-2'] = LayerUtils.depthwise_conv_2d_input_legal

        self.available_model_level_layers["dropout-n"] = LayerUtils.dropout

        self.available_model_level_layers['elu_layer-n'] = LayerUtils.elu_layer
        self.is_input_legal['elu_layer-n'] = LayerUtils.elu_layer_input_legal

        self.available_model_level_layers["flatten-n"] = LayerUtils.flatten

        self.available_model_level_layers["gru-1"] = LayerUtils.gru

        self.available_model_level_layers["gaussian_dropout-n"] = LayerUtils.gaussian_dropout

        # self.available_model_level_layers["gaussian_noise-n"] = LayerUtils.gaussian_noise

        self.available_model_level_layers["global_average_pooling_1d-1"] = LayerUtils.global_average_pooling_1d

        self.available_model_level_layers["global_average_pooling_2d-2"] = LayerUtils.global_average_pooling_2d

        # TODO: https://www.tensorflow.org/versions/r2.2/api_docs/python/tf/keras/layers/GlobalAveragePooling3D
        # self.available_model_level_layers["global_average_pooling_3d"] = LayerUtils.global_average_pooling_3d

        self.available_model_level_layers["global_max_pooling_1d-1"] = LayerUtils.global_max_pooling_1d

        self.available_model_level_layers["global_max_pooling_2d-2"] = LayerUtils.global_max_pooling_2d

        # TODO: https://www.tensorflow.org/versions/r2.2/api_docs/python/tf/keras/layers/GlobalMaxPool3D
        # self.available_model_level_layers["global_max_pooling_3d"] = LayerUtils.global_max_pooling_3d

        self.available_model_level_layers["lstm-1"] = LayerUtils.lstm

        # self.available_model_level_layers["layer_normalization-n"] = LayerUtils.layer_normalization

        self.available_model_level_layers['leaky_relu_layer-n'] = LayerUtils.leaky_relu_layer
        self.is_input_legal['leaky_relu_layer-n'] = LayerUtils.leaky_relu_layer_input_legal

        # self.available_model_level_layers["locally_connected_1d-1"] = LayerUtils.locally_connected_1d

        # self.available_model_level_layers["locally_connected_2d-2"] = LayerUtils.locally_connected_2d

        self.available_model_level_layers['max_pooling_1d-1'] = LayerUtils.max_pooling_1d
        self.is_input_legal['max_pooling_1d-1'] = LayerUtils.max_pooling_1d_input_legal
        self.available_model_level_layers['max_pooling_2d-2'] = LayerUtils.max_pooling_2d
        self.is_input_legal['max_pooling_2d-2'] = LayerUtils.max_pooling_2d_input_legal
        # self.available_model_level_layers['max_pooling_3d'] = LayerUtils.max_pooling_3d
        # self.is_input_legal['max_pooling_3d'] = LayerUtils.max_pooling_3d_input_legal

        self.available_model_level_layers['prelu_layer-n'] = LayerUtils.prelu_layer
        self.is_input_legal['prelu_layer-n'] = LayerUtils.prelu_layer_input_legal

        # self.available_model_level_layers["rnn-1"] = LayerUtils.rnn

        self.available_model_level_layers['relu_layer-n'] = LayerUtils.relu_layer
        self.is_input_legal['relu_layer-n'] = LayerUtils.relu_layer_input_legal

        # self.available_model_level_layers['separable_conv_1d'] = LayerUtils.separable_conv_1d  # disable this cause keras-mxnet will directly crash
        # self.is_input_legal['separable_conv_1d'] = LayerUtils.separable_conv_1d_input_legal
        self.available_model_level_layers['separable_conv_2d-2'] = LayerUtils.separable_conv_2d
        self.is_input_legal['separable_conv_2d-2'] = LayerUtils.separable_conv_2d_input_legal

        self.available_model_level_layers["simple_rnn-2"] = LayerUtils.simple_rnn

        self.available_model_level_layers['softmax_layer-n'] = LayerUtils.softmax_layer
        self.is_input_legal['softmax_layer-n'] = LayerUtils.softmax_layer_input_legal

        self.available_model_level_layers['thresholded_relu_layer-n'] = LayerUtils.thresholded_relu_layer
        self.is_input_legal['thresholded_relu_layer-n'] = LayerUtils.thresholded_relu_layer_input_legal

        self.available_model_level_layers["upsampling_1d-1"] = LayerUtils.upsampling_1d

        self.available_model_level_layers["upsampling_2d-2"] = LayerUtils.upsampling_2d

        # TODO: https://www.tensorflow.org/versions/r2.2/api_docs/python/tf/keras/layers/UpSampling3D
        # self.available_model_level_layers["upsampling_3d"] = LayerUtils.upsampling_3d

        self.available_model_level_layers["zero_padding_1d-1"] = LayerUtils.zero_padding_1d

        self.available_model_level_layers["zero_padding_2d-2"] = LayerUtils.zero_padding_2d

        # TODO: https://www.tensorflow.org/versions/r2.2/api_docs/python/tf/keras/layers/ZeroPadding1D
        # self.available_model_level_layers["zero_padding_3d"] = LayerUtils.zero_padding_3d

        self.available_source_level_layers['activity_regularization_l1-n'] = LayerUtils.activity_regularization_l1
        self.is_input_legal['activity_regularization_l1-n'] = LayerUtils.activity_regularization_input_legal
        self.available_source_level_layers['activity_regularization_l2-n'] = LayerUtils.activity_regularization_l1
        self.is_input_legal['activity_regularization_l2-n'] = LayerUtils.activity_regularization_input_legal

    def is_layer_in_weight_change_white_list(self, layer):
        from tensorflow import keras
        white_list = [keras.layers.Dense, keras.layers.Conv1D, keras.layers.Conv2D, keras.layers.Conv3D,
                      keras.layers.DepthwiseConv2D,
                      keras.layers.Conv2DTranspose, keras.layers.Conv3DTranspose,
                      keras.layers.MaxPooling1D, keras.layers.MaxPooling2D, keras.layers.MaxPooling3D,
                      keras.layers.AveragePooling1D, keras.layers.AveragePooling2D, keras.layers.AveragePooling3D,
                      keras.layers.LeakyReLU, keras.layers.ELU, keras.layers.ThresholdedReLU,
                      keras.layers.Softmax, keras.layers.ReLU
                      ]
        # print(white_list)
        for l in white_list:
            if isinstance(layer, l):
                return True
        return False

    # ============================ general method lists START ==================================
    @staticmethod
    def clone(layer):
        from scripts.tools.utils import ModelUtils
        custom_objects = ModelUtils.custom_objects()
        layer_config = layer.get_config()
        if 'activation' in layer_config.keys():
            activation = layer_config['activation']
            if activation in custom_objects.keys():
                layer_config['activation'] = 'relu'
                clone_layer = layer.__class__.from_config(layer_config)
                clone_layer.activation = custom_objects[activation]
            else:
                clone_layer = layer.__class__.from_config(layer_config)
        else:
            clone_layer = layer.__class__.from_config(layer_config)
        return clone_layer
        # return layer.__class__.from_config(layer.get_config())

    # ============================ general method lists END ==================================

    # ============================ layer method lists START ==================================
    @staticmethod
    def attention(input_shape):
        from tensorflow import keras
        layerStack = LayerStack()
        layerStack._layer_stack.append(keras.layers.Attention())
        # layerStack = DimSolver.check_before_add(layerStack, keras.layers.Attention(), input_shape, 3)
        return layerStack.get_layers()

    @staticmethod
    def alpha_dropout(input_shape):
        import keras
        layerStack = LayerStack()
        layerStack.add(keras.layers.AlphaDropout(0.5))
        return layerStack.get_layers()

    @staticmethod
    def dense(input_shape=None):
        # input_shape = input_shape.as_list()
        import keras
        layerStack = LayerStack()
        if input_shape is not None:
            output_units = input_shape[-1]
            if output_units * output_units > LayerUtils.parameter_size_limit:
                if input_shape[-1] < LayerUtils.parameter_size_limit:
                    output_units = LayerUtils.parameter_size_limit // input_shape[-1]
                else:
                    output_units = 1
            layerStack.add(keras.layers.Dense(output_units))
        else:
            layerStack.add(keras.layers.Dense())
        return layerStack.get_layers()

    @staticmethod
    def dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 2 and input_shape[0] is None and input_shape[1] is not None

    @staticmethod
    def conv1d(input_shape=None):
        import keras
        layerStack = LayerStack()
        if input_shape is not None:
            layerStack = DimSolver.check_before_add(layerStack,
                                                    keras.layers.Conv1D(input_shape[-1], 3, strides=1, padding='same'),
                                                    input_shape, 3)
        else:
            layerStack.add(keras.layers.Conv1D())
        return layerStack.get_layers()

    @staticmethod
    def conv1d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3

    @staticmethod
    def conv2d(input_shape=None):
        import keras
        layerStack = LayerStack()
        if input_shape is not None:
            layerStack = DimSolver.check_before_add(layerStack, keras.layers.Conv2D(input_shape[-1], 3, strides=(1, 1),
                                                                                    padding='same'), input_shape, 4)
        else:
            layerStack.add(keras.layers.Conv2D())
        return layerStack.get_layers()

    @staticmethod
    def conv2d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3

    @staticmethod
    def separable_conv_1d(input_shape=None):
        import keras
        layerStack = LayerStack()
        if input_shape is not None:
            layerStack = DimSolver.check_before_add(layerStack,
                                                    keras.layers.SeparableConv1D(input_shape[-1], 3, strides=1,
                                                                                 padding='same'), input_shape, 3)
        else:
            layerStack.add(keras.layers.SeparableConv1D(10, 3))
        return layerStack.get_layers()

    @staticmethod
    def separable_conv_1d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3

    @staticmethod
    def separable_conv_2d(input_shape=None):
        import keras
        layerStack = LayerStack()
        if input_shape is not None:
            layerStack = DimSolver.check_before_add(layerStack,
                                                    keras.layers.SeparableConv2D(input_shape[-1], 3, strides=(1, 1),
                                                                                 padding='same'), input_shape, 4)
        else:
            layerStack.add(keras.layers.SeparableConv2D(10, 3))
        return layerStack.get_layers()

    @staticmethod
    def separable_conv_2d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3

    @staticmethod
    def depthwise_conv_2d(input_shape=None):
        import keras
        layerStack = LayerStack()
        if input_shape is not None:
            layerStack = DimSolver.check_before_add(layerStack,
                                                    keras.layers.DepthwiseConv2D(3, strides=(1, 1), padding='same'),
                                                    input_shape, 4)
        else:
            layerStack.add(keras.layers.DepthwiseConv2D(3, strides=(1, 1), padding='same'))
        return layerStack.get_layers()

    @staticmethod
    def depthwise_conv_2d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3

    @staticmethod
    def conv_2d_transpose(input_shape=None):
        import keras
        layerStack = LayerStack()
        if input_shape is not None:
            layerStack = DimSolver.check_before_add(layerStack,
                                                    keras.layers.Conv2DTranspose(input_shape[-1], 3, strides=(1, 1),
                                                                                 padding='same'), input_shape, 4)
        else:
            layerStack.add(keras.layers.Conv2DTranspose(10, 3))
        return layerStack.get_layers()

    @staticmethod
    def conv_2d_transpose_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3

    @staticmethod
    def conv_3d(input_shape=None):
        # TODO!!!
        import keras
        if input_shape is not None:
            layer = keras.layers.Conv3D(input_shape[-1], 3, strides=(1, 1, 1), padding='same')
        else:
            layer = keras.layers.Conv3D(10, 3)
        layer.name += '_insert'
        return layer

    @staticmethod
    def conv_3d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None \
               and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3 \
               and input_shape[3] is not None and input_shape[3] >= 3

    @staticmethod
    def conv_3d_transpose(input_shape=None):
        # TODO !!!
        import keras
        if input_shape is not None:
            layer = keras.layers.Conv3DTranspose(input_shape[-1], 3, strides=(1, 1, 1), padding='same')
        else:
            layer = keras.layers.Conv3DTranspose(10, 3)
        layer.name += '_insert'
        return layer

    @staticmethod
    def conv_3d_transpose_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None \
               and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3 \
               and input_shape[3] is not None and input_shape[3] >= 3

    @staticmethod
    def max_pooling_1d(input_shape=None):
        import keras
        layerStack = LayerStack()
        if input_shape is not None:
            layerStack = DimSolver.check_before_add(layerStack,
                                                    keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same'),
                                                    input_shape, 3)
        else:
            layerStack.add(keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same'))
        return layerStack.get_layers()

    @staticmethod
    def max_pooling_1d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3

    @staticmethod
    def max_pooling_2d(input_shape=None):
        import keras
        layerStack = LayerStack()
        if input_shape is not None:
            layerStack = DimSolver.check_before_add(layerStack, keras.layers.MaxPooling2D(pool_size=(3, 3), strides=1,
                                                                                          padding='same'), input_shape,
                                                    4)
        else:
            layerStack.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=1, padding='same'))
        return layerStack.get_layers()

    @staticmethod
    def max_pooling_2d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3

    @staticmethod
    def max_pooling_3d(input_shape=None):
        # TODO!!!
        import keras
        layer = keras.layers.MaxPooling3D(pool_size=(3, 3, 3), strides=1, padding='same')
        layer.name += '_insert'
        return layer

    @staticmethod
    def max_pooling_3d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None \
               and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3 \
               and input_shape[3] is not None and input_shape[3] >= 3

    @staticmethod
    def average_pooling_1d(input_shape=None):
        import keras
        layerStack = LayerStack()
        if input_shape is not None:
            layerStack = DimSolver.check_before_add(layerStack, keras.layers.AveragePooling1D(pool_size=3, strides=1,
                                                                                              padding='same'),
                                                    input_shape, 3)
        else:
            layerStack.add(keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same'))
        return layerStack.get_layers()

    @staticmethod
    def average_pooling_1d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3

    @staticmethod
    def average_pooling_2d(input_shape=None):
        import keras
        layerStack = LayerStack()
        if input_shape is not None:
            layerStack = DimSolver.check_before_add(layerStack,
                                                    keras.layers.AveragePooling2D(pool_size=(3, 3), strides=1,
                                                                                  padding='same'), input_shape, 4)
        else:
            layerStack.add(keras.layers.AveragePooling2D(pool_size=(3, 3), strides=1, padding='same'))
        return layerStack.get_layers()

    @staticmethod
    def average_pooling_2d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3

    @staticmethod
    def average_pooling_3d(input_shape=None):
        # TODO!!!
        import keras
        layer = keras.layers.AveragePooling3D(pool_size=(3, 3, 3), strides=1, padding='same')
        layer.name += '_insert'
        return layer

    @staticmethod
    def average_pooling_3d_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None \
               and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3 \
               and input_shape[3] is not None and input_shape[3] >= 3

    @staticmethod
    def batch_normalization(input_shape=None):
        import keras
        layerStack = LayerStack()
        if input_shape is not None:
            layerStack.add(keras.layers.BatchNormalization())
        else:
            layerStack.add(keras.layers.BatchNormalization())
        return layerStack.get_layers()

    @staticmethod
    def batch_normalization_input_legal(input_shape):
        return True

    @staticmethod
    def leaky_relu_layer(input_shape=None):
        import keras
        layerStack = LayerStack()
        if input_shape is not None:
            layerStack.add(keras.layers.LeakyReLU())
        else:
            layerStack.add(keras.layers.LeakyReLU())
        return layerStack.get_layers()

    @staticmethod
    def leaky_relu_layer_input_legal(input_shape):
        return True

    @staticmethod
    def prelu_layer(input_shape=None):
        import keras
        layerStack = LayerStack()
        if input_shape is not None:
            layerStack.add(keras.layers.PReLU(alpha_initializer='RandomNormal'))
        else:
            layerStack.add(keras.layers.PReLU())
        return layerStack.get_layers()

    @staticmethod
    def prelu_layer_input_legal(input_shape):
        return True

    @staticmethod
    def elu_layer(input_shape=None):
        import keras
        layerStack = LayerStack()
        if input_shape is not None:
            layerStack.add(keras.layers.ELU())
        else:
            layerStack.add(keras.layers.ELU())
        return layerStack.get_layers()

    @staticmethod
    def elu_layer_input_legal(input_shape):
        return True

    @staticmethod
    def thresholded_relu_layer(input_shape=None):
        import keras
        layerStack = LayerStack()
        if input_shape is not None:
            layerStack.add(keras.layers.ThresholdedReLU())
        else:
            layerStack.add(keras.layers.ThresholdedReLU())
        return layerStack.get_layers()

    @staticmethod
    def thresholded_relu_layer_input_legal(input_shape):
        return True

    @staticmethod
    def softmax_layer(input_shape=None):
        import keras
        layerStack = LayerStack()
        if input_shape is not None:
            layerStack.add(keras.layers.Softmax())
        else:
            layerStack.add(keras.layers.Softmax())
        return layerStack.get_layers()

    @staticmethod
    def softmax_layer_input_legal(input_shape):
        return True

    @staticmethod
    def relu_layer(input_shape=None):
        import keras
        layerStack = LayerStack()
        if input_shape is not None:
            layerStack.add(keras.layers.ReLU(max_value=1.0))
        else:
            layerStack.add(keras.layers.ReLU())
        return layerStack.get_layers()

    @staticmethod
    def relu_layer_input_legal(input_shape):
        return True

    @staticmethod
    def activity_regularization_l1(input_shape=None):
        import keras
        layerStack = LayerStack()
        layerStack.add(keras.layers.ActivityRegularization(l1=0.5, l2=0.0))
        return layerStack.get_layers()

    @staticmethod
    def activity_regularization_l2(input_shape=None):
        import keras
        layerStack = LayerStack()
        layerStack.add(keras.layers.ActivityRegularization(l1=0.0, l2=0.5))
        return layerStack.get_layers()

    @staticmethod
    def activity_regularization_input_legal(input_shape):
        return True

    @staticmethod
    def conv_lstm_2d(input_shape):
        import keras
        layerStack = LayerStack()
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.ConvLSTM2D(input_shape[-1], kernel_size=(1, 1),
                                                                                    strides=(1, 1), padding='same',
                                                                                    return_sequences=True), input_shape,
                                                4)
        return layerStack.get_layers()

    @staticmethod
    def cropping1d(input_shape):
        import keras
        layerStack = LayerStack()
        actual_input_shape = DimSolver._get_output_shape(target_ndim=3, input_shape=input_shape)
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.Cropping1D(cropping=(1, 1)), input_shape, 3)
        return layerStack.get_layers()

    @staticmethod
    def flatten(input_shape):
        import keras
        layerStack = LayerStack()
        if len(input_shape) != 2:
            layerStack.add(keras.layers.Flatten())
        if len(input_shape) == 4:
            layerStack.add(keras.layers.Reshape(target_shape=(input_shape[1], input_shape[2], input_shape[3])))
        elif len(input_shape) == 3:
            layerStack.add(keras.layers.Reshape(target_shape=(input_shape[1], input_shape[2])))
        return layerStack.get_layers()

    @staticmethod
    def cropping2d(input_shape):
        import keras
        layerStack = LayerStack()
        actual_input_shape = DimSolver._get_output_shape(target_ndim=4, input_shape=input_shape)
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.Cropping2D(cropping=((1, 1), (1, 1))),
                                                input_shape, 4)
        return layerStack.get_layers()

    @staticmethod
    def dropout(input_shape):
        import keras
        layerStack = LayerStack()
        layerStack.add(keras.layers.Dropout(0.5))
        return layerStack.get_layers()

    @staticmethod
    def gru(input_shape):
        import keras
        layerStack = LayerStack()
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.GRU(50), input_shape, 3)
        return layerStack.get_layers()

    @staticmethod
    def gaussian_dropout(input_shape):
        import keras
        layerStack = LayerStack()
        layerStack.add(keras.layers.GaussianDropout(0.5))
        return layerStack.get_layers()

    @staticmethod
    def gaussian_noise(input_shape):
        import keras
        layerStack = LayerStack()
        layerStack.add(keras.layers.GaussianNoise(0.5))
        return layerStack.get_layers()

    @staticmethod
    def global_average_pooling_1d(input_shape):
        import keras
        layerStack = LayerStack()
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.GlobalAveragePooling1D(), input_shape, 3)
        return layerStack.get_layers()

    @staticmethod
    def global_average_pooling_2d(input_shape):
        import keras
        layerStack = LayerStack()
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.GlobalAveragePooling2D(), input_shape, 4)
        return layerStack.get_layers()

    @staticmethod
    def lstm(input_shape):
        import keras
        layerStack = LayerStack()
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.LSTM(50), input_shape, 3)
        return layerStack.get_layers()

    @staticmethod
    def layer_normalization(input_shape):
        import keras
        layerStack = LayerStack()
        layerStack.add(keras.layers.LayerNormalization())
        return layerStack.get_layers()

    @staticmethod
    def locally_connected_1d(input_shape):
        import keras
        layerStack = LayerStack()
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.LocallyConnected1D(32, 3), input_shape, 3)
        return layerStack.get_layers()

    @staticmethod
    def locally_connected_2d(input_shape):
        import keras
        layerStack = LayerStack()
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.LocallyConnected2D(32, 3), input_shape, 4)
        return layerStack.get_layers()

    @staticmethod
    def rnn(input_shape):
        import keras
        layerStack = LayerStack()
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.RNN(50), input_shape, 3)
        return layerStack.get_layers()

    @staticmethod
    def simple_rnn(input_shape):
        import keras
        layerStack = LayerStack()
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.SimpleRNN(50), input_shape, 3)
        return layerStack.get_layers()

    @staticmethod
    def upsampling_1d(input_shape):
        import keras
        layerStack = LayerStack()
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.UpSampling1D(size=2), input_shape, 3)
        return layerStack.get_layers()

    @staticmethod
    def upsampling_2d(input_shape):
        import keras
        layerStack = LayerStack()
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.UpSampling2D(size=(2, 2)), input_shape, 4)
        return layerStack.get_layers()

    @staticmethod
    def zero_padding_1d(input_shape):
        import keras
        layerStack = LayerStack()
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.ZeroPadding1D(padding=2), input_shape, 3)
        return layerStack.get_layers()

    @staticmethod
    def zero_padding_2d(input_shape):
        import keras
        layerStack = LayerStack()
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.ZeroPadding2D(padding=(2, 2)), input_shape, 4)
        return layerStack.get_layers()

    @staticmethod
    def global_max_pooling_1d(input_shape):
        import keras
        layerStack = LayerStack()
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.GlobalMaxPooling1D(), input_shape, 3)
        return layerStack.get_layers()

    @staticmethod
    def global_max_pooling_2d(input_shape):
        import keras
        layerStack = LayerStack()
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.GlobalMaxPooling2D(), input_shape, 4)
        return layerStack.get_layers()


class LayerMatching:
    concat_size_limit = 1e4

    def __init__(self):
        self.layers = {}
        self.constraints = {}

        # Move this one to LayerUtils
        # self.layers['flatten'] = LayerMatching.flatten
        # self.constraints['flatten'] = LayerMatching.flatten_constraints

        self.layer_concats = {}
        self.input_legal = {}
        self.layer_concats['flatten'] = LayerMatching.flatten_dense
        self.input_legal['flatten'] = LayerMatching.flatten_dense_input_legal
        self.layer_concats['repeat_vector'] = LayerMatching.repeat_vector_dense
        self.input_legal['repeat_vector'] = LayerMatching.repeat_vector_dense_input_legal

        self.layer_concats['cropping1d'] = LayerMatching.cropping1d_dense
        self.input_legal['cropping1d'] = LayerMatching.cropping1d_dense_input_legal
        self.layer_concats['cropping2d'] = LayerMatching.cropping2d_dense
        self.input_legal['cropping2d'] = LayerMatching.cropping2d_dense_input_legal
        # self.layer_concats['cropping3d'] = LayerMatching.cropping3d_dense
        self.input_legal['cropping3d'] = LayerMatching.cropping3d_dense_input_legal

        self.layer_concats['upsampling_1d'] = LayerMatching.upsampling_1d_dense
        self.input_legal['upsampling_1d'] = LayerMatching.upsampling_1d_dense_input_legal
        self.layer_concats['upsampling_2d'] = LayerMatching.upsampling_2d_dense
        self.input_legal['upsampling_2d'] = LayerMatching.upsampling_2d_dense_input_legal
        # self.layer_concats['upsampling_3d'] = LayerMatching.upsampling_3d_dense
        self.input_legal['upsampling_3d'] = LayerMatching.upsampling_3d_dense_input_legal

        self.layer_concats['zeropadding_1d'] = LayerMatching.zeropadding_1d_conv
        self.input_legal['zeropadding_1d'] = LayerMatching.zeropadding_1d_conv_input_legal
        self.layer_concats['zeropadding_2d'] = LayerMatching.zeropadding_2d_conv
        self.input_legal['zeropadding_2d'] = LayerMatching.zeropadding_2d_conv_input_legal
        # self.layer_concats['zeropadding_3d'] = LayerMatching.zeropadding_3d_conv
        self.input_legal['zeropadding_3d'] = LayerMatching.zeropadding_3d_conv_input_legal

        self.layer_concats['global_max_pooling_1d'] = LayerMatching.global_max_pooling_1d_dense
        self.input_legal['global_max_pooling_1d'] = LayerMatching.global_pooling_1d_dense_input_legal
        self.layer_concats['global_average_pooling_1d'] = LayerMatching.global_average_pooling_1d_dense
        self.input_legal['global_average_pooling_1d'] = LayerMatching.global_pooling_1d_dense_input_legal
        self.layer_concats['global_max_pooling_2d'] = LayerMatching.global_max_pooling_2d_dense
        self.input_legal['global_max_pooling_2d'] = LayerMatching.global_pooling_2d_dense_input_legal
        self.layer_concats['global_average_pooling_2d'] = LayerMatching.global_average_pooling_2d_dense
        self.input_legal['global_average_pooling_2d'] = LayerMatching.global_pooling_2d_dense_input_legal
        # self.layer_concats['global_max_pooling_3d'] = LayerMatching.global_max_pooling_3d_dense
        self.input_legal['global_max_pooling_3d'] = LayerMatching.global_pooling_3d_dense_input_legal
        # self.layer_concats['global_average_pooling_3d'] = LayerMatching.global_average_pooling_3d_dense
        self.input_legal['global_average_pooling_3d'] = LayerMatching.global_pooling_3d_dense_input_legal

        self.layer_concats['simple_rnn'] = LayerMatching.simple_rnn_dense
        self.input_legal['simple_rnn'] = LayerMatching.simple_rnn_dense_input_legal
        self.layer_concats['gru'] = LayerMatching.gru_dense
        self.input_legal['gru'] = LayerMatching.gru_dense_input_legal
        self.layer_concats['lstm'] = LayerMatching.lstm_dense
        self.input_legal['lstm'] = LayerMatching.lstm_dense_input_legal
        # self.layer_concats['conv_lstm_2d'] = LayerMatching.conv_lstm_2d_dense  # This require 5D
        self.input_legal['conv_lstm_2d'] = LayerMatching.conv_lstm_2d_dense_input_legal

    @staticmethod
    def flatten_constraints(input_shape):
        input_shape = input_shape.as_list()
        input_shape_len = len(input_shape)
        constraints = []
        if input_shape_len < 2:
            return None
        constraints = []
        dim_size = 1
        for i in range(input_shape_len):
            if i == 0:
                continue
            constraints.append('= input_{} {}'.format(i, input_shape[i]))
            dim_size *= input_shape[i]
        constraint_str = '= output_{} {}'.format(1, dim_size)
        constraints.append(constraint_str)
        return constraints

    @staticmethod
    def squeeze_dimension(layer_concat, input_shape):
        import keras
        if len(input_shape) == 4:
            # if the input_shape has 4 dims, squeeze last two dims together
            input_shape[1] = input_shape[1]
            input_shape[2] = input_shape[2] * input_shape[3]
            layer_concat.append(keras.layers.Reshape((input_shape[1], input_shape[2])))
        return input_shape

    # --------------------------------------------

    @staticmethod
    def simplify_output(units, input_shape):
        # limit dense's output, if output tensor shape is larger than the limit size, we will make it smaller by
        # "subdivided" it
        output_shape = list(input_shape)
        while units > LayerMatching.concat_size_limit:
            units = 1
            for i in range(1, len(output_shape)):
                if output_shape[i] > 10:
                    output_shape[i] -= int(output_shape[i] / 2)
                units *= output_shape[i]
        return tuple(output_shape), units

    @staticmethod
    def flatten_dense_output(input_shape):
        """
        Flat a tensor into two dimensions, then transfer it into a 100 element tensor as the model output
        """
        import keras
        layerStack = LayerStack()
        if len(input_shape) != 2:
            layerStack.add(keras.layers.Flatten())
        layerStack.add(keras.layers.Dense(100))
        return layerStack.get_layers()

    @staticmethod
    def flatten_dense(input_shape):
        import keras
        layerStack = LayerStack()
        layerStack.add(keras.layers.Flatten())
        units = 1
        for i in range(1, len(input_shape)):
            units *= input_shape[i]
        output_shape, units = LayerMatching.simplify_output(units, input_shape)
        layerStack.add(keras.layers.Dense(units))
        layerStack.add(keras.layers.Reshape(output_shape[1:]))
        return layerStack.get_layers()

    @staticmethod
    def flatten_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        is_legal = len(input_shape) > 3 and input_shape[0] is None
        concat_size = 1
        for i, dim in enumerate(input_shape):
            if i == 0:
                continue
            is_legal = is_legal and dim is not None
            if dim is not None:
                concat_size *= dim
        return is_legal and concat_size <= LayerMatching.concat_size_limit

    @staticmethod
    def repeat_vector_dense(input_shape):
        n = 3
        import keras
        layerStack = LayerStack()
        layerStack.add(keras.layers.RepeatVector(n))
        layerStack.add(keras.layers.Reshape((input_shape[1] * n,)))
        layerStack.add(keras.layers.Dense(input_shape[1]))
        return layerStack.get_layers()

    @staticmethod
    def repeat_vector_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 2 and input_shape[0] is None and input_shape[1] is not None \
               and input_shape[1] <= LayerMatching.concat_size_limit

    @staticmethod
    def cropping1d_dense(input_shape):
        import keras
        layerStack = LayerStack()
        actual_input_shape = DimSolver._get_output_shape(target_ndim=3, input_shape=input_shape)
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.Cropping1D(cropping=(1, 1)), input_shape, 3)
        layerStack.add(keras.layers.Dense(input_shape[1]))
        return layerStack.get_layers()

    @staticmethod
    def cropping1d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] > 2 \
               and input_shape[2] is not None and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit

    @staticmethod
    def cropping2d_dense(input_shape):
        import keras
        layerStack = LayerStack()
        actual_input_shape = DimSolver._get_output_shape(target_ndim=4, input_shape=input_shape)
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.Cropping2D(cropping=((1, 1), (1, 1))),
                                                input_shape, 4)
        layerStack.add(
            keras.layers.Reshape(((actual_input_shape[1] - 2) * (actual_input_shape[2] - 2) * actual_input_shape[3],)))
        units = actual_input_shape[1] * actual_input_shape[2] * actual_input_shape[3]
        output_shape, units = LayerMatching.simplify_output(units, actual_input_shape)
        layerStack.add(keras.layers.Dense(units))
        layerStack.add(keras.layers.Reshape(output_shape[1:]))
        return layerStack.get_layers()

    @staticmethod
    def cropping2d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None \
               and input_shape[1] is not None and input_shape[1] > 2 \
               and input_shape[2] is not None and input_shape[2] > 2 \
               and input_shape[3] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] <= LayerMatching.concat_size_limit

    @staticmethod
    def cropping3d_dense(input_shape):
        # TODO!!!
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.Cropping3D(cropping=((1, 1), (1, 1), (1, 1))))
        layer_concat.append(keras.layers.Reshape(
            ((input_shape[1] - 2) * (input_shape[2] - 2) * (input_shape[3] - 2) * input_shape[4],)))
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def cropping3d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None \
               and input_shape[1] is not None and input_shape[1] > 2 \
               and input_shape[2] is not None and input_shape[2] > 2 \
               and input_shape[3] is not None and input_shape[3] > 2 \
               and input_shape[4] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] <= LayerMatching.concat_size_limit

    @staticmethod
    def upsampling_1d_dense(input_shape):
        import keras
        layerStack = LayerStack()
        actual_input_shape = DimSolver._get_output_shape(target_ndim=3, input_shape=input_shape)
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.UpSampling1D(size=2), input_shape, 3)

        layerStack.add(keras.layers.Dense(min(actual_input_shape[1], actual_input_shape[
            2])))  # the original is: actual_input_shape[1] * actual_input_shape[2], but it is too large, change to min(actual_input_shape[1], actual_input_shape[2])
        return layerStack.get_layers()

    @staticmethod
    def upsampling_1d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None \
               and input_shape[2] is not None and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit

    @staticmethod
    def upsampling_2d_dense(input_shape):
        import keras
        layerStack = LayerStack()
        actual_input_shape = DimSolver._get_output_shape(target_ndim=4, input_shape=input_shape)
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.UpSampling2D(size=(2, 2)), input_shape, 4)
        layerStack.add(keras.layers.Flatten())
        units = actual_input_shape[1] * actual_input_shape[2] * actual_input_shape[3]
        output_shape, units = LayerMatching.simplify_output(units, actual_input_shape)
        layerStack.add(keras.layers.Dense(units))
        layerStack.add(keras.layers.Reshape(output_shape[1:]))
        return layerStack.get_layers()

    @staticmethod
    def upsampling_2d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None \
               and input_shape[1] is not None and input_shape[2] is not None and input_shape[3] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] <= LayerMatching.concat_size_limit

    @staticmethod
    def upsampling_3d_dense(input_shape):
        # TODO!!!
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.UpSampling3D(size=(2, 2, 2)))
        layer_concat.append(keras.layers.Flatten())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def upsampling_3d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None \
               and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[3] is not None \
               and input_shape[4] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] <= LayerMatching.concat_size_limit

    @staticmethod
    def zeropadding_1d_conv(input_shape):
        import keras
        layerStack = LayerStack()
        actual_input_shape = DimSolver._get_output_shape(target_ndim=3, input_shape=input_shape)
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.ZeroPadding1D(padding=1), input_shape, 3)
        layerStack.add(keras.layers.Conv1D(actual_input_shape[-1], 3))
        return layerStack.get_layers()

    @staticmethod
    def zeropadding_1d_conv_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None \
               and input_shape[1] is not None and input_shape[2] is not None \
               and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit

    @staticmethod
    def zeropadding_2d_conv(input_shape):
        import keras
        layerStack = LayerStack()
        actual_input_shape = DimSolver._get_output_shape(target_ndim=4, input_shape=input_shape)
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.ZeroPadding2D(padding=(1, 1)), input_shape, 4)
        layerStack.add(keras.layers.Conv2D(actual_input_shape[-1], 3))
        return layerStack.get_layers()

    @staticmethod
    def zeropadding_2d_conv_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None \
               and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[3] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] <= LayerMatching.concat_size_limit

    @staticmethod
    def zeropadding_3d_conv(input_shape):
        # TODO!!!
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.ZeroPadding3D(padding=(1, 1, 1)))
        layer_concat.append(keras.layers.Conv3D(input_shape[-1], 3))
        return layer_concat

    @staticmethod
    def zeropadding_3d_conv_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None \
               and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[3] is not None \
               and input_shape[4] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] <= LayerMatching.concat_size_limit

    @staticmethod
    def global_max_pooling_1d_dense(input_shape):
        import keras
        layerStack = LayerStack()
        actual_input_shape = DimSolver._get_output_shape(target_ndim=3, input_shape=input_shape)
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.GlobalMaxPooling1D(), input_shape, 3)
        units = actual_input_shape[1] * actual_input_shape[2]
        output_shape, units = LayerMatching.simplify_output(units, actual_input_shape)
        layerStack.add(keras.layers.Dense(units))
        layerStack.add(keras.layers.Reshape(output_shape[1:]))
        return layerStack.get_layers()

    @staticmethod
    def global_average_pooling_1d_dense(input_shape):
        import keras
        layerStack = LayerStack()
        actual_input_shape = DimSolver._get_output_shape(target_ndim=3, input_shape=input_shape)
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.GlobalAveragePooling1D(), input_shape, 3)
        units = actual_input_shape[1] * actual_input_shape[2]
        output_shape, units = LayerMatching.simplify_output(units, actual_input_shape)
        layerStack.add(keras.layers.Dense(units))
        layerStack.add(keras.layers.Reshape(output_shape[1:]))
        return layerStack.get_layers()

    @staticmethod
    def global_pooling_1d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None \
               and input_shape[2] is not None and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit

    @staticmethod
    def global_max_pooling_2d_dense(input_shape):
        import keras
        layerStack = LayerStack()
        actual_input_shape = DimSolver._get_output_shape(target_ndim=4, input_shape=input_shape)
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.GlobalMaxPooling2D(), input_shape, 4)
        units = actual_input_shape[1] * actual_input_shape[2] * actual_input_shape[3]
        output_shape, units = LayerMatching.simplify_output(units, actual_input_shape)
        layerStack.add(keras.layers.Dense(units))
        layerStack.add(keras.layers.Reshape(output_shape[1:]))
        return layerStack.get_layers()

    @staticmethod
    def global_average_pooling_2d_dense(input_shape):
        import keras
        layerStack = LayerStack()
        actual_input_shape = DimSolver._get_output_shape(target_ndim=4, input_shape=input_shape)
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.GlobalAveragePooling2D(), input_shape, 4)
        units = actual_input_shape[1] * actual_input_shape[2] * actual_input_shape[3]
        output_shape, units = LayerMatching.simplify_output(units, actual_input_shape)
        layerStack.add(keras.layers.Dense(units))
        layerStack.add(keras.layers.Reshape(output_shape[1:]))
        return layerStack.get_layers()

    @staticmethod
    def global_pooling_2d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None \
               and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[3] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] <= LayerMatching.concat_size_limit

    @staticmethod
    def global_max_pooling_3d_dense(input_shape):
        # TODO!!!
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.GlobalMaxPooling3D())
        layer_concat.append(keras.layers.Flatten())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def global_average_pooling_3d_dense(input_shape):
        # TODO!!!
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.GlobalAveragePooling3D())
        layer_concat.append(keras.layers.Flatten())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def global_pooling_3d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None \
               and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[3] is not None \
               and input_shape[4] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] <= LayerMatching.concat_size_limit

    @staticmethod
    def simple_rnn_dense(input_shape):
        import keras
        layerStack = LayerStack()
        actual_input_shape = DimSolver._get_output_shape(target_ndim=3, input_shape=input_shape)
        layerStack = DimSolver.check_before_add(layerStack,
                                                keras.layers.SimpleRNN(50, input_shape=actual_input_shape[1:],
                                                                       return_sequences=False), input_shape, 3)
        units = actual_input_shape[1] * actual_input_shape[2]
        output_shape, units = LayerMatching.simplify_output(units, actual_input_shape)
        layerStack.add(keras.layers.Dense(units))
        layerStack.add(keras.layers.Reshape(output_shape[1:]))
        return layerStack.get_layers()

    @staticmethod
    def simple_rnn_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None \
               and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit

    @staticmethod
    def gru_dense(input_shape):
        import keras
        layerStack = LayerStack()
        actual_input_shape = DimSolver._get_output_shape(target_ndim=3, input_shape=input_shape)
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.GRU(50), input_shape, 3)
        units = actual_input_shape[1] * actual_input_shape[2]
        output_shape, units = LayerMatching.simplify_output(units, actual_input_shape)
        layerStack.add(keras.layers.Dense(units))
        layerStack.add(keras.layers.Reshape(output_shape[1:]))
        return layerStack.get_layers()

    @staticmethod
    def gru_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None \
               and input_shape[2] is not None and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit

    @staticmethod
    def lstm_dense(input_shape):
        import keras
        layerStack = LayerStack()
        actual_input_shape = DimSolver._get_output_shape(target_ndim=3, input_shape=input_shape)
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.LSTM(50), input_shape, 3)
        layerStack.add(keras.layers.Dense(actual_input_shape[1] * actual_input_shape[2]))
        layerStack.add(keras.layers.Reshape(input_shape[1:]))
        return layerStack.get_layers()

    @staticmethod
    def lstm_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None \
               and input_shape[2] is not None and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit

    @staticmethod
    def conv_lstm_2d_dense(input_shape):
        import keras
        layerStack = LayerStack()
        layerStack = DimSolver.check_before_add(layerStack, keras.layers.ConvLSTM2D(input_shape[-1], kernel_size=(1, 1),
                                                                                    strides=(1, 1), padding='same',
                                                                                    return_sequences=True), input_shape,
                                                4)
        return layerStack.get_layers()

    @staticmethod
    def conv_lstm_2d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None and input_shape[1] is not None \
               and input_shape[2] is not None and input_shape[2] > 3 \
               and input_shape[3] is not None and input_shape[3] > 3 \
               and input_shape[4] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] <= LayerMatching.concat_size_limit
    # ============================ layer method lists START ==================================


def split_number(target_number):
    # """
    import numpy as np
    middle = int(np.sqrt(target_number))
    for i in range(middle, 0, -1):
        if target_number % i == 0:
            j = target_number / i
            break
    assert int(i) * int(j) == target_number
    return int(i), int(j)
    # """
    """tensorflow while_loop version

    import tensorflow as tf
    middle = tf.cast(tf.sqrt(tf.cast(target_number, "float32")), "int32")
    n = tf.constant(0)
    def cond(i, n):
        mod = tf.mod(target_number, i)
        res = tf.not_equal(mod, n)
        return res
    def body(i, n):
        return tf.subtract(i, tf.constant(1)), n
    middle, n = tf.while_loop(
        cond, body, [middle, n]
    )
    tf.cast(middle, "int32")
    return middle, tf.cast(tf.div(target_number, middle), "int32")
    """


class DimSolver:
    """This is a magic class that can solve the dimension inconsistency problem across different layers.

    Args:
        object ([type]): [description]
    """

    def __init__(self):
        pass

    @staticmethod
    def check_before_add(layerStack, inserted_layer, input_shape, required_ndim, output_ndim):
        from scripts.tools.architecture_utils import ArchitectureUtils
        from scripts.generation.layer_pools import LayerInputNdim
        layer_class = inserted_layer.__class__.__name__
        required_ndim_list = LayerInputNdim[layer_class]
        if len(input_shape) in required_ndim_list:
            layerStack.add(inserted_layer)
        else:
            target_shape = [0 for _ in range(required_ndim)]
            # For the expected_shape, we will generate some meaningless shape such as [0, 0, 0]
            # so the get_expand_drop_dim_layer method can return a list of layer to convert the input_shape to the required_ndim
            layer_list = ArchitectureUtils.get_expand_drop_dim_layer(input_shape=input_shape, target_shape=target_shape)
            for _layer in layer_list:
                layerStack.add(_layer)
            layerStack.add(inserted_layer)
        return layerStack

    @staticmethod
    def _get_output_shape(target_ndim, input_shape):
        from scripts.generation.layer_tool_kits import split_number
        if target_ndim == 3 and len(input_shape) == 4:
            # squeeze 4 ndim to 3 ndim
            batch, rows, cols, filters = input_shape
            if None in [rows, cols]:
                output_shape = (batch, None, filters)
            else:
                output_shape = (batch, rows * cols, filters)
        elif target_ndim == 3 and len(input_shape) == 2:
            # expand 2 ndim to 3 ndim
            batch, features = input_shape
            if features is None:
                output_shape = (batch, None, None)
            else:
                sequences, filters = split_number(features)
                output_shape = (batch, sequences, filters)
        elif target_ndim == 3 and len(input_shape) == 5:
            batch, rows, cols, t, filters = input_shape
            output_shape = (batch, rows * cols * t, filters)
        elif target_ndim == 4 and len(input_shape) == 2:
            # expand 2 ndim to 4 ndim
            batch, features = input_shape
            if features is None:
                output_shape = (batch, None, None, None)
            else:
                sequences, filters = split_number(features)
                output_shape_1 = (batch, sequences, filters)
                output_shape = DimSolver._get_output_shape(target_ndim=4, input_shape=output_shape_1)
        elif target_ndim == 4 and len(input_shape) == 3:
            # squeeze 3 ndim to 4 ndim
            batch, sequences, filters = input_shape
            if sequences is None:
                output_shape = (batch, None, None, filters)
            else:
                rows, cols = split_number(sequences)
                output_shape = (batch, rows, cols, filters)
        elif target_ndim == 4 and len(input_shape) == 5:
            batch, rows, cols, t, filters = input_shape
            output_shape = (batch, rows, cols * t, filters)
        elif target_ndim == 2:
            units = 1
            batch = input_shape[0]
            for i in input_shape[1:]:
                units *= i
            output_shape = (batch, units)
        elif target_ndim == 5 and len(input_shape) == 4:
            batch, rows, cols, filters = input_shape
            output_shape = (batch, rows, cols, 1, filters)
        elif target_ndim == 5 and len(input_shape) == 3:
            batch, sequences, filters = input_shape
            output_shape = (batch, sequences, 1, 1, filters)
        elif target_ndim == 5 and len(input_shape) == 2:
            batch, features = input_shape
            sequences, filters = split_number(features)
            output_shape = (batch, sequences, 1, 1, filters)
        else:
            output_shape = input_shape
        return output_shape

    @staticmethod
    def add_reshape_layer(layerStack, target_ndim, input_shape):
        from keras import layers
        output_shape = DimSolver._get_output_shape(target_ndim=target_ndim, input_shape=input_shape)
        layerStack.add(layers.Reshape(target_shape=output_shape[1:]))
        return layerStack
