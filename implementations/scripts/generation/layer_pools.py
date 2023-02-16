from scripts.generation.layer_tool_kits import LayerStack, DimSolver
import json
import numpy as np


class ConfigurationUtils:
    @staticmethod
    def random_config(layer):
        layer_config = layer.get_config()
        layer_class_name = layer.__class__.__name__
        config_pool = json.load(open("boostraps/api_implementations/config_pool_3.json", "rb+"))

        def choose_param(param):
            origin_param = layer_config[param]
            param_indices = np.arange(len(config_pool[param]))
            idx = np.random.choice(param_indices)
            replace_result = config_pool[param][idx]
            if "ZeroPadding" in layer_class_name and "padding" in param:
                replace_result = 0
            if replace_result == 0:
                # this means that we need to randomly generate some values
                if param == "filters" or param == "depth_multiplier" or param == "size":
                    replace_result = np.random.randint(1, 10)
                elif param == "kernel_size":
                    dim = len(origin_param)
                    replace_result = [np.random.randint(1, 10) for i in range(dim)]
                elif param == "strides" or param == "dilation_rate" or param == "padding":
                    dim = len(origin_param)
                    replace_result = [np.random.randint(1, 5) for i in range(dim)]
                elif param == "pool_size" or param == "cropping":
                    dim = len(origin_param)
                    replace_result = [np.random.randint(1, 10) for i in range(dim)]
                elif param == "units":
                    replace_result = np.random.randint(100)
                elif param == "dropout" or param == "recurrent_dropout":
                    replace_result = float(np.random.rand(1))
                elif param == "input_dim":
                    replace_result = np.random.randint(100)
                elif param == "output_dim":
                    replace_result = np.random.randint(100)
                elif param == "stddev" or param == "momentum" or param == "epsilon" or param == "rate":
                    replace_result = float(np.random.rand(1))
            return replace_result

        omit_config = ["strides", "dilation_rate", "cropping"]
        for target_param in layer_config.keys():
            if target_param in config_pool.keys():
                if target_param in omit_config:
                    continue
                new_param = choose_param(target_param)
                layer_config[target_param] = new_param

        if layer_class_name is "LocallyConnected2D":
            if layer_config["implementation"] == 1 and layer_config["padding"] == "same":
                # For LocallyConnected2D, avoid setting padding to be "same" if implementation is 2.
                layer_config["padding"] = "valid"
        new_layer = layer.from_config(layer_config)
        return new_layer

    @staticmethod
    def from_config(layer, config):
        """
        Change the configuration of [layer] to [config]
        Arguments:
            :layer: a keras layer
            :config: target_config
        """
        origin_layer_config = layer.get_config()
        import copy
        new_config = copy.deepcopy(config)
        # For we do not consider filters and units in our parameter, just ignore them.
        if "filters" in origin_layer_config:
            new_config["filters"] = np.random.randint(1, 10)
        if "units" in origin_layer_config:
            new_config["units"] = np.random.randint(1, 10)
        return layer.from_config(new_config)


class ConvLayers:
    def __init__(self):
        # Refer to https://keras.io/api/layers/convolution_layers/
        self.available_layers = {
            'Conv1D': ConvLayers.conv1d,
            'Conv2D': ConvLayers.conv2d,
            'Conv3D': ConvLayers.conv3d,  # Tested
            'SeparableConv1D': ConvLayers.separable_conv_1d,
            'SeparableConv2D': ConvLayers.separable_conv_2d,
            'DepthwiseConv1D': ConvLayers.depthwise_conv_1d,
            'DepthwiseConv2D': ConvLayers.depthwise_conv_2d,
            'Conv2DTranspose': ConvLayers.conv_2d_transpose,
            'Conv3DTranspose': ConvLayers.conv_3d_transpose  # Tested
        }

    @staticmethod
    def conv1d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.Conv1D(input_shape[-1], 3, strides=1, padding='same')
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 3, 3)
        return layerStack.get_layers()

    @staticmethod
    def conv2d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.Conv2D(input_shape[-1], 3, strides=(1, 1), padding='same')
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 4, 4)
        return layerStack.get_layers()

    @staticmethod
    def conv3d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.Conv3D(input_shape[-1], 3, strides=(1, 1, 1), padding='same')
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 5, 5)
        return layerStack.get_layers()

    @staticmethod
    def conv_2d_transpose(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.Conv2DTranspose(input_shape[-1], 3, strides=(1, 1), padding='same')
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 4, 4)
        return layerStack.get_layers()

    @staticmethod
    def conv_3d_transpose(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.Conv3DTranspose(input_shape[-1], 3, strides=(1, 1, 1), padding='same')
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 5, 5)
        return layerStack.get_layers()

    @staticmethod
    def separable_conv_2d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.SeparableConv2D(input_shape[-1], 3, strides=(1, 1), padding='same')
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 4, 4)
        return layerStack.get_layers()

    @staticmethod
    def separable_conv_1d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.SeparableConv1D(input_shape[-1], 3, strides=1, padding='same')
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 3, 3)
        return layerStack.get_layers()

    @staticmethod
    def depthwise_conv_1d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.DepthwiseConv1D(3, strides=1, padding='same')
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 3, 3)
        return layerStack.get_layers()

    @staticmethod
    def depthwise_conv_2d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.DepthwiseConv2D(3, strides=(1, 1), padding='same')
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 4, 4)
        return layerStack.get_layers()


class PoolLayers:
    def __init__(self):
        # tested
        # Refer to https://keras.io/api/layers/pooling_layers/
        self.available_layers = {
            'MaxPooling1D': PoolLayers.max_pooling_1d,
            'MaxPooling2D': PoolLayers.max_pooling_2d,
            'MaxPooling3D': PoolLayers.max_pooling_3d,  # Tested
            'AveragePooling1D': PoolLayers.average_pooling_1d,
            'AveragePooling2D': PoolLayers.average_pooling_2d,
            'AveragePooling3D': PoolLayers.average_pooling_3d,  # Tested
            'GlobalMaxPooling1D': PoolLayers.global_max_pooling_1d,
            'GlobalMaxPooling2D': PoolLayers.global_max_pooling_2d,
            'GlobalMaxPooling3D': PoolLayers.global_max_pooling_3d,  # Tested
            'GlobalAveragePooling1D': PoolLayers.global_average_pooling_1d,
            'GlobalAveragePooling2D': PoolLayers.global_average_pooling_2d,
            'GlobalAveragePooling3D': PoolLayers.global_average_pooling_3d,  # Tested
        }

    @staticmethod
    def max_pooling_1d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same')
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 3, 3)
        return layerStack.get_layers()

    @staticmethod
    def max_pooling_2d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 4, 4)
        return layerStack.get_layers()

    @staticmethod
    def max_pooling_3d(input_shape=None, config=None):
        # TODO!!!
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.MaxPooling3D(pool_size=(3, 3, 3), strides=1, padding='same')
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 5, 5)
        return layerStack.get_layers()

    @staticmethod
    def average_pooling_1d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same')
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 3, 3)
        return layerStack.get_layers()

    @staticmethod
    def average_pooling_2d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 4, 4)
        return layerStack.get_layers()

    @staticmethod
    def average_pooling_3d(input_shape=None, config=None):
        # To be tested
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.AveragePooling3D(pool_size=(3, 3, 3), strides=1, padding='same')
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 5, 5)
        return layerStack.get_layers()

    @staticmethod
    def global_max_pooling_1d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.GlobalMaxPooling1D()
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 3, 2)
        return layerStack.get_layers()

    @staticmethod
    def global_max_pooling_2d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.GlobalMaxPooling2D()
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 4, 2)
        return layerStack.get_layers()

    @staticmethod
    def global_max_pooling_3d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.GlobalMaxPooling3D()
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 5, 2)
        return layerStack.get_layers()

    @staticmethod
    def global_average_pooling_1d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.GlobalAveragePooling1D()
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 3, 2)
        return layerStack.get_layers()

    @staticmethod
    def global_average_pooling_2d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.GlobalAveragePooling2D()
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 4, 2)
        return layerStack.get_layers()

    @staticmethod
    def global_average_pooling_3d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.GlobalAveragePooling3D()
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 5, 2)
        return layerStack.get_layers()


class RNNLayers:
    def __init__(self):
        # https://keras.io/api/layers/recurrent_layers/
        self.available_layers = {
            'LSTM': RNNLayers.lstm,
            'GRU': RNNLayers.gru,
            'SimpleRNN': RNNLayers.simple_rnn,
            'TimeDistributed': RNNLayers.time_distributed,  # Tested
            'Bidirectional': RNNLayers.bidirectional,  # Tested
            "ConvLSTM1D": RNNLayers.conv_lstm_1d,
            "ConvLSTM2D": RNNLayers.conv_lstm_2d,  # Tested
            "ConvLSTM3D": RNNLayers.conv_lstm_3d,
        }

    @staticmethod
    def add_flatten_dense(layerStack, input_shape):
        from scripts.generation.layer_tool_kits import LayerMatching
        import keras
        if len(input_shape) != 2:
            units = 1
            for i in range(1, len(input_shape)):
                units *= input_shape[i]
            output_shape, units = LayerMatching.simplify_output(units, input_shape)
            layerStack.add(keras.layers.Dense(units))
            layerStack.add(keras.layers.Reshape(output_shape[1:]))
        return layerStack

    @staticmethod
    def lstm(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.LSTM(50)
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 3, 2)
        return layerStack.get_layers()

    @staticmethod
    def gru(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.GRU(50)
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 3, 2)
        return layerStack.get_layers()

    @staticmethod
    def time_distributed(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.TimeDistributed(keras.layers.ReLU())
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 3, 3)
        return layerStack.get_layers()

    @staticmethod
    def bidirectional(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.Bidirectional(keras.layers.LSTM(10, return_sequences=True))
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 3, 3)
        return layerStack.get_layers()

    @staticmethod
    def simple_rnn(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.SimpleRNN(50)
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 3, 2)
        return layerStack.get_layers()

    @staticmethod
    def conv_lstm_1d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.ConvLSTM1D(input_shape[-1], kernel_size=1, strides=1, padding="same",
                                                  return_sequences=True)
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 4, 4)
        return layerStack.get_layers()

    @staticmethod
    def conv_lstm_2d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.ConvLSTM2D(
            input_shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same', return_sequences=True)
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 5, 5)
        return layerStack.get_layers()

    @staticmethod
    def conv_lstm_3d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.ConvLSTM3D(
            input_shape[-1], kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', return_sequences=True)
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 6, 6)
        return layerStack.get_layers()

    # @staticmethod
    # def rnn(input_shape=None, config=None):
    #     import keras
    #     layerStack = LayerStack()
    #     layerStack = DimSolver.check_before_add(layerStack, keras.layers.RNN(50), input_shape, 3, 2)
    #     return layerStack.get_layers()


class PreLayers:
    # TBA: https://keras.io/api/layers/preprocessing_layers/
    # This is not available in keras 2.2.4
    pass


class NormLayers:
    def __init__(self):
        # Refer to https://keras.io/api/layers/normalization_layers/
        self.available_layers = {
            'BatchNormalization': NormLayers.batch_normalization,
            'LayerNormalization': NormLayers.layer_normalization  # don't add it cause keras 2.2.4.3 does not have it
        }

    @staticmethod
    def batch_normalization(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.BatchNormalization()
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 4, 2)
        return layerStack.get_layers()

    @staticmethod
    def layer_normalization(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.LayerNormalization()
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack.add(inserted_layer)
        return layerStack.get_layers()


class RegLayers:
    def __init__(self):
        # tested
        # Refer to https://keras.io/api/layers/regularization_layers/
        self.available_layers = {
            'Dropout': RegLayers.dropout,
            'SpatialDropout1D': RegLayers.spatial_dropout_1d,  # Tested
            'SpatialDropout2D': RegLayers.spatial_dropout_2d,  # Tested
            'SpatialDropout3D': RegLayers.spatial_dropout_3d,  # Tested
            'GaussianDropout': RegLayers.gaussian_dropout,
            'GaussianNoise': RegLayers.gaussian_noise,
            'ActivityRegularization': RegLayers.activity_regularization,
            'AlphaDropout': RegLayers.alpha_dropout
        }

    @staticmethod
    def dropout(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.Dropout(0.5)
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack.add(inserted_layer)
        return layerStack.get_layers()

    @staticmethod
    def gaussian_dropout(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.GaussianDropout(0.5)
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack.add(inserted_layer)
        return layerStack.get_layers()

    @staticmethod
    def gaussian_noise(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.GaussianNoise(0.5)
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack.add(inserted_layer)
        return layerStack.get_layers()

    @staticmethod
    def activity_regularization(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.ActivityRegularization()
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack.add(inserted_layer)
        return layerStack.get_layers()

    @staticmethod
    def alpha_dropout(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.AlphaDropout(0.5)
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack.add(inserted_layer)
        return layerStack.get_layers()

    @staticmethod
    def spatial_dropout_1d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.SpatialDropout1D(0.5)
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 3, 3)
        return layerStack.get_layers()

    @staticmethod
    def spatial_dropout_2d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.SpatialDropout2D(0.5)
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 4, 4)
        return layerStack.get_layers()

    @staticmethod
    def spatial_dropout_3d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.SpatialDropout3D(0.5)
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 5, 5)
        return layerStack.get_layers()


class AttentionLayers:
    # Not available in keras 2.2.4
    pass


class ReshapeLayers:
    def __init__(self):
        # tested
        # Refer to https://keras.io/api/layers/reshaping_layers/
        self.available_layers = {
            'Flatten': ReshapeLayers.flatten,
            'RepeatVector': ReshapeLayers.repeat_vector_dense,
            'Permute': ReshapeLayers.permute,  # Tested
            'Cropping1D': ReshapeLayers.cropping1d,
            'Cropping2D': ReshapeLayers.cropping2d,
            'Cropping3D': ReshapeLayers.cropping3d,  # Tested
            'UpSampling1D': ReshapeLayers.upsampling_1d,
            'UpSampling2D': ReshapeLayers.upsampling_2d,
            'UpSampling3D': ReshapeLayers.upsampling_3d,  # Tested
            "ZeroPadding1D": ReshapeLayers.zero_padding_1d,
            "ZeroPadding2D": ReshapeLayers.zero_padding_2d,
            "ZeroPadding3D": ReshapeLayers.zero_padding_3d,  # Tested
        }

    @staticmethod
    def flatten(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        if len(input_shape) != 2:
            # In old keras (2.2.4), the flatten requires min_dim to be 3
            layerStack.add(keras.layers.Flatten())
        return layerStack.get_layers()

    @staticmethod
    def repeat_vector_dense(input_shape=None, config=None):
        n = 2
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.RepeatVector(n)
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 2, 2)
        return layerStack.get_layers()

    @staticmethod
    def permute(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.Permute((2, 1))
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 3, 3)
        return layerStack.get_layers()

    @staticmethod
    def cropping1d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.Cropping1D(cropping=(1, 1))
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 3, 3)
        return layerStack.get_layers()

    @staticmethod
    def cropping2d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.Cropping2D(cropping=((1, 1), (1, 1)))
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 4, 4)
        return layerStack.get_layers()

    @staticmethod
    def cropping3d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.Cropping3D(cropping=((1, 1), (1, 1), (0, 0)))
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 5, 5)
        return layerStack.get_layers()

    @staticmethod
    def upsampling_1d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.UpSampling1D(size=2)
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 3, 3)
        return layerStack.get_layers()

    @staticmethod
    def upsampling_2d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.UpSampling2D(size=(2, 2))
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 4, 4)
        return layerStack.get_layers()

    @staticmethod
    def upsampling_3d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.UpSampling3D(size=(2, 2, 2))
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 5, 5)
        return layerStack.get_layers()

    @staticmethod
    def zero_padding_1d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.ZeroPadding1D(padding=2)
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 3, 3)
        return layerStack.get_layers()

    @staticmethod
    def zero_padding_2d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.ZeroPadding2D(padding=(2, 2))
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 4, 4)
        return layerStack.get_layers()

    @staticmethod
    def zero_padding_3d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.ZeroPadding3D(padding=(2, 2, 2))
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 5, 5)
        return layerStack.get_layers()


class MergeLayers:
    def __init__(self):
        # Refer to https://keras.io/api/layers/merging_layers/
        import keras
        self.available_layers = {
            'Concatenate': keras.layers.Concatenate(),
            'Average': keras.layers.Average(),
            'Maximum': keras.layers.Maximum(),
            'Minimum': keras.layers.Minimum(),
            'Add': keras.layers.Add(),
            'Subtract': keras.layers.Subtract(),
            'Multiply': keras.layers.Multiply(),
            'Dot': keras.layers.Dot(axes=-1)
        }


class LocalConLayers:
    def __init__(self):
        # https://keras.io/api/layers/locally_connected_layers/
        self.available_layers = {
            'LocallyConnected1D': LocalConLayers.locally_connected_1d,
            'LocallyConnected2D': LocalConLayers.locally_connected_2d,
        }

    @staticmethod
    def locally_connected_1d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.LocallyConnected1D(10, 3, padding="valid")
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 3, 3)
        return layerStack.get_layers()

    @staticmethod
    def locally_connected_2d(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.LocallyConnected2D(10, (3, 3), padding="valid")
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack = DimSolver.check_before_add(layerStack, inserted_layer, input_shape, 4, 4)
        return layerStack.get_layers()


class ActLayers:
    def __init__(self):
        # Refer to https://keras.io/api/layers/activation_layers/
        self.available_layers = {
            'ReLU': ActLayers.relu_layer,
            'Softmax': ActLayers.softmax_layer,
            'LeakyReLU': ActLayers.leaky_relu_layer,
            'PReLU': ActLayers.prelu_layer,
            'ELU': ActLayers.elu_layer,
            'ThresholdedReLU': ActLayers.thresholded_relu_layer,
        }

    @staticmethod
    def relu_layer(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.ReLU(max_value=1.0)
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack.add(inserted_layer)
        return layerStack.get_layers()

    @staticmethod
    def softmax_layer(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.Softmax()
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack.add(inserted_layer)
        return layerStack.get_layers()

    @staticmethod
    def leaky_relu_layer(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.LeakyReLU()
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack.add(inserted_layer)
        return layerStack.get_layers()

    @staticmethod
    def prelu_layer(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.PReLU()
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack.add(inserted_layer)
        return layerStack.get_layers()

    @staticmethod
    def elu_layer(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.ELU()
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack.add(inserted_layer)
        return layerStack.get_layers()

    @staticmethod
    def thresholded_relu_layer(input_shape=None, config=None):
        import keras
        layerStack = LayerStack()
        candidate_layer = keras.layers.ThresholdedReLU()
        if config is None:
            inserted_layer = ConfigurationUtils.random_config(candidate_layer)
        else:
            inserted_layer = ConfigurationUtils.from_config(candidate_layer, config)
        layerStack.add(inserted_layer)
        return layerStack.get_layers()


LAYERLIST = {
    "LConv": ConvLayers(),
    "LPool": PoolLayers(),
    "LRnn": RNNLayers(),
    "LNorm": NormLayers(),

    "LReg": RegLayers(),
    "LAct": ActLayers(),

    "LResh": ReshapeLayers(),
    "LMerg": MergeLayers(),  # we do not consider merging layers in our layer pool.
    "LLocal": LocalConLayers(),
}

LayerInputNdim = {
    'Conv1D': [3],
    'Conv2D': [4],
    'Conv3D': [5],  # Tested
    'SeparableConv1D': [3],
    'SeparableConv2D': [4],
    'DepthwiseConv1D': [3],
    'DepthwiseConv2D': [4],
    'Conv2DTranspose': [4],
    'Conv3DTranspose': [5],  # Tested

    'MaxPooling1D': [3],
    'MaxPooling2D': [4],
    'MaxPooling3D': [5],  # Tested
    'AveragePooling1D': [3],
    'AveragePooling2D': [4],
    'AveragePooling3D': [5],  # Tested
    'GlobalMaxPooling1D': [3],
    'GlobalMaxPooling2D': [4],
    'GlobalMaxPooling3D': [5],  # Tested
    'GlobalAveragePooling1D': [3],
    'GlobalAveragePooling2D': [4],
    'GlobalAveragePooling3D': [5],  # Tested

    'LSTM': [3],
    'GRU': [3],
    'SimpleRNN': [3],
    'TimeDistributed': [3, 4, 5, 6],  # Tested
    'Bidirectional': [3],  # Tested
    "ConvLSTM2D": [5],  # Tested
    "ConvLSTM1D": [4],  # Tested
    "ConvLSTM3D": [6],  # Tested
    # 'rnn-1': RNNLayers.rnn,  # this should not be added, it is not a layer but a cell

    'BatchNormalization': [3, 4, 5, 6],
    'LayerNormalization': [2, 3, 4, 5, 6],

    'Dropout': [2, 3, 4, 5, 6],
    'SpatialDropout1D': [3],  # Tested
    'SpatialDropout2D': [4],  # Tested
    'SpatialDropout3D': [5],  # Tested
    'GaussianDropout': [2, 3, 4, 5, 6],
    'GaussianNoise': [2, 3, 4, 5, 6],
    'ActivityRegularization': [2, 3, 4, 5, 6],
    'AlphaDropout': [2, 3, 4, 5, 6],

    'Flatten': [2, 3, 4, 5, 6],
    'RepeatVector': [2],
    'Permute': [3],  # Tested
    'Cropping1D': [3],
    'Cropping2D': [4],
    'Cropping3D': [5],  # Tested
    'UpSampling1D': [3],
    'UpSampling2D': [4],
    'UpSampling3D': [5],  # Tested
    "ZeroPadding1D": [3],
    "ZeroPadding2D": [4],
    "ZeroPadding3D": [5],  # Tested

    'ReLU': [2, 3, 4, 5, 6],
    'Softmax': [2, 3, 4, 5, 6],
    'LeakyReLU': [2, 3, 4, 5, 6],
    'PReLU': [2, 3, 4, 5, 6],
    'ELU': [2, 3, 4, 5, 6],
    'ThresholdedReLU': [2, 3, 4, 5, 6],

    "LocallyConnected1D": [3],
    "LocallyConnected2D": [4]
}

LayerOutputNdim = {
    'Conv1D': [3],
    'Conv2D': [4],
    'Conv3D': [5],  # Tested
    'SeparableConv1D': [3],
    'SeparableConv2D': [4],
    'DepthwiseConv1D': [3],
    'DepthwiseConv2D': [4],
    'Conv2DTranspose': [4],
    'Conv3DTranspose': [5],  # Tested

    'MaxPooling1D': [3],
    'MaxPooling2D': [4],
    'MaxPooling3D': [5],  # Tested
    'AveragePooling1D': [3],
    'AveragePooling2D': [4],
    'AveragePooling3D': [5],  # Tested
    'GlobalMaxPooling1D': [2, 3],
    'GlobalMaxPooling2D': [2, 4],
    'GlobalMaxPooling3D': [2, 5],  # Tested
    'GlobalAveragePooling1D': [2, 3],
    'GlobalAveragePooling2D': [2, 4],
    'GlobalAveragePooling3D': [2, 5],  # Tested

    'LSTM': [3],
    'GRU': [3],
    'SimpleRNN': [3],
    'TimeDistributed': [3, 4, 5, 6],  # Tested
    'Bidirectional': [3],  # Tested
    "ConvLSTM2D": [5, 4],  # Tested
    "ConvLSTM1D": [4, 3],  # Tested
    "ConvLSTM3D": [6, 5],  # Tested
    # 'rnn-1': RNNLayers.rnn,  # this should not be added, it is not a layer but a cell

    'BatchNormalization': [3, 4, 5, 6],
    'LayerNormalization': [2, 3, 4, 5, 6],

    'Dropout': [2, 3, 4, 5, 6],
    'SpatialDropout1D': [3],  # Tested
    'SpatialDropout2D': [4],  # Tested
    'SpatialDropout3D': [5],  # Tested
    'GaussianDropout': [2, 3, 4, 5, 6],
    'GaussianNoise': [2, 3, 4, 5, 6],
    'ActivityRegularization': [2, 3, 4, 5, 6],
    'AlphaDropout': [2, 3, 4, 5, 6],

    'Flatten': [2],
    'RepeatVector': [3],
    'Permute': [3],  # Tested
    'Cropping1D': [3],
    'Cropping2D': [4],
    'Cropping3D': [5],  # Tested
    'UpSampling1D': [3],
    'UpSampling2D': [4],
    'UpSampling3D': [5],  # Tested
    "ZeroPadding1D": [3],
    "ZeroPadding2D": [4],
    "ZeroPadding3D": [5],  # Tested

    'ReLU': [2, 3, 4, 5, 6],
    'Softmax': [2, 3, 4, 5, 6],
    'LeakyReLU': [2, 3, 4, 5, 6],
    'PReLU': [2, 3, 4, 5, 6],
    'ELU': [2, 3, 4, 5, 6],
    'ThresholdedReLU': [2, 3, 4, 5, 6],

    "LocallyConnected1D": [3],
    "LocallyConnected2D": [4]
}
