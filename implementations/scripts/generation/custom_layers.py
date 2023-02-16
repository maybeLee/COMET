import keras
import tensorflow as tf


class CustomCastLayer(keras.layers.Layer):
    """
    Customized Keras Layer To Use tf.cast For Tensor Datatype Casting.
    The Usage of Tensor Casting Is Almost The Same As `tf.cast`.
    The Difference Is That The Parameter Passed To This Customized Layer Is `target_dtype` Instead Of `dtype`

    """

    def __init__(self, target_dtype, **kwargs):
        self.target_dtype = target_dtype
        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        Do Nothing
        """
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        output = tf.cast(inputs, self.target_dtype)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"target_dtype": self.target_dtype})
        return config

    def compute_output_shape(self, input_shape):
        """
        Do Nothing
        """
        return input_shape


# Currently we do not use this CustomPadLayer because the Keras's padding1D, padding2D, padding3D is enough to solve the problem
class CustomPadLayer(keras.layers.Layer):
    """
    Customized Keras Layer To Use tf.padding.
    Follow the same usage of https://www.tensorflow.org/api_docs/python/tf/pad
    For simplicity, we only use the `constant` mode.
    Arguments:
        padding: the option when padding, the dimension of paddings should be dim(input)-1 because
                  padding will not influence the batch dimension.
        constant_values: the constant values that we use to fill
    Example:

        t = np.array([[[1, 2, 3], [4, 5, 6]]])  # shape: (1, 2, 3)
        padding = [[1, 1,], [2, 2]]  # shape: (2, 2)
        # rank of 't' is 2.
        CustomPad(padding)(t)   # [[[0, 0, 0, 0, 0, 0, 0],
                                 #  [0, 0, 1, 2, 3, 0, 0],
                                 #  [0, 0, 4, 5, 6, 0, 0],
                                 #  [0, 0, 0, 0, 0, 0, 0]]]
                                 # shape: (1, 4, 7)
    Test scripts: ./tests/tools_test/test_pad_layer.py
    """

    def __init__(self, padding=[[0, 0]], constant_values=0, **kwargs):
        self.padding = padding  # add [0,0] to padding so we will not change the shape of batch dimension
        self.constant_values = constant_values
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        output = tf.pad(inputs, paddings=tf.constant([[0, 0]] + self.padding), mode="CONSTANT",
                        constant_values=self.constant_values)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"padding": self.padding, "constant_values": self.constant_values})
        return config

    def compute_output_shape(self, input_shape):
        """
        Formula to calculate the output shape
        Suppose the input_shape is (None, N, W, C), the `paddings` is [[a, b], [c, d]].
        The output_shape is (None, N+a+b, W+c+d, C).
        """
        input_shape_list = list(input_shape)
        padding = [[0, 0]] + self.padding
        assert len(input_shape_list) == len(padding)  # Two dimensions should match.
        output_shape = [None]
        for i, pad in zip(input_shape_list[1:], padding[1:]):
            output_shape.append(i + pad[0] + pad[1])
        return tuple(output_shape)


class CustomCropLayer(keras.layers.Layer):
    """
    Customized Keras Layer To Crop Tensor.
    Arguments:
        cropping: the option when cropping, the dimension of cropping should be dim(input)-1 because
                  cropping will not influence the batch dimension.
    Example:

        t = np.array([[...]])  # shape: (1, 4, 5)
        padding = [[1, 1,], [2, 2]]  # shape: (2, 2)
        # rank of 't' is 2.
        CustomPad(padding)(t)  # shape: (1, 2, 1)

    Test scripts: ./tests/tools_test/test_custom_crop_layer.py
    """

    def __init__(self, cropping, **kwargs):
        self.cropping = cropping
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        input_shape = inputs.shape.as_list()
        indices = [slice(None)]
        cropping = [[0, 0]] + self.cropping  # add [0,0] to padding so we will not change the shape of batch dimension
        for shape, crop in zip(input_shape[1:], cropping[1:]):
            indices.append(slice(0 + crop[0], shape - crop[1]))
        return inputs[indices]

    def get_config(self):
        config = super().get_config()
        config.update({"cropping": self.cropping})
        return config

    def compute_output_shape(self, input_shape):
        """
        Formula to calculate the output shape
        Suppose the input_shape is (None, N, W, C), the `cropping` is [[a, b], [c, d]].
        The output_shape is (None, N-a-b, W-c-d, C).
        """
        input_shape_list = list(input_shape)
        cropping = [[0, 0]] + self.cropping
        assert len(input_shape_list) == len(cropping)  # Two dimensions should match.
        output_shape = [None]
        for i, crop in zip(input_shape[1:], cropping[1:]):
            output_shape.append(i - crop[0] - crop[1])
        return tuple(output_shape)


class CustomExpandLayer(keras.layers.Layer):
    """
    Customized Keras Layer To Expand Tensor's Dimension
    Follow the same usage of https://www.tensorflow.org/api_docs/python/tf/expand_dims
    Arguments:
        axis: Integer specifying the dimension index at which to expand the shape of input.
         Given an input of D dimensions, axis must be in range [-(D+1), D] (inclusive).
    Example:
        x = keras.layers.Input((10,10,3))  # x.shape == (None, 10, 10, 3)
        y = CustomExpandLayer(axis=1)(x)  # y.shape == (None, 1, 10, 10, 3)
        y = CustomExpandLayer(axis=2)(x)  # y.shape == (None, 10, 1, 10, 3)

    Test scripts: ./tests/tools_test/test_custom_expand_layer.py
    Currently it is tested in 2D, 3D, 4D, and 5D tensor
    """

    def __init__(self, axis=1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.expand_dims(inputs, self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

    def compute_output_shape(self, input_shape):
        """
        Formula to calculate the output shape
        Suppose the input_shape is [None, N, W, C]:
        axis=0:
            output_shape: [1, None, N, W, C]
        axis=1 (default):
            output_shape: [None, 1, N, W, C]
        axis=2:
            output_shape: [None, N, 1, W, C]
        axis=3:
            output_shape: [None, N, W, 1, C]
        axis=4:
            output_shape: [None, N, W, C, 1]
        axis=5:
            raise Exception
        """
        input_shape_list = list(input_shape)
        if self.axis > len(input_shape_list):
            raise ValueError(f"axis {self.axis} should be smaller than input_shape + 1: {len(input_shape_list) + 1}")
        output_shape = input_shape_list[0:self.axis] + [1] + input_shape_list[self.axis:]
        return tuple(output_shape)  # we should use tuple!!! not list !!!


class CustomDropDimLayer(keras.layers.Layer):
    """
    Customized Keras Layer To Drop Tensor's Dimension
    Follow the idea of https://stackoverflow.com/questions/52453285/drop-a-dimension-of-a-tensor-in-tensorflow
    Arguments:
        axis (default: 1): Integer specifying which dimension index to drop.
        Given an input of D dimension, axis must be in range: [1, D-1] (inclusive).
    Example:
        x = keras.layers.Input((10,10,3))  # x.shape == (None, 10, 5, 3)
        y = CustomDropDimLayer(axis=1)(x)  # y.shape == (None, 5, 3)
        y = CustomDropDimLayer(axis=2)(x)  # y.shape == (None, 10, 3)
        y = CustomDropDimLayer(axis=3)(x)  # y.shape == (None, 10, 5)

    Test scripts: ./tests/tools_test/test_custom_drop_dim_layer.py
    Currently it is tested in 3D, 4D, and 5D tensor.
    """

    def __init__(self, axis=1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Something magic to automatically generate indices for array slicing.
        To determine a specific axis, we can use slice(None) to replace `:`
        """
        dim = len(inputs.shape)
        if self.axis > dim - 1 or self.axis < 1:
            raise ValueError(f"axis: {self.axis} should be within the range: [1, {dim - 1}] for {dim}D tensor")
        indices = [slice(None) for i in range(dim)]
        indices[self.axis] = 0
        return inputs[indices]

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

    def compute_output_shape(self, input_shape):
        """
        Formula to calculate the output shape
        Suppose the input_shape is [None, N, W, C]:
        axis=0:  # Although it is feasible, we don't allow this to happen
            Raise Exception
        axis=1 (default):
            output_shape: [None, W, C]
        axis=2:
            output_shape: [None, N, C]
        axis=3:
            output_shape: [None, N, W]
        axis=4:
            Raise Exception
        """
        input_shape_list = list(input_shape)
        output_shape = input_shape_list[0:self.axis] + input_shape_list[self.axis + 1:]
        return tuple(output_shape)
