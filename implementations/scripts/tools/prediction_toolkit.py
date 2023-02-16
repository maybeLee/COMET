import os
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import argparse
import sys
import gc
from scripts.prediction.custom_objects import custom_objects


def transform_onnx(model_path: str):
    model_name = model_path.split(".h5")[0]
    onnx_path = f"{model_name}.onnx"
    if os.path.exists(onnx_path):
        return
    import keras
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])  # 5G memory limitation
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    from keras import backend as K
    K.set_learning_phase(0)
    import tf2onnx
    model = keras.models.load_model(model_path, custom_objects=custom_objects())
    if tf.__version__.split(".")[0] == 1:
        input_shape = model.layers[0].input_shape
    else:
        input_shape = model.layers[0].input_shape[0]
    spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
    _, _ = tf2onnx.convert.from_keras(model, input_signature=spec, \
        opset=15, output_path=onnx_path)
    del model
    del _


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--model_path", type=str, help="the path of tensorflow model")
    flags, _ = parse.parse_known_args(sys.argv[1:])
    model_path = flags.model_path
    transform_onnx(model_path)
    gc.collect()
