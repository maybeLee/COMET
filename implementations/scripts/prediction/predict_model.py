import os

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1
import argparse
import sys
import warnings
import configparser
import redis
import pickle
import subprocess
import numpy as np
from scripts.logger.logger import Logger


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.simplefilter('ignore', DeprecationWarning)
import gc
import coverage
import os
from scripts.prediction.custom_objects import custom_objects


class Predictor(object):
    def __init__(self, flags):
        super().__init__()
        self.model_path = flags.model_path
        self.model_name = flags.model_name
        self.main_logger = Logger()
        self.onnx_path = f"{self.model_path.rsplit('.h5')[0]}.onnx"
        self.model = None
        self.input = None
        self.bk = flags.backend
        self.arcs = flags.arcs
        self.seed = flags.seed
        np.random.seed(self.seed)
        self.cfg = configparser.ConfigParser()
        self.cfg.read(f"./config/{flags.config_name}")
        pool = redis.ConnectionPool(host=self.cfg['redis']['host'], port=self.cfg['redis']['port'], db=flags.redis_db)
        self.redis_conn = redis.Redis(connection_pool=pool)
        self.parameters = self.cfg['parameters']
        self.gpu_ids = self.parameters["gpu_ids"]
        self.gpu_list = self.gpu_ids.split(",")
        if len(self.gpu_list) >= 2:
            self.gpu_list = self.gpu_list[1:]
        self.gpu_ids = ",".join(self.gpu_list)
        self.model_convertor_python = f"{flags.python_prefix}/model_convertor/bin/python"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore")

    def _transform_onnx(self):
        # transform tensorflow model to onnx model
        cuda_version = "cuda-11"
        envs = f"CUDA_HOME=/usr/local/{cuda_version} CUDA_ROOT=/usr/local/{cuda_version} LD_LIBRARY_PATH=/usr/local/{cuda_version}/lib64:$LD_LIBRARY_PATH PATH=/usr/local/{cuda_version}/bin:$PATH"
        onnx_flag = f'{self.model_path.split(".h5")[0]}.flag'
        while os.path.exists(onnx_flag):
            continue
        open(onnx_flag, "a").close()
        try:
            subprocess.call(
                f"{envs} {self.model_convertor_python} -u -m scripts.tools.prediction_toolkit --model_path {self.model_path}",
                shell=True, timeout=60 * 5)  # set the timeout of model conversion to 300 seconds
        except:
            pass
        try:
            os.remove(onnx_flag)
        except:
            pass

    def _init_input(self, input_shape):
        input_shape = list(input_shape)
        input_shape[0] = 100
        input_shape = tuple(input_shape)
        self.input = np.random.rand(*input_shape)

    def before_predict(self, ):
        input_shape = pickle.loads(self.redis_conn.hget(f"prediction_{self.model_name}", "shape"))
        self._init_input(input_shape)
        if self.bk == "tensorflow":
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
            import tensorflow as tf
            import keras
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            self.main_logger.info(tf.__version__)
            if len(self.gpu_list) >= 2:
                self.gpu_ids = str(self.gpu_list[-1])
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
            if self.bk == "tensorflow":
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
            else:
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20)
                sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            self.main_logger.info(f"TensorFlow is using GPU? {tf.test.gpu_device_name()}")
            self.model = keras.models.load_model(self.model_path, custom_objects=custom_objects())

        elif self.bk == "mxnet" or self.bk == "mxnet_new":
            os.environ["MXNET_ENGINE_TYPE"] = "NaiveEngine"
            os.environ["MXNET_SUBGRAPH_VERBOSE"] = "0"
            os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
            os.environ["MXNET_CUDNN_LIB_CHECKING"] = "0"
            os.environ[
                "MXNET_USE_OPERATOR_TUNING"] = "0"  # add this flag so mxnet's import stage will have low cpu load
            if len(self.gpu_list) >= 2:
                self.gpu_ids = str(self.gpu_list[0])
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
            os.environ["KERAS_BACKEND"] = "mxnet"
            import keras
            import mxnet as mx
            self.main_logger.info(f"MXNet is using GPU? {bool(mx.gpu(0))}")
            self.model = keras.models.load_model(self.model_path, custom_objects=custom_objects())

        elif self.bk == "pytorch":
            if len(self.gpu_list) >= 2:
                self.gpu_ids = str(self.gpu_list[0])
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
            self._transform_onnx()  # change keras model to onnx
            import onnx
            from onnx2pytorch import ConvertModel
            onnx_model = onnx.load(self.onnx_path)
            self.model = ConvertModel(onnx_model, experimental=True)

        elif self.bk == "onnx":
            if len(self.gpu_list) >= 2:
                self.gpu_ids = str(self.gpu_list[0])
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
            self._transform_onnx()
            import onnxruntime as ort
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 10 * 1024 * 1024 * 1024,  # 10G
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider',
            ]
            self.model = ort.InferenceSession(self.onnx_path, providers=providers)

    def predict(self, ):
        if self.bk in ["tensorflow", "mxnet"]:
            if self.bk == "mxnet":
                pred = self.model.predict(self.input, batch_size=2)
            else:
                pred = self.model.predict(self.input)
        if self.bk == "pytorch":
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            import torch
            self.main_logger.info(f"Current GPU device for pytorch: {torch.cuda.current_device()}")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.main_logger.info(f"Now pytorch is using: {device}")
            with torch.no_grad():  # to avoid OOM: https://github.com/pytorch/pytorch/issues/67680
                self.input = torch.from_numpy(self.input)
                self.input = self.input.to(device)
                self.model = self.model.to(device)
                self.model.double()
                pred = self.model(self.input)
                pred = pred.detach().cpu().numpy()
                assert pred.shape[0] == self.input.shape[0]
        if self.bk == "onnx":
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            self.input = self.input.astype('float32')
            pred = self.model.run([output_name], {input_name: self.input})[0]

        self.main_logger.info("SUCCESS:Get prediction for {} successfully on {}!".format(self.model_name, self.bk))
        self.redis_conn.hset(f"prediction_{self.model_name}", self.bk, pickle.dumps(pred))
        self.redis_conn.hset(f"status_{self.model_name}", self.bk, pickle.dumps(0))

    def after_predict(self, ):
        del self.model
        if self.bk == "tensorflow" and self.arcs == 1:
            cov.stop()
            covData = cov.get_data()
            covBranch = [(files, sorted(covData.arcs(files))) for files in covData.measured_files()]  # covBranch: [(files, arcs), ()]
            arcs = {}
            for i in covBranch:
                file_path = i[0]
                arc = i[1]
                arc_list = []
                for ar in arc:
                    l, r = ar
                    arc_list.append(f"{l}_{r}")
                if len(arc_list) == 0:
                    continue
                arcs[file_path] = arc_list
            self.redis_conn.hset(f"arcs_{self.model_name}", self.bk, pickle.dumps(arcs))


if __name__ == "__main__":
    """Parser of command args"""
    parse = argparse.ArgumentParser()
    parse.add_argument("--model_path", type=str, help="the path of tensorflow model")
    parse.add_argument("--model_name", type=str, help="the name of the model")
    parse.add_argument("--backend", type=str, help="specify the backend")
    parse.add_argument("--config_name", type=str)
    parse.add_argument("--redis_db", type=int, default=0, help="the id of redis database")
    parse.add_argument("--seed", type=int, default=None, help="seed for generating dataset")
    parse.add_argument("--python_prefix", type=str, help="the prefix of the base python")
    parse.add_argument("--arcs", type=int, default=0, help="whether to collect arcs")
    flags, _ = parse.parse_known_args(sys.argv[1:])
    predictor = Predictor(flags)
    if flags.backend == "tensorflow" and flags.arcs == 1:
        cov = coverage.Coverage(auto_data=True, branch=True, config_file=f"./config/{flags.backend}_runtime.conf")
        cov.start()
    try:
        predictor.before_predict()
        predictor.predict()
    except Exception:
        predictor.after_predict()
        del predictor
        gc.collect()
        raise ValueError(f"CRASH ON BACKEND {flags.backend}!!!")

    predictor.after_predict()
    del predictor
    gc.collect()
