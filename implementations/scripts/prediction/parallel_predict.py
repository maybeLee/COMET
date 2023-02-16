import warnings
from scripts.logger.logger import Logger
import subprocess
import pickle
import datetime
from scripts.tools import utils
import redis
import os

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
main_logger = Logger()


def parallel_predict(args):
    framework, c_library, model_path, model_name, bk, guidance, python_prefix, config_dir, config_name, config, silent, coverage, guidance_strategy = args
    pool = redis.ConnectionPool(host=config['redis']['host'], port=config['redis']['port'],
                                db=config['redis'].getint('redis_db'))
    redis_conn = redis.Redis(connection_pool=pool)
    redis_db = config['redis'].getint('redis_db')
    predict_st = datetime.datetime.now()
    os.environ["GCOV_PREFIX"] = f"runtime/{bk}"
    # all gcda files will be saved in the runtime dir
    envs_name_list = {
        "pytorch": "pytorch",
        "onnx": "pytorch",
        "mxnet": "mxnet",
        "tensorflow": "tensorflow",
    }
    gcov_prefix_list = {
        "tensorflow": "4",
        "pytorch": "6",
        "onnx": "",
        "mxnet": "6",
    }
    os.environ["GCOV_PREFIX_STRIP"] = gcov_prefix_list[bk]
    envs_name = envs_name_list[bk]
    python_bin = f"{python_prefix}/{envs_name}/bin/python"
    cuda_version = "cuda-11"
    envs = f"COVERAGE_FILE=.coverage.{bk} CUDA_HOME=/usr/local/{cuda_version} CUDA_ROOT=/usr/local/{cuda_version} LD_LIBRARY_PATH=/usr/local/{cuda_version}/lib64:$LD_LIBRARY_PATH PATH=/usr/local/{cuda_version}/bin:$PATH"
    if guidance_strategy == "ARCS" and coverage == bk:
        arcs = 1
        pref = f"{envs} {python_bin} -m"
    else:
        arcs = 0
        pref = f"{envs} {python_bin} -m coverage run -m -a --rcfile {config_dir}/{bk}.conf --branch -m"
    cmd = f"{pref} scripts.prediction.predict_model --backend {bk} --model_path {model_path} --model_name {model_name} --redis_db {redis_db} --config_name {config_name} --seed 2022 --python_prefix {python_prefix} --arcs {arcs} -W ignore::DeprecationWarning"
    main_logger.info(f"Start Predicting on Backend: {bk}")
    pre_status_bk = subprocess.call(
        cmd, shell=True, timeout=60*10)
    # this can be parallel, set time out for each prediction process, the time out is set to 600 seconds.
    predict_et = datetime.datetime.now()
    predict_td = predict_et - predict_st
    h, m, s = utils.ToolUtils.get_HH_mm_ss(predict_td)
    main_logger.info("Prediction Time Used on {} : {}h, {}m, {}s".format(bk, h, m, s))
    predict_output_bk = 0
    if pre_status_bk == 0:
        main_logger.info("loading the redis")
        predict_output_bk = pickle.loads(redis_conn.hget(f"prediction_{model_name}", bk))
        main_logger.info("finish loading from redis")
    else:
        main_logger.info(f"{model_name} crash on backend {bk} when predicting")
    if coverage == "1" or coverage == bk:
        cov_uni_bk = guidance.collect_info(bk, python_bin, model_name, config_dir, framework, c_library,
                                           silent, redis_conn)
    else:
        cov_uni_bk = 0
        if guidance_strategy == "ARCS":
            guidance.arcs[bk] = {}
    if guidance_strategy == "ARCS":
        return pre_status_bk, predict_output_bk, framework, cov_uni_bk, guidance.arcs[bk]
    else:
        return pre_status_bk, predict_output_bk, framework, cov_uni_bk
