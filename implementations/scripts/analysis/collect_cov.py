import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from scripts.logger.logger import Logger
import sys
import argparse
import configparser
import os
import dill
import datetime
from classes.frameworks import Frameworks
from scripts.coverage.coverage_analysis import Coverage, Arcs
from scripts.tools import utils
import traceback
from multiprocessing import Pool


def get_guidance_by_name(guidance_s):
    guidance_strategy_dict = {"COVERAGE": Coverage(), "ARCS": Arcs()}
    return guidance_strategy_dict[guidance_s]


main_logger = Logger()


def parallel_coverage(args):
    framework, c_library, model_name, bk, guidance, python_prefix, config_dir, config_name, config, silent, coverage = args
    os.environ["GCOV_PREFIX"] = f"runtime/{bk}"
    # all gcda files will be saved in the runtime dir
    if bk == "tensorflow":
        envs_name = "tensorflow"
        os.environ["GCOV_PREFIX_STRIP"] = "4"
    elif bk == "mxnet":
        envs_name = "mxnet"
        os.environ["GCOV_PREFIX_STRIP"] = "6"
    elif bk == "pytorch":
        envs_name = "pytorch"
        os.environ["GCOV_PREFIX_STRIP"] = "6"
    elif bk == "onnx":
        envs_name = "pytorch"
    else:
        raise NotImplementedError(f"Backend {bk} Not Implemented When Collecting Code Coverage")
    python_bin = f"{python_prefix}/{envs_name}/bin/python"
    os.environ["COVERAGE_FILE"] = ".coverage.{}".format(bk)
    if coverage is True:
        cov_uni_bk = guidance.collect_info(bk, python_bin, model_name, config_dir, framework, c_library,
                              silent, None)
    elif coverage is False:
        cov_uni_bk = 0
    return framework, cov_uni_bk


class Driver(object):
    def __init__(self, flags):
        super().__init__()
        self.config = configparser.ConfigParser()
        self.config_name = flags.config_name
        self.config_path = f"./config/{self.config_name}"
        self.config_dir = os.path.dirname(self.config_path)
        self.config.read(self.config_path)
        self.guidance_strategy = self.config["parameters"].get("guidance_strategy")
        self.backends = [self.config["parameters"].get("coverage")]
        self.output_dir = self.config["parameters"].get("output_dir")
        self.experiment_dir = os.path.join(self.output_dir, "results")
        self.instrument_result_dir = os.path.join(self.experiment_dir, "instrument_result")
        self.delta_dir = os.path.join(self.experiment_dir, "delta")
        self.python_prefix = self.config["parameters"].get("python_prefix").rstrip("/")

        # set instrumentation configuration
        self.frameworks_trs_path = {}
        self.frameworks = {}
        self.c_libraries = {
                            "tensorflow": self.config["parameters"]["c_tensorflow"],
                            "mxnet": self.config["parameters"]["c_mxnet"],
                            "pytorch": self.config['parameters']["c_pytorch"],
                            "onnx": self.config['parameters']["c_onnx"],
                            }
        self.py_libraries = {"tensorflow": self.config['parameters']["py_tensorflow"],
                             "mxnet": self.config['parameters']["py_mxnet"],
                             "pytorch": self.config['parameters']["py_pytorch"],
                             "onnx": self.config['parameters']["py_onnx"],
                             }
        self.code_json_file = os.path.join("config", os.path.dirname(self.config_name), "codes.json")

        self.initiate_dirs()
        self.guidance = None

    def initiate_dirs(self, ):
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        if not os.path.exists(self.instrument_result_dir):
            os.makedirs(self.instrument_result_dir)
        if not os.path.exists(self.delta_dir):
            os.makedirs(self.delta_dir)

    def initiate_frameworks_trs_path(self):
        """
        Initiate framework's test requirements' path
        """
        for bk in self.backends:
            self.frameworks_trs_path[bk] = os.path.join(self.instrument_result_dir, "{}.pkl".format(bk))

    def initiate_framework(self, bk):
        """
        Initiate Framework
        Args:
            bk (str): backend's name
        """
        py_framework_dir = self.py_libraries[bk]
        c_framework_dir = self.c_libraries[bk]
        self.frameworks[bk] = Frameworks(bk, py_root_dir=py_framework_dir, c_root_dir=c_framework_dir,
                                         python_prefix=self.python_prefix)
        # self.frameworks[bk].clear_and_profile()  # clear all gcda, no need
        with open(os.path.join(self.delta_dir, bk + "_coverage.txt"), "a") as file:
            file.write(f"LineCoverage,BranchCoverage,LineHit,TotalLines,BranchHit,TotalBranches\n")

    def save_framework(self, bk):
        with open(self.frameworks_trs_path[bk], "wb") as file:
            dill.dump(self.frameworks[bk], file)
        # also save the coverage result
        with open(os.path.join(self.delta_dir, bk + "_coverage.txt"), "a") as file:
            cache_py = self.frameworks[bk].coverage(type="py")
            cache_c = self.frameworks[bk].coverage(type="c")
            file.write(f"python result:\n")
            line_coverage, branch_coverage, line_hit, total_lines, branch_hit, total_branches = cache_py
            file.write(f"{line_coverage},{branch_coverage},{line_hit},{total_lines},{branch_hit},{total_branches}\n")
            file.write(f"c result:\n")
            line_coverage, branch_coverage, line_hit, total_lines, branch_hit, total_branches = cache_c
            file.write(f"{line_coverage},{branch_coverage},{line_hit},{total_lines},{branch_hit},{total_branches}\n")

    def coverage(self, model_name: str, coverage: bool):
        silent = {}
        backend_list = ["tensorflow"]  # we hard code the library name that we want to collect code coverage
        for bk in backend_list:
            silent[bk] = False
            main_logger.info("First time, initiate the framework")
            self.initiate_framework(bk)
            silent[bk] = True
        # create args
        args = [(
            self.frameworks[bk], self.c_libraries[bk], model_name,
            bk, self.guidance, self.python_prefix, self.config_dir,
            self.config_name, self.config, silent[bk], coverage
        ) for bk in backend_list]
        # conduct multiprocessing
        with Pool(len(backend_list)+4) as p:
            parallel_return = p.map(parallel_coverage, args)  # return: pre_status_bk, predict_output_bk, framework, cov_uni_bk
        status = {}
        for i, bk in enumerate(backend_list):
            self.frameworks[bk] = parallel_return[i][0]
            self.guidance.cov_uni[bk] = parallel_return[i][1]
            self.save_framework(bk)

    def gen_model(self, ):
        self.guidance = get_guidance_by_name("COVERAGE")
        self.coverage(model_name="fake_name", coverage=True)

    def run(self, ):
        starttime = datetime.datetime.now()
        self.initiate_frameworks_trs_path()
        try:
            self.gen_model()
        except Exception:
            print(traceback.format_exc())
        from keras import backend as K
        K.clear_session()
        endtime = datetime.datetime.now()
        time_delta = endtime - starttime
        h, m, s = utils.ToolUtils.get_HH_mm_ss(time_delta)
        main_logger.info(f"Coverage process is done: Time used: {h} hour,{m} min,{s} sec")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--config_name", type=str, help="config name")
    flags, _ = parse.parse_known_args(sys.argv[1:])
    driver = Driver(flags=flags)
    driver.run()
