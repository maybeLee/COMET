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
from scripts.logger.logger import Logger
import subprocess
import sys
import argparse
import configparser
import os
import numpy as np
import time
from classes.frameworks import Frameworks
import dill
import pickle
import datetime
from scripts.tools import utils
import redis
import traceback
from multiprocessing import Pool
from scripts.prediction.custom_objects import custom_objects
from scripts.generation.run import generate_metrics_result, is_nan_or_inf, partially_nan_or_inf, get_guidance_by_name
from scripts.coverage.architecture_coverage import ArchitectureMeasure
from scripts.prediction.parallel_predict import parallel_predict


main_logger = Logger()


class Driver(object):
    def __init__(self, flags):
        super().__init__()
        self.config = configparser.ConfigParser()
        self.config_name = flags.config_name
        self.config_path = f"./config/{self.config_name}"
        self.config_dir = os.path.dirname(self.config_path)
        self.config.read(self.config_path)
        self.guidance_strategy = self.config["parameters"].get("guidance_strategy")

        # define experiment-related directory
        self.output_dir = self.config["parameters"].get("output_dir")
        self.experiment_dir = os.path.join(self.output_dir, "results")
        self.instrument_result_dir = os.path.join(self.experiment_dir, "instrument_result")
        self.delta_dir = os.path.join(self.experiment_dir, "delta")
        self.python_prefix = self.config["parameters"].get("python_prefix").rstrip("/")
        self.backends = self.config["parameters"].get("backend").split(" ")
        self.gpu_ids = self.config["parameters"]["gpu_ids"]
        self.gpu_list = self.gpu_ids.split(",")
        if len(self.gpu_list) >= 2:
            self.gpu_ids = str(self.gpu_list[0])
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        self.layer_selector = None
        self.introduced_name = None
        self.picked_mutator = None

        # set instrumentation configuration
        self.frameworks_trs_path = {}
        self.frameworks = {}
        self.c_libraries = {"tensorflow": self.config['parameters']["c_tensorflow"],
                            "mxnet": self.config['parameters']["c_mxnet"],
                            "pytorch": self.config['parameters']["c_pytorch"],
                            "onnx": self.config['parameters']["c_onnx"],
                            }
        self.py_libraries = {"tensorflow": self.config['parameters']["py_tensorflow"],
                             "mxnet": self.config['parameters']["py_mxnet"],
                             "pytorch": self.config['parameters']["py_pytorch"],
                             "onnx": self.config['parameters']["py_onnx"],
                             }
        self.code_json_file = os.path.join("config", os.path.dirname(self.config_name), "codes.json")

        # initiate metrics
        self.metrics_list = self.config["parameters"].get("metrics").split(" ")
        self.metrics_results = {k: dict() for k in self.metrics_list}
        self.guidance = None
        # init and clear redis
        pool = redis.ConnectionPool(host=self.config['redis']['host'], port=self.config['redis']['port'],
                                    db=self.config['redis'].getint('redis_db'))
        self.redis_conn = redis.Redis(connection_pool=pool)
        for k in self.redis_conn.keys():
            self.redis_conn.delete(k)
        self.initiate_dirs()

        # part for the search algorithm
        self.mcmc = None
        self.coverage = self.config["parameters"].get("coverage")  # possible option: 1, 0 or library name, string
        self.current_seed_name = None
        self.architecture_measure = ArchitectureMeasure()
        self.start_time = time.time()

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
        self.save_framework(bk)

    def load_framework(self, bk):
        with open(self.frameworks_trs_path[bk], "rb") as file:
            self.frameworks[bk] = dill.load(file)

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

    def save_metrics(self, new_seed_name, metric):
        with open(os.path.join(self.delta_dir, "delta.txt"), "a") as file:
            file.write(f"{new_seed_name}: {metric}\n")

    def run_muffin_mutants(self):
        muffin_dir = self.output_dir
        iterations = os.listdir(muffin_dir)
        for iter in iterations:
            main_logger.info(f"Iteration: {iter}")
            if "100_4_4" in iter or "job" in iter:
                # skip the runtime one
                continue
            model_path = os.path.join(muffin_dir, iter, "models", "tensorflow.h5")
            self.predict(model_path, str(iter))
            from keras import backend as K
            K.clear_session()

    def collect_current_coverage(self):
        subprocess.call(f"python -u -m scripts.analysis.collect_cov --config_name {self.config_name}", shell=True)

    def gen_model(self, ):
        self.guidance = get_guidance_by_name(self.guidance_strategy)
        self.run_muffin_mutants()
        main_logger.info("Stopping the Program ..., Do Not Clear GCDA")
        self.collect_current_coverage()

    def clear_py_coverage(self):
        main_logger.info("Clearing Py Coverage....")
        for bk in self.backends:
            subprocess.call(f"rm -rf .coverage.{bk}", shell=True)

    def clear_gcda_coverage(self):
        main_logger.info(f"Clearing GCDA....")
        for bk in self.backends:
            subprocess.call(f"find runtime/{bk}/ -name '*.gcda' -exec rm -rf " + "{} \\;", shell=True)

    def save_arcs(self, arcs, bk):
        # also save the coverage result
        arc_count = 0
        for file_path in arcs:
            arc_count += len(arcs[file_path])
        with open(os.path.join(self.delta_dir, bk + "_arcs.txt"), "a") as file:
            end_time = time.time()
            file.write(f"Current time (seconds): {end_time - self.start_time}\n")
            file.write(f"arcs result (covered arcs):\n")
            file.write(f"{arc_count}\n")

    def save_api_cov(self, bk="tensorflow"):
        api_pair_cov, config_cov, input_cov, ndims_cov, dtype_cov, shape_cov = self.architecture_measure.coverage()
        with open(os.path.join(self.delta_dir, bk + "_api_cov.txt"), "a") as file:
            end_time = time.time()
            file.write(f"Current time (seconds): {end_time - self.start_time}\n")
            file.write(f"api coverage result (covered api pair; covered config; covered input; covered ndims; covered dtype; covered shape):\n")
            file.write(f"{api_pair_cov}; {config_cov}; {input_cov}; {ndims_cov}; {dtype_cov}; {shape_cov}\n")

    def predict(self, model_path: str, model_name: str):
        # change the prediction to multiprocessing
        import keras
        model = keras.models.load_model(model_path, custom_objects=custom_objects())
        self.architecture_measure.update_diversity(model, model_name)
        self.save_api_cov()
        shape = list(model.input_shape)
        shape[0] = 1
        model.summary()
        self.redis_conn.hset(f"prediction_{model_name}", "shape", pickle.dumps(shape))
        del model
        predict_output = {b: [] for b in self.backends}
        all_backends_predict_status = True
        silent = {}
        for bk in self.backends:
            silent[bk] = False
            if not os.path.exists(self.frameworks_trs_path[bk]):
                main_logger.info("First time, initiate the framework")
                self.initiate_framework(bk)
                self.clear_py_coverage()
                self.clear_gcda_coverage()
                self.guidance.before_predict(self.frameworks[bk])
                silent[bk] = True
            elif len(self.frameworks) != len(self.backends):
                # frameworks have not been initialized yet, load the framework from disk
                main_logger.info("Loading frameworks")
                self.load_framework(bk)
        # create args
        args = [(
            self.frameworks[bk], self.c_libraries[bk], model_path, model_name,
            bk, self.guidance, self.python_prefix, self.config_dir,
            self.config_name, self.config, silent[bk], self.coverage, self.guidance_strategy
        ) for bk in self.backends]
        # conduct multiprocessing
        with Pool(len(self.backends) + 1) as p:
            parallel_return = p.map(parallel_predict,
                                    args)  # return: pre_status_bk, predict_output_bk, framework, cov_uni_bk
        status = {}
        for i, bk in enumerate(self.backends):
            # status[bk] = self._predict_model(model_path=model_path, model_name=model_name, bk=bk,
            #                                  predict_output=predict_output, silent=silent)
            status[bk] = parallel_return[i][0]
            self.frameworks[bk] = parallel_return[i][2]
            self.guidance.cov_uni[bk] = parallel_return[i][3]
            if self.guidance_strategy == "ARCS":
                self.guidance.arcs[bk] = parallel_return[i][4]
                self.save_arcs(self.guidance.arcs[bk], bk)
            if self.guidance_strategy == "COVERAGE":
                # we only save the framework when using coverage as guidance
                self.save_framework(bk)
            if status[bk] == 0:
                predict_output[bk] = parallel_return[i][1]
                main_logger.info(f"Success on backend: {bk} of model {model_name}")
            else:
                all_backends_predict_status = False
                main_logger.info(f"Fail on backend: {bk} of model {model_name}")
        accumulative_incons = 0
        success_bk = [bk for bk in status.keys() if status[bk] == 0]  # get the bk that successes
        crash_bk = [bk for bk in status.keys() if
                    status[bk] != 0]  # WARNING: when a bk crash, its status will not necessarily be 1
        for bk in crash_bk:
            _ = predict_output.pop(bk)
        if (len(predict_output) >= 2 or len(predict_output) == len(self.backends)) and status["tensorflow_new"] == 0:
            # if there are more than one library success and tensorflow should be correct
            predictions = list(predict_output.values())
            self.metrics_results, accumulative_incons = generate_metrics_result(res_dict=self.metrics_results,
                                                                                predict_output=predict_output,
                                                                                model_name=model_name)
            if is_nan_or_inf(accumulative_incons):
                status_crash_nan = True
                nan_or_inf, nan_idxs = partially_nan_or_inf(predictions, len(predictions))
                if nan_or_inf:
                    main_logger.info(f"Error: Found one NaN bug. move NAN model, the NaN idx is: {nan_idxs}")
                else:
                    main_logger.info("Error: Found one NaN Model on all libraries.")
            if accumulative_incons >= 50:
                main_logger.info("New inconsistency issue found!!!")

        return None

    def obtain_metrics(self, ):
        pass

    def run(self, ):
        starttime = datetime.datetime.now()
        self.initiate_frameworks_trs_path()
        try:
            self.gen_model()
        except Exception:
            print(traceback.format_exc())
        endtime = datetime.datetime.now()
        time_delta = endtime - starttime
        h, m, s = utils.ToolUtils.get_HH_mm_ss(time_delta)
        main_logger.info(f"Analysis process is done: Time used: {h} hour,{m} min,{s} sec")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--config_name", type=str, help="config name")
    flags, _ = parse.parse_known_args(sys.argv[1:])
    driver = Driver(flags=flags)
    driver.run()
