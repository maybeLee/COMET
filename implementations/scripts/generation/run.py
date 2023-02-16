import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1
from scripts.logger.logger import Logger
import subprocess
import argparse
import configparser
import numpy as np
import shutil
from classes.frameworks import Frameworks
import dill
import pickle
import datetime
from scripts.coverage.coverage_analysis import Coverage, Arcs
from scripts.tools import utils
import redis
from itertools import combinations
import math
import traceback
from scripts.tools.seed_selection_logic import MutatorSelector
from multiprocessing import Pool
import gc
from scripts.prediction.predict_model import custom_objects
from scripts.prediction.parallel_predict import parallel_predict
from scripts.coverage.architecture_coverage import ArchitectureMeasure
from scripts.tools.architecture_utils import ArchitectureUtils
from utils.utils import *


# Fix: CUDA_ERROR_NOT_INITIALIZED error, see post: https://ai.stackexchange.com/questions/36318/tensorflow-gpu-and-multiprocessing
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
global tf


main_logger = Logger()


def get_guidance_by_name(guidance_s):
    guidance_strategy_dict = {"COVERAGE": Coverage(), "ARCS": Arcs()}
    return guidance_strategy_dict[guidance_s]


def get_seed_selector_by_name(seed_selector_name):
    seed_selector_dict = {"MUTATORS": MutatorSelector}
    return seed_selector_dict[seed_selector_name]


def generate_metrics_result(res_dict, predict_output, model_name):
    main_logger.info("Generating Metrics Result")
    accumulative_incons = 0
    backends_pairs_num = 0
    # Compare results pair by pair
    for pair in combinations(predict_output.items(), 2):
        backends_pairs_num += 1
        backend1, backend2 = pair
        bk_name1, prediction1 = backend1
        bk_name2, prediction2 = backend2
        if prediction1.shape != prediction2.shape:
            # If cases happen when the shape of prediction is already inconsistent, return inconsistency as None to raise a warning
            return res_dict, None
        for metrics_name, metrics_result_dict in res_dict.items():
            metrics_func = utils.MetricsUtils.get_metrics_by_name(metrics_name)
            # metrics_results in list type
            if metrics_name == "D_MAD":
                y_test = np.ones_like(prediction1)
                metrics_results = metrics_func(prediction1, prediction2, y_test)
            else:
                metrics_results = metrics_func(prediction1, prediction2)

            # ACC -> float: The sum of all inputs under all backends
            main_logger.info(f"Inconsistency between {bk_name1} and {bk_name2} is {sum(metrics_results)}")
            accumulative_incons += sum(metrics_results)

            for input_idx, delta in enumerate(metrics_results):
                delta_key = "{}_{}_{}_input{}".format(model_name, bk_name1, bk_name2, input_idx)
                metrics_result_dict[delta_key] = delta

    main_logger.info(f"Accumulative Inconsistency: {accumulative_incons}")
    return res_dict, accumulative_incons


def is_nan_or_inf(t):
    if math.isnan(t) or math.isinf(t):
        return True
    else:
        return False


def is_none(t):
    return t is None


def partially_nan_or_inf(predictions, bk_num):
    """
    Check if there is NAN in the result
    """

    def get_nan_num(nds):
        _nan_num = 0
        nan_idxs = []
        for idx, nd in enumerate(nds):
            if np.isnan(nd).any() or np.isinf(nd).any():
                _nan_num += 1
                nan_idxs.append(idx)
        return _nan_num, nan_idxs

    if len(predictions) == bk_num:
        for input_predict in zip(*predictions):
            nan_num, nan_idxs = get_nan_num(input_predict)
            if 0 < nan_num < bk_num:
                return True, nan_idxs
            else:
                continue
        return False, None
    else:
        raise Exception("wrong backend amounts")


reach_half = False


def continue_checker(**run_stat):
    start_time = run_stat['start_time']
    time_limitation = run_stat['time_limit']
    cur_counters = run_stat['cur_counters']
    counters_limit = run_stat['counters_limit']
    s_mode = run_stat['stop_mode']
    # if timing
    global reach_half
    if s_mode == 'TIMING':
        hours, minutes, seconds = utils.ToolUtils.get_HH_mm_ss(datetime.datetime.now() - start_time)
        total_minutes = hours * 60 + minutes
        main_logger.info(f"INFO: Mutation progress: {total_minutes}/{time_limitation} Minutes!")
        if total_minutes < time_limitation:
            if total_minutes >= time_limitation/2 and reach_half is False:
                reach_half = True
                return True, "half_times"
            if cur_counters == 300:
                main_logger.info(f"INFO: Reach 300 Mutants!")
                return True, "300_models"
            return True, False
        else:
            return False, False
    # if counters
    elif s_mode == 'COUNTER':
        if cur_counters < counters_limit:
            main_logger.info("INFO: Mutation progress {}/{}".format(cur_counters + 1, counters_limit))
            return True, False
        else:
            return False, False
    else:
        raise Exception(f"Error! Stop Mode {s_mode} not Found!")


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
        self.crash_dir = os.path.join(self.experiment_dir, "crashes")
        self.model_dir = os.path.join(self.experiment_dir, "models")
        self.nan_dir = os.path.join(self.experiment_dir, "nan")
        self.instrument_result_dir = os.path.join(self.experiment_dir, "instrument_result")
        self.seed_selector_dir = os.path.join(self.experiment_dir, "seed_selector")
        self.delta_dir = os.path.join(self.experiment_dir, "delta")
        self.python_prefix = self.config["parameters"].get("python_prefix").rstrip("/")
        self.backends = self.config["parameters"].get("backend").split(" ")
        self.gpu_ids = self.config["parameters"]["gpu_ids"]
        self.gpu_list = self.gpu_ids.split(",")
        if len(self.gpu_list) >= 2:
            self.gpu_ids = str(self.gpu_list[0])
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        self.max_iter = int(self.config["parameters"].get("max_iter"))
        self.time_limit = self.config['parameters'].getint("time_limit")
        self.stop_mode = self.config['parameters'].get("stop_mode").upper()
        self.initial_seed_mode = self.config["parameters"].get("initial_seed_mode")
        self.layer_selector = None

        # set instrumentation configuration
        self.frameworks_trs_path = {}
        self.frameworks = {}
        self.c_libraries = {
                            "tensorflow": self.config["parameters"]["c_tensorflow"],
                            "mxnet": self.config['parameters']["c_mxnet"],
                            "pytorch": self.config['parameters']["c_pytorch"],
                            "onnx": self.config['parameters']["c_onnx"],
                            }
        self.py_libraries = {
                             "tensorflow": self.config['parameters']["py_tensorflow"],
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
        self.initiate_frameworks_trs_path()

        # part for the search algorithm
        self.seed_selector = None
        self.seed_selector_path = os.path.join(self.seed_selector_dir, "selector.pkl")
        self.architecture_measure_path = os.path.join(self.seed_selector_dir, "architecture_measure.pkl")
        self.mutate_model_path = os.path.join(self.experiment_dir, "mutated_model.json")
        self.total_iteration_path = os.path.join(self.seed_selector_dir, "total_iteration.txt")
        self.seed_selection_mode = self.config["parameters"]["seed_selection_mode"]  # possible option: mcmc, random
        self.seed_selector_name = self.config["parameters"]["seed_selector_name"]  # possible option: MUTATORS
        self.mutation_operator_mode = self.config["parameters"]["mutation_operator_mode"]  # possible option: diverse, random
        self.mutation_operator_list = self.config["parameters"]["mutation_operator_list"]  # possible option: list of mutation operators, old
        self.coverage = self.config["parameters"].get("coverage")  # possible option: 1, 0 or library name, string
        self.current_seed_name = None
        # add this signal_path so we can directly stop the program by creating this file in the experiment_dir
        self.stop_signal_path = os.path.join(self.experiment_dir, "stop.stop")

        self.architecture_measure = ArchitectureMeasure()
        self.start_time = time.time()

    def initiate_dirs(self, ):
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        if not os.path.exists(self.crash_dir):
            os.makedirs(self.crash_dir)
        if not os.path.exists(self.nan_dir):
            os.makedirs(self.nan_dir)
        if not os.path.exists(self.model_dir): os.makedirs(self.model_dir)
        if not os.path.exists(self.crash_dir): os.makedirs(self.crash_dir)
        if not os.path.exists(self.instrument_result_dir):
            os.makedirs(self.instrument_result_dir)
        if not os.path.exists(self.delta_dir):
            os.makedirs(self.delta_dir)
        if not os.path.exists(self.seed_selector_dir): os.makedirs(self.seed_selector_dir)

    # Framework Related Methods
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
        main_logger.info("Loading the framework")
        with open(self.frameworks_trs_path[bk], "rb") as file:
            self.frameworks[bk] = dill.load(file)

    def save_framework(self, bk):
        with open(self.frameworks_trs_path[bk], "wb") as file:
            dill.dump(self.frameworks[bk], file)
        # also save the coverage result
        with open(os.path.join(self.delta_dir, bk + "_coverage.txt"), "a") as file:
            cache_py = self.frameworks[bk].coverage(type="py")
            cache_c = self.frameworks[bk].coverage(type="c")
            end_time = time.time()
            file.write(f"Current time (seconds): {end_time - self.start_time}\n")
            file.write(f"python result:\n")
            line_coverage, branch_coverage, line_hit, total_lines, branch_hit, total_branches = cache_py
            file.write(f"{line_coverage},{branch_coverage},{line_hit},{total_lines},{branch_hit},{total_branches}\n")
            file.write(f"c result:\n")
            line_coverage, branch_coverage, line_hit, total_lines, branch_hit, total_branches = cache_c
            file.write(f"{line_coverage},{branch_coverage},{line_hit},{total_lines},{branch_hit},{total_branches}\n")

    def save_temporary_frameworks(self, name=None):
        if self.coverage == "0" or self.guidance_strategy == "ARCS":
            # if no coverage have been recorded , we record the coverage first.
            self.collect_current_coverage()
        import shutil
        for bk in self.backends:
            middle_framework_path = os.path.join(self.instrument_result_dir, f"{bk}_{name}.pkl")
            if not os.path.exists(middle_framework_path):
                main_logger.info("Saving Runtime Framework Result")
                shutil.copy(self.frameworks_trs_path[bk], middle_framework_path)

    def check_stop_signal(self):
        return os.path.exists(self.stop_signal_path)

    # Seed Selector Methods
    def seed_selector_exist(self):
        return os.path.exists(self.seed_selector_path)

    def load_seed_selector(self):
        with open(self.seed_selector_path, "rb") as file:
            self.seed_selector = dill.load(file)
        with open(self.total_iteration_path, "r") as file:
            iteration = int(file.read()[:-1])
        return self.seed_selector, iteration

    def save_seed_selector(self, iteration):
        with open(self.seed_selector_path, "wb") as file:
            dill.dump(self.seed_selector, file)
        with open(self.total_iteration_path, "w") as file:
            file.write(f"{iteration}\n")

    def load_architecture_measure(self):
        with open(self.architecture_measure_path, "rb") as file:
            self.architecture_measure = dill.load(file)

    def save_architecture_measure(self):
        with open(self.architecture_measure_path, "wb") as file:
            dill.dump(self.architecture_measure, file)

    def collect_current_coverage(self):
        subprocess.call(f"python -u -m scripts.analysis.collect_cov --config_name {self.config_name}", shell=True)

    def stop(self):
        if self.guidance_strategy == "ARCS":
            main_logger.info("Stopping the Program ..., Do Not Clear GCDA")
            self.collect_current_coverage()
        else:
            if self.coverage == "1":  # if we calculate coverage each iteration, we don't need to save save the gcda
                main_logger.info("Stopping the Program ...")
                self.clear_gcda_coverage()
                self.clear_py_coverage()
            elif self.coverage == "0":  # if we don't calculate the coverage each iteration, we should preserve the gcda.
                main_logger.info("Stopping the Program ...")
                self.collect_current_coverage()

    def before_prediction(self,):
        self.guidance.reset()

    def clear_py_coverage(self):
        main_logger.info("Clearing Py Coverage....")
        for bk in self.backends:
            subprocess.call(f"rm -rf .coverage.{bk}", shell=True)

    def clear_gcda_coverage(self):
        main_logger.info(f"Clearing GCDA....")
        for bk in self.backends:
            subprocess.call(f"find runtime/{bk}/ -name '*.gcda' -exec rm -rf " + "{} \\;", shell=True)

    def clear_code_coverage(self):
        self.clear_gcda_coverage()
        self.clear_py_coverage()

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

    def analyze_inference_result(self, parallel_return, model_path, model_name):
        """
        Analyze the inference result of different DL frameworks
        Arguments:
            parallel_return:
            "COVERAGE": pre_status_bk, predict_output_bk, framework, cov_uni_bk
            "ARCS": pre_status_bk, predict_output_bk, framework, cov_uni_bk, guidance.arcs[bk]
            "ELSE": pre_status_bk, predict_output_bk, framework, cov_uni_bk
            :pre_status_bk: status of prediction {"tensorflow": 0, "pytorch": 0}  0: success, other: crash
            :predict_output_bk: numpy array
            :framework: classes.frameworks.Framework
            :cov_uni_bk: {"tensorflow": 0, "pytorch": 1}  0: no new trs, 1: new trs
            :guidance.arcs[bk]: {file_path: [br_1, br_2, br_3, ...]}
            model_path:
            The path of the model
            model_name:
            The name of the model
        """
        # Step 4.3: Collect output prediction, runtime information.
        predict_output = {b: [] for b in self.backends}
        status = {}
        for i, bk in enumerate(self.backends):
            status[bk] = parallel_return[i][0]
            self.frameworks[bk] = parallel_return[i][2]
            self.guidance.cov_uni[bk] = parallel_return[i][3]
            if self.guidance_strategy == "ARCS":
                self.guidance.arcs[bk] = parallel_return[i][4]
                self.save_arcs(self.guidance.arcs[bk], bk)
            if status[bk] == 0:
                predict_output[bk] = parallel_return[i][1]
                main_logger.info(f"Success on backend: {bk} of model {model_name}")
            else:
                main_logger.info(f"Fail on backend: {bk} of model {model_name}")

        # Step 4.4: Analyze output prediction, runtime information.
        accumulative_incons = 0
        status_crash_nan = False
        crash_bk = [bk for bk in status.keys() if status[bk] != 0]  # WARNING: when a bk crash, its status will not necessarily be 1
        for bk in crash_bk:
            _ = predict_output.pop(bk)
        if (len(predict_output) >= 2 or len(predict_output) == len(self.backends)) and status["tensorflow"] == 0:
            # if there are more than one library success and tensorflow should be correct
            predictions = list(predict_output.values())
            self.metrics_results, accumulative_incons = generate_metrics_result(res_dict=self.metrics_results,
                                                                                predict_output=predict_output,
                                                                                model_name=model_name)
            # Check abnormal output: Not-A-Number / Infinity / InconsistentShape
            if is_nan_or_inf(accumulative_incons):
                status_crash_nan = True
                nan_or_inf, nan_idxs = partially_nan_or_inf(predictions, len(predictions))
                if nan_or_inf:
                    nan_model_path = os.path.join(self.nan_dir, f"{model_name}_NaN_bug.h5")
                    main_logger.info(f"Error: Found one NaN bug. move NAN model, the NaN idx is: {nan_idxs}")
                else:
                    nan_model_path = os.path.join(self.nan_dir, f"{model_name}_NaN_on_all_backends.h5")
                    main_logger.info("Error: Found one NaN Model on all libraries. move NAN model")
                shutil.move(model_path, nan_model_path)
            if is_none(accumulative_incons):
                status_crash_nan = True
                main_logger.info(f"Error: Found one shape inconsistent bug. move buggy model")
                crash_model_path = os.path.join(self.crash_dir, model_name + ".h5")
                shutil.move(model_path, crash_model_path)
        else:
            # Model can only pass one model
            status_crash_nan = True
            main_logger.info("Error: Move Crash model")
            crash_model_path = os.path.join(self.crash_dir, model_name + ".h5")
            shutil.move(model_path, crash_model_path)
        return status_crash_nan, accumulative_incons

    def predict(self, model_path: str, model_name: str):
        """
        Drive multiple deep learning libraries to inference on a same DL model.
        The function is implemented in multiprocessing mode.
        Arguments:
            model_path: the path that stores the DL model
            model_name: the name of the DL model
        Return:
            None
        """
        # Step 4.1: Initiate the framework's code coverage.
        silent = {}
        for bk in self.backends:
            silent[bk] = False
            if not os.path.exists(self.frameworks_trs_path[bk]):
                main_logger.info("First time, initiate the framework")
                self.initiate_framework(bk)
                self.clear_code_coverage()
                self.guidance.before_predict(self.frameworks[bk])
                silent[bk] = True
            elif len(self.frameworks) != len(self.backends):
                # frameworks have not been initialized yet, load the framework from disk
                self.load_framework(bk)
        # Step 4.2: Drive multiple DL framework inference in multiprocessing mode.
        args = [(
            self.frameworks[bk], self.c_libraries[bk], model_path, model_name,
            bk, self.guidance, self.python_prefix, self.config_dir,
            self.config_name, self.config, silent[bk], self.coverage, self.guidance_strategy
        ) for bk in self.backends]
        main_logger.info("Working on Multiprocessing Prediction")
        with Pool(len(self.backends)+1) as p:
            parallel_return = p.map(parallel_predict, args)
        return parallel_return

    def after_prediction(self, model, status_crash_nan, accumulative_incons, reward=None):
        reward = 0.5*reward + 0.5*self.guidance.get_score()
        self.seed_selector.check_inconsistency(self.current_seed_name, accumulative_incons)
        if reward > 0 and self.seed_selection_mode == "mcmc":  # if seed_selection_mode is random, we do nothing
            # if no crash happen, we will replace the mutated seed
            if not status_crash_nan:
                main_logger.info(f"New representative seed found: {model.name}")
                self.seed_selector.add_architecture(new_architecture_name=model.name, new_architecture_config=model.to_json(), old_seed_name=self.current_seed_name)
            self.seed_selector.increase_reward(seed_name=self.current_seed_name, score=reward)
        elif reward == 0 and self.seed_selection_mode == "mcmc":  # 0 means no new trs are found
            main_logger.info(f"No representative seed generated: {model.name}, do not increase the reward")

    def _gen_model(self, iteration=0):
        # choose seed architecture and mutation operators
        selected_seed_name, seed_config, mutation_operator = self.seed_selector.choose_seed()
        main_logger.info(f"Choose seed: {selected_seed_name}")
        main_logger.info("start mutating the generated model")
        self.save_architecture_measure()
        with open(self.mutate_model_path, "w") as file:
            file.write(seed_config)
        new_model_name = f"{selected_seed_name}-{mutation_operator}{iteration}"
        if self.seed_selector_name == "MUTATORS":
            self.current_seed_name = mutation_operator
        else:
            raise NotImplementedError(f"Unknown Seed Selector: {self.seed_selector_name}")
        envs = "CUDA_HOME=/usr/local/cuda-10 CUDA_ROOT=/usr/local/cuda-10 LD_LIBRARY_PATH=/usr/local/cuda-10/lib64:$LD_LIBRARY_PATH PATH=/usr/local/cuda-10/bin:$PATH"
        new_model_path = os.path.join(self.model_dir, new_model_name, new_model_name + ".h5")
        # Step 1~2
        status = subprocess.call(f"{envs} python -u -m scripts.generation.generate_model "
                                 f"--origin_model_path={self.mutate_model_path} "
                                 f"--new_model_path={new_model_path} "
                                 f"--new_model_config_path={self.mutate_model_path} "
                                 f"--new_model_name={new_model_name} "
                                 f"--mutate_op={mutation_operator} "
                                 f"--mutation_operator_mode={self.mutation_operator_mode} "
                                 f"--architecture_measure_path={self.architecture_measure_path} "
                                 f"--config_name={self.config_name}",
                                 shell=True
                                 )
        if status != 0:
            return None, None, None
        # Step 2.4: Update the actual model to the layer interaction coverage and layer configuration coverage
        new_model = ArchitectureUtils.load_json(self.mutate_model_path)
        reward = self.architecture_measure.update_diversity(new_model, new_model_name)
        self.save_api_cov()
        if self.seed_selector_name == "MUTATORS":
            # when we use MUTATORS as the seed_selector, we add the total number of selected seed after the model can be successfully generated
            self.seed_selector.get_seed_by_name(mutation_operator).increase_selected()
        return new_model_path, new_model, reward

    def gen_model(self, last_iteration, start_time):
        def gen(iteration):
            try:
                # Step 1 & 2: Generate/Save Model, save the model to the `model_path`
                new_model_path, new_model, reward = self._gen_model(iteration=iteration)
                if new_model_path is None:
                    # Fail when generating the model
                    return 1
                new_model_name = new_model.name
                # Step 3: Before prediction, clear all outputs from different DL libraries
                self.before_prediction()
                # Step 4: Predict
                parallel_return = self.predict(new_model_path, model_name=new_model_name)
                status_crash_nan, accumulative_incons = self.analyze_inference_result(parallel_return,
                                                                                      model_path=new_model_path,
                                                                                      model_name=new_model_name)
                # Step 5: After prediction, compare the output result, check inconsistency, nan, crash bugs, update seed selector
                self.after_prediction(new_model, status_crash_nan, accumulative_incons, reward=reward)
                del new_model
                gc.collect()
            except Exception:
                # main_logger.info(traceback.format_exc())
                return 1
            return 0

        iteration = 0
        run_stat = {'start_time': start_time, 'time_limit': self.time_limit, 'cur_counters': iteration,
                    'counters_limit': self.max_iter, 'stop_mode': self.stop_mode}
        continue_signal, save_signal = continue_checker(**run_stat)
        while continue_signal:
            # Start the iteration, in each iteration, we will generate one DL model
            status = gen(iteration=last_iteration + iteration)
            iteration -= status*1  # status=0: success when generating model, status=1: fail when generating model
            self.save_seed_selector(iteration=iteration)
            iteration += 1
            # Check if continue
            run_stat["cur_counters"] = iteration
            continue_signal, save_signal = continue_checker(**run_stat)
            if save_signal:
                self.save_temporary_frameworks(save_signal)
            if self.check_stop_signal():
                main_logger.info("Detecting The Stopping Signal! Stopping The Program")
                self.stop()
                return
        self.stop()

    def _save_model(self, model, model_dir):
        """
        If a DL model has multiple outputs, we merge them together, then convert it to a (300,) tensor
        """
        import keras
        # save the appended model
        model_name = model.name
        # Step 2.1: Model input processing
        # For pytorch, it cannot automatically infer the input shape, therefore, we need to send to it through redis
        shape = list(model.input_shape)
        shape[0] = 1
        self.redis_conn.hset(f"prediction_{model_name}", "shape", pickle.dumps(shape))
        x = model.inputs
        # Step 2.2: Model output processing
        # If model has multiple outputs, we merge them together by keras.layers.Add()
        actual_output_lists = []
        for model_output in model.outputs:
            layer_input = model_output
            input_shape = layer_input.shape.as_list()
            from scripts.generation.layer_tool_kits import LayerMatching
            for layer in LayerMatching.flatten_dense_output(input_shape):
                y = layer(layer_input)
                layer_input = y
            actual_output_lists.append(layer_input)
        if len(actual_output_lists) > 1:
            actual_output = keras.layers.Add()(actual_output_lists)
        else:
            actual_output = actual_output_lists[0]
        # Step 2.3: Create and save the model.
        actual_model = keras.Model(x, actual_output)
        if int(keras.__version__.split(".")[1]) >= 7:
            actual_model._name = model.name
        else:
            actual_model.name = model.name
        # actual_model.summary()
        if not os.path.exists(os.path.join(model_dir, actual_model.name)):
            os.makedirs(os.path.join(model_dir, actual_model.name))
        model_path = os.path.join(model_dir, actual_model.name, actual_model.name + ".h5")
        actual_model.save(model_path)
        # Step 2.4: Update the actual model to the layer interaction coverage and layer configuration coverage
        reward = self.architecture_measure.update_diversity(actual_model, actual_model.name)
        # we use the layer type, edge, config reward as guidance
        self.save_api_cov()
        del actual_model
        return model_path, reward

    def run_initial_seeds(self, ):
        # before generating models, we should go through all seed and filter out those that will directly lead to crash
        for arch in self.seed_selector.initial_architectures:
            arch_config = arch.config
            arch_name = arch.name
            main_logger.info(f"Running Initial Architecture: {arch_name}")
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
            import tensorflow as tf
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            import keras
            arch_model = keras.models.model_from_json(arch_config, custom_objects=custom_objects())
            model_path, _ = self._save_model(arch_model, self.model_dir)
            parallel_return = self.predict(model_path, model_name=arch_name)
            status_crash_nan, accumulative_incons = self.analyze_inference_result(parallel_return,
                                                                                  model_path=model_path,
                                                                                  model_name=arch_name)
            arch.check_inconsistency(accumulative_incons)
            arch.update_inconsistency(accumulative_incons)
            if status_crash_nan:
                main_logger.info(f"Popping Initial Architecture: {arch_name}")
                self.seed_selector.pop_architecture_by_name(arch_name)
            del arch_model

    def initiate_seed_selector(self):
        # Initiate seed selector
        if self.seed_selector_exist():
            main_logger.info("Seed Selector Exists! Continue COMET!")
            self.seed_selector, last_iteration = self.load_seed_selector()
        else:
            main_logger.info("Seed Selector Not Exists! Start COMET From Beginning")
            # send the initial seeds to MCMC to construct the seed pool
            self.seed_selector = get_seed_selector_by_name(self.seed_selector_name)(
                initial_seed_mode=self.initial_seed_mode,
                selection_mode=self.seed_selection_mode,
                mutation_operator_list=self.mutation_operator_list
            )
            self.run_initial_seeds()
            self.save_architecture_measure()
            self.seed_selector.initiate_seed()
            self.seed_selector.initiate_mcmc()
            last_iteration = 0
            self.save_temporary_frameworks("initial_models")
        return last_iteration

    def initiate_guidance(self):
        # COMET calculates ARCS (Use coverage.py's module to collect the arcs in python) after each iteration
        self.guidance = get_guidance_by_name(self.guidance_strategy)

    def run(self, ):
        start_time = datetime.datetime.now()
        self.initiate_guidance()
        last_iteration = self.initiate_seed_selector()
        try:
            self.gen_model(last_iteration, start_time)
        except Exception:
            print(traceback.format_exc())
        end_time = datetime.datetime.now()
        time_delta = end_time - start_time
        h, m, s = utils.ToolUtils.get_HH_mm_ss(time_delta)
        main_logger.info(f"COMET Is Finished: Time used: {h} hour,{m} min,{s} sec")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--config_name", type=str, help="config name")
    flags, _ = parse.parse_known_args(sys.argv[1:])
    driver = Driver(flags=flags)
    driver.run()
