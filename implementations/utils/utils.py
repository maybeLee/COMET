import os
from io import StringIO
import sys
import numpy as np
import collections
import time


def deprecated(func):
    """
    This is a decorator for deprecated functions.
    """
    def wrapper(*args, **kwargs):
        print(f"[Warning] Function '{func.__name__}' is deprecated.")
        return func(*args, **kwargs)
    return wrapper


def get_files(framework_dir, file_type):
    # input: root path of specific frameworks
    # output: list format: ["file_dir_1", "file_dir_2", ...]
    # function, go through all files in the framework and only find python files
    file_lists = []
    for root, subdirs, files in os.walk(framework_dir):
        for file in files:
            if not file.endswith(file_type):
                continue
            file_lists.append(os.path.join(root, file))
    return file_lists


def misline_2_range(missline, max_num):
    miss_range = np.ones(max_num + 1)
    miss_branch = []
    for miss in missline:
        if "->" in miss:  # This is a branch
            miss_branch.append(miss)
            continue
        elif "-" in miss:  # This is a statement
            bottom, top = miss.split("-")
            bottom, top = (int(bottom), int(top))
            miss_range[bottom:top + 1] = 0
        else:
            miss_range[int(miss)] = 0

    miss_branch_dict = collections.defaultdict(list)
    for branch in miss_branch:
        start_no = branch.split("->")[0]
        end_no = branch.split("->")[1]
        miss_branch_dict[start_no].append(end_no)
    return miss_range, miss_branch_dict


def parse_miss_line(miss_line, max_line):
    # miss_range: np.array([0,1,1,...]) with shape: max_line+1, miss_branch: {start_no: [end_no], ...}, all keys and values are str
    miss_range, miss_branches = misline_2_range(miss_line, max_line)
    return miss_range, miss_branches


def wait_until(condition, timeout=3000, period=0.25, *args, **kwargs):
    mustend = time.time() + timeout  # set the timeout clock to be 3000 secs, which is 50 minutes, this is enough
    while time.time() < mustend:
        if condition(*args, **kwargs): return True
        time.sleep(period)
    return False


def concatenate_vector(vector_1, vector_2):
    return np.concatenate((vector_1, vector_2), axis=0)


def compare_coverage(framework_1, framework_2):
    new_1 = {}  # {file_name: {"lines":, "branches"}, ...}
    new_2 = {}

    def _compare(file_dict_1, file_dict_2):
        for file_name in file_dict_1.keys():
            if file_name not in file_dict_2.keys():
                new_1[file_name] = {"lines": list(file_dict_1[file_name].lines.keys()), "branches": list(file_dict_1[file_name].branches.keys())}
            file_1 = file_dict_1[file_name]
            file_2 = file_dict_2[file_name]
            result = file_2.compare(file_1)  # what's new in 1 compared with 2
            if result != ([], []):
                new_1[file_name] = {"lines": result[0], "branches": result[1]}
        for file_name in file_dict_2.keys():
            if file_name not in file_dict_1.keys():
                new_2[file_name] = {"lines": list(file_dict_2[file_name].lines.keys()), "branches": list(file_dict_2[file_name].branches.keys())}
            file_1 = file_dict_1[file_name]
            file_2 = file_dict_2[file_name]
            result = file_1.compare(file_2)  # what's new in 2 compared with 1
            if result != ([], []):
                new_2[file_name] = {"lines": result[0], "branches": result[1]}
    _compare(framework_1.py_files, framework_2.py_files)
    _compare(framework_1.c_files, framework_2.c_files)
    return new_1, new_2

