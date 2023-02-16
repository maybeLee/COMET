from collections import defaultdict
from classes.files import CFiles, PyFiles, Files
import classes
import re
import os
from utils.utils import get_files, concatenate_vector
import importlib
import subprocess
import numpy as np


def get_value_from_line(line: str, key: str):
    # target line: xxxxxx key="value"xxxx, we will return the value from line
    return line.split(key + "=")[-1].split("\"")[1]


class Frameworks:
    def __init__(self, name, py_root_dir, c_root_dir, python_prefix, py_trs=None, c_trs=None, algorithm="kdtree"):
        # py_trs, c_trs are input coverage logs
        self.name = name
        self.py_root_dir = py_root_dir  # note that root_dir should end up with "/", this is because both python and c root dir is used to extract file's name
        self.c_root_dir = c_root_dir  # note that root_dir should end up with "/"
        self.py_keras_dir = f"{python_prefix}/diffdelta_{name}/lib/python3.6/site-packages/keras"  # get the directory of keras framework
        self.c_lock = self.c_root_dir + name + "_occupied.flag"
        self.algorithm = algorithm  # algorithm is the algorithm used to set flann and calculate distance
        # self.py_trs = py_trs
        # self.c_trs = c_trs
        self.py_files, self.c_files = {}, {}
        self.py_lines, self.c_lines, self.py_branches, self.c_branches = defaultdict(list), defaultdict(
            list), defaultdict(list), defaultdict(list)
        # specify ignored c files and py files during coverage measurement
        self.c_exclude = ["a", "b"]
        self.py_exclude = ["a", "b"]
        self.coverage_history = {}  # {mutant_name: [], mutant_name: []}
        self.distance = {}
        self.predicate_trs = {}  # will be {key1: <class trs>, key2: <class trs>}

    def clear(self):
        # REALLY DANGEROUS FUNCTION!!!
        # IT WILL DESTROY ALL FIELDS!!!!!! :(
        self.py_files, self.c_files = {}, {}
        self.py_lines, self.c_lines, self.py_branches, self.c_branches = defaultdict(list), defaultdict(
            list), defaultdict(list), defaultdict(list)

    @property
    def vector(self):
        vector = np.zeros_like(list(self.coverage_history.values())[0])
        for value in self.coverage_history.values():
            vector = np.logical_or(vector, value)
        return vector

    def clear_gcda(self):
        """
        Note that this function is not used anymore
        """
        # clear the .gcda file in framework's c library directory
        try:
            if self.name == "tensorflow":
                subprocess.call(
                    f"find {self.c_root_dir}bazel-out/k8-py2-opt/bin/tensorflow -name '*.gcda' -exec rm -rf " + "{} \\;",
                    shell=True)
                if not os.path.islink("./bazel-out"):
                    subprocess.call("rm -rf ./bazel-out", shell=True)
            elif self.name != "theano":
                subprocess.call(f"find {self.c_root_dir} -name '*.gcda' -exec rm -rf " + "{} \\;", shell=True)
        except:
            pass

    def check_gcda(self):
        # check if the c_root_dir still have the gcda, if still have, return true, else return false.
        if self.name == "tensorflow":
            return any(File.endswith(".gcda") for File in
                       os.listdir("{}bazel-out/k8-py2-opt/bin/tensorflow".format(self.c_root_dir)))
        else:
            return any(File.endswith(".gcda") for File in os.listdir(self.c_root_dir))

    def lock_ask(self):
        # ask whether the framework is available currently (for c collection)
        # check whether the occupied.flag exists in the c_root_dir, if occupied.flag exists, return false, else return true
        return not os.path.exists(self.c_lock)

    def lock_acquire(self):
        # require to use the framework
        # create the occupied.flag in the c_root_dir
        self.clear_gcda()
        try:
            open(self.c_lock, "a").close()
        except:
            pass

    def lock_release(self):
        # release the framework
        # self.clear_gcda()
        # delete the created place file
        try:
            os.remove(self.c_lock)
        except:
            pass

    def update_distance(self, distance, silent=False):
        # distance_score will be, for any available distance, if the distance is smaller, it will be 1, else 0
        distance_score = 0
        for key in distance:
            if key in self.distance.keys():
                if distance[key] < self.distance[key]:
                    distance_score += 1
                    self.distance[key] = distance[key]
            else:
                self.distance[key] = distance[key]
        return distance_score

    def update_predicate(self, predicate_trs, silent=False):
        # predicate_score will be, if new predicate is hit, it will be 1, else 0
        predicate_score = 0
        for tr_key in predicate_trs.keys():
            tr = predicate_trs[tr_key]
            if tr_key not in self.predicate_trs.keys():
                # the number of not included trs should be even
                self.predicate_trs[tr_key] = tr
                if tr.hit_status is True:
                    if silent is False:
                        print("New predicate has been reached and invoked: ", tr_key)
                    predicate_score += 1
            elif tr_key in self.predicate_trs.keys() and not self.predicate_trs[tr_key].hit_status and tr.hit_status:
                self.predicate_trs[tr_key].is_visited()
                if silent is False:
                    print("New predicate has been invoked: ", tr_key)
                predicate_score += 1
        return predicate_score

    def __parse_py_xml_content(self, xml_content):
        # xml_content, list containing all lines in xml reported by coverage.py
        # since we prefer speed rather than elegant, we may brutally go through all lines and search for key word in each line,
        # we will not use any xml package (e.g., lxml) to parse the structure
        py_info_files = {}
        file_path = None
        branch_start_line = -1
        miss_branches = []
        for line in xml_content:
            if "filename" in line:
                # e.g., "<class branch-rate="0" complexity="0" filename="/root/anaconda3/envs/diffcu_cntk/lib/python3.6/site-packages/keras/utils/multi_gpu_utils.py" line-rate="0.1446" name="multi_gpu_utils.py">"
                file_path = get_value_from_line(line, "filename")
                if file_path not in py_info_files:
                    file_name = file_path.replace(self.py_root_dir, "")
                    py_file = Files(file_name, file_path)
                    py_info_files[file_path] = py_file

            if branch_start_line != -1 and "<line " in line and "number=" in line:
                # if it's previous line is branch line
                line_no = int(get_value_from_line(line, "number"))
                branch_key_0 = "{}_{}".format(branch_start_line, line_no)
                branch_key_1 = "{}_{}".format(branch_start_line, "else")
                py_file = py_info_files[file_path]
                py_file.add_trs(branch_key=branch_key_0, visit=False)
                py_file.add_trs(branch_key=branch_key_1, visit=False)
                if miss_branches == []:
                    py_file.branches[branch_key_0].is_visited()
                    py_file.branches[branch_key_1].is_visited()
                elif str(line_no) not in miss_branches:
                    # make a really stupid mistake, the line_no is int while miss_branches only contain str, therefore,
                    # the if-true block is always hitted, this mistake will make the performance of our method worse,
                    # and make the improved test requirement less
                    # if at least one branch is missed and 'if' condition is not missed
                    py_file.branches[branch_key_0].is_visited()
                elif str(line_no) in miss_branches and len(miss_branches) == 1:
                    # if only one branch is missed and 'if' condition is in it
                    py_file.branches[branch_key_1].is_visited()
                branch_start_line = -1
                miss_branches = []

            if "<line " in line and "number=" in line:
                line_no = int(get_value_from_line(line, "number"))
                hit_states = bool(int(get_value_from_line(line, 'hits')))  # transfer the hit status from str to boolean
                py_file = py_info_files[file_path]
                if hit_states is True:
                    py_file.add_trs(lineno=line_no, visit=True)
                else:
                    py_file.add_trs(lineno=line_no, visit=False)

            if 'branch=\"true\"' in line:
                line_no = int(get_value_from_line(line, "number"))
                branch_start_line = line_no
                if "missing-branches" in line:
                    miss_branches = get_value_from_line(line, "missing-branches")
                    miss_branches = miss_branches.split(",")  # miss_branches will be [x], or [x,y]
                else:
                    miss_branches = []
        return self.sort_info(py_info_files)

    def __parse_c_json_content(self, json_content):
        # input: json_content generated by gcovr, load by json.load()
        # output: a dict containing all test requirements and their status
        c_info_files = {}

        for file in json_content['files']:
            file_path = file['file']
            if file_path not in c_info_files:
                file_name = file_path.split('/')[-1]
                c_file = Files(file_name, file_path)
                c_info_files[file_path] = c_file

            for line in file['lines']:
                if line['gcovr/noncode'] is True:
                    continue
                line_no = line['line_number']
                hit_states = bool(line['count'])
                c_file = c_info_files[file_path]
                if hit_states is True:
                    c_file.add_trs(lineno=line_no, visit=True)
                else:
                    c_file.add_trs(lineno=line_no, visit=False)

                for branch_id, branch in enumerate(line['branches']):
                    branch_key = f"{line_no}_{branch_id}"
                    branch_hit_states = bool(branch['count'])
                    if branch_hit_states is True:
                        c_file.add_trs(lineno=line_no, branch_key=branch_key, visit=True)
                    else:
                        c_file.add_trs(lineno=line_no, branch_key=branch_key, visit=False)
        return self.sort_info(c_info_files)

    def __parse_c_info_content(self, info_content):
        # parameters: info_content, list containing all lines of tracefile
        # return: info_files: [key=file_path, value=object: Files]

        # Finished: currently we only keep invoked test requirement to list, this
        #  is not reasonable especially when we want to measure total coverage at last.
        #  Therefore, we need to also consider uninvoked test requirement
        info_files = {}
        c_file = None
        file_path = None
        for line in info_content:
            if "SF:" in line:
                file_path = line.split(":")[1]
                if file_path not in info_files:
                    file_name = file_path.replace(self.c_root_dir, "")
                    c_file = Files(file_name, file_path)
                    info_files[file_path] = c_file
            da_pattern = r'DA:.*'
            if re.match(da_pattern, line):
                hit_line = int(line.split(":")[1].split(",")[0])  # get the line number
                c_file = info_files[file_path]
                if int(hit_line) not in c_file.lines.keys() or c_file.lines[hit_line].hit_status is False:
                    # if this test requirement is not recorded or has not been reached yet, we have reason to update it
                    if line.split(",")[-1] != "0":  # if the statement is reached
                        c_file.add_trs(lineno=hit_line, visit=True)
                    else:  # if the statement is not reached in this part and is not reached before
                        c_file.add_trs(lineno=hit_line, visit=False)
            # Additional check on branch
            # Finished: additional check on branch of C files
            br_pattern = r'BRDA:.*'
            if re.match(br_pattern, line):
                hit_line = line.split(":")[1].split(",")[0]
                branch_no = line.split(":")[1].split(",")[2]
                branch_key = "{}_{}".format(hit_line, branch_no)
                c_file = info_files[file_path]
                if branch_key not in c_file.branches.keys() or c_file.branches[branch_key].hit_status is False:
                    # there are two occasion for a branch to be not hit, if it is not reached, it will be "-"
                    # if it is reached but not hit, it will be "0"
                    if line.split(",")[-1] != "0" and line.split(",")[-1] != "-":  # if the branch tr is reached
                        c_file.add_trs(branch_key=branch_key, visit=True)
                    else:  # if the branch tr is not reached
                        c_file.add_trs(branch_key=branch_key, visit=False)
        return self.sort_info(info_files)

    def clear_and_profile(self):
        # In this method, we initially plan to parse the pyc file and collect the test requirements by ourselves,
        # but we further notice that coverage.py have achieved this pretty well, as such we delete all profiling code.
        self.clear_gcda()

    @staticmethod
    def _update_info(given_info, target_files, silent=True):
        change_flag = 0
        for file_path in given_info.keys():
            if file_path not in target_files:
                target_files[file_path] = given_info[file_path]
                change_flag = 1
                if silent is False:
                    print("Find new file: ", file_path)
                continue
            compare_result = target_files[file_path].compare_and_update(given_info[file_path])
            if compare_result[0] != [] or compare_result[1] != []:
                if silent is False:
                    print("New test requirements are invoked on file {}: {}".format(file_path, compare_result))
                change_flag = 1
        return change_flag

    @staticmethod
    def sort_info(given_info):
        given_info = dict(sorted(given_info.items()))
        for file in given_info.keys():
            given_info[file].sort_trs()
        return given_info

    @staticmethod
    def get_tr_vector(given_info):
        # return the activation vector of the given info, given info represents a parse info
        # including all python files or c files, structure: {"file_name_1": File, "file_name_2": File, ...}
        # we will return a 1xN numpy.array vector
        vector = np.array([])
        for file in given_info.keys():
            vector = concatenate_vector(vector, given_info[file].get_tr_vector())
        return vector

    def update_vector(self, idntfr: str, vector: np.array):
        self.coverage_history[idntfr] = vector

    def new_trs(self, idntfr: str, vector: np.array):
        # get whether the newly introduced vector have new
        idx = idntfr.rindex("-")
        pre_idntfr = idntfr[:idx]
        pre_vector = self.coverage_history[pre_idntfr]
        return np.where(vector - pre_vector)[0].any()

    def compare_and_update(self, idntfr, new_py_trs=None, new_c_trs=None, silent=True):
        # given new python xml_content(list) and new c info_content(list), update the Framework's coverage
        # Finished: add python file update method
        # Finished: refine c file update method
        # Finished: transfer the info file into activation vector
        py_change_flag, c_change_flag = 0, 0
        vector = np.array([])
        if new_py_trs is not None:
            info_py_files = self.__parse_py_xml_content(
                new_py_trs)  # info_py_files (sorted): [file_path: Files, file_path: Files, ...]
            py_change_flag = self._update_info(given_info=info_py_files, target_files=self.py_files, silent=silent)
            vector = concatenate_vector(vector, self.get_tr_vector(info_py_files))
        # For C check and update
        if new_c_trs is not None:
            info_c_files = self.__parse_c_info_content(
                new_c_trs)  # info_c_files (sorted): [file_path: Files, file_path: Files, ...]
            c_change_flag = self._update_info(given_info=info_c_files, target_files=self.c_files, silent=silent)
            vector = concatenate_vector(vector, self.get_tr_vector(info_c_files))

        if py_change_flag or c_change_flag:
            print("New test requirements invoked on framework {} when updating {}".format(self.name, idntfr))
            return 1, vector
        else:
            print("no update on framework {} when updating {}".format(self.name, idntfr))
            return 0, vector

    @staticmethod
    def _coverage(file_lists, exclude_lists, cache):
        line_hit, line_miss, branch_hit, branch_miss = cache
        for file_name in file_lists.keys():
            exclude_flag = 0
            for exclude in exclude_lists:
                if re.match(exclude, file_name):
                    exclude_flag = 1
                    break
            if exclude_flag == 1:  # exclude_flag == 1 means that we don't want to consider this file
                print(file_name)
                continue
            file = file_lists[file_name]
            line_hit_delta, line_miss_delta, _, branch_hit_delta, branch_miss_delta, _ = file.coverage
            line_hit += line_hit_delta
            line_miss += line_miss_delta
            branch_hit += branch_hit_delta
            branch_miss += branch_miss_delta
        return line_hit, line_miss, branch_hit, branch_miss

    def coverage(self, py_exclude=None, c_exclude=None, type="all"):
        # type can be: "all", "py", "c"
        py_exclude = self.py_exclude if py_exclude is None else py_exclude
        c_exclude = self.c_exclude if c_exclude is None else c_exclude
        cache = (0, 0, 0, 0)  # cache: line_hit, line_miss, branch_hit, branch_miss
        if type == "all":
            cache = self._coverage(self.c_files, c_exclude, cache)
            cache = self._coverage(self.py_files, py_exclude, cache)
        elif type == "py":
            cache = self._coverage(self.py_files, py_exclude, cache)
        elif type == "c":
            cache = self._coverage(self.c_files, c_exclude, cache)
        line_hit, line_miss, branch_hit, branch_miss = cache
        total_lines = line_hit + line_miss
        total_branches = branch_hit + branch_miss
        line_coverage = line_hit / total_lines if total_lines != 0 else 0
        branch_coverage = branch_hit / total_branches if total_branches != 0 else 0
        return line_coverage, branch_coverage, line_hit, total_lines, branch_hit, total_branches
