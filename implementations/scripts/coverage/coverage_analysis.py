import pickle
import datetime
import os
import subprocess
from utils.filterbr import filter_lcov_trace_file
import json
import traceback


def collect_c(out_file, bk, obj_dir, directory, code_json_file):
    # adding include flag will make the lcov really slow, therefore, we only add the exclude flag in this part,
    # leaving the including part for post processing
    if obj_dir is None:
        raise ValueError("obj_dir cannot be None!")
    codes = json.load(open(code_json_file, "rb+"))
    exclude_list = include_list = []
    if "c_include" in codes[bk].keys():
        include_list = codes[bk]["c_include"]
    if "c_exclude" in codes[bk].keys():
        exclude_list = codes[bk]["c_exclude"]
    exclude_str = ""
    include_str = ""
    for i in exclude_list:
        exclude_str += '--exclude ' + '"' + i + '"' + ' '  # exclude_str should be: '--exclude aa --exclude bb --exclude cc --exclude dd '
    for i in include_list:
        include_str += '--include ' + '"' + i + '"' + ' '  # include_str should be: '--include aa --include bb --include cc --include dd '
    if bk == "tensorflow" or bk == "pytorch":
        exclude_str = "--exclude " if len(exclude_list) != 0 else ""
        include_str = "--include " if len(include_list) != 0 else ""
        for i in exclude_list:
            exclude_str += '"' + i + '"' + ' '
        for i in include_list:
            include_str += '"' + i + '"' + ' '
        use_lcov = f'python -m scripts.coverage.fastcov -b --jobs 16 --quiet --lcov {exclude_str}{include_str} --search-directory {obj_dir} --compiler-directory {directory} -o {out_file}'
    else:
        use_lcov = 'lcov -c --quiet --no-external {}{}--rc lcov_branch_coverage=1 --directory {} --base-directory {} -o {}'.format(
            exclude_str, include_str, obj_dir, directory, out_file)
    return use_lcov


class Guidance(object):
    def __init__(self):
        pass

    def reset(self):
        pass

    def collect_info(self, *args):
        pass

    def before_predict(self, framework):
        pass


class Coverage(Guidance):
    def __init__(self):
        super(Coverage, self).__init__()
        self.cov_uni = {}
        self.diff_delta = {}

    @staticmethod
    def check_and_load(output_file, collection_status, bk):
        if collection_status == 0:
            with open(output_file, "rb") as file:
                content = file.read().decode()
        else:
            raise ValueError(f"Fail when collecting the code coverage {bk}")
        return content

    def collect_c_coverage(self, bk, python_bin, model_idntfr, config_dir, framework, library_path, silent):
        code_json_file = f"{config_dir}/codes.json"
        use_lcov = ""
        info_content = None
        output_file = f"temp_stdout_{bk}"
        if bk in ["mxnet", "pytorch", "tensorflow"]:
            obj_dir = None
            if bk in ["mxnet", "pytorch"]:
                obj_dir = f"runtime/{bk}/"
            elif bk == "tensorflow":
                obj_dir = "runtime/tensorflow/k8-opt/bin/tensorflow/"
            use_lcov = collect_c(output_file, bk, obj_dir, library_path, code_json_file=code_json_file)
            collection_status = subprocess.call(use_lcov, shell=True)
            info_content = self.check_and_load(output_file=output_file, collection_status=collection_status, bk=bk)
            info_content = info_content.split("\n")[:-1]
            os.remove(output_file)
        else:
            info_content = None
        # trim branch
        info_content = filter_lcov_trace_file(info_content)
        return info_content

    def collect_py_coverage(self, bk, python_bin, model_idntfr, config_dir, framework, library_path, silent):
        os.environ["COVERAGE_FILE"] = ".coverage.{}".format(bk)
        output_file = f"temp_stdout_{bk}_py"
        report_coverage = f"{python_bin} -m coverage xml -i --rcfile {config_dir}/{bk}.conf -o {output_file}"
        collection_status = subprocess.call(report_coverage, shell=True)
        xml_content = self.check_and_load(output_file=output_file, collection_status=collection_status, bk=bk)
        os.remove(output_file)
        xml_content = xml_content.split("\n")[:-1]
        return xml_content

    def collect_coverage(self, bk, python_bin, model_idntfr, config_dir, framework, library_path, silent):
        print(f"start collecting {bk}")
        xml_content = info_content = None
        try:
            collect_st = datetime.datetime.now()
            info_content = self.collect_c_coverage(bk, python_bin, model_idntfr, config_dir, framework, library_path,
                                                   silent)
            xml_content = self.collect_py_coverage(bk, python_bin, model_idntfr, config_dir, framework, library_path,
                                                   silent)
            collect_et = datetime.datetime.now()
            print(f"finish collecting {bk}, the time difference is: {collect_et - collect_st}")
        except Exception:
            print("Error when collecting trs: ")
            print(traceback.format_exc())
        # framework.lock_release()
        print("start updating")
        update_st = datetime.datetime.now()
        if silent is True:  # If it is the first time to load, we will not output the difference.
            self.cov_uni[bk], vector = framework.compare_and_update(idntfr=model_idntfr, new_py_trs=xml_content,
                                                                    new_c_trs=info_content,
                                                                    silent=True)  # measure the coverage uniqueness
            print(vector.shape)
        else:
            self.cov_uni[bk], vector = framework.compare_and_update(idntfr=model_idntfr, new_py_trs=xml_content,
                                                                    new_c_trs=info_content,
                                                                    silent=False)  # measure the coverage uniqueness
            print(vector.shape)
        update_et = datetime.datetime.now()
        print("finish updating, the time difference is: ", update_et - update_st)

    def collect_info(self, *args):
        bk, python_bin, model_idntfr, config_dir, framework, library_path, silent, redis_conn = args
        self.collect_coverage(bk, python_bin, model_idntfr, config_dir, framework, library_path, silent)
        return self.cov_uni[bk]

    def before_predict(self, framework):
        # framework.lock_acquire()
        print(f"Clear gcda on dir: runtime/{framework.name}")
        subprocess.call(f"find runtime/{framework.name}/ -name '*.gcda' -exec rm -rf " + "{} \\;", shell=True)


class Arcs(Guidance):
    """
    Use Coverage.py to collect the runtime arcs
    """

    def __init__(self):
        super(Arcs, self).__init__()
        self.arcs = {}
        self.cov_uni = {}

    def reset(self):
        self.cov_uni = {}

    def compare_and_update(self, bk, arcs=None, silent=True):
        if bk not in self.arcs:
            self.arcs[bk] = {}
        change_flag = 0
        origin_arcs = self.arcs[bk]
        for file_path in arcs:
            if file_path not in origin_arcs:
                change_flag = 1
                origin_arcs[file_path] = arcs[file_path]
                if silent is False:
                    print(f"Find new file: {file_path}")
            else:
                new_arcs = []
                for ar in arcs[file_path]:
                    if ar not in origin_arcs[file_path]:
                        change_flag = 1
                        origin_arcs[file_path].append(ar)
                        new_arcs.append(ar)
                if silent is False and len(new_arcs) > 0:
                    print("New test requirements are invoked on file {}: {}".format(file_path, new_arcs))
        self.arcs[bk] = origin_arcs
        return change_flag

    def collect_info(self, *args):
        bk, python_bin, model_name, config_dir, framework, c_library, silent, redis_conn = args
        arcs = pickle.loads(redis_conn.hget(f"arcs_{model_name}", bk))
        self.cov_uni[bk] = self.compare_and_update(bk, arcs, silent)
        if self.cov_uni[bk] == 1:
            print("New test requirements invoked on framework {} when updating {}".format(bk, model_name))
        else:
            print("no update on framework {} when updating {}".format(bk, model_name))
        return self.cov_uni[bk]

    def get_score(self, theta=0.1):
        num_cu = len([v for v in self.cov_uni.values() if v == 1])
        return int(bool(num_cu))  # change reward to a very simple one
