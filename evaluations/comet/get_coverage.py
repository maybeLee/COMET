import sys
sys.path.append("../../implementations")
from classes.frameworks import Frameworks
from collections import defaultdict
import os
import json
import dill


def search_within_files(files, cache):
    for file in files:
        dirname = os.path.dirname(file)
        basename = os.path.basename(file)
        if dirname not in tensorflow_modules:
            continue
        if basename not in tensorflow_modules[dirname]:
            continue
        for module in tensorflow_modules_meta:
            if dirname in tensorflow_modules_meta[module]:
                break
        hit_line, total_line, hit_branch, total_branch = cache[module]
        lt,lm,_,bh,bm,_ = files[file].coverage
        hit_line += lt
        total_line += lt+lm
        hit_branch += bh
        total_branch += bh+bm
        cache[module] = (hit_line, total_line, hit_branch, total_branch)
    return cache


def ceil(x, decimal):
    return math.ceil(x*10**decimal)/10**decimal


tensorflow_modules = json.load(open("../../implementations/scripts/analysis/tensorflow_related_modules.json", "rb+"))
tensorflow_modules_meta = json.load(open("../../implementations/scripts/analysis/tensorflow_modules_meta.json", "rb+"))

bk = "tensorflow"
comet_path = f"./comet_{bk}.pkl"
frameworks = defaultdict(dict)
frameworks[bk]["comet"] = dill.load(open(comet_path, "rb"))
for method in frameworks[bk]:
    cache = {}
    print("\n---- working on method: ", method)
    hit_line, total_line, hit_branch, total_branch = 0,0,0,0
    for module in tensorflow_modules_meta:
        cache[module] = (hit_line, total_line, hit_branch, total_branch)
    cache = search_within_files(frameworks[bk][method].c_files, cache)
    cache = search_within_files(frameworks[bk][method].py_files, cache)
    for module in cache:
        hl, tl, hr, tr = cache[module]
        hit_branch += hr
        total_branch += tr
        hit_line += hl
        total_line += tl
    
    print(f"Branch Coverage: {hit_branch}/{total_branch}({hit_branch/total_branch}), \
          Line Coverage: {hit_line}/{total_line}({hit_line/total_line})")

