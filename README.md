## Notice
The currenct artifact is not ready. We are preparing the final version.

## Bugs

To date, we have detected 32 bugs, including 21 confirmed bugs and 7 out of those confirmed bugs have been fixed. See [here](./evaluations/bugs.csv) for details.

## Implementations

We provide our implementation of COMET in `./implementations` directory. To use them, a few configurations steps are required.

### DL Models

We use 10 published DL models as the seed model for model generation. 
These DL models can be accessed through [here](https://drive.google.com/drive/folders/1d6rk80UvqcRtc6voN3jaux3wTbmUAYaI?usp=sharing). After downloading them, create a directory named `origin_model`  under the  `./data/` directory to save these models.

We also provide the synthesized model using model synthesis algorithm, 
these DL models are stored in the directory `./implementations/data/synthesized_model`.

### Environment

We use Anaconda3 to manage our environment. Please run the following commands to create the main environment

```
conda create -n comet python=3.7
pip install tensorflow-gpu==2.7.0 keras==2.7.0 notebook
pip install -r requirements
```

We further build several conda environment for model conversion and testing different libraries.

```
# build model conversion environment
conda create -n model_convertor python=3.6
pip install tensorflow-gpu==2.6
pip install tf2onnx
pip install keras==2.6.0
```

For tensorflow environment, run the following command to install tensorflow from source

```
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow/
git checkout 76f23e7975ce2bf81721673f20656530e1e609ac
pip install numpy
pip install keras_preprocessing
conda install bazel=3.7.2
```

Then type `cd tensorflow && ./configure`, to configure tensorflow with `--coverage` option. 
After configuring tensorflow with `--coverage` option, run following commands to compile tensorflow

```
cd tensorflow
bazel build --copt=-coverage --linkopt=-lgcov --verbose_failures --spawn_strategy=standalone --config=opt --config=cuda --jobs=24 //tensorflow/tools/pip_package:build_pip_package
```

Then build tensorflow's python environment:

```
# build tensorflow environment
conda create -n tensorflow python=3.6
conda install cudnn=8.2.1 -c anaconda
compile tensorflow (76f23e7975c) from source to collect code coverage
pip install keras==2.8.0
pip install -r requirements
```

Run the following commands to build pytorch and onnx's environment:
```
# build pytorch and onnx environment
conda create -n pytorch python=3.6
pip install -r requirements
pip install onnx2pytorch onnx
pip install pytorch==1.10.0 onnxruntime-gpu==1.10.0

```

Note that we detect some bugs in the current implementation of onnx2pytorch, 
these bugs makes onnx2pytorch fail to convert most of ONNX models to PyTorch. 
To make converting ONNX model to PyTorch model applicable, we applied several patches to ONNX2PyTorch 
(see instructions [here](./patches_on_onnx.md)).

After configuring the conda environment, replace the `/root` annotation on files under the `./implementation/config` 
to the actual path (e.g., replace `python_prefix = /root/anaconda3/envs/` in `comet.conf` 
with `python_prefix = /[anaconda_dir]/anaconda3/envs/`, where `/[anaconda_dir]` is the main directory we store `anaconda3`). 

To collect code coverage of tensorflow, we also need replace the `/root` under `tensorflow_modules_meta.json`, 
`tensorflow_related_modules.json` with the actual directory we used to compile tensorflow.

## Run

`cd implementations/run`

Run `run_comet.sh` to execute COMET.

(If you want to synthesize new initial models, run `synthesize_model.sh`, then run `run_comet.sh` to execute COMET). 

Run `run_old_mutation.sh`, `run_random_selection.sh` to run COMET with old mutation operators or without our MCMC-based search algorithm.

## See results
When running `./run_comet.sh`, the default output directory is stored in 
'./data/working_dir/COMET/results' and this is determined by the 
`output_dir` option (line 4) in the comet's configuration file [here](./implementations/config/comet.conf).

**In './data/working_dir/COMET/results':**

'./instrument_result/' stores the code coverage metadata. Note that only the `tensorflow.pkl` file can be used, 
other files are incorrect.

'./delta/' stores the runtime api call coverage and python arcs collected using [coveragepy](https://coverage.readthedocs.io/en/7.1.0/). 
Again, only the `tensorflow_api_cov.txt` and `tensorflow_arcs.txt` can be used as reference (please note that we do not use the code coverage collected by coveragepy as the reference to evaluate the test adequacy because 
this tool only collects python coverage).

'./models' stores all generated models. If a DL model result in a crash or nan bug, its '.h5' format model will be moved to './crashes' and './nan'.

'./crashes/', './nan/' stores the '.h5' format of DL models that result in library crash or nan outputs.

'./seed_selector' and 'mutated_model.json' stores runtime data which should be ignored.

**When running COMET for six hours, we usually generate around 900 valid DL models that can be used for library testing.**

**Please note that the result of COMET in each run is non-deterministic. Therefore, the achieved api call coverage and code coverage may be different.** 

## Collect Code Coverage
We provide a [script](./evaluations/comet/get_coverage.py) to analyze the branch coverage and line coverage on 
TensorFlow’s model construction and execution modules based on the collected `*tensorflow.pkl`.
For instance, by running the following commands, you can get the branch coverage and line coverage of COMET:

```
>> cd ./evaluations/comet
>> python get_coverage.py

---- working on method:  comet
Branch Coverage: 6080/31390(0.19369225868110862),           Line Coverage: 21823/62558(0.34884427251510597)
```
