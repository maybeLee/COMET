# COMET: Coverage-guided Model Generation For Deep Learning Library Testing
## Bugs

To date, we have detected 32 bugs, including 21 confirmed bugs and 7 out of those confirmed bugs have been fixed. See [here](./evaluations/bugs.csv) for details.

## Implementations

We provide our implementation of COMET in `./implementations` directory. To use them, a few configurations steps are required.

### DL Models

We use 10 published DL models as the seed model for model generation. These DL models can be accessed through [here](https://drive.google.com/drive/folders/1d6rk80UvqcRtc6voN3jaux3wTbmUAYaI?usp=sharing). After downloading them, create a directory named `origin_model`  under the  `./data/` directory to save these models.

We also provide the synthesized model generated using our model synthesis algorithm, these DL models are stored in the directory `./data/synthesized_model`.

### Environment

We use Docker and Conda to manage our environment. Please follow below steps to create the COMET's environment

#### Step 1: Create and enter the container.

We first need to clone this repository via `git clone git@github.com:maybeLee/COMET.git`, then we create the container:

```shell
cd COMET
docker build -t comet:latest -f Dockerfile .
docker run --name=COMET -ti -v ${PWD}:/root comet:latest  # Please note that the ${PWD} should be the COMET's repository directory in your machine.
apt-get update
apt-get install vim wget -y
```

#### Step 2: Build the python environments.

Inside the container, we need to build python environments, below are specific steps.

**Build COMET's python environment:**

```shell
conda create -n comet python=3.7
conda activate comet
pip install tensorflow-gpu==2.7.0 keras==2.7.0 notebook
pip install -r requirements.txt
conda deactivate
```

**Build Tensorflow's python environment:**

To collect code coverage of TensorFlow, we need to install its source code and manually compile it:

First, create the tensorflow environment, install the source code and dependencies: 

```shell
conda create -n tensorflow python=3.7
conda activate tensorflow
pip install -r requirements.txt
cd /root/dl_libraries/tensorflow
./auto_install.sh  # install tensorflow's source code, install necessary dependencies, checkout to the target version: 76f23e7975ce2bf81721673f20656530e1e609ac 
./install_gcc.sh  # install gcc9 to compile tensorflow
```

Second, configure and compile TensorFlow:

```shell
cd /root/dl_libraries/tensorflow/tensorflow
TF_CUDA_VERSION=11 TF_CUDNN_VERSION=8 ./configure  # During the configuration, choose y for the 'compile with cuda' option
cd /root/dl_libraries/tensorflow
./auto_compile.sh  # compile tensorflow with --coverage option.
./auto_binding.sh
pip install keras==2.8.0
conda deactivate
```

**Build model conversion's python environment**

```shell
# build model conversion environment
conda create -n model_convertor python=3.6
conda activate model_convertor
pip install --upgrade pip
pip install tensorflow-gpu==2.6
pip install tf2onnx
pip install keras==2.6.0
conda deactivate
```

**Build pytorch and onnx's python environment:**

```shell
# build pytorch and onnx environment
conda create -n pytorch python=3.6
conda activate pytorch
pip install --upgrade pip
pip install -r requirements.txt
pip install onnx2pytorch onnx
conda install pytorch==1.10.0 -c pytorch
pip install onnxruntime-gpu==1.10.0
conda deactivate
```

Note that we detect some bugs in the current implementation of onnx2pytorch, these bugs makes onnx2pytorch fail to convert most of ONNX models to PyTorch. **To make converting ONNX model to PyTorch model applicable, we applied several patches to ONNX2PyTorch (see instructions [here](./patches_on_onnx.md))**.

#### Step 3: Build dependencies

We need redis and lcov 1.15.0, to install them:

```shell
cd /root/
./install_dependencies.sh
```

## Run

```shell
cd implementations/
./init_redis.sh  # start the redis
cd implementations/run
conda activate comet
```

Run `run_comet.sh` to execute COMET. If you want to synthesize new initial models, run `synthesize_model.sh`, then run `run_comet.sh` to execute COMET.

Run `run_old_mutation.sh`, `run_random_selection.sh` to run COMET with old mutation operators or without our MCMC-based search algorithm.

## See results
When running `./run_comet.sh`, the default output directory is stored in '/root/data/working_dir/COMET' and this is determined by the `output_dir` option (line 4) in the comet's configuration file [here](./implementations/config/comet.conf).

**In '/root/data/working_dir/COMET/results':**

'./instrument_result/' stores the code coverage metadata. Note that only the `tensorflow.pkl` file can be used, other files are invalid.

'./delta/' stores the runtime api call coverage and python arcs collected using [coveragepy](https://coverage.readthedocs.io/en/7.1.0/). Again, only the `tensorflow_api_cov.txt` and `tensorflow_arcs.txt` can be used as reference (`tensorflow_arcs.txt` records the python branch coverage collected by coveragepy, we use it to guide test case generation. `tensorflow_api_cov.txt` records



please note that we do not use the code coverage collected by coveragepy as the reference to evaluate the test adequacy in our evaluation because this tool only collects python coverage).

'./models' stores all generated models. If a DL model result in a crash or nan bug, its '.h5' format model will be moved to './crashes' and './nan'.

'./crashes/', './nan/' stores the '.h5' format of DL models that result in library crash or nan outputs.

'./seed_selector' and 'mutated_model.json' stores runtime data which should be ignored.

**When running COMET for six hours, we usually generate around 800-900 valid DL models that can be used for library testing.**

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
