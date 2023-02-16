cd ../
CUDA_HOME=/usr/local/cuda-11 CUDA_ROOT=/usr/local/cuda-11 LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:$LD_LIBRARY_PATH PATH=/usr/local/cuda-11/bin:$PATH python -u -m scripts.generation.run --config_name random_selection.conf
