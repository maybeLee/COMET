cd tensorflow
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install /tmp/tensorflow_pkg/tensorflow-2.9.0-cp37-cp37m-linux_x86_64.whl
pip install protobuf==3.20.*  # for https://stackoverflow.com/questions/72441758/typeerror-descriptors-cannot-not-be-created-directly

