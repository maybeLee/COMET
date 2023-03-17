cd tensorflow
bazel build --copt=-coverage --linkopt=-lgcov --verbose_failures --spawn_strategy=standalone --config=opt --config=cuda --jobs=16 //tensorflow/tools/pip_package:build_pip_package

