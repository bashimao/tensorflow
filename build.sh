# bazel build --local_cpu_resources=16 --copt=-Wno-error=dangling-pointer= --config=nvpl --verbose_failures //tensorflow/tools/pip_package:build_pip_package
bazel build --local_cpu_resources=16 --copt=-Wno-error=dangling-pointer= --config=nvpl //tensorflow/tools/pip_package:build_pip_package
