"""Starlark macros for MKL.

if_nvpl is a conditional to check if we are building with NVPL.
if_enable_nvpl is a conditional to check if building with NVPL and NVPL is enabled.

nvpl_repository is a repository rule for creating NVPL repository rule that can
be pointed to either a local folder, or downloaded from the internet.
nvpl_repository depends on the following environment variables:
  * `TF_NVPL_ROOT`: The root folder where a copy of libnvpl is located.
"""

load(
    "//tensorflow/tsl/nvpl:build_defs.bzl",
    _if_enable_nvpl = "if_enable_nvpl",
    _if_nvpl = "if_nvpl",
    _nvpl_deps = "nvpl_deps",
    _nvpl_repository = "nvpl_repository",
)

if_nvpl = _if_nvpl
if_enable_nvpl = _if_enable_nvpl
nvpl_deps = _nvpl_deps
nvpl_repository = _nvpl_repository
