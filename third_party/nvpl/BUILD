licenses(["notice"])  # 3-Clause BSD

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(default_visibility = ["//visibility:public"])

alias(
    name = "build_with_nvpl",
    actual = "//tensorflow/tsl/nvpl:build_with_nvpl",
)

alias(
    name = "enable_nvpl",
    actual = "//tensorflow/tsl/mkl:enable_nvpl",
)

alias(
    name = "LICENSE",
    actual = "//tensorflow/tsl/nvpl:LICENSE",
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:public"],
)

#cc_import(
#    name = "libnvpl_blas_lp64_seq",
#    shared_library = "lib/libnvpl_blas_lp64_seq.so.0.2.0",
#)

cc_library(
    name = "nvpl",
    hdrs = glob([
        "include/*.h",
    ]),
    strip_include_prefix = "include",
    srcs = [
        "lib/libnvpl_blas_lp64_seq.so.0.2.0",
        "lib/libnvpl_blas_lp64_seq.so.0",
        "lib/libnvpl_blas_lp64_seq.so",
    ],
    linkopts = ["-lrt"],
    visibility = ["//visibility:public"],
    deps = [],
)
