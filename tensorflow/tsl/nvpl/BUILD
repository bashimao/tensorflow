load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(
    "//tensorflow/tsl:tsl.bzl",
    "clean_dep",
)

licenses(["notice"])  # 3-Clause BSD

config_setting(
    name = "build_with_nvpl",
    define_values = {
        "build_with_nvpl": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "build_with_nvpl_lnx_openmp",
    constraint_values = [
        "@platforms//os:linux",
    ],
    define_values = {
        "build_with_nvpl": "true",
        "build_with_openmp": "true",
    },
    visibility = ["//visibility:public"],
)


config_setting(
    name = "enable_nvpl",
    define_values = {
        "enable_nvpl": "true",
        "build_with_nvpl": "true",
    },
    visibility = ["//visibility:public"],
)

filegroup(
    name = "LICENSE",
    srcs = [
        "NVPL_LICENSE",
        "@llvm_openmp//:LICENSE.txt",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nvpl_libs_linux",
    srcs = [
        "@llvm_openmp//:libiomp5.so",
    ],
    hdrs = ["@llvm_openmp//:config_omp"],
    target_compatible_with = select({
        "//tensorflow/tsl/nvpl:build_with_nvpl": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nvidia_binary_blob",
    target_compatible_with = select({
        "//tensorflow/tsl/nvpl:build_with_nvpl": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//visibility:public"],
    deps = select({
        clean_dep("//tensorflow/tsl:linux_x86_64"): [
            ":nvpl_libs_linux",
        ],
        "//conditions:default": [],
    }),
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:public"],
)