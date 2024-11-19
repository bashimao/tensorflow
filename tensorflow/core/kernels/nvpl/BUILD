load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test_nvpl",
    "tf_nvpl_kernel_library",
)
load(
    "//third_party/nvpl:build_defs.bzl",
    "nvpl_deps",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:__subpackages__",
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

# Public support libraries ----------------------------------------------------
NVPL_SHORT_DEPS = [
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core/framework:bounds_check",
    "//tensorflow/core/kernels:ops_util",
] + ["@nvpl//:nvpl"]

# nvpl_deps()

NVPL_DEPS = NVPL_SHORT_DEPS + [
    "//third_party/eigen3",
    "//tensorflow/core:array_grad",
    "//tensorflow/core:math_grad",
    "//tensorflow/core:nn_grad",
    "//tensorflow/core:protos_all_cc",
    "//tensorflow/core/kernels:concat_lib",
    "//tensorflow/tsl/framework/contraction:eigen_contraction_kernel",
    "//tensorflow/core/kernels:fill_functor",
    "//tensorflow/core/kernels:gather_functor",
    "//tensorflow/core/kernels:transpose_functor",
]

tf_nvpl_kernel_library(
    name = "nvpl_matmul_op",
    srcs = ["nvpl_matmul_op.cc"],
    hdrs = ["nvpl_matmul_ops_common.h"],
    deps = NVPL_DEPS,
)

tf_nvpl_kernel_library(
    name = "nvpl_batch_matmul_op",
    srcs = ["nvpl_batch_matmul_op.cc"],
    hdrs = [
        "nvpl_matmul_ops_common.h",
        "nvpl_batch_matmul_helper.h",
        ],
    deps = NVPL_DEPS,
)

tf_nvpl_kernel_library(
    name = "nvpl_identity_op",
    prefix = "nvpl_identity_op",
    deps = NVPL_DEPS,
)

NVPL_TEST_DEPS = [
    ":nvpl_input_conversion_op",
    "//tensorflow/cc:cc_ops",
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:framework",
    "//tensorflow/core:framework_internal",
    "//tensorflow/core:lib",
    "//tensorflow/core:protos_all_cc",
    "//tensorflow/core:test",
    "//tensorflow/core:test_main",
    "//tensorflow/core:testlib",
    "//tensorflow/core/kernels:ops_testutil",
    "//tensorflow/core/kernels:ops_util",
]

tf_cc_test_nvpl(
    name = "nvpl_matmul_op_benchmark",
    size = "small",
    srcs = ["nvpl_matmul_op_benchmark.cc"],
    linkstatic = 1,
    deps = [
        "//tensorflow/core/kernels:matmul_op",
        "//tensorflow/core/kernels/nvpl:nvpl_matmul_op",
    ] + NVPL_TEST_DEPS,
)
