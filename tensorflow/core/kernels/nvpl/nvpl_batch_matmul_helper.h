#ifndef TENSORFLOW_CORE_KERNELS_NVPL_NVPL_BATCH_MATMUL_HELPER_H_
#define TENSORFLOW_CORE_KERNELS_NVPL_NVPL_BATCH_MATMUL_HELPER_H_

#ifdef NVIDIA_PL

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/nvpl/nvpl_matmul_ops_common.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_bcast.h"

#include "nvpl_blas_cblas.h"
#include "nvpl_blas_service.h"

namespace tensorflow {

template <typename T>
struct NvplBatchMatMulHelper {
  void ExtractMatrixAddresses(Tensor tensor, std::vector<auto *>& matrix_addresses) {
    const int ndims = tensor.dims();
    if (ndims > 2) {
      for (int i = 0; i < tensor.dim_size(0); i++) {
        ExtractMatrixAddresses(tensor.SubSlice(i), matrix_addresses);
      }
    } else if (ndims == 2) {
      auto* const matrix_ptr{tensor.template unaligned_flat<T>().data()};
      matrix_addresses.push_back(matrix_ptr);
    }
  }
};

}  // namespace tensorflow

#endif  // NVIDIA_PL
#endif  // TENSORFLOW_CORE_KERNELS_NVPL_NVPL_BATCH_MATMUL_HELPER_H_
