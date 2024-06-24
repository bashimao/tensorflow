#if defined(NVIDIA_PL)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/nvpl/nvpl_matmul_ops_common.h"
#include "nvpl_blas_cblas.h"
#include <iostream>

namespace tensorflow {

template <typename Device, typename T, bool USE_CUBLAS>
class NvplMatMulOp : public OpKernel {
 public:
  explicit NvplMatMulOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a{ctx->input(0)};
    const Tensor& b{ctx->input(1)};

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a.shape()),
                errors::InvalidArgument("In[0] ndims must be >= 2"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b.shape()),
                errors::InvalidArgument("In[1] ndims must be >= 2"));
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_b_ ? 1 : 0;

    int d1{a.dim_size(dim_pair[0].first)};
    int d2{b.dim_size(dim_pair[0].second)};
    OP_REQUIRES(ctx, d1 == d2,
                errors::InvalidArgument("Matrix size-incompatible: In[0]: ",
                                        a.shape().DebugString(),
                                        ", In[1]: ", b.shape().DebugString()));
    int a_dim_remaining{1 - dim_pair[0].first};
    int b_dim_remaining{1 - dim_pair[0].second};
    TensorShape out_shape({a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
    Tensor* out{};
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

    if (out->NumElements() == 0) {
      // If a has shape [0, x] or b has shape [x, 0], the output shape
      // is a 0-element matrix, so there is nothing to do.
      return;
    }

    if (a.NumElements() == 0 && b.NumElements() == 0) {
      // If a has shape [x, 0] and b has shape [0, y], the
      // output shape is [x, y] where x and y are non-zero, so we fill
      // the output with zeros.
      functor::SetZeroFunctor<Device, T> f;
      f(ctx->eigen_device<Device>(), out->flat<T>());
      return;
    }

    const int m{a.dim_size(1 - dim_pair[0].first)};
    const int k{a.dim_size(dim_pair[0].first)};
    const int n{b.dim_size(1 - dim_pair[0].second)};
    const bool transpose_a{dim_pair[0].first == 0};
    const bool transpose_b{dim_pair[0].second == 1};

    const T* const a_ptr{a.template flat<T>().data()};
    const T* const b_ptr{b.template flat<T>().data()};
    T* const c_ptr{out->template flat<T>().data()};

    NvplSGEMM(ctx, transpose_a, transpose_b,
              m, n, k,
              a_ptr, transpose_a ? m : k,
              b_ptr, transpose_b ? k : n,
              c_ptr, n);
  }

 private:
  bool transpose_a_;
  bool transpose_b_;

  static void NvplSGEMM(OpKernelContext* ctx, bool transa, bool transb,
                        const int m, const int n, const int k,
                        const float* a, const int lda,
                        const float* b, const int ldb,
                        float* c, const int ldc) {
    const float alpha{1.0f}, beta{0.0f};

    const enum CBLAS_TRANSPOSE cblas_transa{transa ? CblasTrans : CblasNoTrans};
    const enum CBLAS_TRANSPOSE cblas_transb{transb ? CblasTrans : CblasNoTrans};

    cblas_sgemm(CblasRowMajor, cblas_transa, cblas_transb, 
                m, n, k,
                alpha, a, lda, b, ldb,
                beta,  c, ldc);
    /*std::cout << transa << " ~ "
              << transb << " ~ "
              << m << " ~ " << n << " ~ " << k
              << " ~ " << alpha << " ~ " << a << " ~ " << lda << " ~ " << b << " ~ " << ldb
              << " ~ " << beta << " ~ " << c << " ~ " << ldc << std::endl;*/
  }
};

#define REGISTER_NVPL_CPU(T)                                   \
  REGISTER_KERNEL_BUILDER(                                \
      Name("NvplMatMul")                                  \
          .Device(DEVICE_CPU)                             \
          .TypeConstraint<T>("T"),                         \
      NvplMatMulOp<CPUDevice, T, false>);

TF_CALL_float(REGISTER_NVPL_CPU);
}  // namespace tensorflow
#endif  // NVIDIA_PL