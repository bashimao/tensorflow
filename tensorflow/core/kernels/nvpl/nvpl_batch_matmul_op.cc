#define EIGEN_USE_THREADS

#if defined(NVIDIA_PL)

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/matmul_op_impl.h"
#include "tensorflow/core/kernels/nvpl/nvpl_batch_matmul_helper.h"
#include "tensorflow/core/kernels/nvpl/nvpl_matmul_ops_common.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_bcast.h"

#include "nvpl_blas_cblas.h"
#include "nvpl_blas_service.h"
#include <iostream>

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

//  The third parameter v2_bcast is set to true if we are using V2 otherwise
//  we set it to false.
template <typename Device, typename Tlhs, typename Trhs, typename Toutput,
          bool v2_bcast>
class BatchMatMulNvpl : public OpKernel {
 public:
  explicit BatchMatMulNvpl(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adj_x", &adj_x_));
    OP_REQUIRES_OK(context, context->GetAttr("adj_y", &adj_y_));
  }

  virtual ~BatchMatMulNvpl() {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& lhs = ctx->input(0);
    const Tensor& rhs = ctx->input(1);

    if (!v2_bcast) {
      // Using V1, so check to make sure lhs and rhs dimensions are correct and
      // no broadcasting is needed.
      OP_REQUIRES(ctx, lhs.dims() == rhs.dims(),
                  errors::InvalidArgument("lhs and rhs has different ndims: ",
                                          lhs.shape().DebugString(), " vs. ",
                                          rhs.shape().DebugString()));
      const int ndims = lhs.dims();
      OP_REQUIRES(
          ctx, ndims >= 2,
          errors::InvalidArgument("lhs and rhs ndims must be >= 2: ", ndims));
      for (int i = 0; i < ndims - 2; ++i) {
        OP_REQUIRES(ctx, lhs.dim_size(i) == rhs.dim_size(i),
                    errors::InvalidArgument(
                        "lhs.dim(", i, ") and rhs.dim(", i,
                        ") must be the same: ", lhs.shape().DebugString(),
                        " vs ", rhs.shape().DebugString()));
      }
    } else {
      OP_REQUIRES(
          ctx, lhs.dims() >= 2,
          errors::InvalidArgument("In[0] ndims must be >= 2: ", lhs.dims()));
      OP_REQUIRES(
          ctx, rhs.dims() >= 2,
          errors::InvalidArgument("In[1] ndims must be >= 2: ", rhs.dims()));
    }

    // lhs and rhs can have different dimensions
    const auto ndims_lhs = lhs.dims();
    const auto ndims_rhs = rhs.dims();

    // Get broadcast info
    MatMulBCast bcast(lhs.shape().dim_sizes(), rhs.shape().dim_sizes());
    OP_REQUIRES(
        ctx, bcast.IsValid(),
        errors::InvalidArgument(
            "In[0] and In[1] must have compatible batch dimensions: ",
            lhs.shape().DebugString(), " vs. ", rhs.shape().DebugString()));

    TensorShape out_shape = bcast.output_batch_shape();

    auto lhs_rows = lhs.dim_size(ndims_lhs - 2);
    auto lhs_cols = lhs.dim_size(ndims_lhs - 1);
    auto rhs_rows = rhs.dim_size(ndims_rhs - 2);
    auto rhs_cols = rhs.dim_size(ndims_rhs - 1);

    if (adj_x_) std::swap(lhs_rows, lhs_cols);
    if (adj_y_) std::swap(rhs_rows, rhs_cols);
    OP_REQUIRES(ctx, lhs_cols == rhs_rows,
                errors::InvalidArgument(
                    "lhs mismatch rhs shape: ", lhs_cols, " vs. ", rhs_rows,
                    ": ", lhs.shape().DebugString(), " ",
                    rhs.shape().DebugString(), " ", adj_x_, " ", adj_y_));

    out_shape.AddDim(lhs_rows);
    out_shape.AddDim(rhs_cols);

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      return;
    }
    if (lhs.NumElements() == 0 || rhs.NumElements() == 0) {
      functor::SetZeroFunctor<Device, Toutput> f;
      f(ctx->eigen_device<Device>(), out->flat<Toutput>());
      return;
    }

    // Compute parameters for DNNL matmul primitive.
    NvplBatchMatMulHelper<float> bmm;
    string prefix = "batchmatmul";
    std::vector<const float*> lhs_matrix_addresses;
    std::vector<const float*> rhs_matrix_addresses;
    std::vector<float*> out_matrix_addresses;
    bmm.ExtractMatrixAddresses(lhs, lhs_matrix_addresses);
    bmm.ExtractMatrixAddresses(rhs, rhs_matrix_addresses);
    bmm.ExtractMatrixAddresses(*out, out_matrix_addresses);

    nvpl_int_t group_count;
    nvpl_int_t group_size = 1;
    if (ndims_lhs > 3) {
      group_count = lhs.dim_size(0);
      for (int i = 1; i < ndims_lhs - 2; i++)
        group_size *= lhs.dim_size(i);
    } else if (ndims_lhs == 3) {
      group_count = 1;
      group_size = lhs.dim_size(0);
    }

    const enum CBLAS_TRANSPOSE cblas_transa{adj_x_ ? CblasTrans : CblasNoTrans};
    const enum CBLAS_TRANSPOSE cblas_transb{adj_y_ ? CblasTrans : CblasNoTrans};
    const float alpha{1.0f}, beta{0.0f};

    std::vector<nvpl_int_t> group_sizes;
    std::vector<float> alpha_array;
    std::vector<float> beta_array;
    std::vector<nvpl_int_t> m_array;
    std::vector<nvpl_int_t> n_array;
    std::vector<nvpl_int_t> k_array;
    std::vector<nvpl_int_t> lda_array;
    std::vector<nvpl_int_t> ldb_array;
    std::vector<nvpl_int_t> ldc_array;
    std::vector<enum CBLAS_TRANSPOSE> trans_a_array;
    std::vector<enum CBLAS_TRANSPOSE> trans_b_array;

    for (int i = 0; i < group_count; i++) {
      group_sizes.push_back(group_size);
      alpha_array.push_back(alpha);
      beta_array.push_back(beta);
      m_array.push_back(lhs_rows);
      n_array.push_back(rhs_cols);
      k_array.push_back(lhs_cols);
      lda_array.push_back(adj_x_ ? lhs_rows : lhs_cols);
      ldb_array.push_back(adj_y_ ? rhs_rows : rhs_cols);
      ldc_array.push_back(rhs_cols);
      trans_a_array.push_back(cblas_transa);
      trans_b_array.push_back(cblas_transb);
    }

    cblas_sgemm_batch(CblasRowMajor,
                      trans_a_array.data(),
                      trans_b_array.data(),
                      m_array.data(), n_array.data(),
                      k_array.data(), alpha_array.data(),
                      lhs_matrix_addresses.data(), lda_array.data(),
                      rhs_matrix_addresses.data(), ldb_array.data(),
                      beta_array.data(),
                      out_matrix_addresses.data(), ldc_array.data(),
                      group_count, group_sizes.data());
  }

 protected:
  virtual void ExtendNvplMatMulParams() {}
  std::vector<string> fused_ops_;

 private:
  bool adj_x_;
  bool adj_y_;

};

#define REGISTER_BATCH_MATMUL_NVPL(T)                                      \
  REGISTER_KERNEL_BUILDER(Name("NvplBatchMatMul")                          \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<T>("T"),                     \
                          BatchMatMulNvpl<CPUDevice, T, T, T, false>);

TF_CALL_float(REGISTER_BATCH_MATMUL_NVPL);

}  // end namespace tensorflow
#endif
