#if defined(NVIDIA_PL)

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/nvpl_util.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class NvplIdentityOp : public OpKernel {
 public:
  explicit NvplIdentityOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // TODO
  }

  bool IsExpensive() override { return false; }
};

#define REGISTER_NVPL_CPU(T)                                   \
  REGISTER_KERNEL_BUILDER(                                \
      Name("_NvplIdentity")                                  \
          .Device(DEVICE_CPU)                             \
          .TypeConstraint<T>("T"),                         \
      NvplIdentityOp<CPUDevice, T>);

TF_CALL_float(REGISTER_NVPL_CPU);
TF_CALL_bfloat16(REGISTER_NVPL_CPU);
}  // namespace tensorflow
#endif  // NVIDIA_PL