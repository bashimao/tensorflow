#ifndef TENSORFLOW_CORE_KERNELS_NVPL_NVPL_UTIL_H_
#define TENSORFLOW_CORE_KERNELS_NVPL_NVPL_UTIL_H_

#if defined(NVIDIA_PL)

using CPUDevice = Eigen::ThreadPoolDevice;

namespace tensorflow {
}

#endif

#endif