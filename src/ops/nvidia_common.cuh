#pragma once

#include "../tensor/tensor.hpp"
#include "../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

namespace llaisys::ops::nvidia {

inline void check_cuda(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << what << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

inline int dtype_kind(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return 0;
    case LLAISYS_DTYPE_F16:
        return 1;
    case LLAISYS_DTYPE_BF16:
        return 2;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

__device__ inline float load_value(const void *ptr, size_t idx, int kind) {
    if (kind == 0) {
        return static_cast<const float *>(ptr)[idx];
    }
    if (kind == 1) {
        return __half2float(static_cast<const __half *>(ptr)[idx]);
    }
#if defined(__CUDA_ARCH__)
    return __bfloat162float(static_cast<const __nv_bfloat16 *>(ptr)[idx]);
#else
    return 0.0f;
#endif
}

__device__ inline void store_value(void *ptr, size_t idx, int kind, float value) {
    if (kind == 0) {
        static_cast<float *>(ptr)[idx] = value;
    } else if (kind == 1) {
        static_cast<__half *>(ptr)[idx] = __float2half_rn(value);
    } else {
        static_cast<__nv_bfloat16 *>(ptr)[idx] = __float2bfloat16_rn(value);
    }
}

inline dim3 grid_1d(size_t n, int block = 256) {
    return dim3(static_cast<unsigned int>((n + block - 1) / block));
}

} // namespace llaisys::ops::nvidia
