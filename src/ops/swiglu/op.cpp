#include "op.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_nvidia.hpp"
#endif

#include "../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace llaisys::ops {
namespace {

template <typename T>
float as_float(T value) {
    return utils::cast<float>(value);
}

template <typename T>
T from_float(float value) {
    return utils::cast<T>(value);
}

} // namespace

namespace {

template <typename T>
void swiglu_impl(tensor_t out, tensor_t gate, tensor_t up) {
    T *y = reinterpret_cast<T *>(out->data());
    const T *g = reinterpret_cast<const T *>(gate->data());
    const T *u = reinterpret_cast<const T *>(up->data());
    for (size_t i = 0; i < out->numel(); ++i) {
        const float gate_val = as_float(g[i]);
        const float silu = gate_val / (1.0f + std::exp(-gate_val));
        y[i] = from_float<T>(as_float(u[i]) * silu);
    }
}

} // namespace

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "SwiGLU: tensors must be contiguous.");
    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
#ifdef ENABLE_NVIDIA_API
        if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
            return nvidia::swiglu(out, gate, up);
        }
#endif
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return swiglu_impl<float>(out, gate, up);
    case LLAISYS_DTYPE_F16:
        return swiglu_impl<fp16_t>(out, gate, up);
    case LLAISYS_DTYPE_BF16:
        return swiglu_impl<bf16_t>(out, gate, up);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
