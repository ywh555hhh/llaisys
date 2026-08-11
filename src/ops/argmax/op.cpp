#include "op.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_nvidia.hpp"
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
void argmax_impl(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    const T *v = reinterpret_cast<const T *>(vals->data());
    int64_t best_idx = 0;
    float best_val = as_float(v[0]);
    for (size_t i = 1; i < vals->numel(); ++i) {
        const float current = as_float(v[i]);
        if (current > best_val) {
            best_val = current;
            best_idx = static_cast<int64_t>(i);
        }
    }
    reinterpret_cast<T *>(max_val->data())[0] = from_float<T>(best_val);
    reinterpret_cast<int64_t *>(max_idx->data())[0] = best_idx;
}

} // namespace

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax index output must be int64.");
    CHECK_ARGUMENT(max_idx->numel() == 1 && max_val->numel() == 1, "Argmax outputs must be single-element tensors.");
    CHECK_ARGUMENT(vals->numel() > 0, "Argmax input must not be empty.");
    ASSERT(vals->isContiguous() && max_idx->isContiguous() && max_val->isContiguous(), "Argmax: tensors must be contiguous.");
    if (vals->deviceType() != LLAISYS_DEVICE_CPU) {
#ifdef ENABLE_NVIDIA_API
        if (vals->deviceType() == LLAISYS_DEVICE_NVIDIA) {
            return nvidia::argmax(max_idx, max_val, vals);
        }
#endif
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
    switch (vals->dtype()) {
    case LLAISYS_DTYPE_F32:
        return argmax_impl<float>(max_idx, max_val, vals);
    case LLAISYS_DTYPE_F16:
        return argmax_impl<fp16_t>(max_idx, max_val, vals);
    case LLAISYS_DTYPE_BF16:
        return argmax_impl<bf16_t>(max_idx, max_val, vals);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
    }
}
} // namespace llaisys::ops
