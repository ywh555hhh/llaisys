#include "op.hpp"

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
void rms_norm_impl(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    const size_t rows = in->shape()[0];
    const size_t cols = in->shape()[1];
    T *y = reinterpret_cast<T *>(out->data());
    const T *x = reinterpret_cast<const T *>(in->data());
    const T *w = reinterpret_cast<const T *>(weight->data());
    for (size_t row = 0; row < rows; ++row) {
        float sum_sq = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            const float v = as_float(x[row * cols + col]);
            sum_sq += v * v;
        }
        const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(cols) + eps);
        for (size_t col = 0; col < cols; ++col) {
            y[row * cols + col] = from_float<T>(as_float(x[row * cols + col]) * inv_rms * as_float(w[col]));
        }
    }
}

} // namespace

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 1, "RMSNorm expects 2D input/output and 1D weight.");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(weight->shape()[0] == in->shape()[1], "RMSNorm weight shape mismatch.");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "RMSNorm: tensors must be contiguous.");
    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_impl<float>(out, in, weight, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_impl<fp16_t>(out, in, weight, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_impl<bf16_t>(out, in, weight, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
