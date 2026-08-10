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
void linear_impl(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    const size_t batch = in->shape()[0];
    const size_t in_features = in->shape()[1];
    const size_t out_features = weight->shape()[0];
    T *y = reinterpret_cast<T *>(out->data());
    const T *x = reinterpret_cast<const T *>(in->data());
    const T *w = reinterpret_cast<const T *>(weight->data());
    const T *b = bias ? reinterpret_cast<const T *>(bias->data()) : nullptr;
    for (size_t row = 0; row < batch; ++row) {
        for (size_t col = 0; col < out_features; ++col) {
            float acc = b ? as_float(b[col]) : 0.0f;
            for (size_t k = 0; k < in_features; ++k) {
                acc += as_float(x[row * in_features + k]) * as_float(w[col * in_features + k]);
            }
            y[row * out_features + col] = from_float<T>(acc);
        }
    }
}

} // namespace

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2, "Linear expects 2D out, input, and weight.");
    const size_t batch = in->shape()[0];
    const size_t in_features = in->shape()[1];
    const size_t out_features = weight->shape()[0];
    CHECK_ARGUMENT(weight->shape()[1] == in_features, "Linear weight shape mismatch.");
    CHECK_ARGUMENT(out->shape()[0] == batch && out->shape()[1] == out_features, "Linear output shape mismatch.");
    if (bias) {
        CHECK_ARGUMENT(bias->ndim() == 1 && bias->shape()[0] == out_features, "Linear bias shape mismatch.");
    }
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && (!bias || bias->isContiguous()), "Linear: tensors must be contiguous.");
    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return linear_impl<float>(out, in, weight, bias);
    case LLAISYS_DTYPE_F16:
        return linear_impl<fp16_t>(out, in, weight, bias);
    case LLAISYS_DTYPE_BF16:
        return linear_impl<bf16_t>(out, in, weight, bias);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
