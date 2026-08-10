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
void rope_impl(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    const size_t seq_len = in->shape()[0];
    const size_t heads = in->shape()[1];
    const size_t head_dim = in->shape()[2];
    const size_t half = head_dim / 2;
    const int64_t *pos = reinterpret_cast<const int64_t *>(pos_ids->data());
    T *y = reinterpret_cast<T *>(out->data());
    const T *x = reinterpret_cast<const T *>(in->data());
    for (size_t s = 0; s < seq_len; ++s) {
        for (size_t h = 0; h < heads; ++h) {
            const size_t base = (s * heads + h) * head_dim;
            for (size_t j = 0; j < half; ++j) {
                const float angle = static_cast<float>(pos[s]) / std::pow(theta, static_cast<float>(2 * j) / static_cast<float>(head_dim));
                const float sinv = std::sin(angle);
                const float cosv = std::cos(angle);
                const float a = as_float(x[base + j]);
                const float b = as_float(x[base + half + j]);
                y[base + j] = from_float<T>(a * cosv - b * sinv);
                y[base + half + j] = from_float<T>(b * cosv + a * sinv);
            }
        }
    }
}

} // namespace

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE position ids must be int64.");
    CHECK_ARGUMENT(out->ndim() == 3 && in->ndim() == 3, "RoPE expects [seq, heads, head_dim].");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(pos_ids->ndim() == 1 && pos_ids->shape()[0] == in->shape()[0], "RoPE position id shape mismatch.");
    CHECK_ARGUMENT(in->shape()[2] % 2 == 0, "RoPE head dimension must be even.");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "RoPE: tensors must be contiguous.");
    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rope_impl<float>(out, in, pos_ids, theta);
    case LLAISYS_DTYPE_F16:
        return rope_impl<fp16_t>(out, in, pos_ids, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_impl<bf16_t>(out, in, pos_ids, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
