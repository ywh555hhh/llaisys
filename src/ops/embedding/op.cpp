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
void embedding_impl(tensor_t out, tensor_t index, tensor_t weight) {
    const int64_t *idx = reinterpret_cast<const int64_t *>(index->data());
    T *dst = reinterpret_cast<T *>(out->data());
    const T *src = reinterpret_cast<const T *>(weight->data());
    const size_t rows = index->shape()[0];
    const size_t dim = weight->shape()[1];
    const size_t vocab = weight->shape()[0];
    for (size_t row = 0; row < rows; ++row) {
        CHECK_ARGUMENT(idx[row] >= 0 && static_cast<size_t>(idx[row]) < vocab, "Embedding index out of range.");
        const size_t src_row = static_cast<size_t>(idx[row]);
        for (size_t col = 0; col < dim; ++col) {
            dst[row * dim + col] = src[src_row * dim + col];
        }
    }
}

} // namespace

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding indices must be int64.");
    CHECK_ARGUMENT(index->ndim() == 1 && weight->ndim() == 2 && out->ndim() == 2, "Embedding expects index[seq], weight[vocab, dim], out[seq, dim].");
    CHECK_ARGUMENT(out->shape()[0] == index->shape()[0] && out->shape()[1] == weight->shape()[1], "Embedding output shape mismatch.");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Embedding: tensors must be contiguous.");
    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return embedding_impl<float>(out, index, weight);
    case LLAISYS_DTYPE_F16:
        return embedding_impl<fp16_t>(out, index, weight);
    case LLAISYS_DTYPE_BF16:
        return embedding_impl<bf16_t>(out, index, weight);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
