#include "op.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.hpp"
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
void self_attention_impl(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    const size_t qlen = q->shape()[0];
    const size_t nh = q->shape()[1];
    const size_t hd = q->shape()[2];
    const size_t kvlen = k->shape()[0];
    const size_t nkvh = k->shape()[1];
    T *out = reinterpret_cast<T *>(attn_val->data());
    const T *query = reinterpret_cast<const T *>(q->data());
    const T *key = reinterpret_cast<const T *>(k->data());
    const T *value = reinterpret_cast<const T *>(v->data());
    std::vector<float> logits(kvlen);
    std::vector<float> probs(kvlen);
    for (size_t qi = 0; qi < qlen; ++qi) {
        const size_t max_k = std::min(kvlen, qi + (kvlen - qlen) + 1);
        for (size_t h = 0; h < nh; ++h) {
            const size_t kvh = h / (nh / nkvh);
            float max_logit = -std::numeric_limits<float>::infinity();
            for (size_t ki = 0; ki < max_k; ++ki) {
                float dot = 0.0f;
                for (size_t d = 0; d < hd; ++d) {
                    dot += as_float(query[(qi * nh + h) * hd + d]) * as_float(key[(ki * nkvh + kvh) * hd + d]);
                }
                logits[ki] = dot * scale;
                max_logit = std::max(max_logit, logits[ki]);
            }
            float sum = 0.0f;
            for (size_t ki = 0; ki < max_k; ++ki) {
                probs[ki] = std::exp(logits[ki] - max_logit);
                sum += probs[ki];
            }
            for (size_t d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (size_t ki = 0; ki < max_k; ++ki) {
                    acc += (probs[ki] / sum) * as_float(value[(ki * nkvh + kvh) * hd + d]);
                }
                out[(qi * nh + h) * hd + d] = from_float<T>(acc);
            }
        }
    }
}

} // namespace

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    CHECK_ARGUMENT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3, "Self-attention expects 3D tensors.");
    const size_t qlen = q->shape()[0];
    const size_t nh = q->shape()[1];
    const size_t hd = q->shape()[2];
    const size_t kvlen = k->shape()[0];
    const size_t nkvh = k->shape()[1];
    CHECK_ARGUMENT(kvlen >= qlen, "Self-attention KV length must be at least query length.");
    CHECK_ARGUMENT(v->shape()[0] == kvlen && v->shape()[1] == nkvh && v->shape()[2] == hd, "Self-attention value shape mismatch.");
    CHECK_ARGUMENT(attn_val->shape()[0] == qlen && attn_val->shape()[1] == nh && attn_val->shape()[2] == hd, "Self-attention output shape mismatch.");
    CHECK_ARGUMENT(nkvh > 0 && nh % nkvh == 0, "Self-attention requires query heads to be a multiple of KV heads.");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), "Self-attention: tensors must be contiguous.");
    if (attn_val->deviceType() != LLAISYS_DEVICE_CPU) {
#ifdef ENABLE_NVIDIA_API
        if (attn_val->deviceType() == LLAISYS_DEVICE_NVIDIA) {
            return nvidia::self_attention(attn_val, q, k, v, scale);
        }
#endif
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
    switch (attn_val->dtype()) {
    case LLAISYS_DTYPE_F32:
        return self_attention_impl<float>(attn_val, q, k, v, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_impl<fp16_t>(attn_val, q, k, v, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_impl<bf16_t>(attn_val, q, k, v, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype());
    }
}
} // namespace llaisys::ops
