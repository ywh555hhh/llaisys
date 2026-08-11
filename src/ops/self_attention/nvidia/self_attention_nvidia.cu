#include "self_attention_nvidia.hpp"
#include "../../nvidia_common.cuh"

namespace llaisys::ops::nvidia {
namespace {
__global__ void attention_kernel(void *out, const void *q, const void *k, const void *v, size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, float scale, int kind) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = qlen * nh * hd;
    if (idx < n) {
        size_t d = idx % hd;
        size_t h = (idx / hd) % nh;
        size_t qi = idx / (hd * nh);
        size_t kvh = h / (nh / nkvh);
        size_t max_k = min(kvlen, qi + (kvlen - qlen) + 1);
        float max_logit = -INFINITY;
        for (size_t ki = 0; ki < max_k; ++ki) {
            float dot = 0.0f;
            for (size_t inner = 0; inner < hd; ++inner) {
                dot += load_value(q, (qi * nh + h) * hd + inner, kind) * load_value(k, (ki * nkvh + kvh) * hd + inner, kind);
            }
            max_logit = fmaxf(max_logit, dot * scale);
        }
        float denom = 0.0f;
        float acc = 0.0f;
        for (size_t ki = 0; ki < max_k; ++ki) {
            float dot = 0.0f;
            for (size_t inner = 0; inner < hd; ++inner) {
                dot += load_value(q, (qi * nh + h) * hd + inner, kind) * load_value(k, (ki * nkvh + kvh) * hd + inner, kind);
            }
            float p = expf(dot * scale - max_logit);
            denom += p;
            acc += p * load_value(v, (ki * nkvh + kvh) * hd + d, kind);
        }
        store_value(out, idx, kind, acc / denom);
    }
}
} // namespace

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    int kind = dtype_kind(attn_val->dtype());
    size_t n = attn_val->numel();
    attention_kernel<<<grid_1d(n, 128), 128>>>(attn_val->data(), q->data(), k->data(), v->data(), q->shape()[0], k->shape()[0], q->shape()[1], k->shape()[1], q->shape()[2], scale, kind);
    check_cuda(cudaGetLastError(), "attention_kernel");
    check_cuda(cudaDeviceSynchronize(), "attention synchronize");
}
} // namespace llaisys::ops::nvidia
