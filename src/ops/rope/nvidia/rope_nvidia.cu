#include "rope_nvidia.hpp"
#include "../../nvidia_common.cuh"

namespace llaisys::ops::nvidia {
namespace {
__global__ void rope_kernel(void *out, const void *in, const int64_t *pos, size_t seq, size_t heads, size_t head_dim, float theta, int kind) {
    size_t half = head_dim / 2;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = seq * heads * half;
    if (idx < n) {
        size_t j = idx % half;
        size_t h = (idx / half) % heads;
        size_t s = idx / (half * heads);
        size_t base = (s * heads + h) * head_dim;
        float angle = static_cast<float>(pos[s]) / powf(theta, static_cast<float>(2 * j) / static_cast<float>(head_dim));
        float sn = sinf(angle);
        float cs = cosf(angle);
        float a = load_value(in, base + j, kind);
        float b = load_value(in, base + half + j, kind);
        store_value(out, base + j, kind, a * cs - b * sn);
        store_value(out, base + half + j, kind, b * cs + a * sn);
    }
}
} // namespace

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    int kind = dtype_kind(out->dtype());
    size_t n = in->shape()[0] * in->shape()[1] * (in->shape()[2] / 2);
    rope_kernel<<<grid_1d(n), 256>>>(out->data(), in->data(), reinterpret_cast<const int64_t *>(pos_ids->data()), in->shape()[0], in->shape()[1], in->shape()[2], theta, kind);
    check_cuda(cudaGetLastError(), "rope_kernel");
    check_cuda(cudaDeviceSynchronize(), "rope synchronize");
}
} // namespace llaisys::ops::nvidia
