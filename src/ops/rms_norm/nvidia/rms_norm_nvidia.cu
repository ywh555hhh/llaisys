#include "rms_norm_nvidia.hpp"
#include "../../nvidia_common.cuh"

namespace llaisys::ops::nvidia {
namespace {
__global__ void rms_norm_kernel(void *out, const void *in, const void *weight, size_t rows, size_t cols, float eps, int kind) {
    size_t row = blockIdx.x;
    if (row < rows) {
        float sum_sq = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            float v = load_value(in, row * cols + col, kind);
            sum_sq += v * v;
        }
        float inv = rsqrtf(sum_sq / static_cast<float>(cols) + eps);
        for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
            float v = load_value(in, row * cols + col, kind) * inv * load_value(weight, col, kind);
            store_value(out, row * cols + col, kind, v);
        }
    }
}
} // namespace

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    int kind = dtype_kind(out->dtype());
    rms_norm_kernel<<<static_cast<unsigned int>(in->shape()[0]), 256>>>(out->data(), in->data(), weight->data(), in->shape()[0], in->shape()[1], eps, kind);
    check_cuda(cudaGetLastError(), "rms_norm_kernel");
    check_cuda(cudaDeviceSynchronize(), "rms_norm synchronize");
}
} // namespace llaisys::ops::nvidia
