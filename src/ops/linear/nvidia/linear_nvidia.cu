#include "linear_nvidia.hpp"
#include "../../nvidia_common.cuh"

namespace llaisys::ops::nvidia {
namespace {
__global__ void linear_kernel(void *out, const void *in, const void *weight, const void *bias, size_t batch, size_t in_features, size_t out_features, int kind) {
    size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = batch * out_features;
    if (linear < n) {
        size_t row = linear / out_features;
        size_t col = linear % out_features;
        float acc = bias ? load_value(bias, col, kind) : 0.0f;
        for (size_t k = 0; k < in_features; ++k) {
            acc += load_value(in, row * in_features + k, kind) * load_value(weight, col * in_features + k, kind);
        }
        store_value(out, linear, kind, acc);
    }
}
} // namespace

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    int kind = dtype_kind(out->dtype());
    size_t batch = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];
    linear_kernel<<<grid_1d(batch * out_features, 128), 128>>>(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, batch, in_features, out_features, kind);
    check_cuda(cudaGetLastError(), "linear_kernel");
    check_cuda(cudaDeviceSynchronize(), "linear synchronize");
}
} // namespace llaisys::ops::nvidia
