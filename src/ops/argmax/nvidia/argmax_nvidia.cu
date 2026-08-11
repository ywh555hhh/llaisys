#include "argmax_nvidia.hpp"
#include "../../nvidia_common.cuh"

namespace llaisys::ops::nvidia {
namespace {
__global__ void argmax_kernel(int64_t *max_idx, void *max_val, const void *vals, size_t n, int kind) {
    int64_t best_idx = 0;
    float best_val = load_value(vals, 0, kind);
    for (size_t i = 1; i < n; ++i) {
        float v = load_value(vals, i, kind);
        if (v > best_val) {
            best_val = v;
            best_idx = static_cast<int64_t>(i);
        }
    }
    *max_idx = best_idx;
    store_value(max_val, 0, kind, best_val);
}
} // namespace

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    int kind = dtype_kind(vals->dtype());
    argmax_kernel<<<1, 1>>>(reinterpret_cast<int64_t *>(max_idx->data()), max_val->data(), vals->data(), vals->numel(), kind);
    check_cuda(cudaGetLastError(), "argmax_kernel");
    check_cuda(cudaDeviceSynchronize(), "argmax synchronize");
}
} // namespace llaisys::ops::nvidia
