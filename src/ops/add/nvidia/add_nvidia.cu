#include "add_nvidia.hpp"
#include "../../nvidia_common.cuh"

namespace llaisys::ops::nvidia {
namespace {
__global__ void add_kernel(void *c, const void *a, const void *b, size_t n, int kind) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        store_value(c, i, kind, load_value(a, i, kind) + load_value(b, i, kind));
    }
}
} // namespace

void add(tensor_t c, tensor_t a, tensor_t b) {
    int kind = dtype_kind(c->dtype());
    add_kernel<<<grid_1d(c->numel()), 256>>>(c->data(), a->data(), b->data(), c->numel(), kind);
    check_cuda(cudaGetLastError(), "add_kernel");
    check_cuda(cudaDeviceSynchronize(), "add synchronize");
}
} // namespace llaisys::ops::nvidia
