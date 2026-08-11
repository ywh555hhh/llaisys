#include "swiglu_nvidia.hpp"
#include "../../nvidia_common.cuh"

namespace llaisys::ops::nvidia {
namespace {
__global__ void swiglu_kernel(void *out, const void *gate, const void *up, size_t n, int kind) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = load_value(gate, i, kind);
        float u = load_value(up, i, kind);
        store_value(out, i, kind, u * g / (1.0f + expf(-g)));
    }
}
} // namespace

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    int kind = dtype_kind(out->dtype());
    swiglu_kernel<<<grid_1d(out->numel()), 256>>>(out->data(), gate->data(), up->data(), out->numel(), kind);
    check_cuda(cudaGetLastError(), "swiglu_kernel");
    check_cuda(cudaDeviceSynchronize(), "swiglu synchronize");
}
} // namespace llaisys::ops::nvidia
