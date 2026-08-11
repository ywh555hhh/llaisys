#include "embedding_nvidia.hpp"
#include "../../nvidia_common.cuh"

namespace llaisys::ops::nvidia {
namespace {
__global__ void embedding_kernel(void *out, const int64_t *idx, const void *weight, size_t rows, size_t dim, int kind) {
    size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = rows * dim;
    if (linear < n) {
        size_t row = linear / dim;
        size_t col = linear % dim;
        size_t src = static_cast<size_t>(idx[row]) * dim + col;
        store_value(out, linear, kind, load_value(weight, src, kind));
    }
}
} // namespace

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    int kind = dtype_kind(out->dtype());
    size_t rows = index->shape()[0];
    size_t dim = weight->shape()[1];
    embedding_kernel<<<grid_1d(rows * dim), 256>>>(out->data(), reinterpret_cast<const int64_t *>(index->data()), weight->data(), rows, dim, kind);
    check_cuda(cudaGetLastError(), "embedding_kernel");
    check_cuda(cudaDeviceSynchronize(), "embedding synchronize");
}
} // namespace llaisys::ops::nvidia
