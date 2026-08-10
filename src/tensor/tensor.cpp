#include "tensor.hpp"

#include "../utils.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = static_cast<ptrdiff_t>(stride);
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

static std::vector<ptrdiff_t> compact_strides(const std::vector<size_t> &shape) {
    std::vector<ptrdiff_t> strides(shape.size());
    ptrdiff_t stride = 1;
    for (size_t i = shape.size(); i > 0; --i) {
        strides[i - 1] = stride;
        stride *= static_cast<ptrdiff_t>(shape[i - 1]);
    }
    return strides;
}

bool Tensor::isContiguous() const {
    ptrdiff_t expected = 1;
    for (size_t i = ndim(); i > 0; --i) {
        const size_t dim = i - 1;
        if (_meta.shape[dim] == 0) {
            return true;
        }
        if (_meta.shape[dim] != 1 && _meta.strides[dim] != expected) {
            return false;
        }
        expected *= static_cast<ptrdiff_t>(_meta.shape[dim]);
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    CHECK_ARGUMENT(order.size() == ndim(), "Permute order rank must match tensor rank.");
    std::vector<bool> seen(ndim(), false);
    TensorMeta meta{_meta.dtype, std::vector<size_t>(ndim()), std::vector<ptrdiff_t>(ndim())};
    for (size_t i = 0; i < order.size(); ++i) {
        CHECK_ARGUMENT(order[i] < ndim(), "Permute dimension is out of range.");
        CHECK_ARGUMENT(!seen[order[i]], "Permute order contains duplicate dimensions.");
        seen[order[i]] = true;
        meta.shape[i] = _meta.shape[order[i]];
        meta.strides[i] = _meta.strides[order[i]];
    }
    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    const size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    CHECK_ARGUMENT(new_numel == numel(), "View shape must preserve the number of elements.");
    if (shape == _meta.shape) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }
    CHECK_ARGUMENT(isContiguous(), "View currently requires a contiguous tensor.");
    TensorMeta meta{_meta.dtype, shape, compact_strides(shape)};
    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    CHECK_ARGUMENT(dim < ndim(), "Slice dimension is out of range.");
    CHECK_ARGUMENT(start <= end && end <= _meta.shape[dim], "Invalid slice range.");
    TensorMeta meta = _meta;
    meta.shape[dim] = end - start;
    const size_t offset = _offset + start * static_cast<size_t>(_meta.strides[dim]) * elementSize();
    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, offset));
}

void Tensor::load(const void *src_) {
    core::context().setDevice(this->deviceType(), this->deviceId());
    const llaisysMemcpyKind_t kind = this->deviceType() == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D;
    core::context().runtime().api()->memcpy_sync(this->data(), src_, this->numel() * this->elementSize(), kind);
}

tensor_t Tensor::contiguous() const {
    if (isContiguous()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }
    CHECK_ARGUMENT(this->deviceType() == LLAISYS_DEVICE_CPU, "Contiguous copy is only implemented for CPU tensors.");
    auto out = Tensor::create(_meta.shape, _meta.dtype, this->deviceType(), this->deviceId());
    for (size_t linear = 0; linear < numel(); ++linear) {
        ptrdiff_t src_elem = 0;
        size_t tmp = linear;
        for (size_t dim = ndim(); dim > 0; --dim) {
            const size_t axis = dim - 1;
            const size_t coord = tmp % _meta.shape[axis];
            tmp /= _meta.shape[axis];
            src_elem += static_cast<ptrdiff_t>(coord) * _meta.strides[axis];
        }
        std::memcpy(out->data() + linear * elementSize(), this->data() + static_cast<size_t>(src_elem) * elementSize(), elementSize());
    }
    return out;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    if (isContiguous()) {
        return view(shape);
    }
    return contiguous()->view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    const int target_device = device < 0 ? this->deviceId() : device;
    auto out = Tensor::create(_meta.shape, _meta.dtype, device_type, target_device);
    CHECK_ARGUMENT(this->isContiguous(), "Device transfer requires a contiguous source tensor.");
    core::context().setDevice(device_type, target_device);
    llaisysMemcpyKind_t kind = LLAISYS_MEMCPY_D2D;
    if (this->deviceType() == LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_H2H;
    } else if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_H2D;
    } else if (device_type == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_D2H;
    }
    core::context().runtime().api()->memcpy_sync(out->data(), this->data(), this->numel() * this->elementSize(), kind);
    return out;
}

} // namespace llaisys
