#include "op.hpp"
#include "../../utils.hpp"
#include <cmath>

namespace llaisys::ops {

namespace cpu {
template <typename T>
void rms_norm_impl(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    size_t num_rows = in->shape()[0];
    size_t dim = in->shape()[1];

    const T* in_ptr = reinterpret_cast<const T*>(in->data());
    const T* w_ptr = reinterpret_cast<const T*>(weight->data());
    T* out_ptr = reinterpret_cast<T*>(out->data());

    for (size_t i = 0; i < num_rows; ++i) {
        // 1. Calculate sum of squares
        float sum_sq = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            float val = utils::cast<float>(in_ptr[i * dim + j]);
            sum_sq += val * val;
        }

        // 2. Calculate RMS
        float rms = std::sqrt(sum_sq / dim + eps);
        float inv_rms = 1.0f / rms;

        // 3. Normalize and scale
        for (size_t j = 0; j < dim; ++j) {
            float val = utils::cast<float>(in_ptr[i * dim + j]);
            float w_val = utils::cast<float>(w_ptr[j]);
            out_ptr[i * dim + j] = utils::cast<T>(val * inv_rms * w_val);
        }
        }
    }
} // namespace cpu

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "RMSNorm: inputs must be contiguous");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (in->dtype()) {
            case LLAISYS_DTYPE_F32: return cpu::rms_norm_impl<float>(out, in, weight, eps);
            case LLAISYS_DTYPE_F16: return cpu::rms_norm_impl<fp16_t>(out, in, weight, eps);
            case LLAISYS_DTYPE_BF16: return cpu::rms_norm_impl<bf16_t>(out, in, weight, eps);
            default: EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
        }
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    }
}