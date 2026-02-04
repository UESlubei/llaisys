#include "op.hpp"
#include "../../utils.hpp"

namespace llaisys::ops {

namespace cpu {
template <typename T>
void linear_impl(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    size_t M = in->shape()[0];      // Batch size
    size_t K = in->shape()[1];      // In features
    size_t N = weight->shape()[0];  // Out features (weight is [N, K])
    
    // Pointers
    const T* in_ptr = reinterpret_cast<const T*>(in->data());
    const T* w_ptr = reinterpret_cast<const T*>(weight->data());
    T* out_ptr = reinterpret_cast<T*>(out->data());
    const T* bias_ptr = bias ? reinterpret_cast<const T*>(bias->data()) : nullptr;

    // Naive MatMul: out[m, n] = sum_k (in[m, k] * w[n, k]) + bias[n]
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float sum = bias_ptr ? utils::cast<float>(bias_ptr[n]) : 0.0f;
            
            for (size_t k = 0; k < K; ++k) {
                float x_val = utils::cast<float>(in_ptr[m * K + k]);
                float w_val = utils::cast<float>(w_ptr[n * K + k]); // Weight is [N, K], row-major
                sum += x_val * w_val;
            }
            out_ptr[m * N + n] = utils::cast<T>(sum);
        }
    }
}
} // namespace cpu

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "Linear: inputs must be contiguous");
    
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (in->dtype()) {
            case LLAISYS_DTYPE_F32: return cpu::linear_impl<float>(out, in, weight, bias);
            case LLAISYS_DTYPE_F16: return cpu::linear_impl<fp16_t>(out, in, weight, bias);
            case LLAISYS_DTYPE_BF16: return cpu::linear_impl<bf16_t>(out, in, weight, bias);
            default: EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
        }
    }
    
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    
}
} // namespace llaisys::ops