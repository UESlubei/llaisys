#include "op.hpp"
#include "../../utils.hpp"
#include <cmath>

namespace llaisys::ops {

namespace cpu {
template <typename T>
void swiglu_impl(tensor_t out, tensor_t gate, tensor_t up) {
    size_t numel = out->numel();
    const T* gate_ptr = reinterpret_cast<const T*>(gate->data());
    const T* up_ptr = reinterpret_cast<const T*>(up->data());
    T* out_ptr = reinterpret_cast<T*>(out->data());

    for (size_t i = 0; i < numel; ++i) {
        float g = utils::cast<float>(gate_ptr[i]);
        float u = utils::cast<float>(up_ptr[i]);
        
        // Swish / SiLU: x * sigmoid(x)
        float swish = g / (1.0f + std::exp(-g));
        
        // SwiGLU: swish(gate) * up
        out_ptr[i] = utils::cast<T>(swish * u);
    }
}
} // namespace cpu

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "SwiGLU: inputs must be contiguous");
    
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
            case LLAISYS_DTYPE_F32: return cpu::swiglu_impl<float>(out, gate, up);
            case LLAISYS_DTYPE_F16: return cpu::swiglu_impl<fp16_t>(out, gate, up);
            case LLAISYS_DTYPE_BF16: return cpu::swiglu_impl<bf16_t>(out, gate, up);
            default: EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    
}
} 