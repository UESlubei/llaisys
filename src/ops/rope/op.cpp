#include "op.hpp"
#include "../../utils.hpp"
#include <cmath>

namespace llaisys::ops {

namespace cpu {
template <typename T>
void rope_impl(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // Shapes: [seqlen, nhead, d]
    size_t seqlen = in->shape()[0];
    size_t nhead = in->shape()[1];
    size_t d = in->shape()[2];
    size_t half_d = d / 2;

    const T* in_ptr = reinterpret_cast<const T*>(in->data());
    const int64_t* pos_ptr = reinterpret_cast<const int64_t*>(pos_ids->data());
    T* out_ptr = reinterpret_cast<T*>(out->data());

    for (size_t s = 0; s < seqlen; ++s) {
        int64_t pos = pos_ptr[s];
        for (size_t h = 0; h < nhead; ++h) {
            for (size_t j = 0; j < half_d; ++j) {
              
                size_t idx_a = (s * nhead * d) + (h * d) + j;
                size_t idx_b = (s * nhead * d) + (h * d) + (j + half_d);

                float a = utils::cast<float>(in_ptr[idx_a]);
                float b = utils::cast<float>(in_ptr[idx_b]);
                
                // a' = a*cos - b*sin
                // b' = b*cos + a*sin
                float freq = pos / std::pow(theta, 2.0f * j / d);
                float cos_val = std::cos(freq);
                float sin_val = std::sin(freq);
                float a_out = a * cos_val - b * sin_val;
                float b_out = b * cos_val + a * sin_val;

                out_ptr[idx_a] = utils::cast<T>(a_out);
                out_ptr[idx_b] = utils::cast<T>(b_out);
            }
        }
    }
}
} // namespace cpu

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    ASSERT(out->isContiguous() && in->isContiguous(), "RoPE: inputs must be contiguous");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (in->dtype()) {
            case LLAISYS_DTYPE_F32: return cpu::rope_impl<float>(out, in, pos_ids, theta);
            case LLAISYS_DTYPE_F16: return cpu::rope_impl<fp16_t>(out, in, pos_ids, theta);
            case LLAISYS_DTYPE_BF16: return cpu::rope_impl<bf16_t>(out, in, pos_ids, theta);
            default: EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
        }
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    
}
} // namespace llaisys::ops