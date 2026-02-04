#include "op.hpp"
#include "../../utils.hpp"
#include <limits>

namespace llaisys::ops {

namespace cpu {
template <typename T>
void argmax_impl(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    size_t numel = vals->numel();
    const T* vals_ptr = reinterpret_cast<const T*>(vals->data());
    int64_t* idx_ptr = reinterpret_cast<int64_t*>(max_idx->data());
    T* val_ptr = reinterpret_cast<T*>(max_val->data());

    float max_v = -std::numeric_limits<float>::infinity();
    int64_t best_idx = 0;
    // 循环获得最大值
    for (size_t i = 0; i < numel; ++i) {
        float curr = utils::cast<float>(vals_ptr[i]);
        if (curr > max_v) {
            max_v = curr;
            best_idx = i;
        }
    }
    
    *idx_ptr = best_idx;
    *val_ptr = utils::cast<T>(max_v);
}
} 
// differnt device and type
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    ASSERT(vals->isContiguous() && max_idx->isContiguous() && max_val->isContiguous(), "Argmax: tensors must be contiguous");
    
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (vals->dtype()) {
            case LLAISYS_DTYPE_F32: return cpu::argmax_impl<float>(max_idx, max_val, vals);
            case LLAISYS_DTYPE_F16: return cpu::argmax_impl<fp16_t>(max_idx, max_val, vals);
            case LLAISYS_DTYPE_BF16: return cpu::argmax_impl<bf16_t>(max_idx, max_val, vals);
            default: EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
        }
    }
    
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops