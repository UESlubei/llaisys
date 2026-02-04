#include "op.hpp"
#include "../../utils.hpp"
#include <cstring>

namespace llaisys::ops {

namespace cpu {
void embedding_impl(tensor_t out, tensor_t index, tensor_t weight) {
    size_t num_rows = index->numel();  //词数
    size_t hidden_dim = weight->shape()[1];  //维度
    size_t vocab_size = weight->shape()[0];
    size_t element_size = weight->elementSize();

    const int64_t* idx_ptr = reinterpret_cast<const int64_t*>(index->data()); // 获取输入
    const std::byte* weight_ptr = weight->data();
    std::byte* out_ptr = out->data();

    for (size_t i = 0; i < num_rows; ++i) {
        int64_t idx = idx_ptr[i];
        ASSERT(idx >= 0 && idx < (int64_t)vocab_size, "Embedding index out of bounds");
        //获取词表中位置
        const std::byte* src = weight_ptr + idx * hidden_dim * element_size;
        std::byte* dst = out_ptr + i * hidden_dim * element_size;
        std::memcpy(dst, src, hidden_dim * element_size);
    }
}
} // namespace cpu

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // Basic checks
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Index must be int64");
    ASSERT(out->dtype() == weight->dtype(), "Output and weight dtype mismatch");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Tensors must be contiguous");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding_impl(out, index, weight);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    
}
} // namespace llaisys::ops