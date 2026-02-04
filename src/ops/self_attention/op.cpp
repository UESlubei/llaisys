#include "op.hpp"
#include "../../utils.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

namespace llaisys::ops {

namespace cpu {
template <typename T>
void self_attention_impl(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {

    // q: [seqlen, nhead, d]
    // k: [kvlen, nkvhead, d]
    // v: [kvlen, nkvhead, dv]
    // attn_val: [seqlen, nhead, dv]
    
    size_t seqlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t d = q->shape()[2];
    
    size_t kvlen = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    size_t dv = v->shape()[2];

    size_t group_size = nhead / nkvhead; // For GQA/MQA

    const T* q_ptr = reinterpret_cast<const T*>(q->data());
    const T* k_ptr = reinterpret_cast<const T*>(k->data());
    const T* v_ptr = reinterpret_cast<const T*>(v->data());
    T* out_ptr = reinterpret_cast<T*>(attn_val->data());

    // Buffer for attention scores: [kvlen]
    std::vector<float> scores(kvlen);

    for (size_t s = 0; s < seqlen; ++s) {
        for (size_t h = 0; h < nhead; ++h) {
            size_t kv_h = h / group_size; 

            // 1. 计算 Q * K^T -> Scores
           
            for (size_t l = 0; l < kvlen; ++l) {
                
                size_t offset = kvlen - seqlen;
                if (l > s + offset) {
                    scores[l] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                float dot = 0.0f;
                for (size_t i = 0; i < d; ++i) {
                    float q_val = utils::cast<float>(q_ptr[(s * nhead * d) + (h * d) + i]);
                    float k_val = utils::cast<float>(k_ptr[(l * nkvhead * d) + (kv_h * d) + i]);
                    dot += q_val * k_val;
                }
                scores[l] = dot * scale;
            }

            // 2. Softmax
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t l = 0; l < kvlen; ++l) {
                if (scores[l] > max_score) max_score = scores[l];
            }
            
            float sum_exp = 0.0f;
            for (size_t l = 0; l < kvlen; ++l) {
                // masked values
                if (scores[l] == -std::numeric_limits<float>::infinity()) {
                    scores[l] = 0.0f; 
                } else {
                    scores[l] = std::exp(scores[l] - max_score);
                    sum_exp += scores[l];
                }
            }
            
            for (size_t l = 0; l < kvlen; ++l) {
                scores[l] /= sum_exp;
            }

            // 3. Scores * V -> Output
            for (size_t i = 0; i < dv; ++i) {
                float acc = 0.0f;
                for (size_t l = 0; l < kvlen; ++l) {
                    float v_val = utils::cast<float>(v_ptr[(l * nkvhead * dv) + (kv_h * dv) + i]);
                    acc += scores[l] * v_val;
                }
                out_ptr[(s * nhead * dv) + (h * dv) + i] = utils::cast<T>(acc);
            }
        }
    }
}
} // namespace cpu

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    ASSERT(q->isContiguous() && k->isContiguous() && v->isContiguous(), "Attention: inputs must be contiguous");

    if (q->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (q->dtype()) {
            case LLAISYS_DTYPE_F32: return cpu::self_attention_impl<float>(attn_val, q, k, v, scale);
            case LLAISYS_DTYPE_F16: return cpu::self_attention_impl<fp16_t>(attn_val, q, k, v, scale);
            case LLAISYS_DTYPE_BF16: return cpu::self_attention_impl<bf16_t>(attn_val, q, k, v, scale);
            default: EXCEPTION_UNSUPPORTED_DATATYPE(q->dtype());
        }
    }

    llaisys::core::context().setDevice(q->deviceType(), q->deviceId());
}
} // namespace llaisys::ops