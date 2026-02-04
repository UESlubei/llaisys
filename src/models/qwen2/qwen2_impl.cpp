#include "qwen2_impl.hpp"

// [新增] 引入内部头文件以访问 LlaisysTensor 的定义
#include "../../llaisys/llaisys_tensor.hpp" 

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils.hpp"
#include <cmath>

namespace llaisys::models {

// 从 C 句柄中提取 C++ shared_ptr
static tensor_t unwrap(llaisysTensor_t t) {
    if (!t) return nullptr;
    return t->tensor; 
}

Qwen2Impl::Qwen2Impl(const LlaisysQwen2Meta* meta, llaisysDeviceType_t device, int device_id)
    : _meta(*meta), _device_type(device), _device_id(device_id), _current_pos(0) {
    allocate_weights();
    init_cache();
}

Qwen2Impl::~Qwen2Impl() {
    free_weights();
}

void Qwen2Impl::allocate_weights() {
    size_t n = _meta.nlayer;
    _weights.attn_norm_w = new llaisysTensor_t[n]();
    _weights.attn_q_w = new llaisysTensor_t[n]();
    _weights.attn_q_b = new llaisysTensor_t[n]();
    _weights.attn_k_w = new llaisysTensor_t[n]();
    _weights.attn_k_b = new llaisysTensor_t[n]();
    _weights.attn_v_w = new llaisysTensor_t[n]();
    _weights.attn_v_b = new llaisysTensor_t[n]();
    _weights.attn_o_w = new llaisysTensor_t[n]();
    _weights.mlp_norm_w = new llaisysTensor_t[n]();
    _weights.mlp_gate_w = new llaisysTensor_t[n]();
    _weights.mlp_up_w = new llaisysTensor_t[n]();
    _weights.mlp_down_w = new llaisysTensor_t[n]();
}

void Qwen2Impl::free_weights() {
    // 注意：这里只释放数组本身。
    delete[] _weights.attn_norm_w;
    delete[] _weights.attn_q_w;
    delete[] _weights.attn_q_b;
    delete[] _weights.attn_k_w;
    delete[] _weights.attn_k_b;
    delete[] _weights.attn_v_w;
    delete[] _weights.attn_v_b;
    delete[] _weights.attn_o_w;
    delete[] _weights.mlp_norm_w;
    delete[] _weights.mlp_gate_w;
    delete[] _weights.mlp_up_w;
    delete[] _weights.mlp_down_w;
}

void Qwen2Impl::init_cache() {
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        _k_cache.push_back(Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id));
        _v_cache.push_back(Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id));
    }
}

int64_t Qwen2Impl::infer(int64_t* token_ids, size_t ntoken) {
    if (ntoken == 0) return _meta.end_token;

    // 1. Prepare Inputs
    auto input_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    input_ids->load(token_ids);

    std::vector<int64_t> pos_vec(ntoken);
    for (size_t i = 0; i < ntoken; ++i) pos_vec[i] = _current_pos + i;
    auto pos_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    pos_ids->load(pos_vec.data());

    // 2. Embedding
    auto hidden_states = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device_type, _device_id);
    
    ops::embedding(hidden_states, input_ids, unwrap(_weights.in_embed));

    // 3. Layers
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        auto residual = hidden_states;

        // --- Attention ---
        auto norm_out = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device_type, _device_id);
        
        ops::rms_norm(norm_out, hidden_states, unwrap(_weights.attn_norm_w[i]), _meta.epsilon);

        auto q = Tensor::create({ntoken, _meta.nh * _meta.dh}, _meta.dtype, _device_type, _device_id);
       
        ops::linear(q, norm_out, unwrap(_weights.attn_q_w[i]), unwrap(_weights.attn_q_b[i]));
        auto q_view = q->view({ntoken, _meta.nh, _meta.dh});

        auto k = Tensor::create({ntoken, _meta.nkvh * _meta.dh}, _meta.dtype, _device_type, _device_id);
        
        ops::linear(k, norm_out, unwrap(_weights.attn_k_w[i]), unwrap(_weights.attn_k_b[i]));
        auto k_view = k->view({ntoken, _meta.nkvh, _meta.dh});

        auto v = Tensor::create({ntoken, _meta.nkvh * _meta.dh}, _meta.dtype, _device_type, _device_id);
       
        ops::linear(v, norm_out, unwrap(_weights.attn_v_w[i]), unwrap(_weights.attn_v_b[i]));
        auto v_view = v->view({ntoken, _meta.nkvh, _meta.dh});

        ops::rope(q_view, q_view, pos_ids, _meta.theta);
        ops::rope(k_view, k_view, pos_ids, _meta.theta);

        // Update KV Cache
        size_t bytes_to_copy = ntoken * _meta.nkvh * _meta.dh * _k_cache[i]->elementSize();
        auto k_slot = _k_cache[i]->slice(0, _current_pos, _current_pos + ntoken);
        auto v_slot = _v_cache[i]->slice(0, _current_pos, _current_pos + ntoken);
        
        core::context().runtime().api()->memcpy_sync(k_slot->data(), k_view->data(), bytes_to_copy, 
            (_device_type == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2D);
        core::context().runtime().api()->memcpy_sync(v_slot->data(), v_view->data(), bytes_to_copy, 
             (_device_type == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2D);

        auto k_full = _k_cache[i]->slice(0, 0, _current_pos + ntoken);
        auto v_full = _v_cache[i]->slice(0, 0, _current_pos + ntoken);

        auto attn_out_view = Tensor::create({ntoken, _meta.nh, _meta.dh}, _meta.dtype, _device_type, _device_id);
        float scale = 1.0f / std::sqrt(static_cast<float>(_meta.dh));
        ops::self_attention(attn_out_view, q_view, k_full, v_full, scale);
        
        auto attn_out = attn_out_view->view({ntoken, _meta.nh * _meta.dh});
        auto attn_proj = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device_type, _device_id);
        
        ops::linear(attn_proj, attn_out, unwrap(_weights.attn_o_w[i]), nullptr);
        ops::add(hidden_states, residual, attn_proj);

        // --- MLP ---
        residual = hidden_states;
        
        ops::rms_norm(norm_out, hidden_states, unwrap(_weights.mlp_norm_w[i]), _meta.epsilon);
        
        auto gate = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device_type, _device_id);
        auto up = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device_type, _device_id);
        
        ops::linear(gate, norm_out, unwrap(_weights.mlp_gate_w[i]), nullptr);
        ops::linear(up, norm_out, unwrap(_weights.mlp_up_w[i]), nullptr);
        ops::swiglu(gate, gate, up);
        
        auto mlp_out = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device_type, _device_id);
        
        ops::linear(mlp_out, gate, unwrap(_weights.mlp_down_w[i]), nullptr);
        ops::add(hidden_states, residual, mlp_out);
    }

    // 4. Output
    auto final_norm = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device_type, _device_id);
    
    ops::rms_norm(final_norm, hidden_states, unwrap(_weights.out_norm_w), _meta.epsilon);

    auto last_token = final_norm->slice(0, ntoken - 1, ntoken);
    auto logits = Tensor::create({1, _meta.voc}, _meta.dtype, _device_type, _device_id);
    
    ops::linear(logits, last_token, unwrap(_weights.out_embed), nullptr);

    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    auto max_val = Tensor::create({1}, _meta.dtype, _device_type, _device_id);
    ops::argmax(max_idx, max_val, logits);

    _current_pos += ntoken;
    
    int64_t next_id;
    core::context().runtime().api()->memcpy_sync(&next_id, max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
    return next_id;
}

} // namespace llaisys::models