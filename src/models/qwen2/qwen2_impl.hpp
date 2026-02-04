#pragma once
#include "llaisys/models/qwen2.h"
#include "../../tensor/tensor.hpp"
#include <vector>
#include <memory>

namespace llaisys::models {

class Qwen2Impl {
public:
    Qwen2Impl(const LlaisysQwen2Meta* meta, llaisysDeviceType_t device, int device_id);
    ~Qwen2Impl();

    LlaisysQwen2Weights* weights() { return &_weights; }
    int64_t infer(int64_t* token_ids, size_t ntoken);

private:
    LlaisysQwen2Meta _meta;
    LlaisysQwen2Weights _weights;
    llaisysDeviceType_t _device_type;
    int _device_id;
    int64_t _current_pos;

    // KV Cache
    std::vector<tensor_t> _k_cache;
    std::vector<tensor_t> _v_cache;

    void allocate_weights();
    void free_weights();
    void init_cache();
};

} // namespace llaisys::models