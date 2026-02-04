#include "llaisys/models/qwen2.h"
#include "../models/qwen2/qwen2_impl.hpp"

// 定义 C api
// 在这里，我们将它定义为持有 Qwen2Impl 指针的包装器
struct LlaisysQwen2Model {
    llaisys::models::Qwen2Impl* impl;
};

extern "C" {

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    auto wrapper = new LlaisysQwen2Model();
    int dev_id = (ndevice > 0 && device_ids != nullptr) ? device_ids[0] : 0;
    wrapper->impl = new llaisys::models::Qwen2Impl(meta, device, dev_id);
    return wrapper;
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (model) {
        delete model->impl;
        delete model;
    }
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    if (!model || !model->impl) return nullptr;
    return model->impl->weights();
}

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    if (!model || !model->impl) return -1; // Error code or end token
    return model->impl->infer(token_ids, ntoken);
}

} // extern "C"