import ctypes
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from .tensor import llaisysTensor_t
from . import LIB_LLAISYS

class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        # 模型的数据类型（如 FP32, BF16），必须使用 C 类型别名
        ("dtype", llaisysDataType_t), 
        ("nlayer", ctypes.c_size_t),    # 层数 (num_hidden_layers)
        ("hs", ctypes.c_size_t),        # 隐藏层维度 (hidden_size)
        ("nh", ctypes.c_size_t),        # 注意力头数 (num_attention_heads)
        ("nkvh", ctypes.c_size_t),      # KV Cache 头数 (num_key_value_heads)
        ("dh", ctypes.c_size_t),        # 单头维度 (head_dim)
        ("di", ctypes.c_size_t),        # 中间层维度 (intermediate_size)
        ("maxseq", ctypes.c_size_t),    # 最大序列长度
        ("voc", ctypes.c_size_t),       # 词表大小 (vocab_size)
        ("epsilon", ctypes.c_float),    # RMS Norm 的 epsilon
        ("theta", ctypes.c_float),      # RoPE 的 theta
        ("end_token", ctypes.c_int64),  # 结束符 ID
    ]

class LlaisysQwen2Weights(ctypes.Structure):
    _fields_ = [ 
        ("in_embed", llaisysTensor_t),  #权重
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_q_w", ctypes.POINTER(llaisysTensor_t)),  #attention
        ("attn_q_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_k_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_k_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_v_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_v_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_o_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_norm_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_gate_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_up_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_down_w", ctypes.POINTER(llaisysTensor_t)),
    ]

# Define Function Prototypes
LIB_LLAISYS.llaisysQwen2ModelCreate.argtypes = [ctypes.POINTER(LlaisysQwen2Meta), llaisysDeviceType_t, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
LIB_LLAISYS.llaisysQwen2ModelCreate.restype = ctypes.c_void_p

LIB_LLAISYS.llaisysQwen2ModelDestroy.argtypes = [ctypes.c_void_p]
LIB_LLAISYS.llaisysQwen2ModelDestroy.restype = None

LIB_LLAISYS.llaisysQwen2ModelWeights.argtypes = [ctypes.c_void_p]
LIB_LLAISYS.llaisysQwen2ModelWeights.restype = ctypes.POINTER(LlaisysQwen2Weights)

LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t]
LIB_LLAISYS.llaisysQwen2ModelInfer.restype = ctypes.c_int64