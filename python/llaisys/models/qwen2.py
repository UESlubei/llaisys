from typing import Sequence
import json
import ctypes
import struct
import os
from pathlib import Path

from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from ..libllaisys.models import LlaisysQwen2Meta, LlaisysQwen2Weights
from ..tensor import Tensor

# --- 加载safetensors张量 ---
class SafeTensorsReader:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        # 1. Read header size
        header_size_bytes = self.f.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"Invalid safetensors file: {path}")
        self.header_size = struct.unpack("<Q", header_size_bytes)[0]
        
        # 2. Read header (JSON)
        header_bytes = self.f.read(self.header_size)
        self.header = json.loads(header_bytes)
        
        # 3. Calculate data start offset
        self.base_offset = 8 + self.header_size
        
    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]
        
    def get_tensor_info(self, name):
        return self.header.get(name)

    def read_tensor_bytes(self, name):
        info = self.header[name]
        start, end = info["data_offsets"]
        length = end - start
        
        self.f.seek(self.base_offset + start)
        data = self.f.read(length)
        return data, info["shape"]

    def close(self):
        self.f.close()
        
    def __del__(self):
        self.close()

# --- Main Model Class ---
class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        
        # 1. Load Config
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
        
        
        self.meta = LlaisysQwen2Meta()
        
        # Set dtype to BF16 (19) as per test requirement
        self.meta.dtype = DataType.BF16 
        
        self.meta.nlayer = config.get("num_hidden_layers", 28)
        self.meta.hs = config.get("hidden_size", 1536)
        self.meta.nh = config.get("num_attention_heads", 12)
        self.meta.nkvh = config.get("num_key_value_heads", 2)
        self.meta.dh = self.meta.hs // self.meta.nh
        self.meta.di = config.get("intermediate_size", 8960)
        self.meta.maxseq = 2048 
        self.meta.voc = config.get("vocab_size", 151936)
        self.meta.epsilon = config.get("rms_norm_eps", 1e-6)
        self.meta.theta = config.get("rope_theta", 10000.0)
        self.meta.end_token = 151643 

        # 3. Create Model
        self.device = device
        self.model_handle = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta), device, None, 0
        )
        
        # 4. Load Weights
        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model_handle)
        weights = weights_ptr.contents

        
        # Map: tensor_name -> (SafeTensorsReader, info)
        self.readers = [] 
        self.tensor_map = {}
        
        for file in sorted(model_path.glob("*.safetensors")):
            reader = SafeTensorsReader(file)
            self.readers.append(reader)
            for key in reader.keys():
                self.tensor_map[key] = reader

        self.keepalive = []

        def load_weight(st_name):
            if st_name not in self.tensor_map: 
                return None
            
            reader = self.tensor_map[st_name]
            raw_bytes, shape = reader.read_tensor_bytes(st_name)
            
            
            t = Tensor(shape, self.meta.dtype, device)
            t.load(raw_bytes)
            
            # Keep python object alive to prevent GC destroying the handle
            self.keepalive.append(t)
            return t._tensor  

        # Load Global Weights
        weights.in_embed = load_weight("model.embed_tokens.weight")
        weights.out_norm_w = load_weight("model.norm.weight")
        weights.out_embed = load_weight("lm_head.weight")

        # Load Layer Weights
        for i in range(self.meta.nlayer):
            weights.attn_norm_w[i] = load_weight(f"model.layers.{i}.input_layernorm.weight")
            weights.mlp_norm_w[i] = load_weight(f"model.layers.{i}.post_attention_layernorm.weight")
            
            weights.attn_q_w[i] = load_weight(f"model.layers.{i}.self_attn.q_proj.weight")
            weights.attn_k_w[i] = load_weight(f"model.layers.{i}.self_attn.k_proj.weight")
            weights.attn_v_w[i] = load_weight(f"model.layers.{i}.self_attn.v_proj.weight")
            weights.attn_o_w[i] = load_weight(f"model.layers.{i}.self_attn.o_proj.weight")
            
            weights.attn_q_b[i] = load_weight(f"model.layers.{i}.self_attn.q_proj.bias")
            weights.attn_k_b[i] = load_weight(f"model.layers.{i}.self_attn.k_proj.bias")
            weights.attn_v_b[i] = load_weight(f"model.layers.{i}.self_attn.v_proj.bias")
            
            weights.mlp_gate_w[i] = load_weight(f"model.layers.{i}.mlp.gate_proj.weight")
            weights.mlp_up_w[i] = load_weight(f"model.layers.{i}.mlp.up_proj.weight")
            weights.mlp_down_w[i] = load_weight(f"model.layers.{i}.mlp.down_proj.weight")

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        tokens = list(inputs)
        
        # 1. Prefill
        input_len = len(tokens)
        c_tokens = (ctypes.c_int64 * input_len)(*tokens)
        
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self.model_handle, c_tokens, input_len)
        tokens.append(next_token)
        
        # 2. Decode Loop
        for _ in range(max_new_tokens - 1):
            if next_token == self.meta.end_token:
                break
            
            c_input = (ctypes.c_int64 * 1)(next_token)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self.model_handle, c_input, 1)
            tokens.append(next_token)
            
        return tokens

    def __del__(self):
        # Close all file readers
        if hasattr(self, "readers"):
            for reader in self.readers:
                reader.close()
        
        if hasattr(self, "model_handle"):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model_handle)