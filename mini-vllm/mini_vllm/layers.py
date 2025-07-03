import numpy as np
from typing import Optional

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的softmax实现"""
    x = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)

def gelu(x: np.ndarray) -> np.ndarray:
    """GELU激活函数"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.weight = np.zeros((out_features, in_features), dtype=np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32) if bias else None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = x @ self.weight.T
        if self.bias is not None:
            out += self.bias
        return out

class RMSNorm:
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.weight = np.ones(hidden_size, dtype=np.float32)
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        norm = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return x / norm * self.weight

class Attention:
    def __init__(self, config: 'ModelConfig'):
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        
        # 注意力投影矩阵
        self.q_proj = Linear(config.hidden_size, config.hidden_size)
        self.k_proj = Linear(config.hidden_size, config.hidden_size)
        self.v_proj = Linear(config.hidden_size, config.hidden_size)
        self.o_proj = Linear(config.hidden_size, config.hidden_size)
        
        # 旋转位置编码
        self.max_seq_len = config.max_seq_len
        self.rotary_emb = self._create_rotary_embedding()

    def _create_rotary_embedding(self) -> np.ndarray:
        """创建旋转位置编码矩阵"""
        freq = 1.0 / (10000 ** (np.arange(0, self.head_dim, 2) / self.head_dim))
        t = np.arange(self.max_seq_len)
        freqs = np.outer(t, freq).astype(np.float32)
        
        # 创建旋转位置编码矩阵
        emb = np.zeros((self.max_seq_len, self.head_dim), dtype=np.float32)
        emb[:, 0::2] = np.cos(freqs)
        emb[:, 1::2] = np.sin(freqs)
        return emb

    def _apply_rotary_embedding(self, x: np.ndarray, seq_pos: int) -> np.ndarray:
        """应用旋转位置编码"""
        batch_size, seq_len, num_heads, head_dim = x.shape
        x = x.reshape(batch_size * seq_len, num_heads, head_dim)
        
        # 获取对应位置的旋转编码
        cos = self.rotary_emb[seq_pos:seq_pos+seq_len].reshape(seq_len, 1, head_dim)
        sin = self.rotary_emb[seq_pos:seq_pos+seq_len].reshape(seq_len, 1, head_dim)
        
        # 应用旋转编码
        x_rot = np.zeros_like(x)
        x_rot[:, :, 0::2] = x[:, :, 0::2] * cos[:, :, 0::2] - x[:, :, 1::2] * cos[:, :, 1::2]
        x_rot[:, :, 1::2] = x[:, :, 1::2] * sin[:, :, 0::2] + x[:, :, 0::2] * sin[:, :, 1::2]
        
        return x_rot.reshape(batch_size, seq_len, num_heads, head_dim)

    def __call__(self, x: np.ndarray, kv_cache: 'KVCache', layer_idx: int, batch_indices: list, seq_pos: int) -> np.ndarray:
        """实现自注意力机制，使用KV缓存优化解码过程"""
        batch_size, seq_len, hidden_size = x.shape
        
        # 线性投影
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 应用旋转位置编码
        q = self._apply_rotary_embedding(q, seq_pos)
        k = self._apply_rotary_embedding(k, seq_pos)
        
        # 更新KV缓存
        kv_cache.update(layer_idx, batch_indices, 
                        k.reshape(batch_size, self.num_heads, seq_len, self.head_dim),
                        v.reshape(batch_size, self.num_heads, seq_len, self.head_dim))
        
        # 获取完整的KV缓存
        cached_k, cached_v = kv_cache.get(layer_idx, batch_indices)
        cached_seq_len = cached_k.shape[2]
        
        # 计算注意力得分
        q = q.reshape(batch_size * seq_len, self.num_heads, self.head_dim)
        k = cached_k.reshape(batch_size * seq_len, self.num_heads, cached_seq_len, self.head_dim)
        
        # 计算注意力权重
        scores = np.matmul(q[:, :, np.newaxis, :], k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        scores = scores.reshape(batch_size, seq_len, self.num_heads, cached_seq_len)
        
        # 应用因果掩码（仅在训练或处理新序列时需要）
        if seq_len > 1:
            mask = np.triu(np.ones((seq_len, cached_seq_len)), k=1) * -1e9
            scores += mask[np.newaxis, :, np.newaxis, :]
        
        # 应用softmax获取注意力权重
        attn_weights = softmax(scores, axis=-1)
        
        # 计算上下文向量
        v = cached_v.reshape(batch_size * seq_len, self.num_heads, cached_seq_len, self.head_dim)
        context = np.matmul(attn_weights.reshape(batch_size * seq_len, self.num_heads, 1, cached_seq_len), v)
        context = context.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # 输出投影
        return self.o_proj(context)

class MLP:
    def __init__(self, config: 'ModelConfig'):
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.down_proj(gelu(self.gate_proj(x)) * self.up_proj(x))

class TransformerLayer:
    def __init__(self, config: 'ModelConfig'):
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

    def __call__(self, hidden_states: np.ndarray, kv_cache: 'KVCache', 
                 layer_idx: int, batch_indices: list, seq_pos: int) -> np.ndarray:
        # 自注意力块
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, kv_cache, layer_idx, batch_indices, seq_pos)
        hidden_states = residual + hidden_states
        
        # MLP块
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states    