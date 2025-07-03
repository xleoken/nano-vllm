import numpy as np

class KVCache:
    def __init__(self, config: 'ModelConfig', batch_size: int):
        self.config = config
        self.batch_size = batch_size
        self.max_seq_len = config.max_seq_len
        
        # 初始化KV缓存为零数组 [batch, num_heads, max_seq_len, head_dim]
        self.k_cache = np.zeros(
            (batch_size, config.num_heads, config.max_seq_len, config.head_dim),
            dtype=np.float32
        )
        self.v_cache = np.zeros_like(self.k_cache)
        
        # 记录每个序列的当前长度
        self.seq_lens = np.zeros(batch_size, dtype=np.int32)

    def update(self, layer_idx: int, batch_indices: list, k: np.ndarray, v: np.ndarray):
        """更新指定层和批次索引的KV缓存"""
        for i, batch_idx in enumerate(batch_indices):
            seq_len = self.seq_lens[batch_idx]
            self.k_cache[batch_idx, :, seq_len:seq_len+1] = k[i:i+1]
            self.v_cache[batch_idx, :, seq_len:seq_len+1] = v[i:i+1]
        self.seq_lens[batch_indices] += 1

    def get(self, layer_idx: int, batch_indices: list, max_context_len: int = None):
        """获取指定层和批次索引的KV缓存"""
        if max_context_len is None:
            max_context_len = self.max_seq_len
        k = np.stack([self.k_cache[i, :, :self.seq_lens[i]] for i in batch_indices], axis=0)
        v = np.stack([self.v_cache[i, :, :self.seq_lens[i]] for i in batch_indices], axis=0)
        return k[:, :, -max_context_len:], v[:, :, -max_context_len:]

    def reset(self, batch_indices: list = None):
        """重置指定批次索引的KV缓存"""
        if batch_indices is None:
            self.seq_lens.fill(0)
        else:
            self.seq_lens[batch_indices] = 0    