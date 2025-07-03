import torch
from torch import nn

from nanovllm.utils.context import get_context


def scaled_dot_product_attention(q, k, v, scale, causal=True):
    attn = (q @ k.transpose(-2, -1)) * scale
    if causal:
        # 应用因果掩码
        seq_len = q.size(-2)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1)
        attn = attn.masked_fill(mask.bool(), float('-inf'))
    attn = attn.softmax(dim=-1)
    return attn @ v


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                  slot_mapping: torch.Tensor):
    """
    将 key/value 存入缓存中对应的位置。
    :param key: 当前生成的 key，shape [N, num_kv_heads, head_dim]
    :param value: 当前生成的 value，shape [N, num_kv_heads, head_dim]
       :param k_cache: 缓存的 key，shape [total_slots, num_kv_heads, head_dim]
    :param v_cache: 缓存的 value，shape [total_slots, num_kv_heads, head_dim]
    :param slot_mapping: 槽位映射，shape [N]
    """
    for i in range(key.size(0)):
        slot = slot_mapping[i]
        k_cache[slot] = key[i]
        v_cache[slot] = value[i]


class Attention(nn.Module):

    def __init__(
            self,
            num_heads,
            head_dim,
            scale,
            num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:
                # 使用 prefix cache 中的 key/value
                k, v = k_cache, v_cache

            # 使用自定义注意力函数替代 flash_attn_varlen_func
            o = scaled_dot_product_attention(q, k, v, scale=self.scale, causal=True)
        else:
            # 解码阶段，使用缓存中的 key/value
            batch_size = q.size(0)
            seq_len = k_cache.size(0) if k_cache.size(0) > k.size(0) else k.size(0)
            k_all = k_cache.unsqueeze(0).expand(batch_size, -1, -1) if k_cache.numel() else k.unsqueeze(0)
            v_all = v_cache.unsqueeze(0).expand(batch_size, -1, -1) if v_cache.numel() else v.unsqueeze(0)

            o = scaled_dot_product_attention(q.unsqueeze(1), k_all, v_all, scale=self.scale, causal=True)

        o = o.view(-1, self.num_heads * self.head_dim)
        return o
