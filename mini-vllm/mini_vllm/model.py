import os
import numpy as np
from typing import List, Dict, Any, Optional
from .layers import TransformerLayer
from .cache import KVCache
from .config import ModelConfig
from .tokenizer import Tokenizer
from .scheduler import Request, Scheduler

class Qwen3Model:
    def __init__(self, config: ModelConfig, model_path: str):
        self.config = config
        self.model_path = model_path
        self.tokenizer = Tokenizer(model_path)
        self.scheduler = Scheduler(max_batch_size=8)
        
        # 初始化模型参数
        self.embed_tokens = np.zeros((config.vocab_size, config.hidden_size), dtype=np.float32)
        self.layers = [TransformerLayer(config) for _ in range(config.num_layers)]
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 加载模型权重
        self._load_model_weights()
        
        # KV缓存
        self.kv_cache = None

    def _load_model_weights(self):
        """从ModelCode格式加载模型权重"""
        print(f"Loading model weights from {self.model_path}")
        # 实际实现中，这里应该从文件加载权重
        # 为简化示例，我们使用随机权重
        
    def _prepare_inputs(self, requests: List[Request]):
        """准备模型输入"""
        input_tokens = []
        batch_indices = []
        seq_positions = []
        
        for i, req in enumerate(requests):
            if len(req.output_tokens) == 0:
                # 处理prompt
                input_tokens.extend(req.prompt_tokens)
                batch_indices.extend([i] * len(req.prompt_tokens))
                seq_positions.extend(list(range(len(req.prompt_tokens))))
            else:
                # 处理生成的token
                input_tokens.append(req.output_tokens[-1])
                batch_indices.append(i)
                seq_positions.append(self.kv_cache.seq_lens[i])
                
        return np.array(input_tokens), batch_indices, seq_positions

    def _forward(self, input_ids: np.ndarray, batch_indices: list, seq_positions: list):
        """模型前向传播"""
        batch_size = max(batch_indices) + 1
        seq_len = len(input_ids)
        
        # 获取嵌入
        hidden_states = self.embed_tokens[input_ids]
        
        # 逐层处理
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, self.kv_cache, i, batch_indices, seq_positions)
        
        # 最终归一化
        hidden_states = self.norm(hidden_states)
        
        # 语言模型头部
        logits = self.lm_head(hidden_states)
        
        return logits

    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_p: float = 0.9):
        """生成文本"""
        # 创建请求
        request_id = f"req_{hash(prompt) % 1000000}"
        request = Request(request_id, prompt, max_new_tokens)
        request.prompt_tokens = self.tokenizer.encode(prompt)
        self.scheduler.add_request(request)
        
        # 初始化KV缓存
        self.kv_cache = KVCache(self.config, batch_size=1)
        
        # 生成过程
        output_text = prompt
        for _ in range(max_new_tokens):
            # 获取批次
            batch, batch_data = self.scheduler.get_batch()
            if not batch:
                break
                
            # 准备输入
            input_ids, batch_indices, seq_positions = self._prepare_inputs(batch)
            
            # 模型推理
            logits = self._forward(input_ids, batch_indices, seq_positions)
            
            # 采样下一个token
            next_tokens = []
            finished = []
            
            for i, req in enumerate(batch):
                if batch_data["is_prompt"][i]:
                    # 处理prompt中的多个token，只关注最后一个
                    logit = logits[seq_positions.index(self.kv_cache.seq_lens[i])]
                else:
                    logit = logits[i]
                    
                # 应用温度
                probs = softmax(logit / temperature, axis=-1)
                
                # 应用top-p采样
                sorted_probs = np.sort(probs)[::-1]
                sorted_indices = np.argsort(probs)[::-1]
                cumulative_probs = np.cumsum(sorted_probs)
                
                # 移除累积概率高于top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                # 保留至少一个token
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                sorted_indices_to_remove[0] = False
                
                # 重新归一化概率
                indices_to_keep = sorted_indices[~sorted_indices_to_remove]
                probs_to_keep = probs[indices_to_keep]
                probs_to_keep = probs_to_keep / np.sum(probs_to_keep)
                
                # 采样
                next_token = np.random.choice(indices_to_keep, p=probs_to_keep)
                
                # 检查是否是结束token
                is_done = next_token == self.tokenizer.eos_token_id
                
                next_tokens.append([next_token])
                finished.append(is_done)
            
            # 更新调度器
            self.scheduler.update_requests(
                [req.request_id for req in batch],
                next_tokens,
                finished
            )
            
            # 更新输出文本
            output_text += self.tokenizer.decode([next_tokens[0][0]])
            
            # 如果当前请求完成，退出循环
            if finished[0]:
                break
                
        return output_text    