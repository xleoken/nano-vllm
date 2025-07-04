import os
import numpy as np
from typing import List, Dict, Any, Optional
from .layers import TransformerLayer, RMSNorm, Linear
from .config import ModelConfig
from .tokenizer import Tokenizer

class Qwen3Model:
    def __init__(self, config: ModelConfig, model_path: str):
        self.config = config
        self.model_path = model_path
        self.tokenizer = Tokenizer(model_path)
        
        # 初始化模型参数
        self.embed_tokens = np.zeros((config.vocab_size, config.hidden_size), dtype=np.float32)
        self.layers = [TransformerLayer(config) for _ in range(config.num_layers)]
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 加载模型权重
        self._load_model_weights()

    def _load_model_weights(self):
        """从ModelCode格式加载模型权重"""
        print(f"Loading model weights from {self.model_path}")
        # 实际实现中，这里应该从文件加载权重
        # 为简化示例，我们使用随机权重
        
    def _forward(self, input_ids: np.ndarray, seq_pos: int):
        """模型前向传播"""
        batch_size, seq_len = input_ids.shape
        
        # 获取嵌入
        hidden_states = self.embed_tokens[input_ids]
        
        # 逐层处理
        for layer in self.layers:
            hidden_states = layer(hidden_states, seq_pos)
        
        # 最终归一化
        hidden_states = self.norm(hidden_states)
        
        # 语言模型头部
        logits = self.lm_head(hidden_states)
        
        return logits

    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_p: float = 0.9):
        """生成文本"""
        # 编码输入
        input_tokens = self.tokenizer.encode(prompt)
        input_ids = np.array([input_tokens], dtype=np.int64)
        
        # 初始前向传播处理整个prompt
        logits = self._forward(input_ids, seq_pos=0)
        
        # 采样第一个生成的token
        next_token = self._sample_next_token(logits[:, -1, :], temperature, top_p)
        generated_tokens = [next_token]
        
        # 逐个生成后续tokens
        for i in range(1, max_new_tokens):
            # 将最新生成的token添加到输入
            input_ids = np.array([[next_token]], dtype=np.int64)
            
            # 前向传播
            logits = self._forward(input_ids, seq_pos=len(input_tokens) + i)
            
            # 采样下一个token
            next_token = self._sample_next_token(logits[:, -1, :], temperature, top_p)
            generated_tokens.append(next_token)
            
            # 检查是否生成了结束token
            if next_token == self.tokenizer.eos_token_id:
                break
                
        # 解码并返回生成的文本
        return prompt + self.tokenizer.decode(generated_tokens)

    def _sample_next_token(self, logits: np.ndarray, temperature: float, top_p: float):
        """采样下一个token"""
        # 应用温度
        logits = logits / temperature
        
        # 应用top-p采样
        probs = softmax(logits, axis=-1)
        sorted_probs = np.sort(probs)[0, ::-1]
        sorted_indices = np.argsort(probs)[0, ::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        
        # 移除累积概率高于top_p的token
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
        sorted_indices_to_remove[0] = False
        
        # 重新归一化概率
        indices_to_keep = sorted_indices[~sorted_indices_to_remove]
        probs_to_keep = probs[0, indices_to_keep]
        probs_to_keep = probs_to_keep / np.sum(probs_to_keep)
        
        # 采样
        next_token = np.random.choice(indices_to_keep, p=probs_to_keep)
        return next_token    