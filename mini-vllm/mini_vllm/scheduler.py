import numpy as np
from typing import List, Dict, Any, Tuple

class Request:
    def __init__(self, request_id: str, prompt: str, max_new_tokens: int = 100):
        self.request_id = request_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.generated_tokens = 0
        self.is_done = False
        self.prompt_tokens = []
        self.output_tokens = []

class Scheduler:
    def __init__(self, max_batch_size: int = 8):
        self.pending_requests = []
        self.active_requests = []
        self.max_batch_size = max_batch_size

    def add_request(self, request: Request):
        self.pending_requests.append(request)

    def get_batch(self) -> Tuple[List[Request], Dict[str, Any]]:
        """获取待处理的批次请求"""
        if not self.pending_requests and not self.active_requests:
            return [], {}

        # 优先处理活跃请求
        batch = []
        for req in self.active_requests:
            if len(batch) >= self.max_batch_size:
                break
            if not req.is_done:
                batch.append(req)

        # 添加新请求
        for req in self.pending_requests[:self.max_batch_size - len(batch)]:
            batch.append(req)
            self.pending_requests.remove(req)
            self.active_requests.append(req)

        # 构建批次数据
        batch_data = {
            "request_ids": [req.request_id for req in batch],
            "is_prompt": [len(req.output_tokens) == 0 for req in batch],
            "tokens": [req.prompt_tokens if len(req.output_tokens) == 0 else [req.output_tokens[-1]] for req in batch],
        }
        return batch, batch_data

    def update_requests(self, request_ids: List[str], output_tokens: List[List[int]], finished: List[bool]):
        """更新请求状态"""
        for req_id, tokens, is_done in zip(request_ids, output_tokens, finished):
            for req in self.active_requests:
                if req.request_id == req_id:
                    req.output_tokens.extend(tokens)
                    req.generated_tokens += len(tokens)
                    req.is_done = is_done or req.generated_tokens >= req.max_new_tokens
                    if req.is_done:
                        self.active_requests.remove(req)
                    break    