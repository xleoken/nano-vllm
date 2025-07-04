import os
import shutil
from modelscope.hub.snapshot_download import snapshot_download
from mini_vllm import Qwen3Model, ModelConfig

def download_model_from_modelscope(model_id: str, cache_dir: str = None):
    """从ModelScope下载模型"""
    print(f"从ModelScope下载模型: {model_id}")
    
    # 如果未指定缓存目录，则使用默认目录
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "models")
    
    # 创建缓存目录
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # 使用ModelScope SDK下载模型
        model_dir = snapshot_download(
            model_id,
            cache_dir=cache_dir,
            revision="master",  # 可以指定特定版本
            user_agent={"mini-vllm": "0.1.0"}
        )
        print(f"模型已下载至: {model_dir}")
        return model_dir
    except Exception as e:
        print(f"下载模型时出错: {e}")
        return None

def main():
    # ModelScope上的Qwen3-0.6B模型ID
    model_id = "qwen/Qwen3-0.6B"
    
    # 从ModelScope下载模型
    model_path = download_model_from_modelscope(model_id)
    if model_path is None:
        print("模型下载失败，程序退出")
        return
    
    # 模型配置（根据Qwen3-0.6B实际参数调整）
    config = ModelConfig(
        vocab_size=64000,       # 词汇表大小
        hidden_size=2048,      # 隐藏层维度
        num_layers=24,         # 层数
        num_heads=32,          # 注意力头数
        head_dim=64,           # 每个头的维度
        max_seq_len=2048       # 最大序列长度
    )
    
    # 加载模型
    model = Qwen3Model(config, model_path)
    
    # 生成示例
    prompt = "Once upon a time"
    generated_text = model.generate(
        prompt=prompt,
        max_new_tokens=100,    # 最大生成token数
        temperature=0.8,       # 温度参数，控制随机性
        top_p=0.9              # top-p采样参数
    )
    
    print("=" * 50)
    print(f"Prompt: {prompt}")
    print("=" * 50)
    print(f"Generated: {generated_text[len(prompt):]}")
    print("=" * 50)

if __name__ == "__main__":
    main()    