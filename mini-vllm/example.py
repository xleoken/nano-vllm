from mini_vllm import Qwen3Model, ModelConfig

def main():
    # 模型配置
    config = ModelConfig(
        vocab_size=64000,
        hidden_size=2048,
        num_layers=24,
        num_heads=32,
        head_dim=64,
        max_seq_len=2048
    )
    
    # 加载模型
    model_path = "path/to/qwen3-0.6b"  # 替换为实际模型路径
    model = Qwen3Model(config, model_path)
    
    # 生成文本
    prompt = "Once upon a time"
    generated_text = model.generate(prompt, max_new_tokens=100)
    
    print("Prompt:", prompt)
    print("Generated:", generated_text[len(prompt):])

if __name__ == "__main__":
    main()    