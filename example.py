# ModelScope 相关导入
from modelscope.hub.snapshot_download import snapshot_download
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


def main(model_id: str = "Qwen/Qwen3-0.6B", revision: str = "master"):
    """
    主函数，自动使用 ModelScope 默认缓存路径。
    """
    print(f"开始从 ModelScope 加载模型：{model_id}@{revision}")
    model_dir = snapshot_download(model_id, revision=revision)
    print(f"模型已下载或加载自默认路径: {model_dir}")

    # 加载 tokenizer 和 LLM
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    llm = LLM(model_dir, enforce_eager=True, tensor_parallel_size=1)

    # 生成参数配置
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

    # 测试 prompt
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]

    # 使用 chat template 构造输入
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]

    # 执行推理
    outputs = llm.generate(prompts, sampling_params)

    # 输出结果
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
