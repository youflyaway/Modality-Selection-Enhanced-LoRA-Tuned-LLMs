import re

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

SYSTEM_PROMPT = """
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""


def process_data(data):
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["instruction"]},
            ],
            "answer": x["output"],
        }
    )
    return data


def extract_answer(text):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def mark_num(text):
    reward = 0
    if text.count("<think>\n") == 1:
        reward += 0.125

    if text.count("</think>\n") == 1:
        reward += 0.125

    if text.count("<answer>\n") == 1:
        reward += 0.125

    if text.count("</answer>\n") == 1:
        reward += 0.125
    return reward



def response_reward(completions, **kwargs):
    responses = [completion for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]

    overall_response = [
            "Yes, the captions should remain",
            "No, the captions should be removed",
        ]

    return_score = []
    
    for response in extracted_responses:
        if (
            overall_response[0].lower() in response.lower()
            or overall_response[1].lower() in response.lower()
        ):
            return_score.append(1)
        else:
            return_score.append(-0.5)

    return return_score



def length_reward(completions, **kwargs):
    return [1.0 if len(completion) > 1024 else 0.0 for completion in completions]


# 格式奖励
def hard_format_reward(completions, **kwargs):
    pattern = r"^<think>\n.*?n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]


# 格式奖励
def soft_format_reward(completions, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]


# 标记奖励（改善格式奖励稀疏问题）
def mark_reward(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    return [mark_num(response) for response in responses]


if __name__ == "__main__":
    model_name = "Qwen/Qwen2___5-7B-Instruct/"

    # 如果使用lora方法训练，取消如下注释
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
    )
    # # 使用lora方法训练
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ds = load_dataset("")
    data = process_data(ds["train"])

    output_dir = "output"

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        fp16=True,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=4,
        num_generations=4,
        max_prompt_length=1024,
        max_completion_length=256,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            mark_reward,
            soft_format_reward,
            hard_format_reward,
            response_reward,
            length_reward
        ],
        args=training_args,
        train_dataset=data,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
