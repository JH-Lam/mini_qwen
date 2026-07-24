import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

# 加载分词器与模型 
model_path = "results/pt"
model_path = "results/pt1"
model_path = "results/pt2"
model_path = "results/pt3"
model_path = "models/Qwen2.5-0.5B"
model_path = "results/sft/checkpoint-7730"
model_path = "results/sft/checkpoint-15461"
model_path = "results/sft"
model_path = "results/dpo/checkpoint-154"
model_path = "results/dpo/checkpoint-309"
model_path = "results/dpo"

model = AutoModelForCausalLM.from_pretrained( # - todo this means the 'pt' model can be used like 'sft/dpo'?
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)


while True:
    prompt = input("用户：")
    
    text = prompt  # 预训练模型
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"  # 微调和直接偏好优化模型 - note appearently it uses function 'generation' of LLM, so given the leading text is enough. see details in @min_qwen_sft.py-dataset
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512) # - note this is the basic usage of LLM, ie. using raw function. the advanced usage is packing tools, like ollama-api or agent
    generated_ids = [
        output_ids[len(input_ids) :] # note: trim the leading question part
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("助手：", response)
