import os
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

# 设置环境变量以优化CUDA内存分配
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 加载分词器与模型
output_path = "demo_results/sft"
model_path = "demo_results/pt" # - generated from demo_pt.py
# - note this model can also be loaded from FastLanguageModel from unsloth lib(see @llm-practice also, which involves 'load_in_4bit' arguemnt too
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(model_path)

def find_files(dirs):
    files = []
    for dir in dirs:
        base_path = os.path.join("mini_data/sft", dir)
        for dirpath, _, filenames in os.walk(base_path):
            for filename in filenames:
                if filename.endswith(".parquet"):
                    full_path = os.path.join(dirpath, filename)
                    files.append(full_path)
    return files

# 加载数据集并进行预处理
directories = ["7M","Gen"]
data_files = find_files(directories)
dataset = load_dataset("parquet", data_files=data_files, split="train", columns=["conversations"]) # 只保留conversations字段（但事实上仍然会出现在结果中，相当于 filter）
dataset = dataset.shuffle(seed=42)
# dataset = dataset.shuffle(seed=42).select(range(20))
print(dataset[:2])

# - act as arg passed into SFTTrainer; different the dataset-processor did in @deno_pt.py, here uses `gpt answer` as labels
#  note: REMOVE `/home/leibnitz/.cache/huggingface/datasets` if it passbys breakpionts during debugs
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["conversations"])):
        for item in example["conversations"][i]:  # - the item 'gpt' always follows to item 'from'
            if item["from"] == "human":
                human_text = item["value"]
            elif item["from"] == "gpt":
                gpt_text = item["value"]
            else:
                raise ValueError(f"Unknown sender: {item['from']}")
        text = f"<|im_start|>user\n{human_text}<|im_end|>\n<|im_start|>assistant\n{gpt_text}<|im_end|>" #260609-1
        output_texts.append(text)
    return output_texts

# 数据整理器 - note:different from the one in @demo_pt.py
response_template = "<|im_start|>assistant\n" # - both Q&A are both in @260609-1, todo why it's one more template here
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:] # the template form that indicates the start of the response
# - completion: is as 'response' style generation
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, mlm=False) # constant usage/固定用法； mlm – Whether or not to use masked language modeling in the underlying DataCollatorForLanguageModeling class. this option currently has no effect but is present for flexibility and backwards-compatibility.

# 训练参数配置
training_args = SFTConfig(
    output_dir=output_path,
    overwrite_output_dir=True,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    num_train_epochs=3, # -it's 1 in @demo_pt.py
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    save_strategy="epoch",  # 保存中间模型
    save_total_limit=2,
    bf16=True,
    save_only_model=True,
    logging_steps=1,
)

# 初始化Trainer - note: uses trl's SFTTreainer instead of transformer.Trainer did in @demo_pt.py
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=128,
    packing=False,
    dataset_num_proc=16,
    dataset_batch_size=5000,
)

# 开始训练 - these mothods will be called after training in @llm-practice, while 'model.save_pretrained()' something should call after training to combine old and adaptor weights into one
trainer.train()
trainer.save_model()  # 保存模型
tokenizer.save_pretrained(output_path)  # 保存分词器


def plot_loss(save_directory, log_history):
    """绘制训练损失曲线并保存图像"""
    plt.switch_backend("agg")  # 使用非交互式后端
    key = "loss"  # 默认使用 'loss' 作为绘图的关键字
    steps, metrics = [], []

    # 提取损失数据
    for log_entry in log_history:
        if key in log_entry:
            steps.append(log_entry["step"])
            metrics.append(log_entry[key])

    # 绘制图像
    plt.figure()
    plt.plot(steps, metrics, color="#1f77b4", label="original")
    plt.title(f"Training {key} of {save_directory}")
    plt.xlabel("Step")
    plt.ylabel(key.capitalize())
    plt.legend()

    # 保存图像
    figure_path = os.path.join(save_directory, f"training_{key.replace('/', '_')}.png")
    plt.savefig(figure_path, format="png", dpi=100)
    print(f"Figure saved at: {figure_path}")


# 绘制并保存损失曲线
plot_loss(output_path, trainer.state.log_history)
