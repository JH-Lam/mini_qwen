import os
import torch
import matplotlib.pyplot as plt
from itertools import chain
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,  # -so this Trainer is different from the one 'SFTTrainer' in trl lib(see @demo_sft.py )
    TrainingArguments,
)

# >>>>>  Note: 如果要在demo目录／ide 下运行，需要修改文件中的model等路径; 或者在ide下配置 “python demo/demo_xx.py “执行脚本 <<<<<

# 设置环境变量以优化CUDA内存分配
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 加载分词器与模型
output_path = "demo_results/pt"
model_path = "models/Qwen2.5-0.5B-Instruct"
config = AutoConfig.from_pretrained(model_path) #-so all 'model,tokenizer and config' can be loaded from model's path
# 调整模型配置 - same is as the one lied in root/min_qwen_pt.py; note this is 'AutoConfig' so it can load via `from_pretrained()`。see details https://blog.csdn.net/gitblog_01134/article/details/151024151
config.num_attention_heads = 16 # 注意力头数量,影响并行计算能力；do only since the latter steps(eg. sft/dpo) will follow these configs which will save
config.num_key_value_heads = 4 # Key-Value头数量，影响内存效率优化；note When num_key_value_heads is smaller, it adopts(改编） Grouped Query Attention, reducing the KV cache size by storing only the unique Key-Value pairs.
config.hidden_size = 1024 # 隐藏层维度,　影响模型容量，参数量
config.num_hidden_layers = 48 # Transformer层数，影响模型深度，计算复杂度
# print(config)
# todo: no significant speed changes if disable flash-atten 2; different model loading Class(eg. FastLanguage) will use different styles. ; note here uses `from_config()` to load model while sft/dpo uses `from_pretrained()`,that means Qwen(and GPT ..) follow the model standard of lib 'transformers'
model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(model_path) # model path > config > model > tokenizer

# # 计算参数量 - cost heavy
# num_params = sum(p.numel() for p in model.parameters())
# print(f"模型参数量: {num_params}") # 模型参数量: 998810624, about 1B

def find_files(dirs):
    files = []
    for dir in dirs:
        base_path = os.path.join("mini_data/pt", dir) # -based on working dir instead of this file's dir, ie. root of this proj
        for dirpath, _, filenames in os.walk(base_path):
            for filename in filenames:
                if filename.endswith(".parquet"):
                    full_path = os.path.join(dirpath, filename)
                    files.append(full_path)
    return files

# 加载数据集并进行预处理 - dirs to filter out
directories = [
    "accommodation_catering_hotel",
    "artificial_intelligence_machine_learning",
    "computer_communication",
    "computer_programming_code",
    "film_entertainment",
    "literature_emotion",
    "news_media",
    "tourism_geography",
    "current_affairs_government_administration",
    "mathematics_statistics",
]
data_files = find_files(directories)
print('data files found:', data_files) #data files found: ['mini_data/pt/accommodation_catering_hotel/english/high/rank_00726.parquet', 'mini_data/pt/accommodation_catering_hotel/chinese/high/rank_00000.parquet']
#- note: split – 拆分对象（需要用到什么就折分什么，从文件结构中提取，而不是filename），Which split of the data to load. If None, will return a dict with all splits (typically datasets.Split.TRAIN and datasets.Split.TEST). If given, will return a single Dataset. Splits can be combined and specified like in tensorflow-datasets.
dataset = load_dataset("parquet", data_files=data_files,
                       split="train", # - split the data on given key of dataset(map)
                       columns=["text"]) # 只保留text字段, else fail
''' structure of `dataset` w/o `split` arg:
 DatasetDict({
    train: Dataset({
        features: ['text'],
        num_rows: 200
    })
 }),
 column_names:{'train': ['text']}
-- structure of dataset with 'split' arg:
 Dataset({
        features: ['text'],
        num_rows: 200
 }),
 column_names:['text']
'''
dataset = dataset.shuffle(seed=42)
# dataset = dataset.shuffle(seed=42).select(range(20))
# print(dataset[:3]);input()

# - reroganize the whole dataset against an alignment block_size. note the `batch_size` is applied after this function
# note: concrete flow: append EOS(w/o BOS,same as Llama） > tokenization > restructure each samples size with block_size.
#  EOS cases in different LLMs:
#   1. both BOS and EOS tokens in Llama/Qwen LLMs are like : BOS <|begin_of_sentence|>, EOS <|end_of_sentence|>.
#   2. EOS token will be appended automatically while using unsloth.apply_chat_template(dataset) to restructure dataset.
def preprocess_dataset(examples):
    """预处理预训练数据集，将文本分词并分块 - '分块' is as `batch_size` #0604-1 ?no , do align each sample sie only. REMOVE `/home/leibnitz/.cache/huggingface/datasets` if it passbys breakpionts, that's what param 'cache_dir' in `lod_dataset()` to do !!
     note: there's no labels in this dataset（other than 'sft/dpo'), maybe it's unnecessar for "pretrained training/PT" (because it learns patterns via autoregression?)
    """
    eos_token = "<|im_end|>" # - todo seems be different from the one in @Deepseek training project, that eos should inherit from qwen's training set
    text_examples = [text + eos_token for text in examples["text"]]  # 添加结束符 - via line by line; 可见即使前置 load_dataset()中指定‘text' property，在此处仍然需要以'text'为key来提取，可见load_dataset()对properties过滤作用而已
    tokenized_examples = tokenizer(text_examples, add_special_tokens=False) # - convert to numeric ids

    # 将分词结果拼接并分块 - struture: {input_ids:[ [..],...], attention_mask: [ [..],..]} . ie. both are 2-d matrix respectively
    concatenated_examples = {
        k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys() # - concatenate 2-d matrix(data, encodings) to 1-d list via keys - ie. reorganize the data by removing redudant properties
    }
    total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]]) # - total amount of elements in input_ids & attention_mask are same ,so use the first one is enough
    block_size = 128  # 分块大小 - is as final `batch_size`? no,相当于重新调整长度统一为block size，因为原样本长度不一，不适合训练。后续再利用batch_size分页 in caller; note:different from 'sft/dpo','pt' procedure has no max_length like argument but block_size instead;  - it's 1024 in mini_qwen_pt.py file
    total_length = (total_length // block_size) * block_size  # 对齐块大小 - ie. no remainer(余数）divied by block_size

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()
    }
    return result

# 应用预处理函数
train_dataset = dataset.map(
    preprocess_dataset,
    batched=True,
    batch_size=5000, #  vs @0604-1: this arg is used when do real batch samples latter in this function `dataset.map()`; Number of examples per batch provided to function if batched=True, batch_size <= 0 or batch_size == None then provide the full dataset as a single batch to function. － note: this arg is different from the 'per_device_train_batch_size' below, the former is a batch to do 'map()'  while the latter is used to build a real examples set to do train.
    remove_columns=dataset.column_names,
    num_proc=16, # note: 如果想清楚知道送入 ‘preproces_dataset()‘中的参数包含的数据量，必须将此参数改为 num_proc=1。否则不同进程拆分数据集
)

# 数据整理器
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 训练参数配置
training_args = TrainingArguments(
    output_dir=output_path,
    overwrite_output_dir=True,
    learning_rate=1e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    per_device_train_batch_size=1, # - real batch size send to gpu/npu... device
    gradient_accumulation_steps=1, # - 模拟加大 'per_device_train_batch_size' 效果，对于小显存来说可能有较好效果？
    save_steps=100_000,  # 保存中间模型
    save_total_limit=3,
    bf16=True,
    save_only_model=True,
    logging_steps=1,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=train_dataset,
)

# 开始训练
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
