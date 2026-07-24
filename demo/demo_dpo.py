import os
import torch
import matplotlib.pyplot as plt
from PIL.ImImagePlugin import split
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

# 设置环境变量以优化CUDA内存分配
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 加载分词器与模型
output_path = "demo_results/dpo"
model_path = "demo_results/sft"  # 从sft 2epoch模型继续训练 - or 3 epoches
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载数据集并进行预处理
data_files = [
    "mini_data/dpo/train-00000-of-00001.parquet",
    "mini_data/dpo/test-00000-of-00001.parquet" # - put 'test set' means consider it as training set?
]
dataset = load_dataset("parquet", data_files=data_files,
                       split="train", # - resulted in 'no test set'
                       # split=None, # - same result as comment this arg
                       )
dataset = dataset.shuffle(seed=42) # total 200 rows in both files
# dataset = dataset.shuffle(seed=42).select(range(20))
'''
 raw dataset structure:
 
 prompt;[那么，我要在我家里种一些银月藤，你能告诉我如何制作适当的土壤和提供给它所需的养分吗？]

 chosen:[{\'content\': \'那么，我要在我家里种一些银月藤，你能告诉我如何制作适当的土壤和提供给它所需的养分吗？\', \'role\': \'user\'}, {\'content\': \'种植银月藤需要提供排水良好、略微酸性的土壤环境，才能帮助它健康生长。 \n\n以下是制作合适土壤的方法：\n\n**土壤配方:**\n\n* **腐殖土:** 40% - 提供丰富的有机质和养分。\n* **珍珠岩或蛭石:** 30% - 改善排水和通气性。\n* **河沙或椰糠:** 30% - 增强排水和土壤结构。\n\n**添加的养分:**\n\n* **缓释肥:** 在种植时加入缓释肥料，可以提供持续的养分供应，例如使用缓释氮磷钾肥或复合肥。\n* **堆肥:** 每月使用一次堆肥水稀释液，可以补充土壤中的有机质和养分。\n\n**其他建议:**\n\n* **pH值:** 银月藤喜欢略微酸性土壤，pH值在5.5-6.5之间最佳。\n* **光照:** 银月藤喜欢明亮散射光，避免强烈的阳光直射。\n* **浇水:** 浇水时要确保土壤湿润，但不要积水。\n* **温度:** 银月藤生长适温为18-25℃，避免寒冷和高温。\n\n**注意:**\n\n* 避免使用过多的肥料，因为过量的肥料会烧根。\n* 定期检查土壤是否有病虫害，并及时进行防治。\n* 每半年左右修剪一次，可以促进新枝条生长，保持株形美观。\n\n\n希望这些信息能帮助你成功种植银月藤！\n\n\n\', \'role\': \'assistant\'}]

 rejcted: [{\'content\': \'那么，我要在我家里种一些银月藤，你能告诉我如何制作适当的土壤和提供给它所需的养分吗？\', \'role\': \'user\'}, {\'content\': \'种植银月藤需要土壤排水良好，略微酸性的环境。以下是一些你可以用来制作合适的土壤的建议：\n\n**土壤成分：**\n\n* **泥炭土:** 约为 1/3 的泥炭土可以提供良好的排水和酸性环境，但要注意避免纯泥炭土，因为它容易缺乏营养。\n* **珍珠岩:**  珍珠岩可以帮助改善土壤的排水性，防止积水。\n* **园土:** 园土提供土壤结构和养分。选择富含有机质的园土。\n\n**混合比例:**  你可以将上述三种材料以 1:1:1 的比例混合，也可以根据你当地的土壤情况进行调整。例如，如果你的土壤排水较好，你可以减少珍珠岩的比例，增加泥炭土的比例。\n\n**养分提供:**\n\n* **缓释型肥料:** 在种植银月藤时，可以加入少量缓释型肥料，以提供初始养分。\n* **腐熟堆肥:** 定期使用腐熟堆肥可以补充土壤中的养分，并且改善土壤结构。\n* **叶面肥:** 生长期间，你可以每月使用一次叶面肥，以补充必要的微量元素。\n\n**其他建议:**\n\n* **pH值:** 银月藤喜欢微酸性的土壤，建议保持土壤pH值在 5.5-6.5 之间。\n* **排水:** 确保你的盆栽有良好的排水孔，避免积水。\n* **阳光:** 银月藤喜欢半阴凉爽的环境，避免强烈的阳光直射。\n\n希望以上信息能够帮助你成功种植银月藤！\n\n\n\', \'role\': \'assistant\'}]
 == so both the 'content' in chosen and rejcted fields are same as the prompt one.
'''
print(dataset[:2]["chosen"]) #;input(); ＃ - resulted dataset is as dataframe,or say table,so here is as filter top rows then choose column 'chosen'.
# result: [[{'content': 'Explain quantum entanglement like I am 5, and include an analogy involving everyday objects to make it easier to understand.', 'role': 'user'},
# {'content': 'Imagine you have two magic coins. These coins are super special because when you flip one, you instantly know what the other coin will land on, even if they\'re miles apart! \n\nIf one coin lands on heads, the other *always* lands on tails, and vice versa. It doesn\'t matter how far apart they are, they\'re always connected in a secret way. \n\nThat\'s kind of like quantum entanglement! Tiny particles can be linked together in a special way, just like our magic coins. Even if you separate them really, really far apart, they still act like they\'re connected. If you measure one particle and know something about it, you instantly know something about the other particle, no matter how far away it is!\n\nScientists are still trying to figure out exactly how this "secret connection" works, but it\'s one of the coolest things about the tiny world of quantum mechanics! \n\n\n', 'role': 'assistant'}], ... ]
#  note: REMOVE `/home/leibnitz/.cache/huggingface/datasets` if it passbys breakpionts
def preprocess_dataset(examples): # - todo:this examples is total 13/200 ratio compared to 200 loaded first above?(maybe some columns miss data,即整合时缺少数据列的行去掉，类似dataframe操作)
    prompt, chosen, rejected = [], [], []

    for i in range(len(examples["prompt"])): # - appearently , "prompt + chosen" or "prompt+rejected" is a pair(a complete Q&A pair)
        text = f"<|im_start|>user\n{examples['prompt'][i]}<|im_end|>\n<|im_start|>assistant\n"
        prompt.append(text)

        assert examples["chosen"][i][1]["role"] == "assistant"
        text = f"{examples['chosen'][i][1]['content']}<|im_end|>" # - same format as demo_pt.py;the [i][0] element is role=user
        chosen.append(text)

        assert examples["rejected"][i][1]["role"] == "assistant"
        text = f"{examples['rejected'][i][1]['content']}<|im_end|>"
        rejected.append(text)

    result = {"prompt": prompt, "chosen": chosen, "rejected": rejected} # - different format compared to the one in @demo_sft.py
    return result


# 应用预处理函数 － todo 为什么处理后，数据量由dataset的200条变成了13条
train_dataset = dataset.map(
    preprocess_dataset,
    batched=True,
    batch_size=5000,
    remove_columns=dataset.column_names, # - Remove a selection of columns while doing the mapping. Columns will be removed before updating the examples with the output of function, i.e. if function is adding columns with names in remove_columns, these columns will be kept.
    num_proc=16, # note：此值将影响送入　‘formatting_prompts_func()' 中的样本数量
)

# 训练参数配置
training_args = DPOConfig(
    output_dir=output_path,
    overwrite_output_dir=True,
    learning_rate=5e-7,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    save_strategy="epoch",  # 保存中间模型
    save_total_limit=3,
    bf16=True,
    save_only_model=True,
    logging_steps=1,
)

# 初始化Trainer - different to the ones did in demo_pt,py and demo_sft.py
trainer = DPOTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    tokenizer=tokenizer,
    dataset_num_proc=16,
    max_length=128, # - The max_prompt_length is the maximum length of the prompt , the max_length is the maximum length of the prompt + chosen or rejected response. Those are used for tokenization, padding and trunctation.
    max_prompt_length=128,
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
