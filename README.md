# 面向作物病虫害识别的大型视觉语言模型 LoRA 微调与优化

本项目使用 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 进行高效的 LLM 微调实验，选择Qwen2.5-VL-7B作为基模型

## 结果文件存放位置说明

所有训练、评估、推理、合并模型等产出的文件都会统一保存在当前工作目录下的 **`output`** 文件夹中


## 1. 环境准备

```bash
# 建议使用独立的虚拟环境
conda create -n lfactory python=3.10 -y
conda activate lfactory

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 基础依赖 + 常用加速包
pip install -e ".[torch,metrics,deepspeed]" --no-build-isolation

# 强烈建议安装（根据需要选装）
pip install flash-attn --no-build-isolation     # FlashAttention-2
pip install vllm                               # 高效推理
pip install bitsandbytes                       # 4bit/8bit 量化

```

## 2. 配置文件存放建议
建议把 yaml 配置文件统一放在项目根目录下的 configs/ 文件夹，并按阶段分类：
```bash
configs/
├── sft/
│   ├── qwen2.5-VL-7b-lora.yaml
│   ├── llama3.1-8b-lora.yaml
│   └── deepseek-v3-lora.yaml
├── dpo/
│   └── qwen2-7b-dpo.yaml
├── full-finetune/
└── inference/
```
## 3. 最常用的几种启动方式

**单卡 LoRA 微调（最常用）**

```bash
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /path/to/your/model \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --dataset your_dataset_name \
    --template llama3 \
    --flash_attn auto \
    --output_dir output/sft/llama3-8b-lora-test \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --save_steps 200 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16
```

**多卡训练**

```bash
deepspeed --num_gpus 4 --master_port 29500 \
    src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --dataset alpaca_zh \
    --template llama3 \
    --flash_attn auto \
    --output_dir output/sft/llama3-8b-lora-multi \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-5 \
    --lr_scheduler_type cosine \
    --num_train_epochs 2.0 \
    --fp16 \
    --deepspeed configs/deepspeed/ds_z3_config.json
```

## 4.推理 / 模型合并 / 量化导出

**1. 合并 LoRA 权重（得到完整模型）**
```bash
llamafactory-cli export \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --adapter_name_or_path output/sft/llama3-8b-lora-test/checkpoint-xxx \
    --template llama3 \
    --finetuning_type lora \
    --export_dir output/checkpoints/llama3-8b-merged \
    --export_size 2 \
    --export_device cpu
```
**2. 直接使用 vLLM 进行高效推理**
```bash
vllm serve output/checkpoints/llama3-8b-merged \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --port 8000
```
