import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import bitsandbytes as bnb

print("🚀 bitsandbytes 버전:", bnb.__version__)

# ───────────────────────────────
# 1. 모델 및 토크나이저 로딩
# ───────────────────────────────
model_name = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # padding 토큰 설정 (필수)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

print("✅ 모델 로딩 완료. 첫 번째 파라미터 위치:", next(model.parameters()).device)

# ───────────────────────────────
# 2. LoRA 설정 및 모델 준비
# ───────────────────────────────
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=64,
    lora_alpha=8,               # ⬅️ 줄임
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.3,           # ⬆️ 증가
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("✅ LoRA 적용 완료")

# ───────────────────────────────
# 3. 데이터셋 로딩 및 전처리
# ───────────────────────────────
dataset = load_dataset("json", data_files="./data/instruct_with_structured_output.json")

def formatting_func(example):
    ins = example["instruction"]
    inp = example.get("input", "")
    out = example["output"]
    prompt = f"{ins}\n\n{inp}\n\n### 답변:" if inp else f"{ins}\n\n### 답변:"
    return {"text": f"{prompt} {out}"}

tokenized_dataset = dataset["train"].map(
    formatting_func,
    remove_columns=dataset["train"].column_names,
)

# ───────────────────────────────
# 4. 학습 설정 및 실행
# ───────────────────────────────
training_args = TrainingArguments(
    output_dir="./eeve_lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1.0,  # ⬅️ 줄임
    learning_rate=2e-4,
    fp16=True,
    save_steps=100,
    logging_steps=10,
    save_total_limit=2,
    optim="paged_adamw_8bit"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    tokenizer=tokenizer,
    packing=False,
    max_seq_length=1024,
)

trainer.train()
