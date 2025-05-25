import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = []
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                d = json.loads(line)
                data.append({"text": d["text"]})
        return data

# Load data
train_data = load_jsonl("./train_sampled.jsonl")
valid_data = load_jsonl("./valid_sampled.jsonl")
train_dataset = Dataset.from_list(train_data)
valid_dataset = Dataset.from_list(valid_data)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(valid_data)}")

# Load model and tokenizer (no quantization)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
print(f"Model loaded on device: {model.device}")

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "dense"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA to model
print("Applying LoRA...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def tokenize(batch):
    model_inputs = tokenizer(
        batch["text"],
        padding=True,
        truncation=True,
        max_length=512
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# Tokenize datasets
print("Tokenizing datasets...")
tokenized_train = train_dataset.map(tokenize, batched=True, batch_size=32)
tokenized_valid = valid_dataset.map(tokenize, batched=True, batch_size=32)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="/mnt/model/chinoa/chinoa_01",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    logging_steps=50,
    save_steps=200,
    eval_steps=200,
    evaluation_strategy="steps",
    fp16=True,  # This will help with memory
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    gradient_checkpointing=True  # This saves memory too
)

# Create trainer
print("Creating trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    data_collator=data_collator
)

# Start training
print("Starting training...")
trainer.train()