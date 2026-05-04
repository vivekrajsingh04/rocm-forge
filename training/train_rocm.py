"""
ROCm Forge — Fine-Tuning Script for AMD GPUs
=============================================
Fine-tunes a code-generation LLM on CUDA→ROCm migration pairs using
LoRA (PEFT) on AMD Instinct GPUs via PyTorch ROCm.

Usage (on AMD Developer Cloud / MI300X):
    source env_rocm.sh
    pip install -r training/requirements.txt
    python training/train_rocm.py

Environment:
    - AMD Instinct MI300X (or MI250X)
    - ROCm 6.2+
    - PyTorch 2.x (ROCm wheel)
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

DEFAULT_MODEL = "codellama/CodeLlama-7b-hf"
DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
OUTPUT_DIR = Path(__file__).parent / "checkpoints"

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "task_type": TaskType.CAUSAL_LM,
}

TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "logging_steps": 5,
    "save_strategy": "epoch",
    "fp16": True,
    "optim": "adamw_torch",
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 1.0,
    "report_to": "none",
}


# ──────────────────────────────────────────────
# GPU Environment Check
# ──────────────────────────────────────────────

def check_amd_gpu():
    """Verify AMD GPU availability via ROCm/HIP backend."""
    print("=" * 60)
    print("  ROCm Forge — Fine-Tuning on AMD GPU")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("\n  ❌ No GPU detected.")
        print("  Ensure ROCm is installed and HIP_VISIBLE_DEVICES is set.")
        print("  Install PyTorch ROCm: pip install torch --index-url https://download.pytorch.org/whl/rocm6.2")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    device_count = torch.cuda.device_count()
    mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9

    print(f"\n  ✅ GPU Detected: {device_name}")
    print(f"  GPU Count:      {device_count}")
    print(f"  VRAM:           {mem_gb:.1f} GB")
    print(f"  PyTorch:        {torch.__version__}")

    # Check if running on ROCm
    hip_version = getattr(torch.version, "hip", None)
    if hip_version:
        print(f"  HIP Version:    {hip_version}")
        print(f"  Backend:        ROCm ✅")
    else:
        cuda_version = getattr(torch.version, "cuda", "unknown")
        print(f"  CUDA Version:   {cuda_version}")
        print(f"  Backend:        CUDA (not AMD — model will still train)")

    print("=" * 60)
    return True


# ──────────────────────────────────────────────
# Dataset Loading & Formatting
# ──────────────────────────────────────────────

def load_dataset_from_jsonl(path: Path) -> Dataset:
    """Load the CUDA→ROCm paired dataset from JSONL."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"\n  📦 Loaded {len(records)} training examples from {path.name}")
    return Dataset.from_list(records)


def format_prompt(example: dict) -> str:
    """Format an instruction/input/output triple into a training prompt."""
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Output:\n{example['output']}"
    )


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int = 1024):
    """Tokenize the dataset for causal LM training."""

    def tokenize_fn(examples):
        prompts = [format_prompt(ex) for ex in [examples]]
        # Handle batched=False case
        prompt = format_prompt(examples)
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = dataset.map(tokenize_fn, remove_columns=dataset.column_names)
    print(f"  🔢 Tokenized {len(tokenized)} examples (max_length={max_length})")
    return tokenized


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """Load the base model with optional 4-bit quantization for memory efficiency."""
    print(f"\n  🤖 Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if use_4bit:
        print("  📉 Using 4-bit quantization (QLoRA) for memory efficiency")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    # Apply LoRA
    print(f"  🔧 Applying LoRA (r={LORA_CONFIG['r']}, alpha={LORA_CONFIG['lora_alpha']})")
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train(
    model_name: str = DEFAULT_MODEL,
    dataset_path: Path = DATASET_PATH,
    output_dir: Path = OUTPUT_DIR,
    use_4bit: bool = True,
    epochs: int = None,
):
    """Main training loop."""

    # 1. Check GPU
    check_amd_gpu()

    # 2. Load dataset
    dataset = load_dataset_from_jsonl(dataset_path)

    # 3. Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, use_4bit)

    # 4. Tokenize
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    # 5. Configure training
    training_args_dict = TRAINING_CONFIG.copy()
    training_args_dict["output_dir"] = str(output_dir)
    if epochs:
        training_args_dict["num_train_epochs"] = epochs

    training_args = TrainingArguments(**training_args_dict)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 6. Train!
    print("\n" + "=" * 60)
    print("  🚀 Starting LoRA Fine-Tuning on AMD GPU...")
    print("=" * 60 + "\n")

    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"  ✅ Training Complete!")
    print(f"  ⏱️  Duration:     {elapsed/60:.1f} minutes")
    print(f"  📉 Final Loss:   {train_result.training_loss:.4f}")
    print(f"  💾 Checkpoint:   {output_dir}")
    print("=" * 60)

    # 7. Save the LoRA adapter
    adapter_path = output_dir / "rocm-forge-lora"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n  💾 LoRA adapter saved to: {adapter_path}")
    print("  To load: model = PeftModel.from_pretrained(base_model, adapter_path)")

    return train_result


# ──────────────────────────────────────────────
# Inference (Test the fine-tuned model)
# ──────────────────────────────────────────────

def test_inference(adapter_path: str = None, model_name: str = DEFAULT_MODEL):
    """Test the fine-tuned model with a sample CUDA code snippet."""
    from peft import PeftModel

    if adapter_path is None:
        adapter_path = str(OUTPUT_DIR / "rocm-forge-lora")

    print(f"\n  🧪 Testing fine-tuned model from: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    test_code = """import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')
model = model.cuda()
torch.backends.cudnn.benchmark = True"""

    prompt = f"### Instruction:\nMigrate the following NVIDIA CUDA Python code to AMD ROCm.\n\n### Input:\n{test_code}\n\n### Output:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_part = result.split("### Output:\n")[-1].strip()

    print("\n  📝 Input (CUDA):")
    print("  " + test_code.replace("\n", "\n  "))
    print("\n  ✅ Output (ROCm):")
    print("  " + output_part.replace("\n", "\n  "))


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROCm Forge — Fine-tune a code LLM for CUDA→ROCm migration")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Base model (default: {DEFAULT_MODEL})")
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="Path to dataset.jsonl")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--test", action="store_true", help="Run inference test on saved adapter")

    args = parser.parse_args()

    if args.test:
        test_inference(model_name=args.model)
    else:
        train(
            model_name=args.model,
            dataset_path=Path(args.dataset),
            output_dir=Path(args.output),
            use_4bit=not args.no_4bit,
            epochs=args.epochs,
        )
