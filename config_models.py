# config_qed.py
"""
Configuration file for QED fine-tuning.
Contains predefined configs for different model types and training scenarios.
"""

import os
from dataclasses import dataclass
from qed_main import ModelArguments, DataArguments, TrainingArguments

# Base configuration paths
DATA_DIR = "./qed_data"
OUTPUT_DIR = "./models_fine_tuned"

def get_llama3_8b_instruct_config(max_examples=None, max_eval_examples=None, epochs=3):
    """Configuration for fine-tuning Meta-Llama-3-8B-Instruct with LoRA."""
    
    model_args = ModelArguments(
        model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        load_in_4bit=True,  # Use 4-bit for memory efficiency
        load_in_8bit=False,
        use_cpu_offload=True,
    )
    
    data_args = DataArguments(
        qed_data_path=DATA_DIR,
        max_examples=max_examples,
        max_eval_examples=max_eval_examples,
        max_source_length=3072,
        max_target_length=1024
    )
    
    output_path = os.path.join(OUTPUT_DIR, "llama3_8b_instruct")
    
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,    # Adjust based on VRAM
        per_device_eval_batch_size=1,     # Adjust based on VRAM
        gradient_accumulation_steps=16,   # Adjust for effective batch size
        eval_accumulation_steps=1,  #
        learning_rate=5e-6,               # Llama 3 can sometimes use higher LRs like 5e-5 or 1e-4 for LoRA
        weight_decay=0.001,
        warmup_ratio=0.2,
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=50,                   # Or match save_steps
        save_strategy="steps",
        save_steps=100,
        save_total_limit=10,
        bf16=True,                        # Llama 3 strongly prefers bf16
        fp16=False,
        optim="paged_adamw_8bit",         # Good default for QLoRA
        report_to="tensorboard",
        max_grad_norm=1.0,                # Enable gradient clipping for stability
        # predict_with_generate=True, # Already default in your TrainingArguments
    )
    
    return model_args, data_args, training_args


def get_mistral_7b_instruct_config(max_examples=None, max_eval_examples=None, epochs=3):
    """Configuration for fine-tuning Mistral-7B-Instruct-v0.3 with LoRA."""
    
    model_args = ModelArguments(
        model_name_or_path="mistralai/Mistral-7B-Instruct-v0.3",
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        load_in_4bit=True,  # Use 4-bit for memory efficiency
        load_in_8bit=False,
        use_cpu_offload=True
    )
    
    data_args = DataArguments(
        qed_data_path=DATA_DIR,
        max_examples=max_examples,
        max_eval_examples=max_eval_examples,
        max_source_length=3072,
        max_target_length=1024
    )
    
    output_path = os.path.join(OUTPUT_DIR, "mistral_7b_instruct")
    
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,    # Adjust based on VRAM
        per_device_eval_batch_size=1,     # Adjust based on VRAM
        gradient_accumulation_steps=16,   # Adjust for effective batch size
        eval_accumulation_steps=1,
        learning_rate=2e-5,               # Mistral works well with standard LoRA LR
        weight_decay=0.001,
        warmup_ratio=0.1,
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=10,
        bf16=True,                        # Mistral supports bf16
        fp16=False,
        optim="paged_adamw_8bit",         # Good default for QLoRA
        report_to="tensorboard",
        max_grad_norm=1.0,                # Enable gradient clipping for stability
    )
    
    return model_args, data_args, training_args


def get_qwen2_5_7b_instruct_config(max_examples=None, max_eval_examples=None, epochs=3):
    """Configuration for fine-tuning Qwen2.5-7B-Instruct with LoRA."""
    
    model_args = ModelArguments(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        load_in_4bit=True,  # Use 4-bit for memory efficiency
        load_in_8bit=False,
        use_cpu_offload=True
    )
    
    data_args = DataArguments(
        qed_data_path=DATA_DIR,
        max_examples=max_examples,
        max_eval_examples=max_eval_examples,
        max_source_length=3072,
        max_target_length=1024
    )
    
    output_path = os.path.join(OUTPUT_DIR, "qwen2_5_7b_instruct")
    
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,    # Adjust based on VRAM
        per_device_eval_batch_size=1,     # Adjust based on VRAM
        gradient_accumulation_steps=16,   # Adjust for effective batch size
        eval_accumulation_steps=1,
        learning_rate=5e-6,               # Qwen2.5 works well with standard LoRA LR
        weight_decay=0.001,
        warmup_ratio=0.2,
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=10,
        bf16=True,                        # Qwen2.5 supports bf16
        fp16=False,
        optim="paged_adamw_8bit",         # Good default for QLoRA
        report_to="tensorboard",
        max_grad_norm=1.0,                # Enable gradient clipping for stability
    )
    
    return model_args, data_args, training_args


def get_few_shot_config(model_name, model_path=None, max_examples=None, max_eval_examples=None, prompt='both', epochs=5):
    """
    Configuration for few-shot fine-tuning (200 examples).
    Adapts parameters for better few-shot learning.
    """
    model_name_lower = model_name.lower()
    if "llama3-8b-instruct" in model_name_lower or "llama3_8b_instruct" in model_name_lower :
        model_args, data_args, training_args = get_llama3_8b_instruct_config(max_examples, max_eval_examples, epochs)
    elif "mistral-7b-instruct" in model_name_lower or "mistral_7b_instruct" in model_name_lower:
        model_args, data_args, training_args = get_mistral_7b_instruct_config(max_examples, max_eval_examples, epochs)
    elif "qwen2.5-7b-instruct" in model_name_lower or "qwen2_5_7b_instruct" in model_name_lower:
        model_args, data_args, training_args = get_qwen2_5_7b_instruct_config(max_examples, max_eval_examples, epochs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    if model_path is not None:
        model_args.model_name_or_path = model_path

    # Modify output path to indicate few-shot
    training_args.output_dir = os.path.join(os.path.dirname(training_args.output_dir), 
                                           f"{os.path.basename(training_args.output_dir)}_fewshot_{prompt}")
    
    return model_args, data_args, training_args
