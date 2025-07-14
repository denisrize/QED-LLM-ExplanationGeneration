import os
import sys
import json
import copy
import torch
import random
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    PeftModel,
    PeftConfig,
)
import evaluate
import re
from dataclasses import dataclass, field
import qed_eval
from qed_eval import load_data, compute_scores, MIN_F1_FOR_NON_STRICT_OVERLAP, load_answer, load_aligned_entities, QEDExample, load_single_line
from config_prompt import *
import uuid
from data_processing import *
from torch.utils.tensorboard import SummaryWriter
from loss_tracking import LossTrackingCallback


from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.plugins.custom_scalar import summary as custom_scalar_summary
CUSTOM_SCALAR_AVAILABLE = True


CACHE_DIR = "/sise/robertmo-group/Denis/Models/QED/models_saved/models"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

"""Fine-tune a language model on the QED dataset for instruction tuning.
Core implementation with the complete fine-tuning and evaluation pipeline"""

#############################
# 1. Config & Argument Parsing
#############################

@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    model_name_or_path: str = field(
        default="meta-llama/Llama-2-7b-hf",  # Default to Phi-2
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models. "
                          "Shortcuts:  'llama2_7b_chat', 'llama3_8b_instruct', 'mistral_7b_instruct', 'qwen2_5_7b_instruct'"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if different from model_name"}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "Rank dimension for LoRA fine-tuning"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "Alpha parameter for LoRA"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout probability for LoRA layers"}
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the model in 8-bit precision"}
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the model in 4-bit precision"}
    )
    use_cpu_offload: bool = field(
        default=False,
        metadata={"help": "Whether to offload model weights to CPU to save GPU memory"}
    )
    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the model checkpoint to load instead of model_name_or_path"}
    )


@dataclass
class DataArguments:
    """Arguments for data processing"""
    qed_data_path: str = field(
        default="./qed_data",
        metadata={"help": "Path to the QED dataset files"}
    )
    train_file: str = field(
        default="qed-train.jsonlines",
        metadata={"help": "Name of the training file"}
    )
    validation_file: str = field(
        default="qed-dev.jsonlines",
        metadata={"help": "Name of the validation file"}
    )
    max_examples: int = field(
        default=None,
        metadata={"help": "Maximum number of examples to use (for quick experiments)"}
    )
    max_eval_examples: int = field(
        default=None,
        metadata={"help": "Maximum number of validation examples to use (for quick experiments)"}
    )
    max_source_length: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length"}
    )
    max_target_length: int = field(
        default=512,
        metadata={"help": "Maximum target sequence length"}
    )
    prompt_examples: str = field(
        default="zero_shot",
        metadata={"help": "Which examples to include in the prompt. Options: 'zero_shot', 'first_only', 'second_only', 'both', 'random_one', 'random_two'"}
    )


@dataclass
class TrainingArguments(TrainingArguments):
    """Arguments for training"""
    output_dir: str = field(
        default="./qed_model_output",
        metadata={"help": "Directory to save the model checkpoints"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=1,  # Reduced batch size
        metadata={"help": "Batch size per device during training"}
    )
    per_device_eval_batch_size: int = field(
        default=1,  # Reduced batch size
        metadata={"help": "Batch size per device during evaluation"}
    )
    gradient_accumulation_steps: int = field(
        default=16,  # Increased gradient accumulation
        metadata={"help": "Number of gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Initial learning rate"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay rate"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Ratio of warmup steps"}
    )
    logging_steps: int = field(
        default=5,
        metadata={"help": "Logging steps during training"}
    )
    evaluation_strategy: str = field(
        default="steps",  
        metadata={"help": "Evaluation strategy"}
    )
    eval_steps: int = field(
        default=5,  # ✅ Only used if evaluation_strategy != "no"
        metadata={"help": "Evaluation steps"}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "Checkpoint saving strategy"}
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Checkpoint saving steps"}
    )
    save_total_limit: int = field(
        default=5,
        metadata={"help": "Maximum number of checkpoints to keep"}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Whether to use bfloat16 precision"}
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use fp16 precision"}
    )
    optim: str = field(
        default="paged_adamw_8bit",
        metadata={"help": "Optimizer to use"}
    )
    report_to: str = field(
        default="tensorboard",
        metadata={"help": "Report results to this platform"}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Whether to use gradient checkpointing"}
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={"help": "Maximum gradient norm for gradient clipping"}
    )
    predict_with_generate: bool = field( # Added this field to override the default
        default=True,
        metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU, etc.)"}
    )


#############################
# 2. Data Processing
#############################

def load_qed_data(data_args):
    """Load and preprocess QED data."""
    
    # Load the training and validation datasets
    train_path = os.path.join(data_args.qed_data_path, data_args.train_file)
    validation_path = os.path.join(data_args.qed_data_path, data_args.validation_file)
    
    train_data = []
    with open(train_path, 'r') as f:
        for line in f:
            train_data.append(json.loads(line))
    
    validation_data = []
    with open(validation_path, 'r') as f:
        for line in f:
            validation_data.append(json.loads(line))
    
    # Limit the number of examples if specified
    if data_args.max_examples is not None:
        train_data = train_data[:data_args.max_examples]
    if data_args.max_eval_examples is not None:
        validation_data = validation_data[:data_args.max_eval_examples]
    
    print(f"Loaded {len(train_data)} training examples")
    print(f"Loaded {len(validation_data)} validation examples")
    
    return train_data, validation_data


#############################
# 2. Model-Specific Instruction Formats
#############################

def get_model_instruction_config(model_name):
    """
    Get model-specific instruction configuration including special tokens and format.
    
    Args:
        model_name: Model name or path
        
    Returns:
        Dict with instruction format configuration
    """
    model_name_lower = model_name.lower()
    
    # LLaMA 3 models
    if "llama-3" in model_name_lower or "llama3" in model_name_lower:
        if "instruct" in model_name_lower:
            return {
                "format_type": "llama3_instruct",
                "special_tokens": ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
                "response_marker": "<|start_header_id|>assistant<|end_header_id|>\n\n",
                "system_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>",
                "user_template": "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>",
                "assistant_template": "<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>",
                "stop_tokens": ["<|eot_id|>"]
            }
        else:
            # Base LLaMA 3 - use custom format
            return {
                "format_type": "custom",
                "special_tokens": ["<|response|>"],
                "response_marker": "<|response|>",
                "template": "{instruction}\n\n{input}\n\n<|response|>{output}",
                "stop_tokens": []
            }
    
    # LLaMA 2 models  
    elif "llama-2" in model_name_lower or "llama2" in model_name_lower:
        if "chat" in model_name_lower:
            return {
                "format_type": "llama2_chat",
                "special_tokens": ["<s>", "</s>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"],
                "response_marker": "[/INST] ",
                "system_template": "<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {assistant} </s>",
                "user_template": "<s>[INST] {user} [/INST] {assistant} </s>",
                "stop_tokens": ["</s>"]
            }
        else:
            # Base LLaMA 2 - use custom format
            return {
                "format_type": "custom", 
                "special_tokens": ["<|response|>"],
                "response_marker": "<|response|>",
                "template": "{instruction}\n\n{input}\n\n<|response|>{output}",
                "stop_tokens": []
            }
    
    # Mistral models
    elif "mistral" in model_name_lower:
        if "instruct" in model_name_lower:
            return {
                "format_type": "mistral_instruct",
                "special_tokens": ["<s>", "</s>", "[INST]", "[/INST]"],
                "response_marker": "[/INST] ",
                "template": "<s>[INST] {instruction}\n\n{input} [/INST] {output}</s>",
                "stop_tokens": ["</s>"]
            }
        else:
            # Base Mistral - use custom format
            return {
                "format_type": "custom",
                "special_tokens": ["<|response|>"],
                "response_marker": "<|response|>",
                "template": "{instruction}\n\n{input}\n\n<|response|>{output}",
                "stop_tokens": []
            }
    
    # Qwen models
    elif "qwen" in model_name_lower:
        if "instruct" in model_name_lower:
            return {
                "format_type": "qwen_instruct",
                "special_tokens": ["<|im_start|>", "<|im_end|>"],
                "response_marker": "<|im_start|>assistant\n",
                "system_template": "<|im_start|>system\n{system}<|im_end|>",
                "user_template": "<|im_start|>user\n{user}<|im_end|>",
                "assistant_template": "<|im_start|>assistant\n{assistant}<|im_end|>",
                "stop_tokens": ["<|im_end|>"]
            }
        else:
            # Base Qwen - use custom format
            return {
                "format_type": "custom",
                "special_tokens": ["<|response|>"],
                "response_marker": "<|response|>",
                "template": "{instruction}\n\n{input}\n\n<|response|>{output}",
                "stop_tokens": []
            }
    
    # Default for unknown models
    else:
        return {
            "format_type": "custom",
            "special_tokens": ["<|response|>"],
            "response_marker": "<|response|>",
            "template": "{instruction}\n\n{input}\n\n<|response|>{output}",
            "stop_tokens": []
        }

def format_instruction_with_model_config(instruction, input_text, output_text, model_config):
    """
    Format instruction using model-specific configuration FOR TRAINING (expects output_text).
    
    Args:
        instruction: Instruction text
        input_text: Input text
        output_text: Expected output text
        model_config: Model configuration from get_model_instruction_config
        
    Returns:
        Formatted text string
    """
    format_type = model_config["format_type"]
    
    if format_type == "llama3_instruct":
        system_msg = instruction
        user_msg = input_text
        assistant_msg = output_text
        
        # This simplified Llama3 formatting was direct and correct for its turn structure:
        # Corrected f-string formatting for special token access
        bos_token = model_config.get('special_tokens', ['<|begin_of_text|>'])[0]
        return (f"{bos_token}"
                f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_msg}<|eot_id|>")

    elif format_type == "llama2_chat":
        if instruction:
            return model_config["system_template"].format(system=instruction, user=input_text, assistant=output_text)
        else:
            return model_config["user_template"].format(user=input_text, assistant=output_text)
    
    elif format_type == "mistral_instruct":
        return model_config["template"].format(instruction=instruction, input=input_text, output=output_text)
    
    elif format_type == "qwen_instruct":
        system_msg = instruction
        user_msg = input_text
        assistant_msg = output_text
        
        return (f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant_msg}<|im_end|>")
    
    elif format_type == "custom":
        return model_config["template"].format(
            instruction=instruction,
            input=input_text,
            output=output_text
        )
    else: # Fallback
        return f"{instruction}\n\n{input_text}\n\n{model_config.get('response_marker', '<|response|>')}{output_text}"

def build_prompt_for_inference(instruction: str, user_input_text: str, model_config: dict) -> str:
    """
    Builds the prompt string for inference, ending where the model should start generating.
    """
    format_type = model_config["format_type"]

    if format_type == "llama3_instruct":
        # Prompt should end with the assistant header, ready for generation
        # Corrected f-string formatting for special token access
        bos_token = model_config.get('special_tokens', ['<|begin_of_text|>'])[0]
        return (f"{bos_token}"
                f"<|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{user_input_text}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n")

    elif format_type == "llama2_chat":
        # Prompt should end with [/INST]
        if instruction: # System prompt provided
            prompt = model_config["system_template"]
            prompt = prompt.replace("{system}", instruction)
            prompt = prompt.replace("{user}", user_input_text)
            # Remove the assistant part and EOS token from the template for inference
            prompt = prompt.split(model_config["response_marker"])[0] + model_config["response_marker"]
            return prompt.strip() # Ensure it ends cleanly with marker
        else: # No system prompt
            prompt = model_config["user_template"]
            prompt = prompt.replace("{user}", user_input_text)
            prompt = prompt.split(model_config["response_marker"])[0] + model_config["response_marker"]
            return prompt.strip()

    elif format_type == "mistral_instruct":
        # Template is: "<s>[INST] {instruction}\\n\\n{input} [/INST] {output}</s>"
        # Prompt should end with [/INST]
        prompt_template_parts = model_config["template"].split("{output}")
        prompt_part = prompt_template_parts[0]
        # If the template uses {instruction} and {input} separately vs combined
        if "{instruction}" in prompt_part and "{input}" in prompt_part:
             return prompt_part.format(instruction=instruction, input=user_input_text)
        else: # Assuming template expects a combined instruction/input if not separate
             combined_instruction = f"{instruction}\\n\\n{user_input_text}"
             # This part needs careful handling if keys don't match
             # For safety, let's assume the template will have {instruction} and we combine user_input_text into it for mistral
             return prompt_part.format(instruction=combined_instruction)

    elif format_type == "qwen_instruct":
        # Prompt should end with the assistant header, ready for generation
        return (f"<|im_start|>system\n{instruction}<|im_end|>\n"
                f"<|im_start|>user\n{user_input_text}<|im_end|>\n"
                f"<|im_start|>assistant\n")

    elif format_type == "custom":
        # Template is "{instruction}\\n\\n{input}\\n\\n<|response|>{output}"
        # Prompt should end with the response marker
        prompt_template_parts = model_config["template"].split("{output}")
        prompt_part = prompt_template_parts[0] # This includes the response marker
        return prompt_part.format(instruction=instruction, input=user_input_text)
        
    else: # Fallback - attempt to use response marker
        return f"{instruction}\n\n{user_input_text}\n\n{model_config.get('response_marker', '<|response|>')}"


#############################
# 3. Model Setup
#############################
def get_target_modules_for_model(model_name):
    """Return appropriate target modules based on model architecture."""
    model_name = model_name.lower()
    
    if any(name in model_name for name in ["llama", "mistral", "qwen"]):
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "phi" in model_name:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]  # Updated for newer Phi models
    elif "falcon" in model_name:
        return ["query_key_value", "dense"]
    else:
        # Default to attention modules
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    
def resolve_model_name(model_name_or_path):
    """
    Resolve model shortcuts to full HuggingFace model names.
    
    Args:
        model_name_or_path: Model name, path, or shortcut
        
    Returns:
        Full model name/path
    """
    shortcuts = {
        "llama3_8b_instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "llama2_7b_chat": "meta-llama/Llama-2-7b-chat-hf",
        "mistral_7b_instruct": "mistralai/Mistral-7B-Instruct-v0.3",
        "qwen2_5_7b_instruct": "Qwen/Qwen2.5-7B-Instruct",
    }
    
    return shortcuts.get(model_name_or_path.lower(), model_name_or_path)

def setup_model_and_tokenizer(model_args):
    """Set up the model and tokenizer with the appropriate configuration."""
    # If model_path is provided, use it as model_name_or_path
    if hasattr(model_args, "model_path") and model_args.model_path:
        model_args.model_name_or_path = model_args.model_path
    else:
        # Resolve model shortcuts only if model_path is not provided
        resolved_model_name = resolve_model_name(model_args.model_name_or_path)
        model_args.model_name_or_path = resolved_model_name
    
    # Get model-specific instruction configuration
    model_config = get_model_instruction_config(model_args.model_name_or_path)
    
    print(f"Loading model: {model_args.model_name_or_path}")
    print(f"Instruction format: {model_config['format_type']}")
    print(f"Response marker: {model_config['response_marker']}")
    
    # Load the tokenizer
    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        padding_side="right",
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )
    
    # Make sure the tokenizer has padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token = " "
    
    # For Llama3, ensure special tokens are properly configured
    if "llama-3" in model_args.model_name_or_path.lower() or "llama3" in model_args.model_name_or_path.lower():
        # Ensure Llama3 special tokens are properly recognized
        special_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
        
        # Check if special tokens are already in vocabulary
        missing_tokens = []
        for token in special_tokens:
            if token not in tokenizer.get_vocab():
                missing_tokens.append(token)
        
        if missing_tokens:
            print(f"Warning: Missing special tokens in tokenizer vocabulary: {missing_tokens}")
            # Add missing special tokens
            tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens})
            print(f"Added missing special tokens to tokenizer")
        else:
            print("All Llama3 special tokens are properly configured in tokenizer")
    
    # Configure quantization
    quantization_config = None  # Only create if quantization is actually requested
    if model_args.load_in_8bit or model_args.load_in_4bit:
        # Use bfloat16 if supported, fallback to float16
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Using compute dtype: {compute_dtype}")
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=model_args.load_in_8bit and not model_args.load_in_4bit,  # Prioritize 4bit over 8bit
            load_in_4bit=model_args.load_in_4bit,
            llm_int8_has_fp16_weight=True,  # ✅ FIXED: LLaMA models have fp16 weights
            llm_int8_enable_fp32_cpu_offload=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        print(f"Using quantization: 8bit={model_args.load_in_8bit and not model_args.load_in_4bit}, 4bit={model_args.load_in_4bit}")
    else:
        print("No quantization - loading in full precision")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        low_cpu_mem_usage=True,  # Reduce CPU memory usage
        offload_folder="./offload" if model_args.use_cpu_offload else None,
    )

    # Resize model embeddings if special tokens were added
    if "llama-3" in model_args.model_name_or_path.lower() or "llama3" in model_args.model_name_or_path.lower():
        # Check if we need to resize embeddings
        if len(tokenizer) > model.config.vocab_size:
            print(f"Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))

    # Only enable gradient checkpointing for training, not inference
    if model_args.use_lora:  
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing for training")
    
    # Set up LoRA if specified
    if model_args.use_lora:
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)  

        target_modules = get_target_modules_for_model(model_args.model_name_or_path)
        
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer, model_config

def setup_finetuned_model_for_evaluation(checkpoint_path: str, model_args):
    """
    Set up a fine-tuned model and tokenizer specifically for evaluation.
    Handles both full fine-tuned models and LoRA checkpoints.
    """
    print(f"Loading fine-tuned model for evaluation from: {checkpoint_path}")
    
    # Check if it's a LoRA checkpoint
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    adapter_model_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
    is_lora = os.path.exists(adapter_config_path) and os.path.exists(adapter_model_path)
    
    if is_lora:
        print("Detected LoRA checkpoint - loading base model + adapter")
        
        # Load adapter config to get base model info
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        base_model_name = adapter_config["base_model_name_or_path"]
        print(f"Base model: {base_model_name}")
        
        # Configure quantization if needed
        quantization_config = None
        if model_args.load_in_8bit or model_args.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=model_args.load_in_8bit and not model_args.load_in_4bit,
                load_in_4bit=model_args.load_in_4bit,
                llm_int8_has_fp16_weight=True,
                llm_int8_enable_fp32_cpu_offload=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print(f"Using quantization: 8bit={model_args.load_in_8bit and not model_args.load_in_4bit}, 4bit={model_args.load_in_4bit}")
        
        # Load base model
        print(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
            low_cpu_mem_usage=True,
            offload_folder="./offload" if model_args.use_cpu_offload else None,
        )
        
        # Load LoRA adapter
        print(f"Loading LoRA adapter from: {checkpoint_path}")
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        print("LoRA model loaded successfully")
        
        # Use base model name for tokenizer and config
        model_name_for_config = base_model_name
        
    else:
        raise ValueError(f"Invalid model path: {checkpoint_path}")
    
    # Get model-specific instruction configuration
    model_config = get_model_instruction_config(model_name_for_config)
    
    print(f"Instruction format: {model_config['format_type']}")
    print(f"Response marker: {model_config['response_marker']}")
    
    # Load the tokenizer
    tokenizer_name = model_args.tokenizer_name or model_name_for_config
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        padding_side="right",
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )
    
    # Make sure the tokenizer has padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token = " "
    
    # For Llama3, ensure special tokens are properly configured
    if "llama-3" in model_name_for_config.lower() or "llama3" in model_name_for_config.lower():
        special_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
        
        missing_tokens = []
        for token in special_tokens:
            if token not in tokenizer.get_vocab():
                missing_tokens.append(token)
        
        if missing_tokens:
            print(f"Warning: Missing special tokens in tokenizer vocabulary: {missing_tokens}")
            tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens})
            print(f"Added missing special tokens to tokenizer")
        else:
            print("All Llama3 special tokens are properly configured in tokenizer")
    
    # Resize model embeddings if special tokens were added
    if "llama-3" in model_name_for_config.lower() or "llama3" in model_name_for_config.lower():
        if len(tokenizer) > model.config.vocab_size:
            print(f"Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))
    
    print("Fine-tuned model loaded successfully for evaluation")
    return model, tokenizer, model_config

#############################
# 4. Evaluation Metrics & Processing
#############################

def extract_json_from_text(text: str) -> str:
    """Extract the largest, first valid JSON object from text."""
    # Pattern for the largest JSON object (outermost curly braces)
    pattern_block = r"\{[\s\S]*\}"
    match_block = re.search(pattern_block, text)
    if match_block:
        json_str = match_block.group(0).strip()
        # Try to parse the JSON string
        try:
            parsed_json = json.loads(json_str)
            return parsed_json
        except json.JSONDecodeError:
            # If no JSON block, try to fix the text by adding/removing quotes
            text_fixed = fix_llm_json_quotes(text)
            match_block = re.search(pattern_block, text_fixed)
            if match_block:
                json_str = match_block.group(0).strip()
                try:
                    parsed_json = json.loads(json_str)
                    return parsed_json
                except json.JSONDecodeError:
                    pass

    first_brace_index = -1
    # Find the first occurrence of '{' that is likely the start of a JSON object.
    first_brace_index = text.find('{')

    if first_brace_index == -1:
        ValueError("No valid JSON object found")

    # Add closing brace if not present
    if text[-1] != '}':
        text += '}'

    # Iterate from the end of the string to find a '}'
    # and try to parse the substring from first_brace_index to this '}'
    # We want the longest valid JSON starting at first_brace_index.
    
    # Iterate from the end of the string (longest potential match first)
    for i in range(len(text) - 1, first_brace_index, -1):
        if text[i] == '}':
            substring_to_try = text[first_brace_index : i + 1]
            try:
                parsed_json = json.loads(substring_to_try)
                return parsed_json  # First one that works is the largest valid
            except json.JSONDecodeError:
                continue
    
    raise ValueError("No valid JSON object found")


def postprocess_qed_response(raw_generated_text: str, question: str, context: str) -> dict:
    """
    Extracts JSON from raw model output and adds character offsets.
    """
    try:
        parsed_json = extract_json_from_text(raw_generated_text)
        # parsed_json = robust_json_loads(json_str) if json_str else {}
    except json.JSONDecodeError:
        message = f"Error: Could not parse JSON from model output: {raw_generated_text[:100]}..."
        print(message)
        raise ValueError(message)

    # Add offsets - this function needs to be robust to missing keys in parsed_json
    processed_prediction = add_offsets_to_prediction(parsed_json, question, context)
    return processed_prediction


class QEDEvaluator:
    """Evaluate QED predictions against ground truth."""
    
    def __init__(self, tokenizer=None, strict=False, debug=False):
        # Metrics
        self.exact_match_metric = evaluate.load("exact_match")
        self.rouge_metric = evaluate.load("rouge")
        self.tokenizer = tokenizer
        self.debug = debug
        self.strict = strict
       
    def set_tokenizer(self, tokenizer):
        """Set the tokenizer after initialization if needed."""
        self.tokenizer = tokenizer

    def normalize_text(self, text):
        """Normalize text by converting to lowercase and normalizing whitespace."""
        if not isinstance(text, str):
            return ""
        # Convert to lowercase and normalize whitespace
        return " ".join(text.lower().split())
    
    def compute_metrics(self, eval_preds):
        """
        Compute QED metrics using the official evaluator with proper QEDExample objects.
        Unpack tuple metrics for TensorBoard logging. Return empty dict on error.
        """
        try:
            if self.tokenizer is None:
                raise ValueError("Tokenizer must be set before computing metrics")
            predictions, labels = eval_preds
            original_examples = getattr(self, '_current_original_examples', None)
            if original_examples is None:
                print("Warning: No original examples available for evaluation")
                return {}
            model_config = getattr(self, 'model_config', None)
            if model_config is None:
                model_config = {
                    "format_type": "custom",
                    "response_marker": "<|response|>",
                    "stop_tokens": []
                }
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            if hasattr(predictions, 'numpy'):
                predictions = predictions.numpy()
            if predictions.ndim == 3:
                predictions = np.argmax(predictions, axis=-1)
            predictions = np.array(predictions)
            labels = np.array(labels)
            model_outputs = []
            response_marker = model_config["response_marker"]
            format_type = model_config.get("format_type", "custom")
            
            for i in range(min(len(original_examples), predictions.shape[0])):
                pred_seq = predictions[i]
                
                # Filter out invalid token IDs to prevent OverflowError
                vocab_size = len(self.tokenizer)
                valid_mask = (pred_seq >= 0) & (pred_seq < vocab_size)
                pred_seq = pred_seq[valid_mask]
                
                # Skip if no valid tokens
                if len(pred_seq) == 0:
                    model_outputs.append("")
                    continue
                
                # First try decoding without skipping special tokens to preserve structure
                try:
                    decoded_pred = self.tokenizer.decode(pred_seq, skip_special_tokens=False)
                    # Remove end of text token
                    end_token = self.tokenizer.eos_token
                    decoded_pred = decoded_pred.replace(end_token, "")
                    model_outputs.append(decoded_pred)
                except (OverflowError, ValueError) as e:
                    print(f"Warning: Could not decode prediction {i}: {e}")
                    continue
                
            annotation_dict = {}
            prediction_dict = {}
            for i, (original_example, model_output) in enumerate(zip(original_examples, model_outputs)):
                example_id = original_example.get("example_id", i)
                try:
                    annotation_dict[example_id] = load_single_line(original_example)
                except Exception as e:
                    print(f"Error creating ground truth QEDExample for {example_id}: {e}")
                    continue
                try:
                    prediction_example = copy.deepcopy(original_example)
                    parsed_pred = extract_json_from_text(model_output)
                    question = original_example["question_text"]
                    context = original_example["paragraph_text"]
                    parsed_pred_with_offsets = add_offsets_to_prediction(parsed_pred, question, context)
                    qed_annotation = self.convert_prediction_to_qed_annotation(parsed_pred_with_offsets)
                    prediction_example["annotation"] = qed_annotation
                    prediction_dict[example_id] = load_single_line(prediction_example)
                except Exception as e:
                    print(f"Error processing prediction for example {example_id}: {e}")
                    continue
            if len(prediction_dict) == 0:
                return {}
            official_metrics = qed_eval.compute_scores(annotation_dict, prediction_dict, self.strict)
            # Unpack tuple metrics for TensorBoard
            unpacked_metrics = {}
            for key, value in official_metrics.items():
                if isinstance(value, tuple):
                    unpacked_metrics[f"{key}_precision"] = value[0]
                    unpacked_metrics[f"{key}_recall"] = value[1]
                    unpacked_metrics[f"{key}_f1"] = value[2]
                else:
                    unpacked_metrics[key] = value
            unpacked_metrics["valid_num_examples"] = len(prediction_dict)
            return unpacked_metrics
        except Exception as e:
            print(f"Error in compute_metrics: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def convert_prediction_to_qed_annotation(self, prediction):
        """
        Convert model prediction to QED annotation format.
        """
        annotation = {
            "referential_equalities": [],
            "answer": [],
            "explanation_type": "single_sentence",
            "selected_sentence": {"start": 0, "end": 0, "string": ""}
        }
        
        # Convert referential equalities
        if "referential_equalities" in prediction:
            for ref_eq in prediction["referential_equalities"]:
                qed_ref_eq = {"question_reference": {}, "sentence_reference": {}}
                
                # Question reference
                if "question_reference" in ref_eq:
                    q_ref = ref_eq["question_reference"]
                    if isinstance(q_ref, dict) and "string" in q_ref:
                        qed_ref_eq["question_reference"] = {
                            "start": q_ref.get("start", 0),
                            "end": q_ref.get("end", len(q_ref["string"])),
                            "string": q_ref["string"]
                        }
                    elif isinstance(q_ref, str):
                        qed_ref_eq["question_reference"] = {
                            "start": 0,
                            "end": len(q_ref),
                            "string": q_ref
                        }
                
                # Sentence reference
                if "sentence_reference" in ref_eq:
                    s_ref = ref_eq["sentence_reference"]
                    if isinstance(s_ref, dict) and "string" in s_ref:
                        qed_ref_eq["sentence_reference"] = {
                            "start": s_ref.get("start", 0),
                            "end": s_ref.get("end", len(s_ref["string"])),
                            "string": s_ref["string"],
                            "bridge": s_ref.get("bridge", False)
                        }
                    elif isinstance(s_ref, str):
                        qed_ref_eq["sentence_reference"] = {
                            "start": 0,
                            "end": len(s_ref),
                            "string": s_ref,
                            "bridge": ref_eq.get("bridge", False)
                        }
                
                annotation["referential_equalities"].append(qed_ref_eq)

        # Convert answer
        if "answer" in prediction:
            answer = prediction["answer"]
            if isinstance(answer, dict) and "string" in answer:
                annotation["answer"] = [{
                    "sentence_reference": {
                        "start": answer.get("start", 0),
                        "end": answer.get("end", len(answer["string"])),
                        "string": answer["string"],
                        "bridge": False
                    },
                    "paragraph_reference": {
                        "start": answer.get("start", 0),
                        "end": answer.get("end", len(answer["string"])),
                        "string": answer["string"]
                    }
                }]
            elif isinstance(answer, str) and answer:
                annotation["answer"] = [{
                    "sentence_reference": {
                        "start": 0,
                        "end": len(answer),
                        "string": answer,
                        "bridge": False
                    },
                    "paragraph_reference": {
                        "start": 0,
                        "end": len(answer),
                        "string": answer
                    }
                }]
        
        # Convert selected sentence
        if "selected_sentence" in prediction:
            sel_sent = prediction["selected_sentence"]
            if isinstance(sel_sent, dict) and "string" in sel_sent:
                annotation["selected_sentence"] = {
                    "start": sel_sent.get("start", 0),
                    "end": sel_sent.get("end", len(sel_sent["string"])),
                    "string": sel_sent["string"]
                }
            elif isinstance(sel_sent, str):
                annotation["selected_sentence"] = {
                    "start": 0,
                    "end": len(sel_sent),
                    "string": sel_sent
                }
        
        return annotation
    
    def evaluate_predictions(self, examples: List[Dict], model, tokenizer, data_args: DataArguments, model_config_override: Optional[Dict] = None, gen_kwargs=None, output_dir=None):
        """
        Run comprehensive evaluation with official metrics on a set of examples.
        `examples` is a list of dicts, each containing:
        'example_id', 'instruction_text', 'user_input_text', 
        'ground_truth_annotation' (the dict), 
        'original_question_text', 'original_paragraph_text'
        """
        annotation_dict: Dict[Any, QEDExample] = {}  # For ground truths
        prediction_dict: Dict[Any, QEDExample] = {}  # For predictions
        all_processed_predictions: List[Dict] = [] # To store final JSON predictions
        problematic_predictions = []

        current_model_config = model_config_override
        if current_model_config is None:
            current_model_config = getattr(self, 'model_config', None)
        if current_model_config is None:
            current_model_config = get_model_instruction_config(model.config.name_or_path)
            if self.debug:
                print("Debug: model_config for evaluate_predictions derived from model name.")
        
        if self.tokenizer is None:
            print("Warning: Tokenizer not set in QEDEvaluator. Setting it now.")
            self.set_tokenizer(tokenizer)

        save_every = data_args.save_every
        all_preds_path = os.path.join(output_dir, "intermediate_predictions.jsonlines")
        problematic_path = os.path.join(output_dir, "intermediate_problematic.jsonlines")

        for idx, eval_sample in enumerate(tqdm(examples, desc="Evaluating predictions with model.generate()")):
            example_id = eval_sample["example_id"]
            
            # 1. Create ground truth QEDExample
            try:
                # Construct a temporary raw_example structure for create_qed_example
                gt_raw_example = {
                    "example_id": example_id,
                    "title_text": eval_sample["original_example_data"]["title_text"],
                    "original_nq_answers": eval_sample["original_example_data"]["original_nq_answers"],
                    "question_text": eval_sample["original_example_data"]["question_text"],
                    "paragraph_text": eval_sample["original_example_data"]["paragraph_text"],
                    "annotation": eval_sample["original_example_data"]["annotation"],
                }
                annotation_dict[example_id] = load_single_line(gt_raw_example)
            except Exception as e:
                print(f"Error creating ground truth QEDExample for {example_id}: {e}")
                # Add it to the problematic predictions
                problematic_predictions.append({
                    "example_id": example_id,
                    "error": str(e),
                    "ground_truth_annotation": eval_sample["ground_truth_annotation"]
                })
                continue

            # 2. Build prompt for inference
            prompt_text = build_prompt_for_inference(
                eval_sample["instruction_text"],
                eval_sample["user_input_text"],
                current_model_config
            )
            # Print prompt_text just for the first example
            if idx == 0:
                print("Prompt text Example:", flush=True)
                print(prompt_text, flush=True)
            
            # 3. Tokenize and Generate model output
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=data_args.max_source_length).to(model.device)
            prompt_token_length = inputs["input_ids"].shape[1]
            
            # Use generation parameters similar to prediction_step
            if gen_kwargs is None:
                gen_kwargs = {
                    "max_new_tokens": data_args.max_target_length,
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id,
                    # Defaulting to more deterministic for this explicit eval, can be made configurable
                    "do_sample": False, 
                    "num_beams": 1,
                    "temperature": 1.0, # Explicitly set temperature to 1.0 for non-sampling
                    "top_p": 1.0, # Explicitly set top_p to 1.0 for non-sampling
                }
            else:
                gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
                gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

            # Check for prompt truncation before model.generate
            max_length = data_args.max_source_length 
            tokens = tokenizer(prompt_text, truncation=False, return_tensors=None)["input_ids"]
            tokens_trunc = tokenizer(prompt_text, truncation=True, max_length=max_length, return_tensors=None)["input_ids"]
            if len(tokens) > max_length:
                problematic_predictions.append({
                    "example_id": eval_sample["example_id"],
                    "error": f"Prompt truncated by {len(tokens) - max_length} tokens (original: {len(tokens)}, truncated: {len(tokens_trunc)})",
                })
                continue  

            # with torch.no_grad():
            raw_generated_output_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs
                )
            
            # Decode only newly generated tokens
            newly_generated_ids = raw_generated_output_ids[0][prompt_token_length:]
            raw_generated_text = tokenizer.decode(newly_generated_ids, skip_special_tokens=True)

            # 4. Post-process response
            generated_json_obj = {}
            try:
                generated_json_obj = postprocess_qed_response(
                    raw_generated_text,
                    eval_sample["original_example_data"]["question_text"],
                    eval_sample["original_example_data"]["paragraph_text"]
                )
                # 5. Create prediction QEDExample

                qed_annotation_for_pred = self.convert_prediction_to_qed_annotation(generated_json_obj)
                gt_raw_example["annotation"] = qed_annotation_for_pred
                prediction_dict[example_id] = load_single_line(gt_raw_example)
                all_processed_predictions.append(gt_raw_example)

            except Exception as e:
                print(e)
                # Save problematic prediction
                problematic_predictions.append({
                    "example_id": example_id,
                    "model_output": raw_generated_text,
                    "error": str(e),
                    "ground_truth_annotation": eval_sample["original_example_data"]
                })

            if (idx + 1) % save_every == 0:
                save_jsonlines(all_processed_predictions, all_preds_path)
                save_jsonlines(problematic_predictions, problematic_path)
        
        if not annotation_dict:
            print("Error: No valid ground truth examples to evaluate against.")
            return {"error": "No ground truth examples."}
        
        try:
            official_metrics = qed_eval.compute_scores(annotation_dict, prediction_dict, self.strict)
            # erase intermediate predictions and problematic predictions if file exists
            if os.path.exists(all_preds_path):
                os.remove(all_preds_path)
            if os.path.exists(problematic_path):
                os.remove(problematic_path)
            return {
                "metrics": official_metrics,
                "predictions_data": all_processed_predictions,
                "problematic_predictions": problematic_predictions
            }
        except Exception as e:
            print(f"Error calling qed_eval.compute_scores: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Failed to compute scores: {e}", "predictions_data": all_processed_predictions, "problematic_predictions": problematic_predictions}


#############################
# 5. Training Pipeline Data Preparation
#############################

def normalize_referential_equalities(ref_equalities):
    """
    Normalize the order of referential equalities to ensure consistent ordering.
    Sort by question_reference offsets (start position in the question).
    """
    if not ref_equalities:
        return ref_equalities
    
    def sort_key(ref_eq):
        
        # For the original QED format with offsets
        q_ref_dict = ref_eq.get("question_reference", {})
        if isinstance(q_ref_dict, dict) and "start" in q_ref_dict:
            start_offset = q_ref_dict["start"]
            # Secondary sort by sentence_reference start if available
            s_ref_dict = ref_eq.get("sentence_reference", {})
            if isinstance(s_ref_dict, dict) and "start" in s_ref_dict:
                s_start = s_ref_dict["start"]
            else:
                s_start = 0
            return (start_offset, s_start)
        
        # Fallback for any other format
        q_ref = str(ref_eq.get("question_reference", ""))
        s_ref = str(ref_eq.get("sentence_reference", ""))
        bridge = ref_eq.get("bridge", False)
        return (q_ref, s_ref, str(bridge))
    
    return sorted(ref_equalities, key=sort_key)

def preprocess_qed_example(example: dict, for_inference: bool = False, prompt_examples: str = "zero_shot", training_data: List[dict] = None, random_seed: int = None) -> dict:
    """
    Transforms a raw QED example into components for training or inference prompt construction.
    If not for_inference, it prepares ground truth target_output_text.
    Output: {
        "example_id": ...,
        "instruction_text": QED_INSTRUCTION_PROMPT_TEMPLATE,
        "user_input_text": "Question: ...\\nContext: ...",
        "target_output_text": "{...json...}" (if not for_inference, else None or empty),
        "original_example_data": example (original raw dict)
    }
    
    Args:
        example: Raw QED example dictionary
        for_inference: Whether this is for inference (True) or training (False)
        prompt_examples: Which examples to include ('zero_shot', 'first_only', 'second_only', 'both', 'random_one', 'random_two')
        training_data: List of training examples for random sampling (required for 'random_one' and 'random_two')
        random_seed: Random seed for reproducible sampling
    """
    example_id = example.get("example_id")
    if example_id is None:
        example_id = example.get("id", str(uuid.uuid4()))
    
    title = example.get('title_text', '')
    question = example['question_text']
    context = example['paragraph_text']
    user_input_text = f"Title: {title}\nQuestion: {question}\nContext: {context}"
    
    target_output_text = None
    if not for_inference:
        if 'annotation' in example and example['annotation']:
            annotation = example['annotation']
            selected_sentence = annotation['selected_sentence'].get('string', '') if 'selected_sentence' in annotation else ""
            ref_equalities = []
            if 'referential_equalities' in annotation:
                # First, collect with full offset information for sorting
                ref_equalities_with_offsets = []
                for ref_eq in annotation['referential_equalities']:
                    q_ref_dict = ref_eq['question_reference']
                    s_ref_dict = ref_eq['sentence_reference']
                    
                    ref_equalities_with_offsets.append({
                        "question_reference": q_ref_dict,
                        "sentence_reference": s_ref_dict,
                        "bridge": s_ref_dict.get('bridge', 'false')
                    })
                
                # Normalize the order using offsets
                normalized_with_offsets = normalize_referential_equalities(ref_equalities_with_offsets)
                
                # Convert to simplified format for training
                for ref_eq in normalized_with_offsets:
                    q_ref = ref_eq['question_reference'].get('string', '')
                    s_ref = ref_eq['sentence_reference'].get('string', '')
                    bridge = ref_eq['sentence_reference'].get('bridge', 'false')
                    ref_equalities.append({
                        "question_reference": q_ref,
                        "sentence_reference": s_ref,
                        "bridge": bridge
                    })
            
            answer = ""
            if 'answer' in annotation and len(annotation['answer']) > 0:
                answer = annotation['answer'][0]['sentence_reference'].get('string', '')
            
            output_json_obj = {
                "answer": answer,
                "selected_sentence": selected_sentence,
                "referential_equalities": ref_equalities
            }
            target_output_text = json.dumps(output_json_obj, indent=2)
        else:
            target_output_text = json.dumps({
                "answer": "", "selected_sentence": "", "referential_equalities": []
            }, indent=2)

    # Build instruction text based on prompt_examples configuration
    instruction_text = QED_INSTRUCTION_PROMPT_TEMPLATE
    
    if prompt_examples == "first_only":
        instruction_text += "\n" + QED_FIRST_EXAMPLE
    elif prompt_examples == "second_only":
        instruction_text += "\n" + QED_SECOND_EXAMPLE
    elif prompt_examples == "both":
        instruction_text += "\n" + QED_FIRST_EXAMPLE + "\n" + QED_SECOND_EXAMPLE
    elif prompt_examples == "random_one":
        # Generate cache path based on training data characteristics
        cache_path = get_training_data_cache_path(training_data)
        sampled_examples = sample_valid_qed_examples(training_data, num_examples=1, random_seed=random_seed, cache_path=cache_path)
        if sampled_examples:
            instruction_text += "\n" + format_qed_example_for_prompt(sampled_examples[0], example_num=1)
        else:
            print("Warning: No valid examples found for random sampling, using zero-shot")
    elif prompt_examples == "random_two":
        # Generate cache path based on training data characteristics
        cache_path = get_training_data_cache_path(training_data)
        sampled_examples = sample_valid_qed_examples(training_data, num_examples=2, random_seed=random_seed, cache_path=cache_path)
        if len(sampled_examples) >= 2:
            instruction_text += "\n" + format_qed_example_for_prompt(sampled_examples[0], example_num=1)
            instruction_text += "\n" + format_qed_example_for_prompt(sampled_examples[1], example_num=2)
        elif len(sampled_examples) == 1:
            instruction_text += "\n" + format_qed_example_for_prompt(sampled_examples[0], example_num=1)
            print("Warning: Only 1 valid example found for random sampling, using one-shot")
        else:
            print("Warning: No valid examples found for random sampling, using zero-shot")
    # For "zero_shot", we just use the base instruction template

    return {
        "example_id": example_id,
        "instruction_text": instruction_text,
        "user_input_text": user_input_text,
        "target_output_text": target_output_text, # Will be None if for_inference=True
        "original_example_data": copy.deepcopy(example) # Store original for GT construction
    }


def format_instruction_dataset(examples, tokenizer, data_args, model_config, include_original_examples=True, training_data_for_sampling=None, output_dir=None):
    """Format the dataset for instruction tuning with model-specific formats."""
    
    formatted_examples_for_dataset = []
    original_examples_for_eval = [] # If needed by evaluator later

    for i, example_data in enumerate(tqdm(examples, desc="Formatting examples for training")):
        # Get prompt_examples from data_args
        prompt_examples = getattr(data_args, 'prompt_examples', 'zero_shot')
        
        # Get components: instruction, input_text, target_output_text
        # Pass training data if random sampling is requested
        processed_components = preprocess_qed_example(
            example_data, 
            for_inference=False, 
            prompt_examples=prompt_examples,
            training_data=training_data_for_sampling if prompt_examples in ['random_one', 'random_two'] else None,
            random_seed=42 + i  # Use different seed for each example to get variety
        )
        
        # Format for training: instruction + input + output
        full_training_text = format_instruction_with_model_config(
            processed_components["instruction_text"],
            processed_components["user_input_text"],
            processed_components["target_output_text"],
            model_config
        )
        
        item = {"text": full_training_text}
        if include_original_examples: # This is for the QEDEvaluator's compute_metrics during training
            item["original_example"] = json.dumps(processed_components["original_example_data"])
        
        formatted_examples_for_dataset.append(item)
        # original_examples_for_eval.append(processed_components["original_example_data"])

    dataset_dict = {"text": [ex["text"] for ex in formatted_examples_for_dataset]}
    if include_original_examples:
        dataset_dict["original_example"] = [ex["original_example"] for ex in formatted_examples_for_dataset]
        
    dataset = Dataset.from_dict(dataset_dict)
    
    # DEBUG: Save sample of formatted training data BEFORE tokenization
    if output_dir:
        debug_save_training_sample(
            formatted_examples_for_dataset,
            output_dir, 
            num_samples=5
        )
    
    # Define tokenization function
    def tokenize_function(examples):
        # Tokenize the formatted text directly
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=data_args.max_source_length + data_args.max_target_length
        )
        
        # Prepare input_ids and labels for causal language modeling
        labels = []
        response_marker = model_config["response_marker"]
        
        for i, tokenized_text in enumerate(tokenized["input_ids"]):
            # Find the response marker to determine boundary
            full_text = examples["text"][i]
            
            # Split on response marker to find instruction part
            if response_marker in full_text:
                instruction_part = full_text.split(response_marker)[0] + response_marker
            else:
                raise ValueError(f"Response marker not found in full_text: {full_text}")
            
            # Tokenize the instruction part with the SAME settings as the full text
            instruction_tokens = tokenizer(
                instruction_part, 
                add_special_tokens=False,  # Don't add BOS again
                padding=False,
                truncation=False
            )["input_ids"]
            
            # Get the full tokenized sequence
            full_text_tokens = tokenized_text
            
            # The instruction length should match the tokenized instruction part
            instruction_input_length = len(instruction_tokens)
            
            # Account for BOS token if present in the full sequence but not in instruction
            if (len(full_text_tokens) > 0 and 
                tokenizer.bos_token_id is not None and 
                full_text_tokens[0] == tokenizer.bos_token_id and
                (len(instruction_tokens) == 0 or instruction_tokens[0] != tokenizer.bos_token_id)):
                instruction_input_length += 1
            
            # Make sure we don't go beyond the length of tokenized_text
            instruction_input_length = min(instruction_input_length, len(tokenized_text))
            
            # Set -100 for instruction and input tokens, actual ids for output tokens
            example_labels = [-100] * instruction_input_length + tokenized_text[instruction_input_length:]
            
            # Make sure labels have exactly the same length as input_ids
            if len(example_labels) < len(tokenized_text):
                example_labels = example_labels + [-100] * (len(tokenized_text) - len(example_labels))
            elif len(example_labels) > len(tokenized_text):
                example_labels = example_labels[:len(tokenized_text)]
            
            labels.append(example_labels)
        
        tokenized["labels"] = labels
        return tokenized
    
    # Apply tokenization - remove text columns, keep original_example only if included
    columns_to_remove = ["text"]
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove,
        desc="Tokenizing examples"
    )
    
    # DEBUG: Save tokenization debug info
    if output_dir:
        debug_save_tokenization_sample(
            tokenized_dataset, 
            tokenizer, 
            model_config, 
            output_dir, 
            num_samples=3
        )
    
    return tokenized_dataset

def debug_save_training_sample(formatted_examples, output_dir, num_samples=5):
    """Save a sample of formatted training data for debugging"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    debug_file = os.path.join(output_dir, "debug_training_sample.txt")
    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write("=== TRAINING DATA SAMPLE ===\n\n")
        for i, example in enumerate(formatted_examples[:num_samples]):
            f.write(f"Example {i+1}:\n")
            f.write("="*50 + "\n")
            f.write(example["text"])
            f.write("\n" + "="*50 + "\n\n")
    
    print(f"Debug: Saved training data sample to {debug_file}")

def debug_save_tokenization_sample(tokenized_dataset, tokenizer, model_config, output_dir, num_samples=3):
    """Save a sample of tokenized data for debugging"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    debug_file = os.path.join(output_dir, "debug_tokenization_sample.txt")
    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write("=== TOKENIZATION DEBUG SAMPLE ===\n\n")
        f.write(f"Response marker: {model_config['response_marker']}\n\n")
        
        for i in range(min(num_samples, len(tokenized_dataset))):
            f.write(f"Example {i+1}:\n")
            f.write("-" * 50 + "\n")
            
            input_ids = tokenized_dataset[i]["input_ids"]
            labels = tokenized_dataset[i]["labels"]
            
            # Decode full sequence
            full_decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
            f.write("Full decoded (with special tokens):\n")
            f.write(full_decoded)
            f.write("\n\n")
            
            # Show label masking
            f.write("Label masking (-100 = masked, others = target):\n")
            masked_count = sum(1 for label in labels if label == -100)
            target_count = len(labels) - masked_count
            f.write(f"Masked tokens: {masked_count}, Target tokens: {target_count}\n")
            
            # Show which tokens are masked vs target
            f.write("Token-by-token breakdown:\n")
            for j, (token_id, label) in enumerate(zip(input_ids, labels)):
                token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                status = "MASKED" if label == -100 else "TARGET"
                f.write(f"  {j:3d}: {token_id:5d} -> '{token_str}' [{status}]\n")
                if j > 50:  # Limit output
                    f.write("  ... (truncated)\n")
                    break
            
            f.write("\n" + "=" * 50 + "\n\n")

def train_qed_model(model_args, data_args, training_args):
    """Main training function."""
    
    # Set up model and tokenizer
    model, tokenizer, model_config = setup_model_and_tokenizer(model_args)
    
    # Load QED data
    train_data, validation_data = load_qed_data(data_args)
    
    # Extract original examples before tokenization for evaluation
    original_eval_examples = []
    for i, example in enumerate(validation_data):
        # Ensure we have an example_id
        if "example_id" not in example:
            example["example_id"] = i
        original_eval_examples.append(example)

    # Format datasets - exclude original examples from both datasets
    # Pass training data for random sampling (use train_data for both train and eval datasets)
    train_dataset = format_instruction_dataset(train_data, tokenizer, data_args, model_config, include_original_examples=False, training_data_for_sampling=train_data, output_dir=training_args.output_dir)
    eval_dataset = format_instruction_dataset(validation_data, tokenizer, data_args, model_config, include_original_examples=False, training_data_for_sampling=train_data, output_dir=None)  # Only debug train dataset
    
    # Set up data collator with memory optimization
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
        pad_to_multiple_of=8,
        return_tensors="pt",
        label_pad_token_id=-100
    )
    
    # Set up evaluator
    evaluator = QEDEvaluator(tokenizer=tokenizer, debug=True)  # Enable debug mode
    # Store model config in evaluator for response extraction
    evaluator.model_config = model_config

    # Set up loss tracking callback
    loss_callback = LossTrackingCallback(
        output_dir=os.path.join(training_args.output_dir, "training_plots"),
        plot_every_steps=50
    )

    # Set up trainer with memory optimizations
    trainer = QEDTrainer(
        original_eval_examples=original_eval_examples,
        data_args=data_args,
        loss_tracking_callback=loss_callback,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=evaluator.compute_metrics,
    )
    
    # Store model_config in trainer for debug purposes
    trainer.model_config = model_config
    
    # Train model
    train_result = trainer.train()
    
    # Save the final model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    return model, tokenizer

#############################
# 6. Training Pipeline
#############################

class QEDTrainer(Trainer):
    """Custom Trainer that passes original examples to compute_metrics"""
    
    def __init__(self, original_eval_examples=None, data_args=None, loss_tracking_callback=None, *args, **kwargs):
        # Store data_args separately since parent Trainer doesn't accept it
        self.data_args = data_args
        self.original_eval_examples = original_eval_examples
        self.loss_tracking_callback = loss_tracking_callback
        
        # Initialize parent
        super().__init__(*args, **kwargs)
        
        # Set up loss tracking callback if provided
        if self.loss_tracking_callback:
            self.add_callback(self.loss_tracking_callback)
    
    def get_tokenizer(self):
        """
        Get tokenizer safely, handling the deprecation of Trainer.tokenizer
        """
        if hasattr(self, 'processing_class') and self.processing_class is not None:
            return self.processing_class
        elif hasattr(self, 'tokenizer') and self.tokenizer is not None:
            return self.tokenizer
        else:
            raise ValueError("No tokenizer or processing_class found in trainer")

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Custom training step with debug analysis.
        """
        # Call parent training step
        result = super().training_step(model, inputs, num_items_in_batch)
        
        return result

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Custom evaluate that preserves original examples for compute_metrics with memory management
        """
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        # Pass original examples to the evaluator if available
        if self.original_eval_examples is not None and hasattr(self.compute_metrics, '__self__'):
            self.compute_metrics.__self__._current_original_examples = self.original_eval_examples
        
        # Clear GPU cache before evaluation to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory before eval: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        # Run evaluation with OOM handling
        try:
            result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        except torch.cuda.OutOfMemoryError as e:
            print("CUDA OOM during evaluation! Returning partial results (if any).")
            torch.cuda.empty_cache()
            # Optionally, you can return a partial result or a custom error dict
            result = {"eval_loss": None, "oom": True}

        return result
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Memory-optimized prediction step that processes sequences one at a time.
        This reduces GPU memory usage during evaluation.
        """
        if not self.args.predict_with_generate:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        
        # Use generation for evaluation
        model.eval()
        
        # Get input_ids and attention_mask
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask", None)
        labels = inputs.get("labels")
        
        if input_ids is None:
            raise ValueError("input_ids must be provided for generation")
        
        batch_size = input_ids.shape[0]
        
        # Get tokenizer safely
        tokenizer = self.get_tokenizer()
        
        # Pre-allocate tensor for final results (more memory efficient than list)
        max_gen_length = 512  # Set a reasonable max length
        predictions = torch.full(
            (batch_size, max_gen_length), 
            tokenizer.pad_token_id, 
            dtype=input_ids.dtype, 
            device=input_ids.device
        )
        
        # Process each sequence individually to minimize memory usage
        for i in range(batch_size):
            try:
                # Extract single sequence
                single_input_ids = input_ids[i:i+1]  # Keep batch dimension
                single_attention_mask = attention_mask[i:i+1] if attention_mask is not None else None
                single_labels = labels[i] if labels is not None else None
                
                # Find generation start position
                if single_labels is not None:
                    non_masked_positions = (single_labels != -100).nonzero(as_tuple=True)[0]
                    if len(non_masked_positions) > 0:
                        generation_start = non_masked_positions[0].item()
                        prompt_input_ids = single_input_ids[:, :generation_start]
                        prompt_attention_mask = single_attention_mask[:, :generation_start] if single_attention_mask is not None else None
                    else:
                        prompt_input_ids = single_input_ids
                        prompt_attention_mask = single_attention_mask
                else:
                    prompt_input_ids = single_input_ids
                    prompt_attention_mask = single_attention_mask
                
                # Memory-optimized generation parameters
                gen_kwargs = {
                    "max_new_tokens": 512,
                    "do_sample": False,
                    "num_beams": 1,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "repetition_penalty": 1.1,
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id,
                    "use_cache": False,  # Disable KV cache to save memory
                }
                
                # Generate with memory management
                with torch.no_grad():
                    try:
                        generated_ids = model.generate(
                            input_ids=prompt_input_ids,
                            attention_mask=prompt_attention_mask,
                            **gen_kwargs
                        )
                        
                        # Extract only newly generated tokens
                        generated_sequence = generated_ids[0][prompt_input_ids.shape[1]:]
                        
                        # Copy to pre-allocated tensor (truncate if necessary)
                        seq_len = min(len(generated_sequence), max_gen_length)
                        predictions[i, :seq_len] = generated_sequence[:seq_len]
                        
                        # Clear intermediate tensors immediately
                        del generated_ids
                        del generated_sequence
                        
                    except torch.cuda.OutOfMemoryError:
                        print(f"OOM during generation for sequence {i}, filling with pad tokens")
                        # Fill with pad tokens on OOM
                        predictions[i, :] = tokenizer.pad_token_id
                
                # Clear single sequence tensors
                del single_input_ids
                if single_attention_mask is not None:
                    del single_attention_mask
                del prompt_input_ids
                if prompt_attention_mask is not None:
                    del prompt_attention_mask
                    
            except Exception as e:
                print(f"Error processing sequence {i}: {e}")
                # Fill with pad tokens on any error
                predictions[i, :] = tokenizer.pad_token_id
        
        # Final memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Compute loss if needed (using teacher forcing)
        loss = None
        if not prediction_loss_only:
            try:
                with torch.no_grad():
                    outputs = model(**inputs)
                    loss = outputs.loss
                    del outputs  # Clear immediately
            except torch.cuda.OutOfMemoryError:
                print("OOM during loss computation, skipping loss calculation")
                loss = None
        
        return (loss, predictions, labels)



#############################
# 7. Full Evaluation Script
#############################

def model_generation_on_qed(model_args, data_args, output_path='qed_model_results', gen_kwargs=None, strict=False, model=None, tokenizer=None, model_config=None):
    """Evaluate the fine-tuned model on the QED validation set using official metrics."""
    
    model_path = model_args.model_name_or_path
    print(f"Starting evaluation for model: {model_path}")
    
    # Load model components if not provided
    if model is None or tokenizer is None or model_config is None:
        model, tokenizer, model_config = setup_model_and_tokenizer(model_args)
    
    model.eval()

    # Create output directory with prompt-specific subdirectory
    prompt_dir_name = getattr(data_args, 'prompt_examples', 'zero_shot')
    output_dir = os.path.join(output_path, model_path, prompt_dir_name)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for existing problematic predictions and filter validation data accordingly
    print("Checking for existing problematic predictions...")
    existing_problematic_ids = load_existing_problematic_predictions(output_dir)
    
    # Load training and validation data (raw)
    print("Loading training and validation data...")
    train_data_raw, validation_data_raw = load_qed_data(data_args)
    if not validation_data_raw:
        print("No validation data found. Exiting evaluation.")
        return {}
    print(f"Loaded {len(validation_data_raw)} raw validation examples.")

    # Filter out examples with only explanation_type in annotation
    filtered_validation_data = [
        ex for ex in validation_data_raw
        if has_valid_qed_annotation(ex.get("annotation", {}))
    ]
    print(f"Filtered validation data: {len(filtered_validation_data)} of {len(validation_data_raw)} examples kept (with valid QED annotation).")

    # Filter validation data if there are existing problematic predictions
    if existing_problematic_ids:
        print("Filtering validation data to include only problematic examples...")
        filtered_validation_data = filter_validation_data_by_problematic_ids(filtered_validation_data, existing_problematic_ids)
        if not filtered_validation_data:
            print("No validation examples match the existing problematic IDs. Exiting evaluation.")
            return {}

    # Get prompt_examples from data_args
    prompt_examples = getattr(data_args, 'prompt_examples', 'zero_shot')

    # Prepare evaluation samples in the new format
    print("Preparing evaluation samples...")
    evaluation_samples_for_evaluator = []
    for i, raw_example in enumerate(tqdm(filtered_validation_data, desc="Preprocessing validation data")):
        # Use for_inference=True if preprocess_qed_example supports it to avoid generating target_output_text
        # Assuming current preprocess_qed_example gives all necessary components
        processed_components = preprocess_qed_example(
            raw_example, 
            for_inference=True, 
            prompt_examples=prompt_examples,
            training_data=train_data_raw if prompt_examples in ['random_one', 'random_two'] else None,
            random_seed=42 + i  # Use different seed for each example to get variety
        )
        
        if "annotation" not in raw_example:
             print(f"Warning: Example {processed_components.get('example_id', '?')} missing ground truth annotation. Skipping.")
             continue

        evaluation_samples_for_evaluator.append({
            "example_id": processed_components["example_id"],
            "instruction_text": processed_components["instruction_text"],
            "user_input_text": processed_components["user_input_text"],
            "original_example_data": processed_components["original_example_data"] # For create_qed_example if it needs more
        })
    
    if not evaluation_samples_for_evaluator:
        print("No valid evaluation samples could be prepared. Exiting evaluation.")
        return {}
    print(f"Prepared {len(evaluation_samples_for_evaluator)} samples for the evaluator.")

    # Initialize QEDEvaluator
    print("Initializing QEDEvaluator...")
    evaluator = QEDEvaluator(tokenizer=tokenizer, strict=strict, debug=False) # Set debug as needed
    # Pass model_config to evaluator if it caches it, or it will be derived inside
    evaluator.model_config = model_config 

    # Run evaluation predictions
    print("Running evaluate_predictions...")
    eval_results_dict = evaluator.evaluate_predictions(
        evaluation_samples_for_evaluator, 
        model, 
        tokenizer, 
        data_args, # Pass data_args for max_target_length etc.
        model_config_override=model_config,# Pass model_config directly
        gen_kwargs=gen_kwargs,
        output_dir=output_dir
    )
        
    official_metrics = eval_results_dict.get("metrics", {})
    predictions_data = eval_results_dict.get("predictions_data", [])
    problematic_predictions = eval_results_dict.get("problematic_predictions", [])

    # Save detailed predictions data (append to existing file if it exists)
    if predictions_data:
        predictions_output_file = os.path.join(output_dir, "detailed_predictions.jsonlines")
        append_to_jsonlines(predictions_data, predictions_output_file)

    # Save problematic predictions (create new file)
    problematic_output_file = os.path.join(output_dir, "problematic_predictions.jsonlines")
    if problematic_predictions:
        save_jsonlines(problematic_predictions, problematic_output_file)
    else:
        # Check if problematic file exists and delete it
        if os.path.exists(problematic_output_file):
            os.remove(problematic_output_file)

    return official_metrics, predictions_output_file

def evaluate_predictions_with_custom_overlap(
    model_name: str,
    predictions_file_path: str,
    validation_data_path: str,
    output_dir: str,
    min_f1_overlap: float = 0.9,
    strict: bool = False
):
    """
    Evaluate already saved model predictions with custom overlap threshold.
    
    Args:
        predictions_file_path: Path to detailed_predictions.jsonlines file
        validation_data_path: Path to qed-dev.jsonlines (ground truth)
        output_dir: Directory to save evaluation results
        min_f1_overlap: Custom MIN_F1_FOR_NON_STRICT_OVERLAP value (default: 0.9)
        strict: Whether to use strict evaluation mode
    """
    
    # Import the qed_eval functions
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    
    print(f"Starting evaluation with MIN_F1_FOR_NON_STRICT_OVERLAP = {min_f1_overlap}")
    print(f"Strict mode: {strict}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Temporarily modify MIN_F1_FOR_NON_STRICT_OVERLAP
        original_min_f1 = MIN_F1_FOR_NON_STRICT_OVERLAP
        import qed_eval
        qed_eval.MIN_F1_FOR_NON_STRICT_OVERLAP = min_f1_overlap
        
        print(f"Modified MIN_F1_FOR_NON_STRICT_OVERLAP from {original_min_f1} to {min_f1_overlap}")
        
        # Load data
        print("Loading data with qed_eval...")
        annotation_dict = load_data(validation_data_path)
        prediction_dict = load_data(predictions_file_path)
        
        print(f"qed_eval loaded {len(annotation_dict)} annotations and {len(prediction_dict)} predictions")
        
        # Compute scores on valid examples only (current behavior)
        print("Computing scores on valid examples only...")
        valid_scores = compute_scores(annotation_dict, prediction_dict, strict)
        
        # Compute normalized scores by treating invalid examples as False
        print("Computing normalized scores (invalid examples as False)...")
        
        # Create dummy predictions for missing examples with empty results
        dummy_prediction = QEDExample(
            example_id=0,
            title="",
            question="",
            answer=[],
            nq_answers=[],
            aligned_nps=[],
            explanation_type="single_sentence"
        )
        
        # Expand prediction_dict to include all annotation examples
        expanded_prediction_dict = prediction_dict.copy()
        for example_id in annotation_dict:
            if example_id not in prediction_dict:
                # Create a dummy prediction with the correct example_id
                dummy_pred = QEDExample(
                    example_id=example_id,
                    title="",
                    question="",
                    answer=[],
                    nq_answers=[],
                    aligned_nps=[],
                    explanation_type="single_sentence"
                )
                expanded_prediction_dict[example_id] = dummy_pred
        
        # Compute scores with expanded predictions
        normalized_scores = compute_scores(annotation_dict, expanded_prediction_dict, strict)
        
        # Count invalid examples
        num_invalid_examples = len(annotation_dict) - len(prediction_dict)
        
        # Prepare results with both metric sets
        results = {
            "metrics": {
                "valid_only": valid_scores,
                "normalized": normalized_scores
            },
            "model_path": model_name,
            "total_number_of_examples": len(annotation_dict),
            "valid_number_of_examples": len(prediction_dict),
            "invalid_number_of_examples": num_invalid_examples,
            "strict_matching": strict,
            "filtered_by_problematic": False,
            "num_existing_problematic": 0,
            "min_f1_overlap": min_f1_overlap
        }
        
        # Save results
        overlap_str = f"overlap_{min_f1_overlap:.2f}"
        strict_str = "strict" if strict else "non_strict"
        results_filename = f"official_eval_results_{overlap_str}_{strict_str}.json"
        results_path = os.path.join(output_dir, results_filename)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_path}")
        
        # Print summary
        print("\n" + "="*70)
        print(f"EVALUATION SUMMARY (overlap={min_f1_overlap}, strict={strict})")
        print("="*70)
        print(f"Total examples: {len(annotation_dict)}")
        print(f"Valid examples: {len(prediction_dict)}")
        print(f"Invalid examples: {num_invalid_examples}")
        print()
        
        print("VALID-ONLY METRICS (computed on valid examples only):")
        print("-" * 50)
        for metric_name, metric_value in valid_scores.items():
            if isinstance(metric_value, tuple):
                print(f"{metric_name}: P={metric_value[0]:.4f}, R={metric_value[1]:.4f}, F1={metric_value[2]:.4f}")
            else:
                print(f"{metric_name}: {metric_value:.4f}")
        
        print()
        print("NORMALIZED METRICS (invalid examples counted as False):")
        print("-" * 50)
        for metric_name, metric_value in normalized_scores.items():
            if isinstance(metric_value, tuple):
                print(f"{metric_name}: P={metric_value[0]:.4f}, R={metric_value[1]:.4f}, F1={metric_value[2]:.4f}")
            else:
                print(f"{metric_name}: {metric_value:.4f}")
        
        print("="*70)
        
        return results
    except Exception as e:
        print(f"Exception during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    finally:
        # Restore original MIN_F1_FOR_NON_STRICT_OVERLAP
        qed_eval.MIN_F1_FOR_NON_STRICT_OVERLAP = original_min_f1


def batch_evaluate_with_different_overlaps(
    model_name: str,
    predictions_file_path: str,
    validation_data_path: str,
    overlap_values: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9],
    strict_values: List[bool] = [False]
):
    """
    Run evaluation with multiple overlap values and strict settings.
    
    Args:
        predictions_file_path: Path to detailed_predictions.jsonlines file
        validation_data_path: Path to qed-dev.jsonlines (ground truth)
        output_dir: Directory to save all evaluation results
        overlap_values: List of MIN_F1_FOR_NON_STRICT_OVERLAP values to test
        strict_values: List of strict mode settings to test
    """
    print(f"Starting batch evaluation with {len(overlap_values)} overlap values and {len(strict_values)} strict settings")
    
    all_results = {}
    output_dir = os.path.dirname(predictions_file_path) # dirctory name of predictions_file_path
    for strict in strict_values:
        for overlap in overlap_values:
            print(f"\n--- Evaluating with overlap={overlap}, strict={strict} ---")
            
            try:
                results = evaluate_predictions_with_custom_overlap(
                    model_name=model_name,
                    predictions_file_path=predictions_file_path,
                    validation_data_path=validation_data_path,
                    output_dir=output_dir,
                    min_f1_overlap=overlap,
                    strict=strict
                )
                
                key = f"overlap_{overlap:.2f}_strict_{strict}"
                all_results[key] = results
                
            except Exception as e:
                print(f"Error in evaluation with overlap={overlap}, strict={strict}: {e}")
                continue
    
    # Save summary of all results
    summary_path = os.path.join(output_dir, "batch_evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nBatch evaluation complete. Summary saved to {summary_path}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("BATCH EVALUATION COMPARISON")
    print("="*80)
    print(f"{'Setting':<25} {'Metric Type':<12} {'Exact Match':<12} {'Answer Acc':<12} {'Pair F1':<12} {'Mention F1':<12}")
    print("-"*80)
    
    for key, results in all_results.items():
        if results and "metrics" in results:
            # Handle new format with valid_only and normalized metrics
            if isinstance(results["metrics"], dict) and "valid_only" in results["metrics"]:
                # New format with both valid_only and normalized
                for metric_type in ["valid_only", "normalized"]:
                    metrics = results["metrics"][metric_type]
                    exact_match = metrics.get("exact_match_accuracy", 0)
                    answer_acc = metrics.get("answer_accuracy", 0)
                    pair_f1 = metrics.get("pair", (0, 0, 0))[2]  # F1 score
                    mention_f1 = metrics.get("all_mention", (0, 0, 0))[2]  # F1 score
                    
                    display_type = "Valid Only" if metric_type == "valid_only" else "Normalized"
                    print(f"{key:<25} {display_type:<12} {exact_match:<12.4f} {answer_acc:<12.4f} {pair_f1:<12.4f} {mention_f1:<12.4f}")
            else:
                # Old format - single metrics
                metrics = results["metrics"]
                exact_match = metrics.get("exact_match_accuracy", 0)
                answer_acc = metrics.get("answer_accuracy", 0)
                pair_f1 = metrics.get("pair", (0, 0, 0))[2]  # F1 score
                mention_f1 = metrics.get("all_mention", (0, 0, 0))[2]  # F1 score
            
                print(f"{key:<25} {'Single':<12} {exact_match:<12.4f} {answer_acc:<12.4f} {pair_f1:<12.4f} {mention_f1:<12.4f}")
    
    print("="*80)
    
    return all_results


