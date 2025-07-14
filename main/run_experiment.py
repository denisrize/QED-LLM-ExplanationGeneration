# run_fewshot_experiment.py
"""
Script to run a few-shot fine-tuning experiment for QED.
This uses a small subset of the training data to fine-tune the model.
"""

import os
import logging
import argparse
from project_code.config_models import get_few_shot_config
from qed_main import train_qed_model, model_generation_on_qed, batch_evaluate_with_different_overlaps, setup_finetuned_model_for_evaluation

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("qed_fewshot_experiment.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run few-shot fine-tuning for QED")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        choices=["llama2_7b_chat", "llama3_8b_instruct", "mistral_7b_instruct", "qwen2_5_7b_instruct"],
        help="Base model to use for fine-tuning. Use specific names for chat/instruct versions."
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Number of training examples to use"
    )
    parser.add_argument(
        "--max_eval_examples",
        type=int,
        default=None,
        help="Number of validation examples to use"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation on a previously fine-tuned model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a fine-tuned model checkpoint directory (required to evaluate finetuned models)"
    )
    # Add argument for prompt examples configuration
    parser.add_argument(
        "--prompt_examples",
        type=str,
        default="zero_shot",
        choices=["zero_shot", "first_only", "second_only", "both", "random_one", "random_two"],
        help="Which examples to include in the prompt: 'zero_shot' (no examples), 'first_only' (Life of Pi example), 'second_only' (hemolytic reaction example), 'both' (both examples)"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Log experiment configuration
    logger.info(f"Running few-shot experiment with the following configuration:")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Number of examples: {args.max_examples}")
    logger.info(f"  Number of validation examples: {args.max_eval_examples}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Prompt examples: {args.prompt_examples}")
    
    if args.eval_only:
                
        # Run evaluation
        model_args, data_args, _ = get_few_shot_config(
            model_name=args.model_name,
            model_path=args.model_path,
            max_eval_examples=args.max_eval_examples,
            prompt=args.prompt_examples
        )
        
        data_args.prompt_examples = args.prompt_examples
        data_args.save_every = 20
        gen_kwargs = {
            "do_sample": False, 
            "num_beams": 1,
            "top_p": 1.0,
            "temperature": 1.0,
            "max_new_tokens": data_args.max_target_length
        }
        output_path = "qed_model_results"

        if args.model_path:
            # Use the specialized function for loading fine-tuned models
            model, tokenizer, model_config = setup_finetuned_model_for_evaluation(args.model_path, model_args)
            
            # Update model_args with the actual model path for proper result organization
            model_args.model_name_or_path = args.model_path
            
            results, predictions_output_file = model_generation_on_qed(model_args, data_args, output_path, gen_kwargs, strict=False, model=model, tokenizer=tokenizer, model_config=model_config)
        else:
            results, predictions_output_file = model_generation_on_qed(model_args, data_args, output_path, gen_kwargs)
            
        results_by_threshold = batch_evaluate_with_different_overlaps(model_args.model_name_or_path, predictions_output_file, os.path.join(data_args.qed_data_path, data_args.validation_file))
        # Log evaluation results
        logger.info("Final evaluation results")

    else:
        # Get configuration for few-shot fine-tuning
        model_args, data_args, training_args = get_few_shot_config(
            model_name=args.model_name,
            max_examples=args.max_examples,
            max_eval_examples=args.max_eval_examples,
            epochs=args.epochs,
            prompt=args.prompt_examples
        )

        data_args.prompt_examples = args.prompt_examples        
        # Log configuration details
        logger.info(f"Model config: {model_args}")
        logger.info(f"Data config: {data_args}")
        logger.info(f"Training config: {training_args}")
        
        # Run training
        logger.info("Starting few-shot fine-tuning...")
        model, tokenizer = train_qed_model(model_args, data_args, training_args)
                
        logger.info("Fine-tuning complete.")

if __name__ == "__main__":
    main()
