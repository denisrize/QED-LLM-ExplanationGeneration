# QED Few-Shot Fine-Tuning and Evaluation

This project provides scripts and configuration for few-shot fine-tuning and evaluation of language models on the QED (Question-Explanation-Data) task. Full dataset description can be found here - [QED Dataset](https://github.com/google-research-datasets/QED)
The project supports various instruction-tuned models and includes a comprehensive evaluation with detailed span extraction and overlap analysis.

---

## Table of Contents

1. [Results and Performance](#results-and-performance)
2. [Project Structure](#project-structure)
3. [Input Arguments](#input-arguments)
4. [Training and Evaluation Pipeline](#training-and-evaluation-pipeline)
5. [Configuration Files](#configuration-files)
6. [Output Results Structure](#output-results-structure)
7. [Model Output Extraction and Span Handling](#model-output-extraction-and-span-handling)
8. [Evaluation Methodology](#evaluation-methodology)
9. [Data Processing](#data-processing)
10. [Example Usage](#example-usage)

---

## 🏆 Results and Performance

### Key Achievements
Our few-shot fine-tuning approach demonstrates significant improvements over base models across all QED evaluation metrics:

**📊 Static vs Randomly Chosen Examples**

<img width="400" height="1475" alt="static_vs_random" src="https://github.com/user-attachments/assets/8beabe6d-f96b-4020-aefa-b5631d9878d8" />

**📊 Model Performance Comparison Overview**
Comparison of base vs fine-tuned models across key metrics (All Mention F1, Pair F1, Answer Accuracy) and Prompt Type (Number of Examples, Randomly chosen or Not)

<img width="400" height="1180" alt="fine_tuned_vs_regular" src="https://github.com/user-attachments/assets/016a9330-47be-43a3-a320-0c56c301855b" />

#### Llama3-8B-Instruct Fine-tuned Results
- **All Mention F1**: 25.6% → **35.8%** (+10.2%, **+39.8% relative improvement**)
- **Pair Alignment F1**: 17.0% → **22.1%** (+5.1%, **+30.0% relative improvement**)  
- **Answer Accuracy**: 51.9% → **58.8%** (+6.9%, **+13.3% relative improvement**)
- **Question Mention F1**: 68.1% → **71.8%** (+3.7%, **+5.4% relative improvement**)

#### Qwen2.5-7B-Instruct Fine-tuned Results
- **All Mention F1**: 25.6% → **36.3%** (+10.7%, **+41.8% relative improvement**)
- **Pair Alignment F1**: 17.0% → **22.6%** (+5.6%, **+32.9% relative improvement**)
- **Answer Accuracy**: 51.9% → **59.4%** (+7.5%, **+14.4% relative improvement**)  
- **Question Mention F1**: 68.1% → **72.4%** (+4.3%, **+6.3% relative improvement**)

**📊 Model Performance Comparison Overview**

### Performance Across Different Evaluation Thresholds
Our evaluation reveals that fine-tuned models maintain superior performance even at stricter overlap thresholds:

| Overlap Threshold | Llama3 Base F1 | Llama3 Fine-tuned F1 | Qwen Base F1 | Qwen Fine-tuned F1 |
|------------------|-----------------|----------------------|--------------|-------------------|
| 0.5 (Lenient)    | 25.6%          | **35.8%**           | 25.6%       | **36.3%**        |
| 0.7 (Moderate)   | 23.1%          | **32.4%**           | 23.1%       | **33.1%**        |
| 0.9 (Strict)     | 19.8%          | **27.6%**           | 19.8%       | **28.3%**        |

**📊 Model Answers Length Distribution**
Comparative analysis of actual versus predicted output distributions reveals significant differences between base and fine-tuned models:

Sentence and questions reference length distributions for Qwen models:

<img width="400" height="1181" alt="quen_q_dis" src="https://github.com/user-attachments/assets/88179a56-3b90-46d1-8c92-429d0a5cade4" />

<img width="400" height="1181" alt="quen_s_dis" src="https://github.com/user-attachments/assets/9612cb1f-5c02-4f49-9edb-9c10ad882bf5" />

Sentence reference length distributions for Llama models:

<img width="400" height="1181" alt="llama_q_dis" src="https://github.com/user-attachments/assets/0644cf1c-4dc2-4976-87ad-4d39b155091e" />

<img width="400" height="1181" alt="llama_s_dis" src="https://github.com/user-attachments/assets/0d598dd0-95e0-4f2c-b6ba-7e31a8b987bf" />

### Entity Recognition and Alignment Improvements

Our fine-tuned models show particularly strong improvements in:
- **Entity Identification**: **41.8% relative improvement** in All Mention F1 (Qwen2.5)
- **Entity Relationship Mapping**: **32.9% relative improvement** in Pair Alignment F1 (Qwen2.5)
- **Answer Extraction**: Consistent **13-14% relative improvements** in Answer Accuracy

---

## 🚀 Quick Results Summary

| Model | All Mention F1 | Pair Alignment F1 | Answer Accuracy | Training Examples |
|-------|----------------|-------------------|-----------------|-------------------|
| **Qwen2.5-7B Fine-tuned** | **36.3%** (+41.8%) | **22.6%** (+32.9%) | **59.4%** (+14.4%) | 50 |
| **Llama3-8B Fine-tuned** | **35.8%** (+39.8%) | **22.1%** (+30.0%) | **58.8%** (+13.3%) | 50 |
| Base Models (Zero-shot) | 25.6% | 17.0% | ~52% | N/A |

## 🤗 Pre-trained Models

Our fine-tuned models are available on Hugging Face:

### Ready-to-Use Models:
- **[Llama3-8B-QED-FineTuned](https://huggingface.co/DenisRz/llama3_8b_instruct_qed)** 
  - Base: meta-llama/Meta-Llama-3-8B-Instruct
  - Performance: +10.2% All Mention F1, +39.8% relative improvement

- **[Qwen2.5-7B-QED-FineTuned](https://huggingface.co/DenisRz/qwen2.5-7b-qed)**
  - Base: Qwen/Qwen2.5-7B-Instruct  
  - Performance: +10.7% All Mention F1, +41.8% relative improvement

---

## Project Structure

- **`project_code/`**: Main project directory containing all core scripts
  - `run_experiment.py`: **Main entry point script** for training and evaluation
  - `config_models.py`: Model configurations for different fine-tuning setups
  - `qed_main.py`: Core functions for training, generation, and evaluation (2157 lines)
  - `qed_eval.py`: Official QED evaluation metrics and scoring functions
  - `data_processing.py`: Data handling, filtering, and loading utilities
  - `config_prompt.py`: Prompt configuration and few-shot examples management
  - `loss_tracking.py`: Training loss monitoring and visualization
- **`qed_data/`**: Directory for QED datasets (validation: `qed-dev.jsonlines`, training data)
- **`models_fine_tuned/`**: Stores fine-tuned models and their checkpoints
- **`qed_model_results/`**: Stores output predictions and evaluation results organized by model and prompt configuration

---

## Input Arguments

The main script `run_experiment.py` accepts the following arguments:

### Core Model Configuration
- **`--model_name`**:  
  **Type:** str  
  **Default:** None  
  **Choices:** `"llama2_7b_chat"`, `"llama3_8b_instruct"`, `"mistral_7b_instruct"`, `"qwen2_5_7b_instruct"`  
  **Description:** Specifies the base model for fine-tuning or evaluation. Uses specific names for chat/instruct versions to ensure proper prompt formatting.

### Training Configuration
- **`--max_examples`**:  
  **Type:** int  
  **Default:** None  
  **Description:** Number of training examples to use for few-shot fine-tuning. If not specified, uses all available training data.

- **`--max_eval_examples`**:  
  **Type:** int  
  **Default:** None  
  **Description:** Number of validation examples to use for evaluation. If not specified, uses all available validation data.

- **`--epochs`**:  
  **Type:** int  
  **Default:** None  
  **Description:** Number of training epochs. If not specified, uses default configuration from `config_models.py`.

### Evaluation Configuration
- **`--eval_only`**:  
  **Type:** flag  
  **Description:** If set, only runs evaluation on a previously fine-tuned model. Skips the training phase entirely.

- **`--model_path`**:  
  **Type:** str  
  **Default:** None  
  **Description:** Path to a previously fine-tuned model directory (required if `--eval_only` is set). Should point to the model directory containing the fine-tuned weights.

### Prompt Configuration
- **`--prompt_examples`**:  
  **Type:** str  
  **Default:** `"zero_shot"`  
  **Choices:** `"zero_shot"`, `"first_only"`, `"second_only"`, `"both"`, `"random_one"`, `"random_two"`  
  **Description:** Which prompt examples to include in the input:  
    - `zero_shot`: No examples in the prompt (direct inference)
    - `first_only`: Includes only the "Life of Pi" example
    - `second_only`: Includes only the "Hemolytic reaction" example  
    - `both`: Includes both example cases for few-shot prompting
    - `random_one`: Includes one randomly selected example
    - `random_two`: Includes two randomly selected examples

---

## Training and Evaluation Pipeline

### Training Pipeline

1. **Configuration Loading:**  
   The script loads model, data, and training configurations using `get_few_shot_config()` from `config_models.py`, which automatically sets optimal hyperparameters based on the model type and training regime.

2. **Data Preparation:**  
   - Loads training data from `qed_data/` directory
   - Applies prompt formatting based on `--prompt_examples` setting using templates from `config_prompt.py`
   - Filters data to the specified number of examples if `--max_examples` is provided

3. **Model Fine-Tuning:**  
   - Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning with 4-bit quantization
   - Supports various instruction-tuned models with model-specific configurations
   - Saves checkpoints and final model under `models_fine_tuned/[model_name]_fewshot_[prompt_type]/`

4. **Logging:**  
   - All training progress is logged to `qed_fewshot_experiment.log`
   - TensorBoard logs are generated for monitoring training metrics

### Evaluation Pipeline

1. **Model Loading:**  
   - If `--eval_only` is set, loads the specified pre-trained model from `--model_path`
   - Uses `setup_finetuned_model_for_evaluation()` for loading fine-tuned models

2. **Prediction Generation:**  
   - Loads validation data (typically `qed-dev.jsonlines`)
   - Generates predictions using `model_generation_on_qed()` with specified generation parameters
   - Saves detailed predictions to `qed_model_results/[model_path]/[prompt_config]/detailed_predictions.jsonlines`

3. **Comprehensive Evaluation:**  
   - Runs `batch_evaluate_with_different_overlaps()` to compute metrics across multiple overlap thresholds
   - Generates evaluation results for overlap ratios: [0.5, 0.6, 0.7, 0.8, 0.9]
   - Produces both "valid_only" and "normalized" evaluation modes

---

## Configuration Files

### `config_models.py`
Contains pre-configured settings for different models and training scenarios:

- **`get_llama3_8b_instruct_config()`**: Optimized settings for Meta-Llama-3-8B-Instruct with LoRA parameters (r=16, alpha=32)
- **`get_mistral_7b_instruct_config()`**: Settings for Mistral-7B-Instruct-v0.3 with standard LoRA configuration
- **`get_qwen2_5_7b_instruct_config()`**: Configuration for Qwen2.5-7B-Instruct with adaptive learning rates
- **`get_few_shot_config()`**: Dynamically selects appropriate configuration and adjusts output paths for few-shot experiments

### `config_prompt.py`
Manages prompt templates and example configurations:
- Defines QED instruction prompt structure with JSON output formatting
- Contains three demonstration examples: "Life of Pi", "Hemolytic reaction", and "Black Mass"
- Handles prompt formatting for different models and example configurations

---

## Output Results Structure

### Directory Organization
```
qed_model_results/
├── [model_name_or_path]/
│   ├── [prompt_config]/
│   │   ├── detailed_predictions.jsonlines
│   │   ├── batch_evaluation_summary.json
│   │   ├── official_eval_results_overlap_[X]_strict_[True/False].json
│   │   └── problematic_predictions.jsonlines (if any)
```

### Predictions Output File (`detailed_predictions.jsonlines`)
Each line contains a JSON object with:
```json
{
  "example_id": "unique_identifier",
  "title_text": "document_title",
  "original_nq_answers": [["span_list"]],
  "question_text": "original_question",
  "paragraph_text": "context_paragraph", 
  "annotation": {
    "referential_equalities": [
      {
        "question_reference": {"start": 8, "end": 40, "string": "entity_in_question"},
        "sentence_reference": {"start": 0, "end": 32, "string": "corresponding_entity_in_sentence", "bridge": false}
      }
    ],
    "answer": [{"sentence_reference": {"start": 56, "end": 78, "string": "answer_text", "bridge": false}, "paragraph_reference": {"start": 56, "end": 78, "string": "answer_text"}}],
    "explanation_type": "single_sentence",
    "selected_sentence": {"start": 0, "end": 167, "string": "selected_sentence_text"}
  }
}
```

### Evaluation Results Files
- **`batch_evaluation_summary.json`**: Comprehensive results across all overlap thresholds and strict settings
- **Individual result files**: Detailed metrics for each overlap/strict combination
- **Metrics included**: 
  - Exact match accuracy
  - Answer accuracy
  - Question mention F1 (Precision, Recall, F1)
  - Context mention F1 (Precision, Recall, F1)
  - All mention F1 (combined entity extraction performance)
  - Pair F1 (entity alignment scores)

---

## Model Output Extraction and Span Handling

### JSON Extraction Process

1. **Response Parsing:**  
   - The model is prompted to output structured JSON containing answers, selected sentences, and referential equalities
   - The system first attempts to extract JSON from markdown code blocks (```json...```)
   - If not found, uses regex to locate JSON objects within free-form text

2. **Fallback Strategies:**  
   - If JSON is malformed, attempts partial parsing to recover usable information
   - Handles common JSON formatting errors (missing braces, trailing commas, etc.)
   - Logs problematic responses for manual inspection in `problematic_predictions.jsonlines`

### Span Offset Calculation

1. **Answer Span Extraction:**  
   - For each predicted answer, finds exact character offsets (start, end) in the source context
   - Uses fuzzy string matching when exact matches are not found
   - Handles multiple answer spans when the model predicts multiple answers

2. **Selected Sentence Mapping:**  
   - Maps model-selected sentences back to original paragraph offsets
   - Supports both sentence ID-based selection and full-text matching
   - Validates sentence boundaries and handles sentence segmentation edge cases

3. **Entity Span Alignment:**  
   - Extracts question entities and their corresponding context entities from referential equalities
   - Computes precise character offsets for all referential equality pairs
   - Handles multi-word entities and nested entity recognition

---

## Evaluation Methodology

### Understanding QED Task Structure

A **QED example** consists of:
- **Question**: A natural language question
- **Context**: A paragraph containing the answer
- **Answer**: Spans in the context that answer the question
- **Selected Sentence**: The sentence that entails the answer
- **Referential Equalities**: Mappings between question entities and context entities

### Strict vs. Non-Strict Matching

1. **Strict Evaluation (`strict=True`):**  
   - Predicted spans must **exactly match** ground truth spans (character-perfect alignment)
   - No tolerance for boundary differences or partial matches
   - Most conservative evaluation setting

2. **Non-Strict Evaluation (`strict=False`):**  
   - Uses **overlap-based matching** with configurable F1 thresholds
   - Entities are considered equivalent if they satisfy:
     - Normalized text matching (after lowercasing, removing punctuation/articles)
     - Sufficient span overlap based on F1 score

### Overlap Ratio Calculation

The overlap ratio is computed using **F1 score between predicted and ground truth spans**:

```
F1 = tp / (tp + (fp + fn) / 2)

Where:
- tp (true positive): Character overlap between spans
- fp (false positive): Characters in prediction but not in ground truth  
- fn (false negative): Characters in ground truth but not in prediction
```

**Example:**
```
Ground truth: [10, 25] "machine learning"
Prediction:   [12, 23] "chine learnin"
F1 = 11 / (11 + (2 + 4) / 2) = 11 / 14 ≈ 0.79
```

### Evaluation Thresholds

The system evaluates at multiple overlap thresholds:
- **0.5**: Requires ≥50% F1 overlap (lenient)
- **0.6**: Requires ≥60% F1 overlap
- **0.7**: Requires ≥70% F1 overlap  
- **0.8**: Requires ≥80% F1 overlap
- **0.9**: Requires ≥90% F1 overlap (strict)

### Performance Analysis Results

Our comprehensive evaluation demonstrates that fine-tuned models consistently outperform base models across all overlap thresholds, with particularly strong improvements in entity relationship tasks:

| Metric | Threshold | Base Performance | Fine-tuned Performance | Improvement |
|--------|-----------|------------------|------------------------|-------------|
| All Mention F1 | 0.5 | 25.6% | **36.3%** | **+41.8%** |
| All Mention F1 | 0.7 | 23.1% | **33.1%** | **+43.3%** |
| All Mention F1 | 0.9 | 19.8% | **28.3%** | **+43.0%** |
| Pair Alignment F1 | 0.5 | 17.0% | **22.6%** | **+32.9%** |
| Answer Accuracy | 0.5 | 51.9% | **59.4%** | **+14.4%** |

### Evaluation Modes

1. **Valid-Only Metrics:**  
   - Computed only on examples where the model produced valid JSON output
   - Excludes examples with parsing failures or malformed responses

2. **Normalized Metrics:**  
   - Treats invalid/missing predictions as False
   - Provides a complete evaluation across all ground truth examples
   - More conservative and comprehensive measure

---

## Data Processing

### Ground Truth Filtering

1. **Missing Annotation Removal:**  
   - Ground truth examples lacking required annotation fields are automatically filtered out
   - Ensures fair evaluation by excluding malformed ground truth data

2. **Validation:**  
   - Validates that all ground truth examples have required fields: `annotation`, `question_text`, `paragraph_text`
   - Ensures entity spans and answer spans are properly formatted and within document boundaries

### Model Response Processing

1. **JSON Validation:**  
   - Attempts to parse model responses as JSON
   - Handles various JSON formatting issues automatically
   - Logs problematic responses for analysis

2. **Span Extraction:**  
   - Extracts answer spans, selected sentences, and referential equalities
   - Computes character offsets for all predicted spans
   - Handles edge cases like empty bridges, multiple spans, and missing context

---

## Example Usage

### Few-Shot Fine-Tuning
```bash
# Train Llama-3-8B-Instruct with 200 examples for 3 epochs using both prompt examples
python project_code/run_experiment.py \
    --model_name llama3_8b_instruct \
    --max_examples 200 \
    --epochs 3 \
    --prompt_examples both
```

### Evaluation Only
```bash
# Evaluate a previously fine-tuned model
python project_code/run_experiment.py \
    --eval_only \
    --model_name llama3_8b_instruct \
    --model_path models_fine_tuned/llama3_8b_instruct_fewshot_both \
    --prompt_examples both
```

### Zero-Shot Evaluation
```bash
# Evaluate base model without fine-tuning
python project_code/run_experiment.py \
    --eval_only \
    --model_name qwen2_5_7b_instruct \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --prompt_examples zero_shot
```

### Different Prompt Configurations
```bash
# Compare different prompting strategies
for prompt in zero_shot first_only second_only both random_one random_two; do
    python project_code/run_experiment.py \
        --eval_only \
        --model_name mistral_7b_instruct \
        --model_path mistralai/Mistral-7B-Instruct-v0.3 \
        --prompt_examples $prompt
done
```

### Quick Evaluation with Limited Examples
```bash
# Evaluate with limited examples for quick testing
python project_code/run_experiment.py \
    --eval_only \
    --model_name llama3_8b_instruct \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --max_eval_examples 100 \
    --prompt_examples both
```

---

## Understanding Output Files

After running evaluation, check the results in:

1. **`batch_evaluation_summary.json`**: 
   - Comprehensive metrics across all overlap thresholds
   - Comparison between valid-only and normalized scores
   - Includes example counts and filtering information

2. **Individual Evaluation Files**: 
   - `official_eval_results_overlap_[X]_strict_[True/False].json`
   - Detailed metrics for specific overlap/strict combinations

3. **`detailed_predictions.jsonlines`**: 
   - Per-example predictions with full annotation structure
   - Useful for debugging and error analysis

4. **`problematic_predictions.jsonlines`**: 
   - Examples where JSON parsing failed or other issues occurred
   - Helps identify model response formatting problems

### Key Metrics to Monitor

- **Answer Accuracy**: Percentage of questions with correct answer spans
- **Exact Match Accuracy**: Percentage of perfect prediction matches
- **Pair F1**: Entity alignment performance between questions and context
- **All Mention F1**: Overall entity extraction performance

---

## Contact

For questions or issues, please contact the project maintainer. 
