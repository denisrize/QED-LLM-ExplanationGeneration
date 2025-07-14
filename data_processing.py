import json
import os
import re
import random
from typing import List

"""
This file functions for saving and loading, filtering, and managing data.
"""

def save_jsonlines(data, filepath):
    """Save a list of dicts to a .jsonlines file."""
    try:
        with open(filepath, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {len(data)} items to {filepath}")
    except Exception as e:
        print(f"Error saving to {filepath}: {e}")

#############################
# 9. Helper Functions for File Management and Filtering
#############################

def load_existing_problematic_predictions(output_dir):
    """Load existing problematic predictions if they exist."""
    problematic_file = os.path.join(output_dir, "problematic_predictions.jsonlines")
    if os.path.exists(problematic_file):
        print(f"Found existing problematic predictions at {problematic_file}")
        try:
            problematic_ids = set()
            with open(problematic_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if 'example_id' in data:
                        problematic_ids.add(data['example_id'])
            print(f"Loaded {len(problematic_ids)} problematic example IDs")
            return problematic_ids
        except Exception as e:
            print(f"Error loading problematic predictions: {e}")
            return set()
    return set()

def filter_validation_data_by_problematic_ids(validation_data, problematic_ids):
    """Filter validation data to include only examples with IDs in problematic_ids."""
    if not problematic_ids:
        return validation_data
    
    filtered_data = []
    for example in validation_data:
        # Try different possible ID fields
        example_id = example.get('example_id') or example.get('id') or hash(str(example))
        if example_id in problematic_ids:
            filtered_data.append(example)
    
    print(f"Filtered validation data from {len(validation_data)} to {len(filtered_data)} examples based on problematic IDs")
    return filtered_data

def get_next_available_filename(base_path, base_name, extension):
    """Get the next available filename with incremental numbering."""
    if not os.path.exists(os.path.join(base_path, f"{base_name}.{extension}")):
        return os.path.join(base_path, f"{base_name}.{extension}")
    
    counter = 1
    while True:
        filename = f"{base_name}_{counter}.{extension}"
        filepath = os.path.join(base_path, filename)
        if not os.path.exists(filepath):
            return filepath
        counter += 1


def has_valid_qed_annotation(annotation):
    """
    Returns True if annotation has all required QED fields with valid content.
    Checks for:
    - 'answer' field with at least one non-empty answer
    - 'selected_sentence' field with non-empty content
    - 'referential_equalities' field (can be empty list)
    """
    if not isinstance(annotation, dict):
        return False
    
    # Check for required fields
    required_fields = ["answer", "selected_sentence", "referential_equalities"]
    if not all(field in annotation for field in required_fields):
        return False
    
    # Check answer field - should have at least one non-empty answer
    answer = annotation["answer"]
    if isinstance(answer, list):
        if not answer or not any(ans.get("sentence_reference", {}).get("string", "").strip() for ans in answer):
            return False
    elif isinstance(answer, str):
        if not answer.strip():
            return False
    else:
        return False
    
    # Check selected_sentence field - should be non-empty
    selected_sentence = annotation["selected_sentence"]
    if isinstance(selected_sentence, dict):
        if not selected_sentence.get("string", "").strip():
            return False
    elif isinstance(selected_sentence, str):
        if not selected_sentence.strip():
            return False
    else:
        return False
    
    # Check referential_equalities field - should be a list (can be empty)
    if not isinstance(annotation["referential_equalities"], list):
        return False
    
    return True


def append_to_jsonlines(data, filepath):
    """Append data to an existing .jsonlines file or create new if it doesn't exist."""
    try:
        mode = 'a' if os.path.exists(filepath) else 'w'
        with open(filepath, mode) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        action = "Appended" if mode == 'a' else "Saved"
        print(f"{action} {len(data)} items to {filepath}")
    except Exception as e:
        print(f"Error saving to {filepath}: {e}")

def fix_llm_json_quotes(s):
    # Only add a double quote at the start of any value that starts with : ``
    # Do not add or modify the end of the value
    s = re.sub(
        r'(:\s*)``',
        r'\1"``',
        s
    )
    return s

def get_training_data_cache_path(training_data: List[dict], base_cache_dir: str = "./cache") -> str:
    """
    Generate a unique cache path based on training data characteristics.
    
    Args:
        training_data: List of training examples
        base_cache_dir: Base directory for cache files
        
    Returns:
        Unique cache file path
    """
    import hashlib
    import os
    
    # Create a simple hash based on dataset size and first/last example IDs
    data_size = len(training_data)
    
    # Get identifiers from first and last examples to create a unique signature
    first_id = training_data[0].get('example_id', training_data[0].get('id', '0')) if training_data else '0'
    last_id = training_data[-1].get('example_id', training_data[-1].get('id', '0')) if training_data else '0'
    
    # Create hash string
    hash_input = f"{data_size}_{first_id}_{last_id}"
    dataset_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    cache_filename = f"valid_qed_examples_{data_size}_{dataset_hash}.json"
    return os.path.join(base_cache_dir, cache_filename)


def sample_valid_qed_examples(training_data: List[dict], num_examples: int = 2, random_seed: int = None, cache_path: str = None) -> List[dict]:
    """
    Sample random examples from training data that have valid QED annotations.
    Uses caching to avoid recomputing valid examples on every call.
    
    Args:
        training_data: List of training examples
        num_examples: Number of examples to sample
        random_seed: Random seed for reproducibility
        cache_path: Optional path to cache valid examples. If provided and file exists,
                   loads from cache. If file doesn't exist, computes and saves to cache.
        
    Returns:
        List of sampled valid examples
    """
    
    if random_seed is not None:
        random.seed(random_seed)
    
    valid_examples = []
    
    # Try to load from cache if cache_path is provided
    if cache_path is not None:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    valid_examples = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Warning: Failed to load cache from {cache_path}: {e}")
                print("Will recompute valid examples...")
                valid_examples = []
    
    # If no valid examples loaded from cache, compute them
    if not valid_examples:
        print("Filtering valid examples from training data...")
        for example in training_data:
            if 'annotation' in example and has_valid_qed_annotation(example['annotation']):
                valid_examples.append(example)
        
        print(f"Found {len(valid_examples)} valid examples out of {len(training_data)} total examples")
        
        # Save to cache if cache_path is provided
        if cache_path is not None:
            try:
                print(f"Saving valid examples to cache: {cache_path}")
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(valid_examples, f, indent=2, ensure_ascii=False)
                print(f"Cached {len(valid_examples)} valid examples")
            except Exception as e:
                print(f"Warning: Failed to save cache to {cache_path}: {e}")
    
    if len(valid_examples) < num_examples:
        print(f"Warning: Only {len(valid_examples)} valid examples found, but {num_examples} requested")
        return valid_examples
    
    # Sample random examples
    sampled_examples = random.sample(valid_examples, num_examples)
    return sampled_examples


def format_qed_example_for_prompt(example: dict, example_num: int = 1) -> str:
    """
    Format a QED example into the same format as the static examples in config_prompt.py
    
    Args:
        example: QED example with annotation
        example_num: Example number for display
        
    Returns:
        Formatted string for prompt
    """
    title = example.get('title_text', '')
    question = example['question_text']
    context = example['paragraph_text']
    annotation = example['annotation']
    
    # Extract annotation components
    selected_sentence = annotation['selected_sentence'].get('string', '') if 'selected_sentence' in annotation else ""
    
    # Extract answer
    answer = ""
    if 'answer' in annotation and len(annotation['answer']) > 0:
        answer = annotation['answer'][0]['sentence_reference'].get('string', '')
    
    # Extract referential equalities
    ref_equalities = []
    if 'referential_equalities' in annotation:
        for ref_eq in annotation['referential_equalities']:
            q_ref = ref_eq['question_reference'].get('string', '')
            s_ref = ref_eq['sentence_reference'].get('string', '')
            bridge = ref_eq['sentence_reference'].get('bridge', 'false')
            
            ref_equalities.append({
                "question_reference": q_ref,
                "sentence_reference": s_ref,
                "bridge": bridge
            })
    
    # Create the formatted example
    formatted_example = f"""
Demonstration Example {example_num}:
Title:
{title}

Question:
{question}

Context:
{context}

Expected JSON:
{{
  "answer": "{answer}",
  "selected_sentence": "{selected_sentence}",
  "referential_equalities": {json.dumps(ref_equalities, indent=4)}
}}
"""
    
    return formatted_example


def find_best_match(needle, haystack, threshold=0.7):
    """
    Find the best matching substring in haystack for the given needle.
    
    Args:
        needle: The string to search for
        haystack: The string to search in
        threshold: Minimum similarity score to consider a match
        
    Returns:
        Tuple of (start_index, end_index) if a good match is found, or None if no match
    """
    if not needle or not haystack:
        return None
    
    def remove_trailing_dots(text):
        """Remove trailing dots and spaces from text"""
        # Remove patterns like ' .' or '.' at the end
        import re
        return re.sub(r'\s*\.\s*$', '', text)
    
    # Try exact match first
    start_idx = haystack.find(needle)
    if start_idx >= 0:
        return (start_idx, start_idx + len(needle))
    
    # Try case insensitive match
    needle_lower = needle.lower()
    haystack_lower = haystack.lower()
    
    start_idx = haystack_lower.find(needle_lower)
    if start_idx >= 0:
        # Use the original case from haystack
        return (start_idx, start_idx + len(needle))
    
    # Try matching after removing trailing dots
    needle_no_dots = remove_trailing_dots(needle)
    haystack_no_dots = remove_trailing_dots(haystack)
    
    # Try exact match without trailing dots
    start_idx = haystack_no_dots.find(needle_no_dots)
    if start_idx >= 0:
        return (start_idx, start_idx + len(needle_no_dots))
    
    # Try case insensitive match without trailing dots
    start_idx = haystack_no_dots.lower().find(needle_no_dots.lower())
    if start_idx >= 0:
        return (start_idx, start_idx + len(needle_no_dots))
    
    # Try to find needle_no_dots in original haystack (in case only needle had trailing dots)
    start_idx = haystack_lower.find(needle_no_dots.lower())
    if start_idx >= 0:
        # Find the actual end position in the original haystack
        # Look for the end of the match, allowing for trailing dots
        end_idx = start_idx + len(needle_no_dots)
        # Extend end_idx if there are trailing dots in haystack
        while end_idx < len(haystack) and haystack[end_idx] in ' .':
            end_idx += 1
        return (start_idx, end_idx)
    
    # Try to find original needle in haystack_no_dots (in case only haystack had trailing dots)
    start_idx = haystack_no_dots.lower().find(needle_lower)
    if start_idx >= 0:
        return (start_idx, start_idx + len(needle))
    
    # If all else fails, try fuzzy matching with length-based chunks
    try:
        import difflib
        
        # Create chunks based on needle length rather than punctuation
        chunk_size = len(needle)
        chunks = []
        
        for i in range(0, len(haystack) - chunk_size + 1):
            chunk = haystack[i:i + chunk_size]
            chunks.append((chunk, i))  # Store chunk and its start position
        
        # Also add the last chunk if it wasn't fully covered
        if len(haystack) > chunk_size:
            last_chunk = haystack[-chunk_size:]
            last_start = len(haystack) - chunk_size
            chunks.append((last_chunk, last_start))
        
        best_ratio = 0
        best_match = None
        best_start = 0
        
        for chunk, start_pos in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
                
            # Try direct comparison
            ratio = difflib.SequenceMatcher(None, needle_lower, chunk.lower()).ratio()
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = chunk
                best_start = start_pos
            
            # Also try with trailing dots removed
            chunk_no_dots = remove_trailing_dots(chunk)
            needle_no_dots_lower = remove_trailing_dots(needle).lower()
            
            if chunk_no_dots and needle_no_dots_lower:
                ratio = difflib.SequenceMatcher(None, needle_no_dots_lower, chunk_no_dots.lower()).ratio()
                if ratio > best_ratio and ratio >= threshold:
                    best_ratio = ratio
                    best_match = chunk_no_dots
                    best_start = start_pos
        
        if best_match:
            # Find the exact position of best_match in haystack
            match_start = haystack.find(best_match, best_start)
            if match_start >= 0:
                return (match_start, match_start + len(best_match))
            else:
                # If exact match not found, use the chunk position
                return (best_start, best_start + len(best_match))
                
    except ImportError:
        pass
    
    # No good match found
    return None


def add_offsets_to_prediction(prediction, question, context, remove_unmatched_references: bool = True):
    """
    Add character offsets to a QED prediction that only contains the exact strings.
    Enhanced to handle variations in the model's output.
    
    Args:
        prediction: Dictionary with answer, selected_sentence, and referential_equalities
        question: Original question text
        context: Original context text
        
    Returns:
        Updated prediction with all character offsets added
    """
    updated_prediction = {
        "answer": None,
        "selected_sentence": None,
        "referential_equalities": []
    } #copy.deepcopy(prediction)
    
    # Add offsets for selected sentence
    if "selected_sentence" in prediction and isinstance(prediction["selected_sentence"], str):
        sentence = prediction["selected_sentence"]
        
        # Find the best match for the selected sentence in the context
        match = find_best_match(sentence, context)
        
        if match:
            sentence_start, sentence_end = match
            updated_prediction["selected_sentence"] = {
                "string": context[sentence_start:sentence_end],  # Use the actual text from context
                "start": sentence_start,
                "end": sentence_end
            }
        else:
            # If no match found, use the original string but with dummy offsets
            updated_prediction["selected_sentence"] = {
                "string": sentence,
                "start": 0,
                "end": len(sentence)
            }
    else:
        raise ValueError("No selected sentence found in prediction")
    
    # Add offsets for referential equalities
    if "referential_equalities" in prediction and isinstance(prediction["referential_equalities"], list):
        for i, ref_eq in enumerate(prediction["referential_equalities"]):
            # Process question reference
            if "question_reference" in ref_eq and isinstance(ref_eq["question_reference"], str):
                q_ref = ref_eq["question_reference"]
                
                # Find the best match for the question reference in the question
                q_match = find_best_match(q_ref, question)
                
                if q_match:
                    q_start, q_end = q_match
                    updated_prediction["referential_equalities"].append({
                        "question_reference": {
                            "string": question[q_start:q_end],  # Use the actual text from question
                            "start": q_start,
                            "end": q_end
                        }
                    })
                else:
                    if remove_unmatched_references:
                        continue
                    # If no match found, use the original string but with dummy offsets
                    updated_prediction["referential_equalities"].append({
                        "question_reference": {
                            "string": q_ref,
                            "start": 0,
                            "end": len(q_ref)
                        }
                    })
            
            # Process sentence reference
            if "sentence_reference" in ref_eq and isinstance(ref_eq["sentence_reference"], str):
                s_ref = ref_eq["sentence_reference"]
                bridge = ref_eq.get("bridge", False)

                if s_ref == "":
                    # Special case: empty string means bridge to whole sentence, use (-1, -1)
                    updated_prediction["referential_equalities"][-1]["sentence_reference"] = {
                        "string": "",
                        "start": -1,
                        "end": -1,
                        "bridge": bridge
                    }
                else:
                    # If not found in sentence, look in entire context
                    c_match = find_best_match(s_ref, context)
                    
                    if c_match:
                        s_start_in_context, s_end_in_context = c_match
                        updated_prediction["referential_equalities"][-1]["sentence_reference"] = {
                            "string": context[s_start_in_context:s_end_in_context],  # Use the actual text from context
                            "start": s_start_in_context,
                            "end": s_end_in_context,
                            "bridge": bridge
                        }
                    else:
                        if remove_unmatched_references:
                            # Check that the reference is not empty and there is only question reference
                            if len(updated_prediction["referential_equalities"]):
                                updated_prediction["referential_equalities"].pop()
                            continue
                        # If still not found
                        updated_prediction["referential_equalities"][i]["sentence_reference"] = {
                            "string": s_ref,
                            "start": 0,
                            "end": len(s_ref),
                            "bridge": bridge
                        }
    else:
        raise ValueError("No referential equalities found in prediction")
    
    # Add offsets for answer
    if "answer" in prediction and isinstance(prediction["answer"], str):
        answer = prediction["answer"]
        
        # Find the best match for the answer in the context
        a_match = find_best_match(answer, context)
        
        if a_match:
            a_start, a_end = a_match
            # Store the answer with offsets in the prediction
            # Note: The official evaluator expects the answer in a specific format
            # We'll keep the simple string format for basic metrics and add a structured format for official evaluation
            updated_prediction["answer"] = {
                "string": context[a_start:a_end],  # Use the actual text from context
                "start": a_start,
                "end": a_end
            }
        else:
            # Fall down to the selected sentence
            updated_prediction["answer"] = updated_prediction["selected_sentence"]

    else:
        raise ValueError("No answer found in prediction")
    
    return updated_prediction