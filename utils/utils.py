import os
import json
import base64
import warnings
from copy import deepcopy

def is_multimodal_dataset(sample):
    """Check if a sample contains multimodal data (has image_path field)"""
    return "image_path" in sample

def encode_image_to_base64(image_path):
    """Encode image file to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except FileNotFoundError:
        warnings.warn(f"Image file not found: {image_path}", category=UserWarning, stacklevel=3)
        return None
    except PermissionError:
        warnings.warn(f"Permission denied accessing image file: {image_path}", category=UserWarning, stacklevel=3)
        return None
    except Exception as e:
        warnings.warn(f"Error encoding image {image_path}: {e}", category=UserWarning, stacklevel=3)
        return None

def prepare_multimodal_content(sample):
    """Prepare multimodal content for LLM input"""
    if not is_multimodal_dataset(sample):
        return sample["query"]
    
    # Encode image to base64
    image_base64 = encode_image_to_base64(sample["image_path"])
    if image_base64 is None:
        raise FileNotFoundError(f"Failed to encode image: {sample['image_path']}")
    
    # Create multimodal content
    multimodal_content = [
        {
            "type": "text",
            "text": sample["query"]
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        }
    ]
    
    return multimodal_content

def compose_multimodal_input(text, multimodal_content):
    """Attach text to multimodal template so downstream calls can include images."""
    if multimodal_content is None:
        return text

    payload = deepcopy(multimodal_content)
    text_slot_found = False
    for part in payload:
        if isinstance(part, dict) and part.get("type") == "text":
            part["text"] = text
            text_slot_found = True
            break

    if not text_slot_found:
        payload.insert(0, {"type": "text", "text": text})

    return payload

def load_model_api_config(model_api_config, model_name):
    with open(model_api_config, "r") as f:
        model_api_config = json.load(f)
    for model_name in model_api_config:
        actual_max_workers = model_api_config[model_name]["max_workers_per_model"] * len(model_api_config[model_name]["model_list"])
        model_api_config[model_name]["max_workers"] = actual_max_workers
    return model_api_config

def write_to_jsonl(lock, file_name, data):
    with lock:
        with open(file_name, 'a') as f:
            json.dump(data, f)
            f.write('\n')

def reserve_unprocessed_queries(output_path, test_dataset):
    processed_queries = set()
    keep_lines = []
    needs_rewrite = False
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                infered_sample = json.loads(line)
                response = infered_sample.get("response")
                has_error = bool(infered_sample.get("error"))
                if response is None or has_error:
                    needs_rewrite = True
                    continue
                processed_queries.add(infered_sample.get("query"))
                keep_lines.append(line)

    if needs_rewrite and keep_lines:
        with open(output_path, "w") as f:
            f.writelines(keep_lines)

    test_dataset = [sample for sample in test_dataset if sample["query"] not in processed_queries]
    return test_dataset