import json
import os
import random
import argparse
import ast
from datasets import Dataset
import pandas as pd
from PIL import Image
import base64
import io

from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="MATH")
parser.add_argument("--dataset_path", type=str, default=None)
parser.add_argument("--num2sample", type=int, default=500)
args = parser.parse_args()

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f"{args.dataset_name}.json")
os.makedirs(os.path.dirname(save_path), exist_ok=True)

image_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",args.dataset_name)
os.makedirs(os.path.dirname(image_save_path), exist_ok=True)


def shuffle_and_sample(data_list, num2sample):
    data_list = deduplicate(data_list)
    random.seed(2024)
    random.shuffle(data_list)
    return data_list[:num2sample]

def deduplicate(data_list):
    seen_queries = set()
    unique_data = []

    for item in data_list:
        if item["query"] not in seen_queries:
            unique_data.append(item)
            seen_queries.add(item["query"])
    if len(unique_data) < len(data_list):
        print(f">> Duplicate samples removed: {len(data_list) - len(unique_data)}")
    return unique_data

# load MATH-500
if args.dataset_name == "MATH":
    load_dataset_path = args.dataset_path if args.dataset_path else "HuggingFaceH4/MATH-500"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["problem"],
            "gt": example["answer"],
            "tag": [args.dataset_name, "math", example["subject"], f"Level {example['level']}", ],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load GSM-Hard
elif args.dataset_name == "GSM-Hard":
    load_dataset_path = args.dataset_path if args.dataset_path else "reasoning-machines/gsm-hard"
    dataset = load_dataset(load_dataset_path, split="train", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = [
        {
            "query": example["input"],
            "gt": str(example["target"]),
            "tag": ["math", "GSM-Hard"],
            "source": "GSM-Hard"
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load MedMCQA
elif args.dataset_name == "MedMCQA":
    load_dataset_path = args.dataset_path if args.dataset_path else "openlifescienceai/medmcqa"
    dataset = load_dataset(load_dataset_path, split="validation", trust_remote_code=True)
    filtered_dataset = dataset.filter(lambda example: example['choice_type'] != 'multi')
    print(f"{'='*50}\n", filtered_dataset)
    def format_medmcqa_query(example):
        query = example["question"]
        query += "\n\nChoose the correct answer from the following options:"
        query += f"\n(A) {example['opa']}"
        query += f"\n(B) {example['opb']}"
        query += f"\n(C) {example['opc']}"
        query += f"\n(D) {example['opd']}"
        return query
    def format_medmcqa_gt(example):
        answer_list = [f"(A) {example['opa']}", f"(B) {example['opb']}", f"(C) {example['opc']}", f"(D) {example['opd']}"]
        answer = f"The correct answer is: {answer_list[example['cop']]}"
        return answer
    data_list = [
        {
            "query": format_medmcqa_query(example),
            "gt": format_medmcqa_gt(example),
            "tag": ["medical", example['subject_name'], example['topic_name']],
            "source": "MedMCQA"
        }
        for example in filtered_dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load MedQA
elif args.dataset_name == "MedQA":
    load_dataset_path = args.dataset_path if args.dataset_path else "bigbio/med_qa"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    def format_medqa_query(example):
        query = example["question"]
        query += " Choose the correct answer from the following options:"
        for option in example["options"]:
            query += f"\n({option['key']}) {option['value']}"
        return query
    def format_medqa_gt(example):
        answer = f"The correct answer is: ({example['answer_idx']}) {example['answer']}"
        return answer
    data_list = [
        {
            "query": format_medqa_query(example),
            "gt": format_medqa_gt(example),
            "tag": ["medical"],
            "source": "MedQA"
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load MMLU
elif args.dataset_name == "MMLU":
    load_dataset_path = args.dataset_path if args.dataset_path else "cais/mmlu"
    dataset = load_dataset(load_dataset_path, "all", split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)

    def format_mmlu_query(example):
        query = f"""The following is a multiple-choice question:
{example["question"]}

Choose the correct answer from the following options:
(A) {example["choices"][0]}
(B) {example["choices"][1]}
(C) {example["choices"][2]}
(D) {example["choices"][3]}"""
        return query
    
    def format_mmlu_gt(example):
        choice_list = ["A", "B", "C", "D"]
        answer = f"({choice_list[example['answer']]})"
        return answer
    
    data_list = [
        {
            "query": format_mmlu_query(example),
            "gt": format_mmlu_gt(example),
            "tag": ["mmlu", example['subject']],
            "source": "MMLU"
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load MMLU-Pro
elif args.dataset_name == "MMLU-Pro":
    load_dataset_path = args.dataset_path if args.dataset_path else "TIGER-Lab/MMLU-Pro"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    option_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    def format_mmlu_query(example):
        query = "The following is a multiple-choice question:\n"
        query += example["question"]
        query += "\n\nChoose the correct answer from the following options:"
        for idx, option in enumerate(example["options"]):
            query += f"\n({option_list[idx]}) {option}"
        return query
    
    def format_mmlu_gt(example):
        answer = f"The answer is ({option_list[example['answer_index']]}) {example['options'][example['answer_index']]}"
        return answer
    
    data_list = [
        {
            "query": format_mmlu_query(example),
            "gt": format_mmlu_gt(example),
            "tag": ["MMLU-Pro", example['category'], example['src']],
            "source": "MMLU-Pro",
            "num_choices": len(example["options"])
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)  # 1001, 2004

elif args.dataset_name.startswith("GPQA"):
    load_dataset_path = args.dataset_path if args.dataset_path else "Idavidrein/gpqa"
    if args.dataset_name == "GPQA-Diamond":
        dataset = load_dataset(load_dataset_path, "gpqa_diamond", split="train", trust_remote_code=True)
    else:
        dataset = load_dataset(load_dataset_path, "gpqa_main", split="train", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    def format_gpqa_query(example):
        query = example["Question"]
        query += "\n\nChoose the correct answer from the following options:"
        query += f"\n(A) {example['Correct Answer']}"
        query += f"\n(B) {example['Incorrect Answer 1']}"
        query += f"\n(C) {example['Incorrect Answer 2']}"
        query += f"\n(D) {example['Incorrect Answer 3']}"
        return query
    def format_gpqa_gt(example):
        answer = f"(A) {example['Correct Answer']}"
        return answer
    data_list = [
        {
            "query": format_gpqa_query(example),
            "gt": format_gpqa_gt(example),
            "tag": [args.dataset_name, example["High-level domain"], example["Subdomain"], example["Writer's Difficulty Estimate"]],
            "source": args.dataset_name
        }
        for example in dataset
    ]
    data_list = shuffle_and_sample(data_list, args.num2sample)

# Multimodal datasets
# load MME
elif args.dataset_name == "MME":
    load_dataset_path = args.dataset_path if args.dataset_path else "lmms-lab/MME"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    
    data_list = []
    for example in dataset:
        os.makedirs(os.path.join(image_save_path,example['question_id'].split('/')[0]), exist_ok=True)
        image_path = example['question_id']
        local_image_path = os.path.join(image_save_path, image_path)
        img = example["image"]
        img.save(local_image_path)

        data = {
            "image_path": local_image_path,
            "query": example["question"],
            "gt": example["answer"],
            "tag": [args.dataset_name, "comprehensive", example["category"]],
            "source": args.dataset_name
        }
        data_list.append(data)
    data_list = shuffle_and_sample(data_list, args.num2sample)


elif args.dataset_name == "ScienceQA":  
    load_dataset_path = args.dataset_path if args.dataset_path else "derek-thomas/ScienceQA"

# load MathVista
elif args.dataset_name == "MathVista": 
    load_dataset_path = args.dataset_path if args.dataset_path else "AI4Math/MathVista"
    dataset = load_dataset(load_dataset_path, split="testmini", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = []
    
    def format_dynamic_mathvista_gt(example):
        num_choices = len(example["choices"])
        choice_list = [chr(65 + i) for i in range(num_choices)]

        index = example['choices'].index(example['answer'])
        answer = f"({choice_list[index]})"
        return answer

    for example in dataset:
        os.makedirs(os.path.join(image_save_path,example['image'].split('/')[0]), exist_ok=True)
        image_path = example['image']
        local_image_path = os.path.join(image_save_path, image_path)
        img = example["decoded_image"]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(local_image_path)

        data = {
            "image_path": local_image_path,
            "query": example["query"],
            "gt":  format_dynamic_mathvista_gt(example) if example["question_type"]=="multi_choice" else example["answer"],
            "tag": ["math"],
            "source": args.dataset_name
        }
        data_list.append(data)
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load MathVision
elif args.dataset_name == "MathVision":
    load_dataset_path = args.dataset_path if args.dataset_path else "MathLLMs/MathVision"
    # Use download_mode to force re-download
    try:
        dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True, download_mode="force_redownload")
    except Exception as e:
        # If force_redownload fails, try ignoring cache
        print(f"Warning: force_redownload failed: {e}")
        print("Trying to load with ignore_verifications=True...")
        dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True, ignore_verifications=True)
    print(f"{'='*50}\n", dataset)
    data_list = []

    def format_dynamic_mathvision_query(example):
        num_choices = len(example["options"])
        
        query = f"""The following is a multiple-choice question:
{example["question"]}

Choose the correct answer from the following options:"""
        
        for i in range(num_choices):
            query += f"\n({chr(65 + i)}) {example['options'][i]}"
        return query

    for example in dataset:
        os.makedirs(os.path.join(image_save_path,example['image'].split('/')[0]), exist_ok=True)
        image_path = example['image']
        local_image_path = os.path.join(image_save_path, image_path)
        img = example["decoded_image"]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(local_image_path)

        if len(example["options"])>1 and example["options"][0]!="A":
            query = format_dynamic_mathvision_query(example)
            gt = f"({example['answer']})"
        elif len(example["options"])>1 and example["options"][0]=="A":
            query = f"""The following is a multiple-choice question: 
{example["question"]} 

Choose the correct answer from the image."""
            gt = f"({example['answer']})"
        else:
            query = example["question"]
            gt = example["answer"]

        data = {
            "image_path": local_image_path,
            "query": query,
            "gt": gt,
            "tag": ["math",example["subject"],f"level:{example['level']}"],
            "source": args.dataset_name
        }
        data_list.append(data)
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load PathVQA
elif args.dataset_name == "PathVQA":
    load_dataset_path = args.dataset_path if args.dataset_path else "flaviagiammarino/path-vqa"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = []

    cnt = 1
    for example in dataset:
        local_image_path = os.makedirs(image_save_path, exist_ok=True)
        local_image_path = os.path.join(image_save_path, f"image_{cnt}.png")
        img = example["image"]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(local_image_path)
        cnt+=1
        data = {
            "image_path": local_image_path,
            "query": example["question"],
            "gt": example["answer"],
            "tag": ["medical"],
            "source": args.dataset_name
        }
        data_list.append(data)
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load TextVQA 还没有修改
elif args.dataset_name == "TextVQA":
    load_dataset_path = args.dataset_path if args.dataset_path else "lmms-lab/textvqa"
    dataset = load_dataset(load_dataset_path, split="test", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = []

    cnt = 1
    for example in dataset:
        local_image_path = os.makedirs(image_save_path, exist_ok=True)
        local_image_path = os.path.join(image_save_path, f"image_{cnt}.png")
        img = example["image"]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(local_image_path)
        cnt+=1
        data = {
            "image_path": local_image_path,
            "query": example["question"],
            "gt": example["answers"][0],
            "tag": ["OCR"],
            "source": args.dataset_name
        }
        data_list.append(data)
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load MME-Reasoning
elif args.dataset_name == "MME-reasoning":
    tsv_path = "/home/usr/datasets/MME_Reasoning.tsv"
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"{'='*50}\nLoaded {len(df)} examples from {tsv_path}")
    data_list = []

    save_root = "/home/usr/liao/MAD-exp/datasets/data/MME_reasoning/images"
    os.makedirs(save_root, exist_ok=True)

    for i, example in df.iterrows():
        img_data = base64.b64decode(example["image"])
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        image_name = f"MME_{i:05d}.png"
        local_image_path = os.path.join(save_root, image_name)
        img.save(local_image_path)

        if example["question_type"] == "choice":
            query = f"""The following is a multiple-choice question: 
{example["question"]} 
Choose the correct answer from the image."""
        else:
            query = example["question"]
        gt = f"{example['answer']}"

        data = {
            "image_path": local_image_path,
            "query": query,
            "gt": gt,
            "tag": [example["difficulty"], "reasoning", example["reasoning_type"], example["capability"]],
            "source": args.dataset_name
        }
        data_list.append(data)
    data_list = shuffle_and_sample(data_list, args.num2sample)

# load VisualPuzzles
elif args.dataset_name == "VisualPuzzles":
    load_dataset_path = args.dataset_path if args.dataset_path else "neulab/VisualPuzzles"
    dataset = load_dataset(load_dataset_path, split="train", trust_remote_code=True)
    print(f"{'='*50}\n", dataset)
    data_list = []

    def format_dynamic_visualpuzzles_query(example):
        num_choices = len(example["options"])
        
        query = f"""The following is a multiple-choice question:
{example["question"]}

Choose the correct answer from the following options:"""
        
        for i in range(num_choices):
            query += f"\n({chr(65 + i)}) {example['options'][i]}"
        return query

    cnt = 1
    for example in dataset:
        local_image_path = os.makedirs(image_save_path, exist_ok=True)
        local_image_path = os.path.join(image_save_path, f"image_{cnt}.png")
        img = example["image"]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(local_image_path)
        cnt+=1

        if example["options"]!=None:
            query = format_dynamic_visualpuzzles_query(example)
        else:
            query = f"""The following is a multiple-choice question: 
{example["question"]} 

Choose the correct answer from the image."""
        gt = f"({example['answer']})"
        data = {
            "image_path": local_image_path,
            "query": query,
            "gt": gt,
            "tag": [example["category"],"reasoning"],
            "source": args.dataset_name
        }
        data_list.append(data)
    data_list = shuffle_and_sample(data_list, args.num2sample)

else:
    raise ValueError(f"Dataset {args.dataset_name} not supported.")

print(f">> A data sample from the pool:\n{data_list[0]}")

print(f"{'='*50}\n There are {len(data_list)} queries in the pool.")

with open(save_path, 'w') as output_json:
    json.dump(data_list, output_json, indent=4)
