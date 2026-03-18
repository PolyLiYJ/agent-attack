import os
from load_coco2017_dataset import load_dataset, read_image_ids, load_pil_image
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
)
from accelerate import Accelerator
import torch
import random
from tqdm import tqdm
from PIL import Image
import requests
import json
from typing import List, Dict
from qwen_vl_utils import process_vision_info

# Unset proxies
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

# DeepSeek-V3 API Config
DEEPSEEK_API_URL = "https://api.deepseek.com/v3/chat/completions"
DEEPSEEK_API_KEY = "sk-30819ddcc707468e87303f6eecc293b7"  # Replace with your actual key of deepseek

from openai import OpenAI  # DeepSeek-compatible SDK
import json
from tqdm import tqdm
import random

# ====================
# 1. Configuration
# ====================

# Initialize clients
deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# ====================
# 2. Model Loading
# ====================
def load_model(model_id, adapter_path=None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    # adapter_path = "/home/yjli/Agent/agent-attack/qwen2-7b-instruct-caption-advtraining"
    if adapter_path is not None:
        model.load_adapter(adapter_path) # load the adapter and activate
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor

# ====================
# 3. Dataset Preparation
# ====================
def format_data(sample):
    system_message = "Give the caption of the image."

    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [
                    # {"type": "text", "text": f"Caption: {sample['caption']}"},
                    {"type": "image", "image": sample["image"]}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["caption"]}]
            }
        ]
    }

# ====================
# 4. Caption Generation
# ====================

accelerator = Accelerator()

def generate_description(sample, model, processor):
    # Prepare messages
    # messages = sample["messages"]
    messages = format_data(sample)["messages"]

    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    

    # Prepare inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.dtype)

    # Use accelerator to prepare inputs (handles device placement)
    inputs = accelerator.prepare(inputs)

    # Ensure all inputs are on the same device
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

    # Inference: Generation of the output
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated_ids = model.generate(**inputs, max_new_tokens=256, top_p=1.0, do_sample=True, temperature=0.8)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

# ====================
# 5. DeepSeek Evaluation
# ====================
def evaluate_caption(reference: str, generated: str) -> dict:
    prompt = f"""Evaluate this image caption:

    **Reference Caption**: {reference}
    **Generated Caption**: {generated}

    Provide scores (1-5) for:
    - Accuracy
    - Detail
    - Fluency
    - Overall Quality

    Return JSON format only, example:
    {{
        "accuracy": 4,
        "detail": 3,
        "fluency": 5,
        "overall": 4,
        "feedback": "Good but missed some details"
    }}"""
    
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are an image caption evaluator. Respond only with JSON."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.2
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"error": "Failed to parse evaluation"}

# ====================
# 6. Main Evaluation Loop
# ====================
def run_evaluation(test_samples, model, processor, num_samples=100 ):
    results = []
    random.seed(42)

    samples = random.sample(test_samples, min(num_samples, len(test_samples)))
    
    for sample in tqdm(samples, desc="Evaluating"):
        img = sample["image"]
        true_caption = sample["caption"]
        
        # Generate caption
        pred_caption = generate_description(sample, model, processor)
        
        # Evaluate with DeepSeek
        evaluation = evaluate_caption(true_caption, pred_caption)
        
        # Save results
        results.append({
            "image_name": sample["img_name"],
            "true_caption": true_caption,
            "generated_caption": pred_caption,
            "evaluation": evaluation
        })
        
        # Print progress
        print(f"\nImage: {sample['img_name']}")
        print(f"True: {true_caption}")
        print(f"Generated: {pred_caption}")
        print(f"Evaluation: {json.dumps(evaluation, indent=2)}\n")
    
        # Save full results
        with open("COCO2017/Qwen2_5_evaluation_results_advtrained_model.json", "w") as f:
            json.dump(results, f, indent=2)
    
    return results

def calculate_avg_scores(results):
    return {
        "accuracy": sum(r["evaluation"].get("accuracy", 0) for r in results) / len(results),
        "detail": sum(r["evaluation"].get("detail", 0) for r in results) / len(results),
        "fluency": sum(r["evaluation"].get("fluency", 0) for r in results) / len(results),
        "overall": sum(r["evaluation"].get("overall", 0) for r in results) / len(results),
    }
    
# ====================
# 7. Run Evaluation
# ====================
if __name__ == "__main__":
    # train_ids_file = "COCO2017/train_image_ids.txt"  # Update with your train IDs file path
    # test_ids_file = "COCO2017/test_image_ids.txt" 
    system_message = "Give the caption of the image."

    new_train_ids_file = "COCO2017/negtive_correct_train_ids.txt"  # 新训练集文件
    new_test_ids_file = "COCO2017/negtive_correct_test_ids.txt"    # 新测试集文件
    # img_folder = "/home/yjli/Agent/agent-attack/COCO2017/val2017"
    img_folder = "/home/yjli/Agent/agent-attack/COCO2017/adv_images/step_0099"
    train_samples, test_samples, train_dataset, test_dataset = load_dataset(new_train_ids_file, new_test_ids_file, img_folder, system_message=system_message, format_data=format_data, negtive_prompt=False)

    print("=== Baseline Evaluation ===")
    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    model, processor = load_model(model_id = MODEL_ID, adapter_path=None )
    baseline_results = run_evaluation(test_samples, model, processor)
    print("baseline results:", calculate_avg_scores(baseline_results))

    # Load adapter and evaluate
    adapter_path = "/home/yjli/Agent/agent-attack/qwen2_5-7b-instruct-caption-adv_training"
    model, processor = load_model(model_id = MODEL_ID, adapter_path=adapter_path)
    finetuned_results = run_evaluation(test_samples, model, processor)
    print("\n=== Final Scores ===")
    print("adv trained model results:", calculate_avg_scores(finetuned_results))
    
    print("Fine-tuned:", calculate_avg_scores(finetuned_results))

    # # Load adapter and evaluate
    # adapter_path = "/home/yjli/Agent/agent-attack/qwen2-7b-instruct-caption-advtraining"
    # model.load_adapter(adapter_path)
    
    # print("\n=== Fine-tuned Model Evaluation ===")
    # finetuned_results = evaluate_model(
    #     model=model,
    #     processor=processor,
    #     samples=test_samples,
    #     evaluator=deepseek_evaluator,
    #     num_samples=5
    # )
    
    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump({
            "baseline": baseline_results,
            "finetuned": finetuned_results,
            "average_score_baseline": calculate_avg_scores(baseline_results),
            "average_score_finetuned": calculate_avg_scores(finetuned_results),
        }, f, indent=2)

    print("Evaluation complete. Results saved to evaluation_results.json")
