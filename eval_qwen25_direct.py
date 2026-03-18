import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

class ImageCaptionEvaluator:
    SYSTEM_MESSAGE = """Determine if the given image and text caption match. 
    Respond with only 'Yes' or 'No'.
    """

    def __init__(self, args, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", adapter_path: str = None):
        self.accelerator = Accelerator()
        self.args = args
        self.model, self.processor = self._load_model(model_id, adapter_path)

    def _load_model(self, model_id: str, adapter_path: str = None) -> Tuple[AutoModelForVision2Seq, AutoProcessor]:
        """Load the model and processor with quantization."""
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

        if adapter_path is not None:
            model.load_adapter(adapter_path)

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        return model, processor

    def _format_data(self, image: Image.Image, caption: str) -> Dict:
        """Format the input data for the model."""
        return {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.SYSTEM_MESSAGE}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Does this caption match the image?: {caption}"},
                        {"type": "image", "image": image}
                    ]
                }
            ]
        }

    def _generate_response(self, sample: Dict) -> str:
        """Generate a single response for the given sample."""
        messages = self._format_data(sample["image"], sample["caption"])["messages"]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.dtype)

        inputs = self.accelerator.prepare(inputs)
        inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated_ids = self.model.generate(**inputs, max_new_tokens=50)

        full_response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # Extract the part after 'assistant'
        if 'assistant' in full_response.lower():
            response = full_response.lower().split('assistant')[-1].strip()
        else:
            response = full_response.strip()
        return response

    def evaluate_caption(self, image_path: str, caption: str) -> Tuple[bool, str]:
        """Evaluate if a caption matches an image and return both decision and raw response."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        sample = {"image": image, "caption": caption}
        response = self._generate_response(sample)
        
        # Directly check for Yes/No in the response
        response_lower = response.lower()
        has_yes = "yes" in response_lower
        has_no = "no" in response_lower
        
        # If both Yes and No appear, or neither appears, consider it as no match
        if has_yes and not has_no:
            match = True
        else:
            match = False
            
        return match, response

    def evaluate_dataset(self, test_data: List[Dict], image_dir: str) -> Dict:
        """Evaluate performance on a test dataset with resume capability"""
        if os.path.exists(self.args.output_json):
            with open(self.args.output_json, "r") as f:
                results = json.load(f)
            print(f"Loaded existing results with {len(results['per_item_results'])} items")
        else:
            results = {
                "TP": 0, "TN": 0, "FP": 0, "FN": 0,
                "per_item_results": {},
            }

        for item in tqdm(test_data):
            img_id = item.get("image_id")
            if img_id is None and "image_name" in item:
                img_id = item["image_name"].split("_")[0]  
            existing_item = results["per_item_results"].get(img_id)

            # Skip if already evaluated unless forced
            if existing_item:
                prev_gt_status = existing_item["gt_match"]
                prev_target_status = existing_item["target_match"]
                
                if not (prev_gt_status == False or prev_target_status == True):
                    print(f"Skipping already correct item: {img_id}")
                    continue
                    
                results["TP" if prev_gt_status else "FN"] -= 1
                results["TN" if not prev_target_status else "FP"] -= 1

            # Process image evaluation based on dataset type
            if self.args.dataset == "COCO2017":
                # COCO2017 format
                img_path = os.path.join(image_dir, item["image_name"])
            else:
                # ImageNet format
                img_path = os.path.join(image_dir, f"ILSVRC2012_val_{img_id}.png")
            
            if not os.path.exists(img_path):
                print(f"Image {img_id} not found at {img_path}")
                continue

            # Evaluate captions and get both decision and raw response
            gt_match, gt_response = self.evaluate_caption(img_path, item["gt_caption"])
            target_match, target_response = self.evaluate_caption(img_path, item["target_caption"])

            # Update counters
            results["TP" if gt_match else "FN"] += 1
            results["TN" if not target_match else "FP"] += 1

            # Build new result record with responses
            item_result = {
                "image_id": img_id,
                "gt_caption": item["gt_caption"],
                "gt_match": gt_match,
                "gt_response": gt_response,
                "target_caption": item["target_caption"],
                "target_match": target_match,
                "target_response": target_response,
            }
            
            results["per_item_results"][img_id] = item_result

            print(
                f"Image: {img_id}",
                f"GT Match: {'✅' if gt_match else '❌'}",
                f"Target Match: {'⚠️' if target_match else '✓'}"
            )

            # Save incremental updates
            with open(self.args.output_json, "w") as f:
                json.dump({
                    "TP": results["TP"],
                    "TN": results["TN"],
                    "FP": results["FP"],
                    "FN": results["FN"],
                    "per_item_results": results["per_item_results"],
                }, f, indent=4, ensure_ascii=False)

        # Calculate metrics
        total = len(test_data)
        results["accuracy"] = (results["TP"] + results["TN"]) / (2 * total) if total > 0 else 0
        results["recall"] = results["TP"] / (results["TP"] + results["FN"]) if (results["TP"] + results["FN"]) > 0 else 0
        results["fscore"] = (2 * results["TP"]) / (2 * results["TP"] + results["FP"] + results["FN"]) if (2 * results["TP"] + results["FP"] + results["FN"]) > 0 else 0

        return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate image-caption matching with Qwen2.5-VL model")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                       help="HuggingFace model ID")
    parser.add_argument("--adapter_path", type=str, default=None,
                       help="Path to adapter weights (optional)")
    parser.add_argument("--dataset", type=str, choices=["ImageNet", "COCO2017"], default="COCO2017",
                       help="Dataset to evaluate on")
    parser.add_argument("--test_json", type=str, default="test_images_ids.json",
                       help="Path to JSON file containing test data")
    parser.add_argument("--image_dir", type=str, default="/home/yjli/Agent/agent-attack/COCO2017/adv_images/step_0299",
                       help="Directory containing test images", choices=["ImageNet/MF_clipattack/data/adversarial_images", "/home/yjli/Agent/agent-attack/COCO2017/adv_images/step_0299"])
    parser.add_argument("--output_json", type=str, default="evaluation_results_direct.json",
                       help="Path to save evaluation results")
    # COCO2017 specific arguments
    return parser.parse_args()


def load_coco_data(args):
    """Load COCO2017 test data."""
    with open(f"{args.dataset}/{args.test_json}") as f:
        test_ids = json.load(f)
    
    test_data = []
    for item in test_ids:
        # img_id = str(item['image_id']).zfill(12)
        img_id = str(item['image_id'])
        test_data.append({
            'image_name': f'{img_id}.png',
            'gt_caption': item['gt_caption'],
            'target_caption': item['target_caption']
        })
    return test_data

def main():
    args = parse_args()

    # Load test data based on dataset
    if args.dataset == "COCO2017":
        test_data = load_coco_data(args)
        image_dir = args.image_dir
    else:  # ImageNet
        with open(args.test_json) as f:
            test_data = json.load(f)
        image_dir = args.image_dir
        
    args.output_json = f"{args.dataset}/{args.output_json}"

    evaluator = ImageCaptionEvaluator(args=args, model_id=args.model_id, adapter_path=args.adapter_path)
    results = evaluator.evaluate_dataset(test_data, image_dir)

    # Save results
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\nEvaluation Summary:")
    print(f"Dataset: {args.dataset}")
    print(f"TP: {results['TP']}, TN: {results['TN']}, FP: {results['FP']}, FN: {results['FN']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F-score: {results['fscore']:.4f}")


if __name__ == "__main__":
    main()