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
    Qwen2VLProcessor
)
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

class ImageCaptionEvaluator:
    SYSTEM_MESSAGE = """Determine if the given image and text caption match. Carefully check the color and the object.  
    Firstly, give the analysis carefully step by step. The caption does not need to strictly reflect all elements of the image. 
    For instance, when a picture contains both grassland and bare ground but is predominantly grassland, 'grassland' can be used as the caption. 
    Finally, give Yes or No.
    The response should be in json format. The json dictionary should have two keywords, 'Analysis' and 'Final Answer'
    Answer example: 
    ```json\n
    {"Analysis":"The provided caption is ##A red motocycle parked on a green floor.##\n 
    I will analysis these caption and the image step by step.\n
    **Step 1: The objects in the images:** The objects in the image are a motorcycle and a floor.
    **Step 2: Color of the Motorcycles**: The motorcycles in the image is indeed red. 
    **Step 3. Color of the Floor included red, green and possibly other shades but the major color is green.
    Based on the analysis:
    - The motorcycles are red.
    - The floor is mixture of red and green but the main color is green.
    Given these discrepancies, the provided caption can accurately describe the image.",
    "Final Answer": "Yes"}
    ```
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
                        {"type": "text", "text": f"Do you think the caption: {caption} matches with the image?"},
                        {"type": "image", "image": image}
                    ]
                }
            ]
        }

    def _generate_responses(self, sample: Dict, num_sequences: int = 5) -> List[str]:
        """Generate multiple responses for the given sample."""
        messages = self._format_data(sample["image"], sample["caption"])["messages"]

        # Preparation for inference
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        

        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.dtype)

        # Use accelerator to prepare inputs (handles device placement)
        inputs = self.accelerator.prepare(inputs)

        # Ensure all inputs are on the same device
        inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}

        # Inference: Generation of the output
        # with torch.autoccast(device_type="cuda", dtype=torch.bfloat16):
        generated_ids = self.model.generate(**inputs, max_new_tokens=256, top_p=1.0, do_sample=True, temperature=0.8, num_return_sequences=num_sequences )

        return [
            text.split("assistant")[-1].split("<|im_end|>")[0].strip() 
            for text in self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=False  # 必须保留特殊标记用于分割
            )
        ]

    @staticmethod
    def _majority_vote(responses: List[str]) -> bool:
        """Enhanced majority vote with fallback text parsing when JSON fails."""
        yes_count = 0
        valid_responses = 0
        yes_keywords = {"yes"}
        no_keywords = {"no"}
        
        for response in responses:
            if not response.strip():
                continue
                
            answer = None
            response_lower = response.lower()
            
            # First try JSON parsing
            try:
                # Extract JSON part (handle markdown code blocks)
                json_str = response[response.find('{'):response.rfind('}')+1]
                data = json.loads(json_str)
                
                # Case-insensitive key search for final answer
                final_answer_key = next(
                    (k for k in data if k.lower().replace(" ", "_") == "final_answer"),
                    None
                )
                
                if final_answer_key:
                    answer = str(data[final_answer_key]).lower().strip()
                    
            except (json.JSONDecodeError, KeyError, AttributeError, ValueError):
                # JSON parse failed, fallback to text pattern matching
                pass
                
            # If JSON parsing didn't find answer, try direct text matching
            if answer is None:
                final_answer_idx = response_lower.find("final answer")
                # Extract text after Final Answer marker
                if final_answer_idx >= 0:
                    answer_start = final_answer_idx + len("final answer")
                    answer_text = response_lower[answer_start:]
                    # Check for affirmative keywords
                    if any(kw in answer_text for kw in yes_keywords):
                        answer = "yes"
                    # Check for negative keywords only if no affirmative found
                    elif any(kw in answer_text for kw in no_keywords):
                        answer = "no"
                    
            # Count valid responses
            if answer is not None:
                if any(kw in answer for kw in yes_keywords):
                    yes_count += 1
                    valid_responses += 1
                elif any(kw in answer for kw in no_keywords):
                    valid_responses += 1
                    
        if valid_responses == 0:
            yes_count = 0
        
        print("valid responses number:", valid_responses)
            
        # Return True only when majority is clearly "yes"
        return yes_count > valid_responses // 2

    def evaluate_caption(self, image_path: str, caption: str, num_sequences: int = 5) -> Tuple[bool, List[str]]:
        """Evaluate if a caption matches an image."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        sample = {"image": image, "caption": caption}
        responses = self._generate_responses(sample, num_sequences)
        return self._majority_vote(responses), responses

    def evaluate_dataset(self, test_data: List[Dict], image_dir: str) -> Dict:
        """Evaluate performance on a test dataset with resume capability"""
        # Initialize or load existing results
        if os.path.exists(self.args.output_json):
            with open(self.args.output_json, "r") as f:
                results = json.load(f)
            print(f"Loaded existing results with {len(results['per_item_results'])} items")
        else:
            results = {
                "TP": 0, "TN": 0, "FP": 0, "FN": 0,
                "per_item_results": {},
                "all_responses": []
            }

        for item in tqdm(test_data):
            img_id = item.get("image_id")
            if img_id is None and "image_name" in item:
                img_id = item["image_name"].split("_")[0] if self.args.dataset == "ImageNet" else item["image_name"].replace(".png", "")
            existing_item = results["per_item_results"].get(img_id)

            # Skip correct predictions unless forced
            if existing_item:
                prev_gt_status = existing_item["gt_match"]
                prev_target_status = existing_item["target_match"]
                
                # Check if need re-evaluation
                if not (prev_gt_status == False or prev_target_status == True):
                    print(f"Skipping already correct item: {img_id}")
                    continue
                    
                # Rollback previous counters
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

            # Re-evaluate captions
            gt_match, gt_responses = self.evaluate_caption(img_path, item["gt_caption"])
            target_match, target_responses = self.evaluate_caption(img_path, item["target_caption"])

            # Update counters
            results["TP" if gt_match else "FN"] += 1
            results["TN" if not target_match else "FP"] += 1

            # Build new result record
            item_result = {
                "image_id": img_id,
                "gt_caption": item["gt_caption"],
                "gt_match": gt_match,
                "gt_responses": gt_responses,
                "target_caption": item["target_caption"],
                "target_match": target_match,
                "target_responses": target_responses
            }
            
            # Update records
            results["per_item_results"][img_id] = item_result

            print(
                f"TP: {results['TP']}(+{1 if gt_match else 0})",
                f"TN: {results['TN']}(+{1 if not target_match else 0})",
                f"FP: {results['FP']}(+{1 if target_match else 0})",
                f"FN: {results['FN']}(+{1 if not gt_match else 0})",
                f"\nImage: {img_id}",
                f"\nGT Caption: {item['gt_caption']} → {'✅' if gt_match else '❌'}",
                f"\nTarget Caption: {item['target_caption']} → {'⚠️' if target_match else '✓'}"
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

def load_coco_data(args):
    """Load COCO2017 test data."""
    with open(args.test_json) as f:
        test_ids = json.load(f)
    
    test_data = []
    for item in test_ids:
        img_id = str(item['image_id'])
        test_data.append({
            'image_name': f'{img_id}.png',
            'gt_caption': item['gt_caption'],
            'target_caption': item['target_caption'],
            'image_id': img_id
        })
    return test_data

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
    parser.add_argument("--output_json", type=str, default="evaluation_results_CoT.json",
                       help="Path to save evaluation results")
    # COCO2017 specific arguments
    parser.add_argument("--num_sequences", type=int, default=5,
                       help="Number of sequences to generate per evaluation")
    return parser.parse_args()

def main():
    args = parse_args()
    args.output_json = f"{args.dataset}/{args.output_json}"
    args.test_json = f"{args.dataset}/{args.test_json}"

    # Load test data based on dataset type
    if args.dataset == "COCO2017":
        test_data = load_coco_data(args)
        # Set correct image directory format for COCO
        args.image_dir = os.path.join(args.image_dir, "")  # Ensures trailing slash
    else:
        # Load ImageNet data as before
        with open(args.test_json) as f:
            test_data = json.load(f)
        
    # Initialize evaluator
    evaluator = ImageCaptionEvaluator(args=args, model_id=args.model_id, adapter_path=args.adapter_path)

    # Run evaluation
    results = evaluator.evaluate_dataset(test_data, args.image_dir)

    # Save results
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # Print summary
    print("\nEvaluation Summary:")
    print(f"Dataset: {args.dataset}")
    print(f"TP: {results['TP']}, TN: {results['TN']}, FP: {results['FP']}, FN: {results['FN']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F-score: {results['fscore']:.4f}")

if __name__ == "__main__":
    main()
