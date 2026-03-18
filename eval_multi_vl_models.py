import argparse
import json
import os
from typing import Dict, List, Optional, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    CLIPImageProcessor,
    AutoProcessor,
    pipeline
)
from PIL import Image
import time
from tqdm import tqdm
import logging
from huggingface_hub import login
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModelEvaluator:
    def __init__(self, model_configs: List[Dict], device: str = "cuda", hf_token: Optional[str] = None):
        """
        Initialize evaluator with multiple model configurations.
        
        Args:
            model_configs: List of dictionaries containing model configurations
            device: Device to run models on
            hf_token: HuggingFace API token for accessing gated models
        """
        self.device = device
        self.models = {}
        self.processors = {}  # Combined tokenizer and image processor
        
        # Login to HuggingFace if token is provided
        if hf_token:
            try:
                login(hf_token)
                logger.info("Successfully logged in to HuggingFace")
            except Exception as e:
                logger.error(f"Failed to login to HuggingFace: {e}")
        
        for config in model_configs:
            try:
                self._load_model(config)
            except Exception as e:
                logger.error(f"Failed to load model {config['model_id']}: {e}")
                continue

    def _load_model(self, config: Dict):
        """Load a specific model and its associated components."""
        model_id = config["model_id"]
        logger.info(f"\nLoading model: {model_id}")
        
        try:
            # Load processor and model based on model type
            if "qwen" in model_id.lower():
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from transformers.models.qwen2.processing_qwen2 import Qwen2Processor
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                processor = Qwen2Processor.from_pretrained(model_id, trust_remote_code=True)
            
            elif "llama" in model_id.lower():
                from transformers import LlamaForCausalLM, LlamaTokenizer
                
                model = LlamaForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    use_auth_token=True
                )
                processor = AutoProcessor.from_pretrained(model_id, use_auth_token=True)
            
            elif "pixtral" in model_id.lower():
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            
            elif "molmo" in model_id.lower():
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            
            else:
                raise ValueError(f"Unsupported model type: {model_id}")
            
            self.models[model_id] = model
            self.processors[model_id] = processor
            
            logger.info(f"Successfully loaded {model_id}")
            
        except Exception as e:
            logger.error(f"Error loading {model_id}: {e}")
            raise

    def evaluate_image(self, model_id: str, image_path: str, caption: str) -> Tuple[bool, List[str]]:
        """
        Evaluate a single image with a specific model.
        
        Args:
            model_id: Model identifier
            image_path: Path to image file
            caption: Caption to evaluate
            
        Returns:
            Tuple of (match_result, model_responses)
        """
        try:
            model = self.models[model_id]
            processor = self.processors[model_id]
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Prepare prompt based on model type
            if "llama" in model_id.lower():
                prompt = f"<image>\nHuman: Does this image show {caption}? Please answer with yes or no and explain why.\nAssistant: "
            elif "qwen" in model_id.lower():
                prompt = f"<image>Does this image show {caption}? Please answer with yes or no and explain why."
            elif "pixtral" in model_id.lower():
                prompt = f"<image>\nDoes this image show {caption}? Please answer with yes or no and explain why."
            elif "molmo" in model_id.lower():
                prompt = f"<image>\nQuestion: Does this image show {caption}? Please answer with yes or no and explain why.\nAnswer: "
            else:
                prompt = f"Does this image show {caption}? Please answer with yes or no and explain why."
            
            # Process inputs
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            response = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Check if response indicates a match
            response_lower = response.lower()
            match = "yes" in response_lower[:50]  # Check first 50 chars for yes/no
            
            return match, [response]
            
        except Exception as e:
            logger.error(f"Error evaluating image with {model_id}: {e}")
            return False, [f"Error: {str(e)}"]

    def evaluate_dataset(self, test_data: List[Dict], image_dir: str, output_dir: str) -> Dict:
        """
        Evaluate entire dataset with all models.
        
        Args:
            test_data: List of test examples
            image_dir: Directory containing images
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing evaluation results for each model
        """
        results = {}
        
        for model_id in self.models.keys():
            logger.info(f"\nEvaluating model: {model_id}")
            model_results = {
                "TP": 0, "TN": 0, "FP": 0, "FN": 0,
                "per_item_results": {},
                "successful_examples": []
            }
            
            for item in tqdm(test_data):
                img_name = item["image_name"]
                img_id = img_name.split("_")[0].zfill(8)
                img_path = os.path.join(image_dir, f"ILSVRC2012_val_{img_id}.JPEG")
                
                if not os.path.exists(img_path):
                    logger.warning(f"Image {img_id} not found at {img_path}")
                    continue
                
                try:
                    # Evaluate ground truth caption
                    gt_match, gt_responses = self.evaluate_image(model_id, img_path, item["gt_caption"])
                    # Evaluate target caption
                    target_match, target_responses = self.evaluate_image(model_id, img_path, item["target_caption"])
                    
                    # Update counters
                    model_results["TP"] += 1 if gt_match else 0
                    model_results["TN"] += 1 if not target_match else 0
                    model_results["FP"] += 1 if target_match else 0
                    model_results["FN"] += 1 if not gt_match else 0
                    
                    # Record result
                    item_result = {
                        "image_id": img_id,
                        "gt_caption": item["gt_caption"],
                        "gt_match": gt_match,
                        "gt_responses": gt_responses,
                        "target_caption": item["target_caption"],
                        "target_match": target_match,
                        "target_responses": target_responses
                    }
                    model_results["per_item_results"][img_id] = item_result
                    
                    # Save successful examples
                    if gt_match and not target_match:
                        model_results["successful_examples"].append({
                            "image_name": img_name,
                            "image_id": img_id,
                            "gt_caption": item["gt_caption"],
                            "target_caption": item["target_caption"],
                            "gt_responses": gt_responses,
                            "target_responses": target_responses
                        })
                    
                    # Print progress
                    logger.info(
                        f"\nImage: {img_id}"
                        f"\nGT Caption: {item['gt_caption']} → {'✅' if gt_match else '❌'}"
                        f"\nTarget Caption: {item['target_caption']} → {'⚠️' if target_match else '✓'}"
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing image {img_id} with {model_id}: {e}")
                    continue
                
                # Save intermediate results every 10 images
                if len(model_results["per_item_results"]) % 10 == 0:
                    results[model_id] = model_results
                    self._save_results(results, os.path.join(output_dir, f"intermediate_results_{model_id}_{int(time.time())}.json"))
            
            # Calculate metrics
            total = len(test_data)
            model_results["accuracy"] = (model_results["TP"] + model_results["TN"]) / (2 * total) if total > 0 else 0
            model_results["recall"] = model_results["TP"] / (model_results["TP"] + model_results["FN"]) if (model_results["TP"] + model_results["FN"]) > 0 else 0
            model_results["fscore"] = (2 * model_results["TP"]) / (2 * model_results["TP"] + model_results["FP"] + model_results["FN"]) if (2 * model_results["TP"] + model_results["FP"] + model_results["FN"]) > 0 else 0
            
            results[model_id] = model_results
            
            # Save model-specific results
            self._save_results(
                {model_id: model_results},
                os.path.join(output_dir, f"results_{model_id}_{int(time.time())}.json")
            )
        
        return results

    def _save_results(self, results: Dict, output_file: str):
        """Save evaluation results to file."""
        try:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multiple VL models on ImageNet")
    parser.add_argument("--test_json", type=str, default="ImageNet/test_images.json",
                       help="Path to test data JSON file")
    parser.add_argument("--image_dir", type=str, default="ImageNet/clean_image",
                       help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Directory to save results")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="HuggingFace API token for accessing gated models")
    parser.add_argument("--models", nargs="+", default=[
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "mistralai/Pixtral-12B-2409",
        "Qwen/Qwen2-VL-7B",
        "allenai/Molmo-7B-D-0924"
    ], help="List of model IDs to evaluate")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define model configurations
    model_configs = [{"model_id": model_id} for model_id in args.models]
    
    # Load test data
    try:
        with open(args.test_json) as f:
            test_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        sys.exit(1)
    
    # Initialize evaluator
    evaluator = MultiModelEvaluator(model_configs, hf_token=args.hf_token)
    
    if not evaluator.models:
        logger.error("No models were successfully loaded. Exiting.")
        sys.exit(1)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(test_data, args.image_dir, args.output_dir)
    
    # Save final results
    output_file = os.path.join(args.output_dir, f"final_results_{int(time.time())}.json")
    evaluator._save_results(results, output_file)
    
    # Print summary
    logger.info("\nEvaluation Summary:")
    for model_id, model_results in results.items():
        logger.info(f"\nModel: {model_id}")
        logger.info(f"Total examples evaluated: {len(test_data)}")
        logger.info(f"Successful examples: {len(model_results['successful_examples'])}")
        logger.info(f"TP: {model_results['TP']}, TN: {model_results['TN']}, FP: {model_results['FP']}, FN: {model_results['FN']}")
        logger.info(f"Accuracy: {model_results['accuracy']:.4f}")
        logger.info(f"Recall: {model_results['recall']:.4f}")
        logger.info(f"F-score: {model_results['fscore']:.4f}")

if __name__ == "__main__":
    main()