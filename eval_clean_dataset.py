import argparse
import json
import os
from typing import Dict, List

from eval_qwen25_CoT import ImageCaptionEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate clean dataset with Qwen2.5-VL model")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                       help="HuggingFace model ID")
    parser.add_argument("--adapter_path", type=str, default=None,
                       help="Path to adapter weights (optional)")
    parser.add_argument("--test_json", type=str, default="/home/yjli/Agent/agent-attack/ImageNet/test_images.json",
                       help="Path to JSON file containing test data")
    parser.add_argument("--image_dir", type=str, default="/home/yjli/Agent/agent-attack/ImageNet/clean_image",
                       help="Directory containing clean images")
    parser.add_argument("--output_json", type=str, default="ImageNet/evaluation_results_clean.json",
                       help="Path to save evaluation results")
    parser.add_argument("--success_json", type=str, default="ImageNet/successful_examples.json",
                       help="Path to save successful examples")
    parser.add_argument("--num_sequences", type=int, default=1,
                       help="Number of sequences to generate per evaluation")
    return parser.parse_args()

def evaluate_and_save_successful(test_data: List[Dict], evaluator: ImageCaptionEvaluator, args) -> Dict:
    """Evaluate dataset and save successful examples."""
    # Load existing results if available
    if os.path.exists(args.output_json):
        with open(args.output_json, "r") as f:
            results = json.load(f)
        print(f"Loaded existing results with {len(results['per_item_results'])} evaluated images")
    else:
        results = {
            "TP": 0, "TN": 0, "FP": 0, "FN": 0,
            "per_item_results": {},
            # "successful_examples": []
        }

    # Load existing successful examples if available
    # if os.path.exists(args.success_json):
    #     with open(args.success_json, "r") as f:
    #         results["successful_examples"] = json.load(f)
    #     print(f"Loaded {len(results['successful_examples'])} existing successful examples")
        
    # # Get the last image ID from success_json if available
    # last_img_name = None
    # if os.path.exists(args.success_json):
    #     with open(args.success_json, "r") as f:
    #         success_data = json.load(f)
    #         if success_data and isinstance(success_data, list) and len(success_data) > 0:
    #             last_item = success_data[-1]
    #             if "image_name" in last_item:
    #                 last_img_name = last_item["image_name"]
                    
    # Find the index of the last evaluated image in test_data
    
    last_img_name = "00036682_n02107683"
    start_index = 0
    if last_img_name:
        for index, item in enumerate(test_data):
            if item["image_name"] == last_img_name:
                start_index = index + 1
                break
            
    print("start_index:", start_index)

    for item in test_data[start_index:-1]:
        img_name = item["image_name"]
        img_id = img_name.split("_")[0].zfill(8)  # 确保ID是8位数字
        
        # Skip if already evaluated
        if img_id in results["per_item_results"]:
            print(f"Skipping already evaluated image {img_id}")
            continue
            
        img_path = os.path.join(args.image_dir, f"ILSVRC2012_val_{img_id}.JPEG")
        
        if not os.path.exists(img_path):
            print(f"Image {img_id} not found at {img_path}")
            continue

        # Evaluate ground truth caption
        gt_match, gt_responses = evaluator.evaluate_caption(img_path, item["gt_caption"], args.num_sequences)
        # Evaluate target caption
        target_match, target_responses = evaluator.evaluate_caption(img_path, item["target_caption"], args.num_sequences)

        # Update counters
        results["TP"] += 1 if gt_match else 0
        results["TN"] += 1 if not target_match else 0
        results["FP"] += 1 if target_match else 0
        results["FN"] += 1 if not gt_match else 0

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
        results["per_item_results"][img_id] = item_result

        # Save successful examples (where gt_match is True and target_match is False)
        # if gt_match and not target_match:
        #     results["successful_examples"].append({
        #         "image_name": img_name,
        #         "image_id": img_id,
        #         "gt_caption": item["gt_caption"],
        #         "target_caption": item["target_caption"],
        #         "gt_responses": gt_responses,
        #         "target_responses": target_responses
        #     })

        # Print progress
        print(
            f"\nImage: {img_id}",
            f"\nGT Caption: {item['gt_caption']} → {'✅' if gt_match else '❌'}",
            f"\nTarget Caption: {item['target_caption']} → {'⚠️' if target_match else '✓'}"
        )

        # Save intermediate results
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        # Save successful examples separately only when we have a new successful example
        # if gt_match and not target_match:
        #     with open(args.success_json, "w") as f:
        #         json.dump(results["successful_examples"], f, indent=4, ensure_ascii=False)
        #     print(f"Saved {len(results['successful_examples'])} successful examples")

    # Calculate metrics
    total = len(test_data)
    results["accuracy"] = (results["TP"] + results["TN"]) / (2 * total) if total > 0 else 0
    results["recall"] = results["TP"] / (results["TP"] + results["FN"]) if (results["TP"] + results["FN"]) > 0 else 0
    results["fscore"] = (2 * results["TP"]) / (2 * results["TP"] + results["FP"] + results["FN"]) if (2 * results["TP"] + results["FP"] + results["FN"]) > 0 else 0

    return results

def main():
    args = parse_args()

    # Load test data
    with open(args.test_json) as f:
        test_data = json.load(f)

    # Initialize evaluator
    evaluator = ImageCaptionEvaluator(args=args, model_id=args.model_id, adapter_path=args.adapter_path)

    # Run evaluation and save successful examples
    results = evaluate_and_save_successful(test_data, evaluator, args)

    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total examples evaluated: {len(test_data)}")
    print(f"Successful examples (GT correct & Target incorrect): {len(results['successful_examples'])}")
    print(f"TP: {results['TP']}, TN: {results['TN']}, FP: {results['FP']}, FN: {results['FN']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F-score: {results['fscore']:.4f}")

if __name__ == "__main__":
    main()