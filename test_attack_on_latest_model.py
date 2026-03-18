"""
Script to load previously successful attack prompts or webpage images 
and test them on the latest model (e.g., gpt-5).
"""
import argparse
import json
import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import requests
import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from scripts.adaptive_pipeline_attack import load_adv, test_single_case
from scripts.LLMwapper import OpenAIWrapper, QwenWrapper, get_llm_wrapper


def load_successful_attacks(attack_dir: str = "exp_data/agent_adv/") -> List[str]:
    """
    Load task IDs of successful attacks from the attack directory.
    
    Args:
        attack_dir: Directory containing attack results
        
    Returns:
        List of task IDs that had successful attacks
    """
    successful_tasks = []
    
    # Look for successful attack directories
    for task_dir in os.listdir(attack_dir):
        task_path = os.path.join(attack_dir, task_dir)
        if os.path.isdir(task_path):
            # Check if this is a successful attack by looking for attack files
            attack_caption_file = os.path.join(task_path, "bim_caption_attack_caption.txt")
            attack_image_file = os.path.join(task_path, "bim_caption_attack_image.png")
            
            if os.path.exists(attack_caption_file) or os.path.exists(attack_image_file):
                successful_tasks.append(task_dir)
    
    return successful_tasks


def load_attack_prompts_and_images(task_id: str, attack_dir: str = "exp_data/agent_adv/") -> Dict:
    """
    Load the attack prompt and image for a specific task.
    
    Args:
        task_id: The task ID to load
        attack_dir: Directory containing attack results
        
    Returns:
        Dictionary containing attack prompt, image, and related data
    """
    task_path = os.path.join(attack_dir, task_id)
    
    result = {
        "task_id": task_id,
        "attack_prompt": None,
        "attack_image": None,
        "config_data": None,
        "attack_data": None
    }
    
    # Load attack caption (prompt)
    attack_caption_file = os.path.join(task_path, "bim_caption_attack_caption.txt")
    if os.path.exists(attack_caption_file):
        with open(attack_caption_file, "r") as f:
            result["attack_prompt"] = f.read().strip()
    
    # Load attack image
    attack_image_file = os.path.join(task_path, "bim_caption_attack_image.png")
    if os.path.exists(attack_image_file):
        result["attack_image"] = Image.open(attack_image_file)
    
    # Load config data
    config_file = os.path.join(task_path, "config.json")
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            result["config_data"] = json.load(f)
    
    # Load attack-specific data
    data_file = os.path.join(task_path, "data.json")
    if os.path.exists(data_file):
        with open(data_file, "r") as f:
            result["attack_data"] = json.load(f)
    
    return result


def test_on_latest_model(
    task_data: Dict, 
    model_name: str = "gpt-4o", 
    temperature: float = 0.7
) -> Dict:
    """
    Test the loaded attack on the latest model.
    
    Args:
        task_data: Dictionary containing attack data
        model_name: Name of the model to test on
        temperature: Temperature for generation
        
    Returns:
        Dictionary containing test results
    """
    print(f"Testing on model: {model_name}")
    
    # Get the LLM wrapper for the specified model
    try:
        llm = get_llm_wrapper(model_name, temperature)
    except ValueError as e:
        print(f"Error initializing model {model_name}: {e}")
        # Fallback to a default model
        llm = get_llm_wrapper("gpt-4o-mini", temperature)
        print(f"Falling back to gpt-4o-mini")
    
    # Prepare the test
    config_file_path = f"exp_data/agent_adv/{task_data['task_id']}/config.json"
    
    # Create args namespace for compatibility with existing functions
    class Args:
        def __init__(self):
            self.attack = "bim_caption"  # Default attack type
            self.model = model_name
            self.captioning_model = "liuhaotian/llava-v1.5-7b"
            self.eval_captioning_model = "liuhaotian/llava-v1.5-7b"
            self.observation_type = "image_som"
            self.action_set_tag = "som"
            self.max_steps = 30
            self.parsing_failure_th = 3
            self.repeating_action_failure_th = 5
            self.viewport_width = 1280
            self.viewport_height = 2048
            self.save_trace_enabled = False
            self.current_viewport_only = True
            self.result_dir = "./test_results"
    
    args = Args()
    
    # If we have an attack prompt, test it
    if task_data["attack_prompt"]:
        print(f"Testing attack prompt: {task_data['attack_prompt'][:100]}...")
        
        # Try to run the test using the existing function
        try:
            # Note: We're adapting the test_single_case function to work with our data
            user_intent, obs_text, agent_output, score = test_single_case(
                args, 
                config_file_path, 
                task_data["attack_prompt"]
            )
            
            return {
                "task_id": task_data["task_id"],
                "model": model_name,
                "attack_prompt": task_data["attack_prompt"],
                "user_intent": user_intent,
                "observation_text": obs_text,
                "agent_output": agent_output,
                "score": score,
                "status": "completed"
            }
        except Exception as e:
            print(f"Error during testing: {e}")
            return {
                "task_id": task_data["task_id"],
                "model": model_name,
                "attack_prompt": task_data["attack_prompt"],
                "error": str(e),
                "status": "error"
            }
    else:
        print("No attack prompt found for this task")
        return {
            "task_id": task_data["task_id"],
            "model": model_name,
            "status": "no_attack_prompt"
        }


def test_on_multiple_models(task_data: Dict, model_list: List[str]) -> List[Dict]:
    """
    Test the attack on multiple models.
    
    Args:
        task_data: Dictionary containing attack data
        model_list: List of model names to test
        
    Returns:
        List of test results for each model
    """
    results = []
    
    for model_name in model_list:
        print(f"\n--- Testing {task_data['task_id']} on {model_name} ---")
        result = test_on_latest_model(task_data, model_name)
        results.append(result)
        
        # Print summary
        if result.get("status") == "completed":
            print(f"Score on {model_name}: {result['score']}")
        else:
            print(f"Status on {model_name}: {result.get('status', 'unknown')}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test previous successful attacks on latest models")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to test on (e.g., gpt-4o, gpt-4o-mini)")
    parser.add_argument("--all-models", action="store_true", help="Test on multiple models")
    parser.add_argument("--task-id", type=str, help="Specific task ID to test (otherwise test all successful attacks)")
    parser.add_argument("--output-dir", type=str, default="./test_results", help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which tasks to test
    if args.task_id:
        task_ids = [args.task_id]
    else:
        print("Loading successful attacks...")
        task_ids = load_successful_attacks()
        print(f"Found {len(task_ids)} successful attack tasks")
    
    if not task_ids:
        print("No successful attacks found to test")
        return
    
    # Determine which models to test
    if args.all_models:
        model_list = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        if args.model not in model_list:
            model_list.append(args.model)
    else:
        model_list = [args.model]
    
    print(f"Testing on models: {model_list}")
    
    # Test each task
    all_results = []
    
    for task_id in task_ids[:5]:  # Limit to first 5 for initial testing
        print(f"\n{'='*60}")
        print(f"Testing task: {task_id}")
        print(f"{'='*60}")
        
        # Load attack data
        task_data = load_attack_prompts_and_images(task_id)
        
        if not task_data["attack_prompt"] and not task_data["attack_image"]:
            print(f"No attack data found for {task_id}, skipping...")
            continue
        
        # Test on selected models
        results = test_on_multiple_models(task_data, model_list)
        all_results.extend(results)
        
        # Save intermediate results
        intermediate_file = os.path.join(args.output_dir, f"intermediate_results_{task_id}.json")
        with open(intermediate_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved intermediate results to {intermediate_file}")
    
    # Save all results
    results_file = os.path.join(args.output_dir, "all_test_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll results saved to {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful_transfers = 0
    total_tests = 0
    
    for result in all_results:
        if result.get("status") == "completed":
            total_tests += 1
            if result["score"] == 1:  # Assuming 1 means successful attack transfer
                successful_transfers += 1
    
    if total_tests > 0:
        transfer_success_rate = successful_transfers / total_tests
        print(f"Attack transfer success rate: {transfer_success_rate:.2%} ({successful_transfers}/{total_tests})")
    
    print(f"Tested on models: {set(r.get('model') for r in all_results if 'model' in r)}")


if __name__ == "__main__":
    main()