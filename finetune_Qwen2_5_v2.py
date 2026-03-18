import argparse
import json
import os
import torch
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLProcessor
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login
from qwen_vl_utils import process_vision_info

class VisionCaptionFinetuner:
    def __init__(self, args):
        self.args = args
        self.setup_environment()
        self.load_model_and_processor()
        self.load_datasets()
        self.setup_trainer()

    def setup_environment(self):
        os.environ.pop('http_proxy', None)
        os.environ.pop('https_proxy', None)
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)
        login(token=self.args.hf_token, add_to_git_credential=True)
        torch.backends.cuda.matmul.allow_tf32 = True
        # Unset proxies


    def load_model_and_processor(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.args.model_id,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.args.model_id, 
            trust_remote_code=True
        )

    def format_caption_data(self, sample):
        return {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Give the caption of the image."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "image", "image": sample["image"]}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["caption"]}]
                }
            ]
        }

    def load_datasets(self):
        if self.args.dataset == "COCO2017":
            from load_coco2017_dataset import load_dataset
            img_folder = self.args.adv_folder if self.args.use_adv else self.args.img_folder
            self.train_samples, self.test_samples, self.train_dataset, self.test_dataset = load_dataset(
                train_ids_file=self.args.train_ids_file,
                test_ids_file=self.args.test_ids_file,
                img_folder=img_folder,
                system_message="Give the caption of the image.",
                format_data=self.format_caption_data,
                negtive_prompt=False
            )
        elif self.args.dataset == "ImageNet":
            from load_imagenet_dataset import load_dataset
            img_folder = self.args.adv_folder if self.args.use_adv else self.args.img_folder
            self.train_samples, self.test_samples, self.train_dataset, self.test_dataset = load_dataset(
                train_file=self.args.train_ids_file,
                test_file=self.args.test_ids_file,
                img_folder=img_folder,
                system_message="Give the caption of the image.",
                format_data=self.format_caption_data,
                negtive_prompt=False
            )

    def collate_fn(self, examples):
        texts = [self.processor.apply_chat_template(
            example["messages"], tokenize=False) for example in examples]
        image_inputs = [process_vision_info(example["messages"])[0] for example in examples]

        batch = self.processor(
            text=texts, 
            images=image_inputs, 
            return_tensors="pt", 
            padding=True
        )

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        if isinstance(self.processor, Qwen2VLProcessor):
            image_tokens = [151652, 151653, 151655]
        else: 
            image_tokens = [self.processor.tokenizer.convert_tokens_to_ids(
                self.processor.image_token)]
            
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
            
        batch["labels"] = labels
        return batch

    def setup_trainer(self):
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=8,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM", 
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=SFTConfig(
                output_dir=self.args.adapter_path,
                num_train_epochs=self.args.epochs,
                per_device_train_batch_size=self.args.batch_size,
                gradient_accumulation_steps=4,
                gradient_checkpointing=True,
                optim="adamw_torch_fused",
                logging_steps=5,
                save_strategy="epoch",
                learning_rate=2e-5,
                bf16=True,
                tf32=True,
                max_grad_norm=0.3,
                warmup_ratio=0.03,
                lr_scheduler_type="constant",
                push_to_hub=True,
                report_to="tensorboard",
                gradient_checkpointing_kwargs={"use_reentrant": False},
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                remove_unused_columns=False
            ),
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            data_collator=self.collate_fn,
            peft_config=peft_config,
        )

    def finetune(self):
        self.trainer.train()
        self.trainer.save_model(self.args.adapter_path)
        print(f"Model saved to {self.args.adapter_path}")

    def evaluate(self):
        from eval_qwen2_5 import run_evaluation, calculate_avg_scores
        
        print("\n=== Baseline Evaluation ===")
        baseline_results = run_evaluation(
            self.test_samples, 
            self.model, 
            self.processor,
            num_samples=30,
        )
        print("Baseline results:", calculate_avg_scores(baseline_results))

        print("\n=== Fine-tuned Evaluation ===")
        self.model.load_adapter(self.args.adapter_path)
        finetuned_results = run_evaluation(
            self.test_samples, 
            self.model, 
            self.processor,
            num_samples=30,
        )
        print("Fine-tuned results:", calculate_avg_scores(finetuned_results))
        with open("ImageNet/evaluation_results.json", "w") as f:
            json.dump({
                "baseline": baseline_results,
                "finetuned": finetuned_results,
                "average_score_baseline": calculate_avg_scores(baseline_results),
                "average_score_finetuned": calculate_avg_scores(finetuned_results),
            }, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description="Vision-Language Model Finetuning")
    
    # Model arguments
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="qwen2_5-7b-instruct-caption-adv_training-ImageNet")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"), help="HuggingFace access token")
    
    # Dataset arguments
    parser.add_argument("--dataset", default="ImageNet", choices=["COCO2017", "ImageNet"])
    parser.add_argument("--train_ids_file", default="ImageNet/train_images.json", type=str)
    parser.add_argument("--test_ids_file", default="ImageNet/test_images.json", type=str)
    parser.add_argument("--img_folder", default="ImageNet/clean_image", type=str)
    parser.add_argument("--adv_folder", default="ImageNet/MF_clipattack/data/adversarial_images", type=str, help="Path to adversarial images")
    parser.add_argument("--use_adv", default=True, help="Use adversarial images")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_only", default=False, help="Skip training")
    
    return parser.parse_args()

def main():
    args = parse_args()
    finetuner = VisionCaptionFinetuner(args)
    
    if not args.eval_only:
        finetuner.finetune()
    
    finetuner.evaluate()

if __name__ == "__main__":
    main()
