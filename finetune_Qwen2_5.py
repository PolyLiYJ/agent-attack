

from PIL import Image
from datasets import Dataset, Image
import random
import os
from trl import SFTTrainer
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import torch
from huggingface_hub import login
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import LoraConfig

login(
  token=os.getenv("HF_TOKEN"), # ADD YOUR TOKEN HERE
  add_to_git_credential=True
)
from eval_qwen import evaluate

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

DATASET="ImageNet"
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"


# 定义caption生成任务的数据格式
def format_caption_data(sample):
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Give the caption of the image."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["caption"]}]
            }
        ]
    }

# 加载数据集（caption生成任务）

if DATASET == "COCO2017":
    from load_coco2017_dataset import load_dataset
    train_ids_file = "COCO2017/correct_train_ids.txt"
    test_ids_file = "COCO2017/correct_test_ids.txt"
    new_train_ids_file = "COCO2017/negtive_correct_train_ids.txt"  # 新训练集文件
    new_test_ids_file = "COCO2017/negtive_correct_test_ids.txt"    # 新测试集文件
    img_folder = "/home/yjli/Agent/agent-attack/COCO2017/val2017"
    adv_folder = "/home/yjli/Agent/agent-attack/COCO2017/adv_images/step_0099"
    train_samples, test_samples, train_dataset, test_dataset = load_dataset(
        train_ids_file, 
        test_ids_file, 
        img_folder=adv_folder,
        system_message="Give the caption of the image.",
        format_data=format_caption_data
    )
    
elif DATASET == "ImageNet":
    from load_imagenet_dataset import load_dataset
    train_ids_file = "ImageNet/train_images.json"
    test_ids_file = "ImageNet/test_images.json"
    img_folder = "/home/yjli/Agent/agent-attack/ImageNet/MF_clipattack/data/imageNet-1K_validation"
    adv_folder = "/home/yjli/Agent/agent-attack/ImageNet/MF_clipattack/data/adversarial_images"
    train_samples, test_samples, train_dataset, test_dataset = load_dataset(
        train_ids_file, 
        test_ids_file, 
        img_folder=adv_folder,
        system_message="Give the caption of the image.",
        format_data=format_caption_data
    )


# Hugging Face model id
# model_id = "Qwen/Qwen2-VL-7B-Instruct" 


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

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM", 
)

from trl import SFTConfig
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

adapter_path = "qwen2_5-7b-instruct-caption-adv_training-ImageNet"

args = SFTConfig(
    output_dir=adapter_path, # directory to save and repository id
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=2,          # batch size per device during training
    gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=5,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-5,                     # learning rate, based on QLoRA paper
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=True,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
    gradient_checkpointing_kwargs = {"use_reentrant": False}, # use reentrant checkpointing
    dataset_text_field="", # need a dummy field for collator
    dataset_kwargs = {"skip_prepare_dataset": True} # important for collator
)
args.remove_unused_columns=False

# Create a data collator to encode text and image pairs
def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example["messages"])[0] for example in examples]

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100  #
    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):
        image_tokens = [151652,151653,151655]
    else: 
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collate_fn,
    # dataset_text_field="", # needs dummy value
    peft_config=peft_config,
    # tokenizer=processor.tokenizer,
)

# start training, the model will be automatically saved to the hub and the output directory
trainer.train()

# save model 
trainer.save_model(args.output_dir)

from eval_qwen import evaluate
# adapter_path = "qwen2_5-7b-instruct-caption-adv_training"

# Load model and tokenizer
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    device_map="auto",
    # attn_implementation="flash_attention_2", # not supported for training
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)

model.config.use_cache = False  # Required!

print("=== Baseline Evaluation ===")
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

from eval_qwen2_5 import run_evaluation, calculate_avg_scores
baseline_results = run_evaluation(test_samples, model, processor)
print("baseline results:", calculate_avg_scores(baseline_results))

# Load adapter and evaluate
adapter_path = "/home/yjli/Agent/agent-attack/qwen2_5-7b-instruct-caption-adv_training"
model, processor = load_model(model_id = MODEL_ID, adapter_path=adapter_path)
finetuned_results = run_evaluation(test_samples, model, processor)
print("\n=== Final Scores ===")
print("adv trained model results:", calculate_avg_scores(finetuned_results))

print("Fine-tuned:", calculate_avg_scores(finetuned_results))

# adapter_path = "qwen2_5-7b-instruct-caption-adv_training"
# model.load_adapter(adapter_path) # load the adapter and activate
# eval_results = evaluate(
#     model=model,
#     processor=processor,
#     eval_dataset=test_dataset
# )
# # Save evaluation results
# with open(f"{adapter_path}/eval_results_adv_training.txt", "w") as f:
#     f.write(f"Overall Accuracy: {eval_results['accuracy']:.2%}\n")
#     f.write(f"'Yes' Accuracy: {eval_results['yes_accuracy']:.2%} ({eval_results['yes_correct']}/{eval_results['yes_samples']})\n")
#     f.write(f"'No' Accuracy: {eval_results['no_accuracy']:.2%} ({eval_results['no_correct']}/{eval_results['no_samples']})\n")

# # Print the same results to console
# print(f"\nEvaluation Results:")
# print(f"Overall Accuracy: {eval_results['accuracy']:.2%}")
# print(f"'Yes' Accuracy: {eval_results['yes_accuracy']:.2%} ({eval_results['yes_correct']}/{eval_results['yes_samples']})")
# print(f"'No' Accuracy: {eval_results['no_accuracy']:.2%} ({eval_results['no_correct']}/{eval_results['no_samples']})")

# save the total model


