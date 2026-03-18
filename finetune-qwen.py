import os
# Unset HTTP/HTTPS proxy variables
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
# Verify proxy is unset
print("Current proxy settings:", os.environ.get('http_proxy'))

from huggingface_hub import login
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import LoraConfig

login(
  token=os.getenv("HF_TOKEN"), # ADD YOUR TOKEN HERE
  add_to_git_credential=True
)

import os
from PIL import Image
from datasets import Dataset, Image
import random
from load_coco2017_dataset import load_dataset
import os
from trl import SFTTrainer
# train_img_ids, test_img_ids = get_image_ids("./COCO2017/adv_images/step_0299")
train_ids_file = "COCO2017/train_image_ids.txt"  # Update with your train IDs file path
test_ids_file = "COCO2017/test_image_ids.txt"    # Update with your test IDs file path
# train_dataset, test_dataset = load_dataset(train_ids_file,test_ids_file, img_folder="COCO2017/adv_images/step_0299")


system_message = "Determine if the given image and text caption match. Answer only 'yes' or 'no'."

def format_data(sample):
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Do you think the caption: {sample['caption']} match with the image?"},
                    {"type": "image", "image": sample["image"]}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["label"]}]
            }
        ]
    }
    
train_dataset, test_dataset = load_dataset(train_ids_file,test_ids_file, img_folder="COCO2017/val2017", system_message=system_message, format_data=format_data)


import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

# Hugging Face model id
model_id = "Qwen/Qwen2-VL-7B-Instruct" 

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    device_map="auto",
    # attn_implementation="flash_attention_2", # not supported for training
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)

model.config.use_cache = False  # Required!

processor = AutoProcessor.from_pretrained(model_id)

# Preparation for inference
text = processor.apply_chat_template(
    train_dataset[2]["messages"], tokenize=False, add_generation_prompt=False
)
print(text)




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

args = SFTConfig(
    output_dir="qwen2-7b-instruct-caption-normaltraining", # directory to save and repository id
    num_train_epochs=5,                     # number of training epochs
    per_device_train_batch_size=4,          # batch size per device during training
    gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=5,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
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
    dataset_text_field="", # needs dummy value
    peft_config=peft_config,
    tokenizer=processor.tokenizer,
)

# start training, the model will be automatically saved to the hub and the output directory
trainer.train()

# save model 
trainer.save_model(args.output_dir)

from eval_qwen import evaluate
adapter_path = "/home/yjli/Agent/agent-attack/qwen2-7b-instruct-caption-normaltraining"

eval_results = evaluate(
    model=model,
    processor=processor,
    eval_dataset=test_dataset
)
# Save evaluation results
with open(f"/home/yjli/Agent/agent-attack/qwen2-7b-instruct-caption-advtraining/eval_results_normal_training.txt", "w") as f:
    f.write(f"normal trained model Accuracy: {eval_results['accuracy']:.2%}\n")
print(f"normal trained model Accuracy: {eval_results['accuracy']:.2%}\n")

# Hugging Face model id
model_id = "Qwen/Qwen2-VL-7B-Instruct"

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)

processor = AutoProcessor.from_pretrained(model_id)

# Run evaluation after training
eval_results = evaluate(
    model=model,
    processor=processor,
    eval_dataset=test_dataset
)

# Save evaluation results
with open(f"qwen2-7b-instruct-caption-advtraining/eval_results.txt", "w") as f:
    f.write(f"Base Accuracy: {eval_results['accuracy']:.2%}\n")
    
print(f"Base Accuracy: {eval_results['accuracy']:.2%}\n")
    
# adapter_path = "/home/yjli/Agent/agent-attack/qwen2-7b-instruct-caption-advtraining"
    
# model.load_adapter(adapter_path) # load the adapter and activate
# eval_results = evaluate(
#     model=model,
#     processor=processor,
#     eval_dataset=test_dataset
# )

# # Save evaluation results
# with open(f"/home/yjli/Agent/agent-attack/qwen2-7b-instruct-caption-advtraining/eval_results_new.txt", "w") as f:
#     f.write(f"adv trained model Accuracy: {eval_results['accuracy']:.2%}\n")

    
# model.load_adapter(adapter_path) # load the adapter and activate
# eval_results = evaluate(
#     model=model,
#     processor=processor,
#     eval_dataset=test_dataset
# )
# # Save evaluation results
# with open(f"/home/yjli/Agent/agent-attack/qwen2-7b-instruct-caption-advtraining/eval_results_new.txt", "w") as f:
#     f.write(f"adv trained model Accuracy: {eval_results['accuracy']:.2%}\n")

