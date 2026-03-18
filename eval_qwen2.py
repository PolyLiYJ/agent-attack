from load_coco2017_dataset import load_dataset
from trl import SFTConfig
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from accelerate import Accelerator
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info
import os
from load_coco2017_dataset import read_image_ids,load_pil_image
# Unset HTTP/HTTPS proxy variables
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
# Verify proxy is unset
print("Current proxy settings:", os.environ.get('http_proxy'))


def load_captions(file_path):
    captions = {}
    with open(file_path, 'r') as f:
        for line in f:
            img_id, caption = line.strip().split(',', 1)
            captions[img_id] = caption
    return captions

true_captions = load_captions("/home/yjli/Agent/agent-attack/COCO2017/val2017-captions.txt")


train_ids_file = "COCO2017/train_image_ids.txt"  # Update with your train IDs file path
test_ids_file = "COCO2017/test_image_ids.txt"    # Update with your test IDs file path
system_message = "Give the caption of the image."
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
    
test_img_ids = read_image_ids(test_ids_file)
img_folder = "/home/yjli/Agent/agent-attack/COCO2017/val2017"

test_samples = []
for img_id in test_img_ids:
    path1 = f"{img_folder}/{img_id}.png"
    path2 = f"{img_folder}/{int(img_id):012d}.jpg"
    if os.path.exists(path1):
        img_path = path1
    elif os.path.exists(path2):
        img_path = path2
    else:
        print(f"No image file {path1} or {path2} exists!")
    if os.path.exists(img_path):
        # Positive sample (matched caption)
        pil_image = load_pil_image(img_path)

        test_samples.append({
            "image": pil_image,
            "caption": true_captions[img_id],
            "label": "yes"
        })

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
    )

    # Use accelerator to prepare inputs (handles device placement)
    inputs = accelerator.prepare(inputs)

    # Ensure all inputs are on the same device
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256, top_p=1.0, do_sample=True, temperature=0.8)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

# Helper function to process predictions
def postprocess_text(preds, labels):
    preds = [pred.strip().lower() for pred in preds]
    labels = [label.strip().lower() for label in labels]
    
    # Map variations to yes/no
    preds = ["yes" if p in ("yes", "true", "1") else "no" for p in preds]
    labels = ["yes" if l in ("yes", "true", "1") else "no" for l in labels]
    
    return preds, labels

import random
from tqdm import tqdm
from IPython.display import display, Image, HTML

# Evaluation function
def evaluate(model, processor, eval_dataset, num_samples=10):
    # Set random seed for reproducibility
    random.seed(42)
    
    # Randomly sample from evaluation dataset

    eval_samples = range(len(eval_dataset))
    
    predictions = []
    true_labels = []
    
    model.eval()
    
    for i in tqdm(eval_samples):
        sample = eval_dataset[i]
        
        # Prepare inputs
        pred = generate_description(sample, model, processor)
        print("pred:", pred)
        print("true caption:", sample["caption"])

        img = sample['image'] # Replace with the correct key for image path
        display(img)


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
evaluate(
    model=model,
    processor=processor,
    eval_dataset=test_samples
)

    
adapter_path = "/home/yjli/Agent/agent-attack/qwen2-7b-instruct-caption-advtraining"
    
model.load_adapter(adapter_path) # load the adapter and activate
evaluate(
    model=model,
    processor=processor,
    eval_dataset=test_samples
)