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
# Unset HTTP/HTTPS proxy variables
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
# Verify proxy is unset
print("Current proxy settings:", os.environ.get('http_proxy'))




accelerator = Accelerator()


def generate_description(sample, model, processor):
    # Prepare messages
    messages = sample["messages"]

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

def evaluate(model, processor, eval_dataset, num_samples=100):
    import random
    from tqdm import tqdm
    
    # 评估所有样本或指定数量样本
    # eval_samples = range(len(eval_dataset)) if len(eval_dataset) <= num_samples else random.sample(range(len(eval_dataset)), num_samples)
    eval_samples = range(len(eval_dataset))
    
    predictions = []
    true_labels = []
    
    model.eval()
    
    for i in tqdm(eval_samples, desc="Evaluating"):
        sample = eval_dataset[i]
        
        # 生成预测
        pred = generate_description(sample, model, processor)
        
        predictions.append(pred)
        true_labels.append(sample["messages"][-1]["content"][0]["text"])
    
    # 后处理文本，统一格式
    preds, labels = postprocess_text(predictions, true_labels)
    
    # 初始化统计量
    yes_correct = 0
    yes_total = 0
    no_correct = 0
    no_total = 0
    
    for p, l in zip(preds, labels):
        if l == "yes":
            yes_total += 1
            if p == l:
                yes_correct += 1
        elif l == "no":
            no_total += 1
            if p == l:
                no_correct += 1
    
    # 计算各项指标
    accuracy = (yes_correct + no_correct) / len(preds)
    yes_accuracy = yes_correct / yes_total if yes_total > 0 else 0.0
    no_accuracy = no_correct / no_total if no_total > 0 else 0.0
    
    print(f"\nEvaluation results (sample size: {len(preds)}):")
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"Accuracy on 'yes' samples: {yes_accuracy:.2%} ({yes_correct}/{yes_total})")
    print(f"Accuracy on 'no' samples: {no_accuracy:.2%} ({no_correct}/{no_total})")
    
    # 打印样例
    print("\nSample predictions:")
    for i in range(min(3, len(preds))):
        print(f"Input: {eval_dataset[eval_samples[i]]['messages'][1]['content'][0]['text']}")
        print(f"Predicted: {preds[i]}, True: {labels[i]}\n")
    
    return {
        "accuracy": accuracy,
        "yes_correct": yes_correct,
        "no_correct": no_correct,
        "yes_accuracy": yes_accuracy,
        "no_accuracy": no_accuracy,
        "yes_samples": yes_total,
        "no_samples": no_total
    }




if __name__ == "__main__":
    train_ids_file = "COCO2017/train_image_ids.txt"  # Update with your train IDs file path
    test_ids_file = "COCO2017/test_image_ids.txt"    # Update with your test IDs file path
    train_dataset, test_dataset = load_dataset(train_ids_file,test_ids_file, img_folder = "COCO2017/val2017")
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
        
    adapter_path = "/home/yjli/Agent/agent-attack/qwen2-7b-instruct-caption-advtraining"
        
    model.load_adapter(adapter_path) # load the adapter and activate
    eval_results = evaluate(
        model=model,
        processor=processor,
        eval_dataset=test_dataset
    )

    # Save evaluation results
    with open(f"/home/yjli/Agent/agent-attack/qwen2-7b-instruct-caption-advtraining/eval_results_new.txt", "w") as f:
        f.write(f"adv trained model Accuracy: {eval_results['accuracy']:.2%}\n")

