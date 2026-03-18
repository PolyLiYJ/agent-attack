import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from agent_attack.models.llava import LLaVa
from torchvision.transforms import Compose

torch.cuda.set_device(0)
# Set visible CUDA devices (already done via torch.cuda.set_device(0))
# This ensures only GPU 0 is used for computation

def captioning_fn(images, hub_path = "liuhaotian/llava-v1.5-7b", ):
    captioning_model = LLaVa(hub_path)
    prompt_fn = captioning_model.get_captioning_prompt_fn()
    all_gen_texts = []
    for image in images:
        inputs = [prompt_fn()]
        if isinstance(captioning_model.image_processor, Compose) or hasattr(
            captioning_model.image_processor, "is_prismatic"
        ):
            # This is a standard `torchvision.transforms` object or custom PrismaticVLM wrapper
            adv_pixel_values = captioning_model.image_processor(image).unsqueeze(0)
        else:
            # Assume `image_transform` is an HF ImageProcessor...
            adv_pixel_values = captioning_model.image_processor(image, return_tensors="pt")["pixel_values"]
        adv_pixel_values = adv_pixel_values.to(captioning_model.distributed_state.device)

        gen_texts = captioning_model.generate_answer(adv_pixel_values, inputs)
        all_gen_texts.append(gen_texts[0])
    return all_gen_texts
# def load_llava_model(model_name="llava-hf/llava-1.5-7b-hf"):
#     """Load LLaVA model and processor"""
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     model = LlavaForConditionalGeneration.from_pretrained(
#         model_name, 
#         torch_dtype=torch.float16 if "cuda" in device else torch.float32,
#         low_cpu_mem_usage=True
#     ).to(device)
    
#     processor = LlavaProcessor.from_pretrained(model_name)
#     captioning_model = LLaVa(hub_path)
#     prompt_fn = captioning_model.get_captioning_prompt_fn()
#     return model, processor, device

# def run_inference(image_path: str, prompt: str = "Describe this image in detail."):
#     """Run inference on an image with LLaVA"""
#     # Load model
#     model, processor, device = load_llava_model()
    
#     # Load image
#     image = Image.open(image_path)
    
#     # Prepare inputs
#     inputs = processor(
#         text=prompt, 
#         images=image, 
#         return_tensors="pt",
#         padding=True
#     ).to(device)
    
#     # Generate response
#     with torch.no_grad():
#         output = model.generate(**inputs, max_new_tokens=200)
    
#     # Decode output
#     response = processor.decode(output[0][2:], skip_special_tokens=True)
#     return response.strip()

# Example Usage
if __name__ == "__main__":
    image_path = "/home/yjli/Agent/agent-attack/ImageNet/clean_image/ILSVRC2012_val_00000001.JPEG"  # Replace with your image
    prompt = "What is happening in this image?"
    
    # response = run_inference(image_path, prompt)
     # Load image (note: images parameter expects list of images)
    pil_image = Image.open(image_path).convert("RGB")
    
    # Get captions (returns list of strings)
    captions = captioning_fn(images=[pil_image])
    
    print(f"LLaVA Response: {captions[0]}")
    
