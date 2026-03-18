import os
import torch
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def calculate_clip_similarity(image_folder, prompt):
    # Load the CLIP model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # List all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Prepare preprocess function
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    similarities = []

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert("RGB")

        # Create inputs
        inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

        # Calculate cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(image_embeds, text_embeds)
        similarity = cosine_sim.item()  # Convert similarity to distance
        similarities.append(similarity)

    # Calculate and return the average distance
    average_distance = sum(similarities) / len(similarities) if similarities else float('inf')
    return average_distance

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate average CLIP distance between images and a prompt.")
    parser.add_argument('image_folder', type=str, help='Folder containing images.')
    parser.add_argument('prompt', type=str, help='Prompt string for comparison.')

    args = parser.parse_args()
    
    avg_distance = calculate_clip_similarity(args.image_folder, args.prompt)
    print(f"Average CLIP similarity: {avg_similarity}")
