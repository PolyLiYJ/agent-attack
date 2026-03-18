from PIL import Image
import os
import random
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from torchvision import transforms
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import pandas as pd
from torchvision.transforms.functional import InterpolationMode

def load_pil_image(img_path):
    """Load an image using PIL."""
    return Image.open(img_path)

def load_dataset(train_file, test_file, img_folder, system_message=None, format_data=None, negtive_prompt=True):
    # Load JSON data
    train_data = json.load(open(train_file))
    test_data = json.load(open(test_file))
    print("Training IDs:", len(train_data))
    print("Testing IDs:", len(test_data))
    
    # Build paired samples (match + non-match)
    train_samples = []
    
    for item in train_data:
        img_name = item["image_name"]
        img_id = img_name.split("_")[0]
        img_path = os.path.join(img_folder, f"ILSVRC2012_val_{img_id}.png")
        
        if os.path.exists(img_path):
            pil_image = load_pil_image(img_path)

            # Positive sample (matched caption)
            train_samples.append({
                "image": pil_image,
                "caption": item["gt_caption"],
                "label": "yes",
                "img_name": img_name,
                "target_caption": item["target_caption"]
            })
            
            if negtive_prompt:
                # Negative sample (mismatched caption)
                train_samples.append({
                    "image": pil_image,
                    "caption": item["target_caption"],
                    "label": "no",
                    "img_name": img_name,
                })
        else:
            print(f"Image {img_id} not found.")

    test_samples = []
    for item in test_data:
        img_name = item["image_name"]
        img_id = img_name.split("_")[0]
        img_path = os.path.join(img_folder, f"ILSVRC2012_val_{img_id}.png")
        
        if os.path.exists(img_path):
            pil_image = load_pil_image(img_path)

            # Positive sample (matched caption)
            test_samples.append({
                "image": pil_image,
                "caption": item["gt_caption"],
                "label": "yes",
                "img_name": img_name,
                "target_caption": item["target_caption"]
            })
            
            if negtive_prompt:
                # Negative sample (mismatched caption)
                test_samples.append({
                    "image": pil_image,
                    "caption": item["target_caption"],
                    "label": "no",
                    "img_name": img_name
                })

    if system_message is None or format_data is None:
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
                            {"type": "text", "text": f"Caption: {sample['caption']}"},
                            {"type": "image", "image": sample["image"]}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": sample["label"]}]
                    }
                ]
            }


    
    train_dataset = [format_data(sample) for sample in train_samples]
    print(train_dataset[0])  # Verify first sample
    print(train_dataset[0]["messages"])
    
    test_dataset = [format_data(sample) for sample in test_samples]
    
    return train_samples, test_samples, train_dataset, test_dataset

# Example call (make sure to specify the correct paths)
# train_samples, test_samples, train_dataset, test_dataset = load_dataset('train_images.json', 'test_images.json', '/path/to/images')

if __name__ == "__main__":
    import torch
    from PIL import Image
    import os


    """
    # Assuming you have a list of image names
    data = pd.read_csv("/home/yjli/Agent/agent-attack/ImageNet/MF_clipattack/data/imagenet1000.csv")

    # Create a list of dictionaries for each image
    image_names = []
    for index, row in data.iterrows():
        print(row)
        image_names.append(row['Image Names'])
    clean_image_folder = '/home/yjli/Agent/agent-attack/ImageNet/clean_image'
    adv_image_folder = '/home/yjli/Agent/agent-attack/ImageNet/MF_clipattack/data/adversarial_images'  # Folder to save adversarial images

    # Create the folder if it doesn't exist
    os.makedirs(adv_image_folder, exist_ok=True)
    vis_processors = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        ])

    # Process each image
    for image_name in image_names:
        # Process clean image
        image_id = image_name.split("_")[0]
        print(f"img_id: {image_id}")
        image_path = os.path.join(clean_image_folder, f"ILSVRC2012_val_{image_id}.JPEG")
        if os.path.exists(image_path)==False:
            print(f"Image not found: {image_path}")
            continue
    
        image_rgb = Image.open(image_path).convert('RGB')

        image_tensor = vis_processors(image_rgb)

        # Load delta and compute adversarial image
        img_size = 224
        delta_path = f"ImageNet/MF_clipattack/delta/delta{img_size}/MF_ii_{image_name}.pt"
        
        if os.path.exists(delta_path):
            delta = torch.load(delta_path)
            adv_images_tensor = image_tensor + delta.detach().squeeze(0)
            adv_image_rgb = (adv_images_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
            adv_pil_image = Image.fromarray(adv_image_rgb)

            # Save adversarial image
            adv_img_pth = os.path.join(adv_image_folder, f"ILSVRC2012_val_{image_id}.png")
            adv_pil_image.save(adv_img_pth)

            # # Get captions
            # attacked_text = blip2(adv_img_pth, device, model, processor)
            # clean_text = blip2(img_pth, device, model, processor)
            
            # # Optionally, you can print or log the results
            # print(f"Processed {image_name}: Clean text: {clean_text}, Attacked text: {attacked_text}")
        else:
            print(f"Delta file not found for {image_name}: {delta_path}")
    """    
        
    ### split the dataset into train and test
    
    # Load the CSV file
    file_path = '/home/yjli/Agent/agent-attack/ImageNet/MF_clipattack/data/imagenet1000.csv'  # Replace with your actual file path
    data = pd.read_csv(file_path)

    # Create a list of dictionaries for each image
    image_data = []
    for index, row in data.iterrows():
        print(row)
        image_data.append({
            'image_name': row['Image Names'],
            'gt_caption': row['Clean Text'],
            'target_caption': row['Target Text']
        })

    # Split the data into training and testing sets (80% train, 20% test)
    train_data, test_data = train_test_split(image_data, test_size=0.2, random_state=42)

    # Save only image IDs to JSON files
    train_image_ids = [{'image_name': item['image_name']} for item in train_data]
    test_image_ids = [{'image_name': item['image_name']} for item in test_data]

    with open('ImageNet/train_images.json', 'w') as train_file:
        json.dump(train_data, train_file, indent=4)

    with open('ImageNet/test_images.json', 'w') as test_file:
        json.dump(test_data, test_file, indent=4)

    print("Data has been split and saved to 'train_images.json' and 'test_images.json'.")
    img_folder ="/home/yjli/Agent/agent-attack/ImageNet/MF_clipattack/data/adversarial_images"
    train_samples, test_samples, train_dataset, test_dataset = load_dataset('ImageNet/train_images.json', 'ImageNet/test_images.json', img_folder=img_folder, negtive_prompt=False)