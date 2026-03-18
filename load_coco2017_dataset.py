from PIL import Image
import os
import random
def read_image_ids(file_path):
    # Initialize an empty list to store image IDs
    image_ids = []
    
    # Open the file and read the IDs
    with open(file_path, "r") as file:
        for line in file:
            # Strip whitespace and add to the list
            image_ids.append(line.strip())
    
    return image_ids

def load_captions(file_path):
    captions = {}
    with open(file_path, 'r') as f:
        for line in f:
            img_id, caption = line.strip().split(',', 1)
            captions[img_id] = caption
    return captions



# Modified image loading function
def load_pil_image(image_path):
    try:
        with Image.open(image_path) as img:
            return img.convert("RGB")  # Ensure RGB format
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None


def get_image_ids(image_folder,training_ratio=0.8):
    # Get all files in the specified folder
    files = os.listdir(image_folder)
    
    # Filter for image files (you can adjust extensions as needed)
    image_ids = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Remove the file extension to get just the IDs
    image_ids = [os.path.splitext(image_id)[0] for image_id in image_ids]
    
    random.shuffle(image_ids)
    
    # Calculate the split index
    split_index = int(len(image_ids) * training_ratio)
    
    # Split the list into two parts
    train_img_ids = image_ids[:split_index]
    test_img_ids = image_ids[split_index:]
    # Save train and test IDs to text files
    with open("COCO2017/train_image_ids.txt", "w") as train_file:
        for img_id in train_img_ids:
            train_file.write(f"{img_id}\n")
    
    with open("COCO2017/test_image_ids.txt", "w") as test_file:
        for img_id in test_img_ids:
            test_file.write(f"{img_id}\n")
    
    return train_img_ids, test_img_ids

def load_dataset(train_ids_file, test_ids_file, img_folder, system_message=None, format_data=None, negtive_prompt=True):
        
    train_img_ids = read_image_ids(train_ids_file)
    test_img_ids = read_image_ids(test_ids_file)
    
    true_captions = load_captions("/home/yjli/Agent/agent-attack/COCO2017/val2017-captions.txt")
    false_captions = load_captions("/home/yjli/Agent/agent-attack/COCO2017/target_caption.txt")

    # Build paired samples (match + non-match)
    train_samples = []
    # train_img_ids, test_img_ids = get_image_ids("./COCO2017/adv_images/step_0299")
    for img_id in train_img_ids:
        # img_id = int(img_id)
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

            train_samples.append({
                "image": pil_image,
                "caption": true_captions[img_id],
                "label": "yes",
                "img_id": img_id
            })
            
            if negtive_prompt:
                # Negative sample (mismatched caption)
                train_samples.append({
                    "image": pil_image,
                    "caption": false_captions.get(img_id, ""),
                    "label": "no",
                    "img_id": img_id

                })
            
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
                "label": "yes",
                "img_id": img_id

            })
            if negtive_prompt:
                # Negative sample (mismatched caption)
                test_samples.append({
                    "image": pil_image,
                    "caption": false_captions.get(img_id, ""),
                    "label": "no",
                    "img_id": img_id

                })


    if system_message == None or format_data==None:
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

    print("Training IDs:", len(train_img_ids))
    print("Testing IDs:", len(test_img_ids))
    train_dataset = [format_data(sample) for sample in train_samples]
    print(train_dataset[0])  # Verify first sample
    print(train_dataset[0]["messages"])
    test_dataset = [format_data(sample) for sample in test_samples]
    return train_samples, test_samples, train_dataset, test_dataset
