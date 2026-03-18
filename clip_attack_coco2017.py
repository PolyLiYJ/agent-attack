
# from pycocotools.coco import COCO
# import numpy as np
# import skimage.io as io
# import matplotlib.pyplot as plt
# import pylab
# import os
# from agent_attack.attacks.clip_attack import clip_attack
# from PIL import Image

# pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# # Load dataset 
# dataDir = 'COCO2017'
# dataType = 'val2017'
# annFile = os.path.join(dataDir, 'val2017-annotations-3', 'instances_val2017.json')
# coco = COCO(annFile)

# # Initialize COCO API for caption annotations
# annFile_caps = os.path.join(dataDir, 'val2017-annotations-3', f'captions_{dataType}.json')
# coco_caps = COCO(annFile_caps)

# # Get all image IDs
# imgIds = coco.getImgIds()

# # Check the number of images found
# print(f"Number of images found: {len(imgIds)}")

# # Create and open a document to write captions
# with open('COCO2017/target_caption.txt', 'r', encoding='utf-8') as file:
#     for line in file:
#         # Each line is "img_id,target_caption"
#         img_id, target_caption = line.strip().split(',')
#         img_id = int(img_id)  # Convert img_id to integer

#         # Load annotations for the current image
#         annIds = coco_caps.getAnnIds(imgIds=img_id)
#         anns = coco_caps.loadAnns(annIds)

#         if len(anns) == 0:
#             print(f"No captions found for image ID {img_id}. Skipping...")
#             continue

#         ori_caption = anns[0]["caption"]  # Original caption
#         print(f"Original Caption: {ori_caption}")
#         print(f"Target Caption: {target_caption}")

#         # Load the image
#         img = coco.loadImgs(img_id)[0]
#         # image = io.imread(img['coco_url'])  # Read the image using its URL
#         image = io.imread('/home/yjli/Agent/agent-attack/COCO2017/val2017/%s'%(img['file_name']))
#         image = Image.fromarray(image)


#         # Apply the adversarial attack
#         attack_out_dict = clip_attack(image, target_caption, ori_caption, epsilon=16 / 255, alpha=1 / 255, iters=300, size=180)

#         adv_images = attack_out_dict["adv_images"]

#         # Create a directory for saving adversarial images
#         save_dir = os.path.join(dataDir, "adv_images")
#         os.makedirs(save_dir, exist_ok=True)

#         # Iterate through adversarial samples and save them
#         for step, adv_image in adv_images.items():
#             # Generate filename with attack step
#             filename = os.path.join(save_dir, f"{img_id}_{step:04d}.png")  # Zero-padded to 4 digits
#             # Save as lossless PNG format
#             adv_image.save(
#                 filename,
#                 format="PNG",
#                 compress_level=0  # Disable compression to preserve adversarial integrity
#             )

#         print(f"Adversarial images saved for image ID {img_id}.")



# from pycocotools.coco import COCO
# import numpy as np
# import skimage.io as io
# import matplotlib.pyplot as plt
# import pylab
# import os
# from agent_attack.attacks.clip_attack import clip_attack
# from PIL import Image
# import concurrent.futures

# pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# # Load dataset 
# dataDir = 'COCO2017'
# dataType = 'val2017'
# annFile = os.path.join(dataDir, 'val2017-annotations-3', 'instances_val2017.json')
# coco = COCO(annFile)

# # Initialize COCO API for caption annotations
# annFile_caps = os.path.join(dataDir, 'val2017-annotations-3', f'captions_{dataType}.json')
# coco_caps = COCO(annFile_caps)

# # Get all image IDs
# imgIds = coco.getImgIds()
# print(f"Number of images found: {len(imgIds)}")

# # Create a directory for saving adversarial images
# save_dir = os.path.join(dataDir, "adv_images")
# os.makedirs(save_dir, exist_ok=True)

# def process_image(line):
#     img_id, target_caption = line.strip().split(',')
#     img_id = int(img_id)  # Convert img_id to integer

#     # Load annotations for the current image
#     annIds = coco_caps.getAnnIds(imgIds=img_id)
#     anns = coco_caps.loadAnns(annIds)

#     if len(anns) == 0:
#         print(f"No captions found for image ID {img_id}. Skipping...")
#         return

#     ori_caption = anns[0]["caption"]  # Original caption
#     print(f"Processing Image ID: {img_id} | Original Caption: {ori_caption} | Target Caption: {target_caption}")

#     # Load the image
#     img = coco.loadImgs(img_id)[0]
#     image = io.imread(os.path.join(dataDir, 'val2017', img['file_name']))
#     image = Image.fromarray(image)

#     # Apply the adversarial attack
#     attack_out_dict = clip_attack(image, target_caption, ori_caption, epsilon=16 / 255, alpha=1 / 255, iters=300, size=180)

#     adv_images = attack_out_dict["adv_images"]

#     # Iterate through adversarial samples and save them
#     for step, adv_image in adv_images.items():
#         filename = os.path.join(save_dir, f"{img_id}_{step:04d}.png")  # Zero-padded to 4 digits
#         adv_image.save(filename, format="PNG", compress_level=0)  # Save as lossless PNG format

#     print(f"Adversarial images saved for image ID {img_id}.")

# # Read the target captions
# with open('COCO2017/target_caption.txt', 'r', encoding='utf-8') as file:
#     lines = file.readlines()

# # Use ThreadPoolExecutor to process images in parallel
# with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
#     executor.map(process_image, lines)


from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
from agent_attack.attacks.clip_attack import clip_attack_batch
from PIL import Image

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# Load dataset 
dataDir = 'COCO2017'
dataType = 'val2017'
annFile = os.path.join(dataDir, 'val2017-annotations-3', 'instances_val2017.json')
coco = COCO(annFile)

# Initialize COCO API for caption annotations
annFile_caps = os.path.join(dataDir, 'val2017-annotations-3', f'captions_{dataType}.json')
coco_caps = COCO(annFile_caps)

# Get all image IDs
imgIds = coco.getImgIds()
print(f"Number of images found: {len(imgIds)}")

# Create a directory for saving adversarial images
save_dir = os.path.join(dataDir, "adv_images_budget_8_255")
os.makedirs(save_dir, exist_ok=True)

# Batch size for processing
batch_size = 10
batch_images = []
batch_target_captions = []
batch_ori_captions = []
batch_img_ids = []

# Create and open a document to write captions
with open('COCO2017/target_caption.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()  # Read all lines at once
    for line in lines:
    # for line in file:
        # img_id, target_caption = line.strip().split(',')
        img_id, target_caption = line.strip().split(',', 1)
        img_id = int(img_id)  # Convert img_id to integer
        print("img_id:", img_id)

        # Load annotations for the current image
        annIds = coco_caps.getAnnIds(imgIds=img_id)
        anns = coco_caps.loadAnns(annIds)

        if len(anns) == 0:
            print(f"No captions found for image ID {img_id}. Skipping...")
            continue

        ori_caption = anns[0]["caption"]  # Original caption
        print(f"Original Caption: {ori_caption}, Target Caption: {target_caption}")

        # Load the image and convert to PIL format
        img = coco.loadImgs(img_id)[0]
        image = io.imread(os.path.join(dataDir, 'val2017', img['file_name']))
        image = Image.fromarray(image)

        # Append to batch
        batch_images.append(image)
        batch_target_captions.append(target_caption)
        batch_ori_captions.append(ori_caption)
        batch_img_ids.append(img_id)

        # Process the batch if it reaches the specified batch size
        if len(batch_images) == batch_size:
            # Apply the adversarial attack in batch
            attack_results = clip_attack_batch(batch_images, batch_target_captions, batch_ori_captions, 
                                           epsilon=8 / 255, alpha=1 / 255, iters=100, size=(180, 130))

             # Save all steps for all images in batch
            for step, adv_images in attack_results["adv_images"].items():
                step_dir = os.path.join(save_dir, f"step_{step:04d}")
                os.makedirs(step_dir, exist_ok=True)
                
                for filename, adv_img in zip(batch_img_ids, adv_images):
                    adv_img.save(os.path.join(step_dir, f"{filename}.png"), quality=100)
            print(f"Saved adversarial images for steps: {list(attack_results['adv_images'].keys())}")

            # Clear the batch after processing
            batch_images.clear()
            batch_target_captions.clear()
            batch_ori_captions.clear()
            batch_img_ids.clear()
            
if batch_images:
    print(f"\nProcessing final batch of {len(batch_images)} images...")
    attack_results = clip_attack_batch(
        batch_images,
        batch_target_captions,
        batch_victim_texts=batch_ori_captions,
        epsilon=16/255,
        alpha=1/255,
        iters=300,
        size=180
    )
    for step, adv_images in attack_results["adv_images"].items():
        step_dir = os.path.join(save_dir, f"step_{step:04d}")
        os.makedirs(step_dir, exist_ok=True)
        
        for filename, adv_img in zip(batch_img_ids, adv_images):
            adv_img.save(os.path.join(step_dir, f"{filename}.png"), quality=100)
    
    print(f"Saved final batch adv images for steps: {list(attack_results['adv_images'].keys())}")

