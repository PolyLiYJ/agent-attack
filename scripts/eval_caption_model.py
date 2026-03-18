import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import glob
from PIL import Image

import sys
# from agent_attack.models import get_captioning_model
from LLMwapper import QwenWrapper
# captioning_model = "gpt-4o"
# captioning_model = "liuhaotian/llava-v1.5-7b"
# caption_image_fn = get_captioning_model(captioning_model)

# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path
# from llava.eval.run_llava import eval_model

# model_path = "liuhaotian/llava-v1.5-7b"

# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path=model_path,
#     model_base=None,
#     model_name=get_model_name_from_path(model_path)
# )

model = QwenWrapper(model="qwen-2.5", temperature=0.7)

task_id_list = [os.path.splitext(os.path.basename(png))[0] for png in glob.glob('exp_data/clean_html/*.png')]

for task_id in task_id_list[:-1]:
    if "mail" not in task_id:
        continue
    img_path = f"/home/yjli/Agent/agent-attack/pipeline_test_gpt4o_som_typography/typography_attack/typography_{task_id}.png"
    # # Read the image file in binary mode
    image = Image.open(img_path)
    # Send the image to OpenAI API for captioning
    # caption = caption_image_fn(images=[image])
    
    # with open(img_path, "rb") as image_file:
    #     encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    # print(f"Base64 encoded image:
    # {encoded_string}")
    
    
    # model_path = "liuhaotian/llava-v1.5-7b"
    # prompt = "What is the Email in the Image?"
    prompt = "Describe the image content"
    caption = model.generate(prompt=prompt, system=None, image=img_path)

    # args = type('Args', (), {
    #     "model_path": model_path,
    #     "model_base": None,
    #     "model_name": get_model_name_from_path(captioning_model),
    #     "query": prompt,
    #     "conv_mode": None,
    #     "image_file": [img_path],
    #     "sep": ",",
    #     "temperature": 0.7,
    #     "top_p": 10,
    #     "num_beams": 1,
    #     "max_new_tokens": 512
    # })()

    # caption = eval_model(args)
    
    # Process or store the caption as needed
    print(img_path)

    print(f"Task ID: {task_id}, Caption: {caption}")



