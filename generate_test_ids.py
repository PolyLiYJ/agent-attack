import json

def read_captions(file_path):
    """读取target_caption.txt文件，返回字典 {image_id: caption}"""
    captions = {}
    with open(file_path, 'r') as f:
        for line in f:
            if ',' in line:
                image_id, caption = line.strip().split(',', 1)
                captions[image_id] = caption.strip()
    return captions


def main():
    # 读取目标标题
    # read all ids from /home/yjli/Agent/agent-attack/COCO2017/negtive_correct_test_ids.txt
    with open("/home/yjli/Agent/agent-attack/COCO2017/negtive_correct_test_ids.txt", 'r') as f:
        image_ids = {line.strip() for line in f}    
    
    target_captions = read_captions("COCO2017/target_caption.txt")
    gt_captions = read_captions("COCO2017/val2017-captions.txt")
    
    # 生成测试数据
    test_data = []
    for image_id in image_ids:
        test_data.append({
            "image_id": image_id,
            "gt_caption": gt_captions[image_id],
            "target_caption": target_captions[image_id]
        })
    
    # 保存为JSON文件
    with open("COCO2017/test_images_ids.json", "w") as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
    
    print(f"Generated test_ids.json with {len(test_data)} items")

if __name__ == "__main__":
    main()