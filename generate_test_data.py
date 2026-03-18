import os
import json
from collections import defaultdict

def load_attack_data(data_json_path):
    """
    从攻击目录下的data.json加载攻击相关信息。
    
    Args:
        data_json_path (str): data.json文件的路径
        
    Returns:
        dict: 包含victim_som_id和target_caption的字典，加载失败则返回None
    """
    try:
        with open(data_json_path, 'r') as f:
            data = json.load(f)
            return {
                'victim_som_id': data.get('victim_som_id'),
                'target_caption': data.get('target_caption'),
                'adversarial_goal': data.get('adversarial_goal')
            }
    except Exception as e:
        print(f"Error reading {data_json_path}: {e}")
        return None

def extract_description_from_obs_text(file_path):
    """
    从obs_text.txt文件中提取图像描述。
    
    Args:
        file_path (str): obs_text.txt文件的路径
        
    Returns:
        str or None: 提取的图像描述，如果未找到则返回None
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if '[IMG]' in line and 'description:' in line:
                    desc_start = line.find('description:') + len('description:')
                    desc_end = min(
                        line.find(',', desc_start) if line.find(',', desc_start) != -1 else len(line),
                        line.find(']', desc_start) if line.find(']', desc_start) != -1 else len(line)
                    )
                    return line[desc_start:desc_end].strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

def main():
    """
    主函数：生成测试数据集并统计信息。
    
    处理流程：
    1. 建立原始图像索引：
       - 扫描clean_dir下所有图像目录
       - 提取每个图像的描述信息
       - 建立som_id到图像信息的映射
       
    2. 处理攻击样本：
       - 扫描adv_dir下的攻击目录
       - 读取data.json获取victim_som_id
       - 关联对应的原始图像信息
       
    3. 生成测试数据：
       - 整合原始图像和攻击信息
       - 记录统计信息
       - 输出JSON格式数据
    """
    # 定义关键路径
    clean_dir = "exp_data/agent_clean"
    adv_dir = "exp_data/agent_adv"
    output_file = "test_data.json"

    # 初始化数据结构
    test_data = []
    stats = {
        "scene_types": defaultdict(int),
        "attack_types": defaultdict(int),
        "total_clean_images": 0,
        "total_attack_images": 0
    }

    # 第一步：建立原始图像索引
    print("Building clean image index...")
    clean_image_index = {}  # som_id -> image_info mapping
    for item in sorted(os.listdir(clean_dir)):
        item_path = os.path.join(clean_dir, item)
        if not os.path.isdir(item_path):
            continue

        # 提取som_id（从目录名中的数字部分）
        som_id = int(item.split('_')[1])
        scene_type = item.split('_')[0]
        
        # 获取ground truth caption
        obs_text_path = os.path.join(item_path, "obs_text.txt")
        gt_caption = extract_description_from_obs_text(obs_text_path)
        
        if gt_caption:
            clean_image_index[som_id] = {
                "directory": item,
                "scene_type": scene_type,
                "gt_caption": gt_caption,
                "clean_image_path": os.path.join(item, "obs_screenshot.png")
            }
            stats["scene_types"][scene_type] += 1
            stats["total_clean_images"] += 1

    # 第二步：处理攻击样本
    print("\nProcessing attack samples...")
    for adv_item in sorted(os.listdir(adv_dir)):
        adv_item_path = os.path.join(adv_dir, adv_item)
        if not os.path.isdir(adv_item_path):
            continue

        # 读取攻击数据
        data_json_path = os.path.join(adv_item_path, "data.json")
        attack_data = load_attack_data(data_json_path)
        
        if not attack_data or attack_data['victim_som_id'] is None:
            print(f"Warning: Invalid attack data in {adv_item}")
            continue

        # 获取原始图像信息
        victim_som_id = attack_data['victim_som_id']
        if victim_som_id not in clean_image_index:
            print(f"Warning: No clean image found for som_id {victim_som_id} in {adv_item}")
            continue

        # 提取攻击类型
        attack_parts = adv_item.split('_')
        attack_type = '_'.join(attack_parts[2:-1]) if len(attack_parts) > 3 else attack_parts[-2]
        stats["attack_types"][attack_type] += 1

        # 处理攻击目录中的每个图像
        for img_file in sorted(os.listdir(adv_item_path)):
            if img_file.endswith('.png'):
                stats["total_attack_images"] += 1
                clean_info = clean_image_index[victim_som_id]
                
                test_data.append({
                    "image_id": clean_info["directory"],
                    "scene_type": clean_info["scene_type"],
                    "attack_type": attack_type,
                    "gt_caption": clean_info["gt_caption"],
                    "target_caption": attack_data["target_caption"],
                    "adversarial_goal": attack_data["adversarial_goal"],
                    "image_path": os.path.join(adv_item, img_file),
                    "clean_image_path": clean_info["clean_image_path"],
                    "som_id": victim_som_id
                })

    print("Processing clean image directories...")
    for item in sorted(os.listdir(clean_dir)):
        item_path = os.path.join(clean_dir, item)
        if not os.path.isdir(item_path):
            continue

        # Get scene type (classifieds, reddit, shopping)
        scene_type = item.split('_')[0]
        stats["scene_types"][scene_type] += 1
        stats["total_clean_images"] += 1

        # Get ground truth caption
        obs_text_path = os.path.join(item_path, "obs_text.txt")
        gt_caption = extract_description_from_obs_text(obs_text_path)
        
        if not gt_caption:
            print(f"Warning: No caption found for {item}")
            continue

        # Find corresponding attack directories
        for adv_item in os.listdir(adv_dir):
            if adv_item.startswith(item + "_") and os.path.isdir(os.path.join(adv_dir, adv_item)):
                # Extract attack type from directory name
                attack_parts = adv_item.split('_')
                attack_type = '_'.join(attack_parts[2:-1]) if len(attack_parts) > 3 else attack_parts[-2]
                stats["attack_types"][attack_type] += 1

                # Process each attack image in the directory
                adv_dir_path = os.path.join(adv_dir, adv_item)
                for img_file in sorted(os.listdir(adv_dir_path)):
                    if img_file.endswith('.png'):
                        stats["total_attack_images"] += 1
                        test_data.append({
                            "image_id": item,
                            "scene_type": scene_type,
                            "attack_type": attack_type,
                            "gt_caption": gt_caption,
                            "image_path": os.path.join(adv_item, img_file),
                            "clean_image_path": os.path.join(item, "obs_screenshot.png")
                        })

    # Sort test data by image_id
    test_data.sort(key=lambda x: x["image_id"])

    # Save to JSON file
    print(f"\nSaving test data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(test_data, f, indent=2)

    # Print statistics
    print("\nTest Data Statistics:")
    print(f"Total test cases: {len(test_data)}")
    print(f"Total clean images: {stats['total_clean_images']}")
    print(f"Total attack images: {stats['total_attack_images']}")
    
    print("\nScene type distribution:")
    for scene_type, count in sorted(stats["scene_types"].items()):
        print(f"  {scene_type}: {count} items")
    
    print("\nAttack type distribution:")
    for attack_type, count in sorted(stats["attack_types"].items()):
        print(f"  {attack_type}: {count} instances")

if __name__ == "__main__":
    main()