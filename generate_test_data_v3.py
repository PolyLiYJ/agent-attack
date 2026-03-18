import os
import json
from collections import defaultdict
import ast

def parse_som_id(som_id_raw):
    """
    解析som_id，支持数字和列表格式。
    
    Args:
        som_id_raw: 原始som_id值，可能是数字或字符串表示的列表
        
    Returns:
        list: som_id列表
    """
    try:
        if isinstance(som_id_raw, (int, float)):
            return [int(som_id_raw)]
        elif isinstance(som_id_raw, str):
            # 尝试解析为列表
            if som_id_raw.startswith('[') and som_id_raw.endswith(']'):
                return [int(x.strip()) for x in ast.literal_eval(som_id_raw)]
            else:
                return [int(som_id_raw)]
        elif isinstance(som_id_raw, list):
            return [int(x) for x in som_id_raw]
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing som_id {som_id_raw}: {e}")
    return []

def load_attack_data(data_json_path):
    """从攻击目录下的data.json加载攻击相关信息。"""
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

def extract_description_by_som_id(file_path, som_id):
    """从obs_text.txt文件中提取指定som_id的图像描述。"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                # 支持多种som_id格式的匹配
                if f"[{som_id}]" in line and "[IMG]" in line and "description:" in line:
                    desc_start = line.find('description:') + len('description:')
                    desc_end = min(
                        line.find(',', desc_start) if line.find(',', desc_start) != -1 else len(line),
                        line.find(']', desc_start) if line.find(']', desc_start) != -1 else len(line)
                    )
                    return line[desc_start:desc_end].strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

def find_clean_image_info(clean_dir, clean_item, som_ids):
    """在clean目录中查找指定som_id的图像信息。"""
    for som_id in som_ids:
        item_path = os.path.join(clean_dir, clean_item)
        if not os.path.isdir(item_path):
            continue
        try:
            obs_text_path = os.path.join(item_path, "obs_text.txt")
            gt_caption = extract_description_by_som_id(obs_text_path, som_id)
            
            if gt_caption:
                return {
                    "directory": clean_item,
                    "scene_type": clean_item.split('_')[0],
                    "gt_caption": gt_caption,
                    "screen_shot_path": os.path.join(item_path, "obs_screenshot.png"),
                    "som_id": som_id
                }
        except (IndexError, ValueError):
            continue
                
    return None

def get_threat_level(attack_type: str) -> int:
    """根据攻击类型返回威胁等级(1-3)"""
    high_threat = ["choose_me", "do_not_choose", "upvote", "wrong_email"]
    medium_threat = [
        "modify_comment", "add_comment", "wrong_price", "wrong_rank",
        "wrong_location", "wrong_city", "wrong_position", "wrong_star"
    ]
    
    if any(ht in attack_type for ht in high_threat):
        return 3
    elif any(mt in attack_type for mt in medium_threat):
        return 2
    return 1

def main():
    """主函数：生成测试数据集并统计信息。"""
    clean_dir = "exp_data/agent_clean"
    adv_dir = "exp_data/agent_adv"
    output_file = "exp_data/test_data_detail.json"
    error_log_file = "generate_test_data_errors.log"

    test_data = []
    stats = {
        "scene_types": defaultdict(int),
        "attack_types": defaultdict(int),
        "threat_levels": defaultdict(int),
        "total_clean_images": set(),
        "total_attack_images": 0,
        "errors": defaultdict(list)
    }

    # 处理攻击样本
    print("Processing attack samples...")
    for adv_item in sorted(os.listdir(adv_dir)):
        adv_item_path = os.path.join(adv_dir, adv_item)
        if not os.path.isdir(adv_item_path):
            continue

        # 读取攻击数据
        data_json_path = os.path.join(adv_item_path, "data.json")
        attack_data = load_attack_data(data_json_path)
        
        if not attack_data or attack_data['victim_som_id'] is None:
            stats["errors"]["invalid_data"].append(adv_item)
            continue

        # 解析som_id
        som_ids = parse_som_id(attack_data['victim_som_id'])
        if not som_ids:
            stats["errors"]["invalid_som_id"].append(
                f"{adv_item}: {attack_data['victim_som_id']}")
            continue

        # 获取原始图像信息
        # Extract clean item from adv_item by removing '_adv' suffix if present
        clean_item = adv_item.split("_")[0]+"_"+adv_item.split("_")[1]
        clean_info = find_clean_image_info(clean_dir, clean_item, som_ids)

        if not clean_info:
            stats["errors"]["missing_clean_image"].append(
                f"{adv_item}: {som_ids}")
            continue

        # 提取攻击类型和威胁等级
        attack_parts = adv_item.split('_')
        attack_type = '_'.join(attack_parts[2:-1]) if len(attack_parts) > 3 else attack_parts[-2]
        threat_level = get_threat_level(attack_type)
        
        stats["attack_types"][attack_type] += 1
        stats["threat_levels"][f"Level {threat_level}"] += 1
        stats["scene_types"][clean_info["scene_type"]] += 1
        stats["total_clean_images"].add(clean_info["som_id"])
    
        for img_file in sorted(os.listdir(adv_item_path)):
            if img_file.endswith('.png') and (img_file.startswith('bim_caption_attack') or img_file.startswith('clip_attack')):
                stats["total_attack_images"] += 1
                test_data.append({
                    "image_id": adv_item,
                    "scene_type": clean_info["scene_type"],
                    "attack_type": attack_type,
                    "threat_level": threat_level,
                    "gt_caption": clean_info["gt_caption"],
                    "target_caption": attack_data["target_caption"],
                    "adversarial_goal": attack_data["adversarial_goal"],
                    "atack_image_path": os.path.join(adv_item_path, img_file),
                    "clean_image_path": os.path.join(adv_item_path, "victim_image.png"),
                    "som_id": clean_info["som_id"]
                })

    # 按照多个键值排序测试数据
    test_data.sort(key=lambda x: (x["scene_type"], x["som_id"], x["attack_type"]))

    # 保存测试数据到JSON文件
    print(f"\nSaving test data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump({
            "metadata": {
                "total_samples": len(test_data),
                "total_clean_images": len(stats["total_clean_images"]),
                "total_attack_images": stats["total_attack_images"],
                "scene_types": dict(stats["scene_types"]),
                "attack_types": dict(stats["attack_types"])
            },
            "test_cases": test_data
        }, f, indent=2)

    # 保存错误日志
    with open(error_log_file, 'w') as f:
        json.dump(stats["errors"], f, indent=2)

    # 打印统计信息
    print("\nTest Data Statistics:")
    print(f"Total test cases: {len(test_data)}")
    print(f"Total unique clean images: {len(stats['total_clean_images'])}")
    print(f"Total attack images: {stats['total_attack_images']}")
    
    print("\nScene type distribution:")
    for scene_type, count in sorted(stats["scene_types"].items()):
        print(f"  {scene_type}: {count} items")
        scene_attacks = [item for item in test_data if item["scene_type"] == scene_type]
        if scene_attacks:
            print(f"    - Total attacks for {scene_type}: {len(scene_attacks)}")
    
    print("\nAttack type distribution:")
    for attack_type, count in sorted(stats["attack_types"].items()):
        threat_level = get_threat_level(attack_type)
        print(f"  {attack_type} (Level {threat_level}): {count} instances")
        type_attacks = [item for item in test_data if item["attack_type"] == attack_type]
        if type_attacks:
            scene_breakdown = defaultdict(int)
            for item in type_attacks:
                scene_breakdown[item["scene_type"]] += 1
            print("    Scene breakdown:")
            for scene, scene_count in sorted(scene_breakdown.items()):
                print(f"    - {scene}: {scene_count} attacks")

    print("\nThreat level distribution:")
    for level, count in sorted(stats["threat_levels"].items()):
        print(f"  {level}: {count} attacks")
    
    print(f"\nError details have been saved to {error_log_file}")
    print(f"Total errors: {sum(len(errors) for errors in stats['errors'].values())}")
    for error_type, errors in stats["errors"].items():
        print(f"  {error_type}: {len(errors)} cases")

if __name__ == "__main__":
    main()