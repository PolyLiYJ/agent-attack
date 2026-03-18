import json
from collections import defaultdict

# 加载JSON文件
# with open('/home/yjli/Agent/agent-attack/cache/results_20250627235723_gpt-4omini-adv_img+adv_caption/target_correct_bim_caption_cap.json', 'r') as f:
# with open('/home/yjli/Agent/agent-attack/cache/results_20250709220016_gpt-4o-mini_typography_attack_0/target_correct_typography_attack_cap.json', 'r') as f:
with open("/home/yjli/Agent/agent-attack/pipeline_test_gpt4o_som/target_correct_bim_caption_cap.json") as f:
    data = json.load(f)

# 按攻击前缀和类型分类
attacks = defaultdict(list)

for key, success in data.items():
    # 提取前缀（如classifieds/reddit/shopping）
    prefix = key.split('_')[0]
    # 提取攻击类型（如wrong_color、wrong_email等）
    attack_type = '_'.join(key.split('_')[2:-1])
    attacks[(prefix, attack_type)].append(success)

# 计算总ASR（所有攻击的平均成功率）
total_successes = sum(sum(successes) for successes in attacks.values())
total_attempts = sum(len(successes) for successes in attacks.values())
total_asr = total_successes / total_attempts if total_attempts > 0 else 0
print("Total ASR:", total_asr)
# 计算每个攻击类型的ASR（均值）
asr_results = {}
for (prefix, attack_type), successes in attacks.items():
    asr = sum(successes) / len(successes)
    asr_results[f"{prefix}_{attack_type}"] = asr

# 输出结果
print("攻击类型及ASR统计结果:")
for attack, rate in sorted(asr_results.items()):
    print(f"{attack}: {rate:.2%}")

# 可选的汇总统计（按前缀）
prefix_asr = defaultdict(list)
for (prefix, _), successes in attacks.items():
    prefix_asr[prefix].extend(successes)

print("\n按前缀的汇总ASR:")
for prefix, successes in prefix_asr.items():
    print(f"{prefix}: {sum(successes)/len(successes):.2%}")
    
# 可选的汇总统计（按后缀）
suffix_asr = defaultdict(list)
for key, success in data.items():
    # 提取后缀（如wrong_color、wrong_email等）
    suffix = '_'.join(key.split('_')[2:-1])
    suffix_asr[suffix].append(success)

print("\n按后缀的汇总ASR:")
for suffix, successes in suffix_asr.items():
    print(f"{suffix}: {sum(successes)/len(successes):.2%}")