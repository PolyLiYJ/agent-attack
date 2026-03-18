import json
# with open("/home/yjli/Agent/agent-attack/cache/LifeLong—results_20250902192358_gpt-4o-mini_typography_attack_0/task_id_score.json", "r") as f:

with open("//home/yjli/Agent/agent-attack/cache/LifeLong—results_20250902192358_gpt-4o-mini_typography_attack_0/task_id_score.json", "r") as f:
    data = json.load(f)
    
# Count total number of tasks
total_tasks = len(data)

# Count tasks with score > 0.8
high_score_tasks = 0
for key, value in data.items():
    if value > 0.8:
        high_score_tasks += 1

# Calculate accuracy
accuracy = high_score_tasks / total_tasks if total_tasks > 0 else 0

print(f"Total tasks: {total_tasks}")
print(f"Tasks with score > 0.8: {high_score_tasks}")
print(f"Accuracy (score > 0.8): {accuracy:.4f} ({accuracy*100:.2f}%)")
