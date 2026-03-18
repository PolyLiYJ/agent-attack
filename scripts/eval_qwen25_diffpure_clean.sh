#!/bin/bash
# Run the evaluation script for qwen25_diffpure on clean data
python eval_qwen25_diffpure.py --eval_mode clean \
    --model_id "Qwen/Qwen2.5-VL-7B-Instruct" \
    --test_json "exp_data/test_data.json" \
    --image_dir "./" \
    --output_json "exp_data/evaluation_results_diffpure_clean.json" \
    --output_mode "direct" \
    --sample_step 1 \
    --t 150 \
    --guidance_scale 7.5 \
    --model_path "models" \
    --config "/home/yjli/Agent/agent-attack/DiffPure/configs/agent.yml" \
    --log_dir "/home/yjli/Agent/agent-attack/exp_data/logs_dir_clean"