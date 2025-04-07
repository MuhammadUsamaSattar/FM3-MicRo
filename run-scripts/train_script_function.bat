@echo off

REM Run the Python script the first time with delta_r for first run
python src/FM3_MicRo/rl.py --batch-size 64 --exc train --goal-reset False --model-id PATH_LLAMA_8B --model-quant fp16 --num-eval 100 --prompt-file llm_prompt_continuous_rewards_zero_shot.yaml --reward-type llm --rollout-steps 2048 --text-verbosity True --total-timesteps 1000000 --train-episode-time-limit 15 --train-fps None --train-render-fps None --train-verbosity True --goal-reset True