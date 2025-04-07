@echo off

REM Activate the Conda environment
CALL conda activate FM3-MicRo

REM Run the Python script the first time with delta_r for first run
python src/FM3_MicRo/rl.py --batch-size 64 --exc test --goal-reset False --num-eval 100 --reward-type delta_r --rollout-steps 2048 --text-verbosity True --total-timesteps 2500000 --test-episode-time-limit 15 --train-fps None --train-render-fps None --train-verbosity True --test-model-path src/FM3_MicRo/control_models/2025-01-24_14-26-37_llm_triton_qwen_14b_continuous_rewards_1_example_explanation_1000000-steps_5-obs_ep-time-360.0/model.zip --env-type simulator --goal-reset True