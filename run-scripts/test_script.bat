@echo off

REM Activate the Conda environment
CALL conda activate FM3-MicRo

REM Run the Python script the first time with delta_r for first run
python src/FM3_MicRo/rl.py --exc test --goal-reset False --text-verbosity True --test-episode-time-limit 15 --test-model-path src/FM3_MicRo/control_models/2025-02-02_18-32-35_llm_triton_qwen_32b_binary_rewards_1_example_explanation_1000000-steps_5-obs_ep-time-360.0/model.zip --env-type simulator