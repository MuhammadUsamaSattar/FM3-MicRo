@echo off

set input_point=(112,122)

REM Run the Python script the first time with delta_r for first run
python src/FM3_MicRo/reward_map.py --mode plot --input-point %input_point% --file-name 2025-02-16_08-05-46_map_triton_qwen_32b_binary_rewards_1_example_explanation_r_64.0_resolution_8.0.json --verbose True
