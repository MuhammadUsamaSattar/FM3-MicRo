@echo off
setlocal enabledelayedexpansion

Define file names individually
set file_names[0]=2025-02-03_13-43-26_map_triton_qwen_3b_continuous_rewards_zero_shot_r_64.0_resolution_8.0.json
set file_names[1]=2025-02-03_17-26-09_map_triton_qwen_3b_continuous_rewards_5_examples_r_64.0_resolution_8.0.json
set file_names[2]=2025-02-04_08-20-11_map_triton_qwen_3b_continuous_rewards_1_example_explanation_r_64.0_resolution_8.0.json
set file_names[3]=2025-02-04_08-23-36_map_triton_qwen_7b_continuous_rewards_zero_shot_r_64.0_resolution_8.0.json
set file_names[4]=2025-02-05_16-57-53_map_triton_qwen_7b_continuous_rewards_5_examples_r_64.0_resolution_8.0.json
set file_names[5]=2025-02-08_17-42-34_map_triton_qwen_7b_continuous_rewards_1_example_explanation_r_64.0_resolution_8.0.json
set file_names[6]=2025-02-04_21-47-17_map_triton_qwen_14b_continuous_rewards_zero_shot_r_64.0_resolution_8.0.json
set file_names[7]=2025-02-11_17-08-05_map_triton_qwen_14b_continuous_rewards_5_examples_r_64.0_resolution_8.0.json
set file_names[8]=2025-02-08_17-44-17_map_triton_qwen_14b_continuous_rewards_1_example_explanation_r_64.0_resolution_8.0.json
set file_names[9]=2025-02-08_17-55-12_map_triton_qwen_32b_continuous_rewards_5_examples_r_64.0_resolution_8.0.json
set file_names[10]=2025-02-11_17-11-53_map_triton_qwen_32b_continuous_rewards_1_example_explanation_r_64.0_resolution_8.0.json
set file_names[11]=2025-02-11_17-11-53_map_triton_qwen_32b_continuous_rewards_zero_shot_r_64.0_resolution_8.0.json
set file_names[12]=2025-02-16_13-23-49_map_triton_qwen_3b_binary_rewards_zero_shot_r_64.0_resolution_8.0.json
set file_names[13]=2025-02-16_16-44-39_map_triton_qwen_3b_binary_rewards_5_examples_r_64.0_resolution_8.0.json
set file_names[14]=2025-02-16_21-18-52_map_triton_qwen_3b_binary_rewards_1_example_explanation_r_64.0_resolution_8.0.json
set file_names[15]=2025-02-17_12-48-19_map_triton_qwen_7b_binary_rewards_zero_shot_r_64.0_resolution_8.0.json
set file_names[16]=2025-02-17_13-30-23_map_triton_qwen_7b_binary_rewards_5_examples_r_64.0_resolution_8.0.json
set file_names[17]=2025-02-21_22-33-20_map_triton_qwen_7b_binary_rewards_1_example_explanation_r_64.0_resolution_8.0.json
set file_names[18]=2025-02-21_22-33-20_map_triton_qwen_14b_binary_rewards_zero_shot_r_64.0_resolution_8.0.json
set file_names[19]=2025-02-18_09-47-47_map_triton_qwen_14b_binary_rewards_5_examples_r_64.0_resolution_8.0.json
set file_names[20]=2025-03-06_03-21-43_map_triton_qwen_14b_binary_rewards_1_example_explanation_r_64.0_resolution_8.0.json
set file_names[21]=2025-02-16_00-05-41_map_triton_qwen_32b_binary_rewards_zero_shot_r_64.0_resolution_8.0.json
set file_names[22]=2025-02-16_04-33-09_map_triton_qwen_32b_binary_rewards_5_examples_r_64.0_resolution_8.0.json
set file_names[23]=2025-02-16_08-05-46_map_triton_qwen_32b_binary_rewards_1_example_explanation_r_64.0_resolution_8.0.json

REM Loop over input points
for %%p in ("(0,0)" "(112,112)" "(112,-112)" "(-112,-112)" "(-112,112)") do (
    REM Loop over file names using index
    for /L %%i in (0,1,23) do (
        set file_name=!file_names[%%i]!
        python src/FM3_MicRo/reward_map.py --mode plot --input-point %%p --file-name !file_name! --show-plot-after-save False --verbose True
    )
)
