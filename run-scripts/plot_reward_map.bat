@echo off

set input_point=(-113,-113)

REM Run the Python script the first time with delta_r for first run
python src/FM3_MicRo/reward_map.py --mode plot --input-point %input_point% --file-name 2025-02-02_17-26-22_map_triton_qwen_3b__r_16.0_resolution_8.0.json --verbose True
