@echo off

REM Run the Python script the first time with delta_r for first run
python src/FM3_MicRo/reward_map.py --mode calculate_data_points --workspace-radius 320 --reward-calculation-radius 0.5 --resolution 0.5 --verbose True