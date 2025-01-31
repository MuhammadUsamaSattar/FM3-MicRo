@echo off

REM Run the Python script the first time with delta_r for first run
python src/FM3_MicRo/reward_map.py --mode generate --model-id PATH_LLAMA_8B --workspace-radius 2 --reward-calculation-radius 1 --resolution 1 --verbose True