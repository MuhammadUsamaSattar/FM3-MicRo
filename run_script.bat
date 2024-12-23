@echo off
REM Run the Python script the first time with delta_r for first run
python src/FM3_MicRo/rl.py --reward-type delta_r --num-eval 50 409600

REM Run the Python script the second time with delta_r for second run
python src/FM3_MicRo/rl.py --reward-type delta_r --num-eval 50 409600

REM Run the Python script the first time with llm for third run
python src/FM3_MicRo/rl.py --reward-type llm --num-eval 50 409600

REM Run the Python script the second time with llm  for fourth run
python src/FM3_MicRo/rl.py --reward-type llm --num-eval 50 409600