@echo off

REM Run the Python script the first time with delta_r for first run
python src/FM3_MicRo/rl.py --batch-size 64 --exc train --goal-reset False --num-eval 100 --reward-type delta_r --rollout-steps 2048 --text-verbosity True --total-timesteps 2500000 --train-episode-time-limit 15 --train-fps None --train-render-fps None --train-verbosity True