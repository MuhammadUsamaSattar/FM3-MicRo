#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --gpus=h100:1
#SBATCH --mem=2G
#SBATCH --output=outputs/%A/%a_output.out
#SBATCH --error=outputs/%A/%a_error.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=muhammad.sattar@aalto.fi
#SBATCH --job-name=llama_8b_rew
#SBATCH --array=0-5

module restore FM3-MicRo
source activate FM3-MicRo

# Define two different parameter sets using associative arrays
declare -A PARAMS

# Parameter set 0
PARAMS[0]="--batch-size 64 --exc train --goal-reset False --model-id TRITON_LLAMA_8B --model-quant fp16 --num-eval 100 --prompt-file llm_prompt_continuous_rewards_zero_shot.yaml --reward-type llm --rollout-steps 2048 --text-verbosity True --total-timesteps 1000000 --train-episode-time-limit 360 --train-fps 10 --train-render-fps 120 --train-render-fps None --train-verbosity True"

# Parameter set 1
PARAMS[1]="--batch-size 64 --exc train --goal-reset False --model-id TRITON_LLAMA_8B --model-quant fp16 --num-eval 100 --prompt-file llm_prompt_continuous_rewards_5_examples.yaml --reward-type llm --rollout-steps 2048 --text-verbosity True --total-timesteps 1000000 --train-episode-time-limit 360 --train-fps 10 --train-render-fps 120 --train-render-fps None --train-verbosity True"

# Parameter set 2
PARAMS[2]="--batch-size 64 --exc train --goal-reset False --model-id TRITON_LLAMA_8B --model-quant fp16 --num-eval 100 --prompt-file llm_prompt_continuous_rewards_1_example_explanation.yaml --reward-type llm --rollout-steps 2048 --text-verbosity True --total-timesteps 1000000 --train-episode-time-limit 360 --train-fps 10 --train-render-fps 120 --train-render-fps None --train-verbosity True"

# Parameter set 3
PARAMS[3]="--batch-size 64 --exc train --goal-reset False --model-id TRITON_LLAMA_8B --model-quant fp16 --num-eval 100 --prompt-file llm_prompt_binary_rewards_zero_shot.yaml --reward-type llm --rollout-steps 2048 --text-verbosity True --total-timesteps 1000000 --train-episode-time-limit 360 --train-fps 10 --train-render-fps 120 --train-render-fps None --train-verbosity True"

# Parameter set 4
PARAMS[4]="--batch-size 64 --exc train --goal-reset False --model-id TRITON_LLAMA_8B --model-quant fp16 --num-eval 100 --prompt-file llm_prompt_binary_rewards_5_examples.yaml --reward-type llm --rollout-steps 2048 --text-verbosity True --total-timesteps 1000000 --train-episode-time-limit 360 --train-fps 10 --train-render-fps 120 --train-render-fps None --train-verbosity True"

# Parameter set 5
PARAMS[5]="--batch-size 64 --exc train --goal-reset False --model-id TRITON_LLAMA_8B --model-quant fp16 --num-eval 100 --prompt-file llm_prompt_binary_rewards_1_example_explanation.yaml --reward-type llm --rollout-steps 2048 --text-verbosity True --total-timesteps 1000000 --train-episode-time-limit 360 --train-fps 10 --train-render-fps 120 --train-render-fps None --train-verbosity True"

# Choose the parameter set to use by index (e.g., 0 or 1)
SELECTED_PARAMS=${PARAMS[$SLURM_ARRAY_TASK_ID]}

# Prints filename and parameters for debugging
echo '####################'
echo $0
echo '####################'
echo $SELECTED_PARAMS
echo '####################'
echo

# Run the Python script with the selected parameters
srun python src/FM3_MicRo/rl.py $SELECTED_PARAMS
