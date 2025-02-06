#!/bin/bash
#SBATCH --time=35:00:00
#SBATCH --gpus=h100:1
#SBATCH --mem=20G
#SBATCH --output=outputs/%A/%a_output.out
#SBATCH --error=outputs/%A/%a_error.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=muhammad.sattar@aalto.fi
#SBATCH --job-name=gen_rew_map
#SBATCH --array=1-2

module restore FM3-MicRo
source activate FM3-MicRo

# Define two different parameter sets using associative arrays
declare -A PARAMS

# Parameter set 0
PARAMS[0]="--mode generate --model-id TRITON_QWEN_7B --workspace-radius 320 --reward-calculation-radius 64 --resolution 8 --prompt-file llm_prompt_continuous_rewards_zero_shot.yaml --verbose True"

# Parameter set 1
PARAMS[1]="--mode generate --model-id TRITON_QWEN_7B --workspace-radius 320 --reward-calculation-radius 64 --resolution 8 --prompt-file llm_prompt_continuous_rewards_5_examples.yaml --verbose True"

# Parameter set 2
PARAMS[2]="--mode generate --model-id TRITON_QWEN_7B --workspace-radius 320 --reward-calculation-radius 64 --resolution 8 --prompt-file llm_prompt_continuous_rewards_1_example_explanation.yaml --verbose True"

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
srun python src/FM3_MicRo/reward_map.py $SELECTED_PARAMS