#!/bin/bash
#SBATCH --time=25:00:00
#SBATCH --gpus=h100:1
#SBATCH --mem=10G
#SBATCH --output=outputs/%A/%a_output.out
#SBATCH --error=outputs/%A/%a_error.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=muhammad.sattar@aalto.fi
#SBATCH --job-name=gen_map_3b
#SBATCH --array=0-2

module restore FM3-MicRo
source activate FM3-MicRo

# Common parameters
COMMON_PARAMS="--mode generate --model-id TRITON_QWEN_3B --workspace-radius 320 --reward-calculation-radius 64 --resolution 8 --verbose True"

# List of different prompt files
PROMPT_FILES=(
    "llm_prompt_binary_rewards_zero_shot.yaml"
    "llm_prompt_binary_rewards_5_examples.yaml"
    "llm_prompt_binary_rewards_1_example_explanation.yaml"
)

# Select the prompt file based on the array task ID
SELECTED_PROMPT_FILE=${PROMPT_FILES[$SLURM_ARRAY_TASK_ID]}

# Print debug information
echo '####################'
echo $0
echo '####################'
echo "Selected prompt file: $SELECTED_PROMPT_FILE"
echo "Running with parameters: $COMMON_PARAMS --prompt-file $SELECTED_PROMPT_FILE"
echo '####################'
echo

# Run the Python script with selected parameters
srun python src/FM3_MicRo/reward_map.py $COMMON_PARAMS --prompt-file "$SELECTED_PROMPT_FILE"
