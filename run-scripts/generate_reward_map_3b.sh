#!/bin/bash
#SBATCH --time=0:10:00
#SBATCH --gpus=h100:1
#SBATCH --mem=10G
#SBATCH --output=outputs/%A/%a_output.out
#SBATCH --error=outputs/%A/%a_error.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=muhammad.sattar@aalto.fi
#SBATCH --job-name=gen_rew_map

module restore FM3-MicRo
source activate FM3-MicRo

SELECTED_PARAMS="--mode generate --model-id TRITON_QWEN_3B --workspace-radius 320 --reward-calculation-radius 128 --resolution 128 --prompt-file llm_prompt_continuous_rewards_zero_shot.yaml --verbose True"

# Prints filename and parameters for debugging
echo '####################'
echo $0
echo '####################'
echo $SELECTED_PARAMS
echo '####################'
echo

# Run the Python script with the selected parameters
srun python src/FM3_MicRo/reward_map.py $SELECTED_PARAMS