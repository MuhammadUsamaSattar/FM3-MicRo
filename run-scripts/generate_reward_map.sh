#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --gpus=h100:1
#SBATCH --mem=10G
#SBATCH --output=outputs/%A/%a_output.out
#SBATCH --error=outputs/%A/%a_error.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=muhammad.sattar@aalto.fi
#SBATCH --job-name=gen_rew_map
#SBATCH --array=0-4

module restore FM3-MicRo
source activate FM3-MicRo

# Define two different parameter sets using associative arrays
declare -A PARAMS

# Parameter set 0
PARAMS[0]="--mode generate --model-id TRITON_QWEN_3B --workspace-radius 2 --reward-calculation-radius 1 --resolution 1 --verbose True"

# Parameter set 1
PARAMS[1]="--mode generate --model-id TRITON_QWEN_7B --workspace-radius 2 --reward-calculation-radius 1 --resolution 1 --verbose True"

# Parameter set 2
PARAMS[2]="--mode generate --model-id TRITON_QWEN_14B --workspace-radius 2 --reward-calculation-radius 1 --resolution 1 --verbose True"

# Parameter set 3
PARAMS[3]="--mode generate --model-id TRITON_QWEN_32B --workspace-radius 2 --reward-calculation-radius 1 --resolution 1 --verbose True"

# Parameter set 4
PARAMS[4]="--mode generate --model-id TRITON_QWEN_72B --workspace-radius 2 --reward-calculation-radius 1 --resolution 1 --verbose True"

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