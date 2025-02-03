#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=10G
#SBATCH --output=outputs/%A/%a_output.out
#SBATCH --error=outputs/%A/%a_error.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=muhammad.sattar@aalto.fi
#SBATCH --job-name=calc_points
#SBATCH --array=0-2

module restore FM3-MicRo
source activate FM3-MicRo

# Define two different parameter sets using associative arrays
declare -A PARAMS

# Parameter set 0
PARAMS[0]="--mode calculate_data_points --workspace-radius 320 --reward-calculation-radius 1 --resolution 1 --verbose True"

# Parameter set 1
PARAMS[1]="--mode calculate_data_points --workspace-radius 320 --reward-calculation-radius 2 --resolution 1 --verbose True"

# Parameter set 2
PARAMS[2]="--mode calculate_data_points --workspace-radius 320 --reward-calculation-radius 5 --resolution 1 --verbose True"

# Parameter set 3
PARAMS[3]="--mode calculate_data_points --workspace-radius 320 --reward-calculation-radius 2 --resolution 0.2 --verbose True"

# Parameter set 4
PARAMS[4]="--mode calculate_data_points --workspace-radius 320 --reward-calculation-radius 2 --resolution 0.1 --verbose True"

# Parameter set 5
PARAMS[5]="--mode calculate_data_points --workspace-radius 320 --reward-calculation-radius 5 --resolution 0.25 --verbose True"

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