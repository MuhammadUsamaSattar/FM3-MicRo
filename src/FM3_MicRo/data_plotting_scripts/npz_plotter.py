import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class NPZPlotter:
    def __init__(self, npz_files, text_verbosity=False):
        """Initializes the NPZPlotter class and loads data from the specified .npz file(s).

        Args:
            npz_files (str or dict): Either a single file path (str) or a dictionary where keys are legend labels and values are file paths.
            text_verbosity (bool, optional): Verbosity of textual outputs. Defaults to False.
        """
        file_path = Path(__file__).resolve().parent.parent.parent.parent
        self.text_verbosity = text_verbosity
        
        # If a single file path is provided, convert it to a dictionary without labels
        if isinstance(npz_files, str):
            self.npz_files = {None: file_path / npz_files}  # No legend labels
        else:
            self.npz_files = {label: file_path / file for label, file in npz_files.items()}

        # Load data for all files
        self.data = {}
        for label, file_path in self.npz_files.items():
            data = np.load(file_path)
            self.data[label] = {
                "timesteps": data["timesteps"],
                "results": data["results"],
                "ep_lengths": data["ep_lengths"] if "ep_lengths" in data else None,
                "time_elapsed": data["time_elapsed"] if "time_elapsed" in data else None,
            }
    
    def print(self):
        """
        Prints the contents of the .npz files in a readable format.
        """
        for label, dataset in self.data.items():
            label_str = f" ({label})" if label else ""
            print(f"\nDetailed evaluation data{label_str}:")
            print(f"Timesteps (shape={dataset['timesteps'].shape}): {dataset['timesteps']}")
            print(f"Results (shape={dataset['results'].shape}):\n{dataset['results']}")
            if dataset["ep_lengths"] is not None:
                print(f"Episode Lengths (shape={dataset['ep_lengths'].shape}):\n{dataset['ep_lengths']}")
            else:
                print("Episode Lengths: Not available")
            if dataset["time_elapsed"] is not None:
                print(f"Time Elapsed (shape={dataset['time_elapsed'].shape}):\n{dataset['time_elapsed']}")
            else:
                print("Time Elapsed: Not available")
    
    def save(self, path):
        """
        Saves the plot.
        """
        self._gen_plot()
        os.makedirs(path, exist_ok=True)
        output_path = os.path.join(path, "rewards_plot.png")
        plt.savefig(output_path)
        if self.text_verbosity:
            print(f"Plot saved to {output_path}")
    
    def plot(self):
        """
        Shows the plot.
        """
        self._gen_plot()
        plt.show()
    
    def _gen_plot(self):
        """Generates the plot figure using the data from the .npz files."""
        plt.close("all")
        plt.figure(figsize=(8, 6))
        
        for label, dataset in self.data.items():
            mean_rewards = np.mean(dataset["results"], axis=1)
            plt.plot(dataset["timesteps"], mean_rewards, label=label if label else None, linewidth=2.5)
        
        font = {
        'size': 24,
        }
        tick_fontsize=16

        plt.xlabel("Timesteps", font)
        plt.ylabel("Mean Reward", font)
        plt.title("Evaluation Results", font)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.grid()
        plt.ylim(bottom=min(np.min(dataset["results"]) for dataset in self.data.values()), top=None)
        
        if any(self.npz_files.keys()):  # Only show legend if labels exist
            plt.legend(fontsize=tick_fontsize)


if __name__ == "__main__":
    npz_files_zero_shot_binary = {
        "QWEN-3b-Instruct": "src/FM3_MicRo/control_models/2025-01-20_13-49-52_llm_triton_qwen_3b_binary_rewards_zero_shot_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "QWEN-7b-Instruct": "src/FM3_MicRo/control_models/2025-01-22_04-50-18_llm_triton_qwen_7b_binary_rewards_zero_shot_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "QWEN-14b-Instruct": "src/FM3_MicRo/control_models/2025-01-24_14-30-15_llm_triton_qwen_14b_binary_rewards_zero_shot_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "QWEN-32b-Instruct": "src/FM3_MicRo/control_models/2025-01-31_15-25-46_llm_triton_qwen_32b_binary_rewards_zero_shot_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "sparse": "src/FM3_MicRo/control_models/2025-01-23_15-57-24_sparse_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "delta_r": "src/FM3_MicRo/control_models/2025-01-23_15-57-24_delta_r_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
    }

    npz_files_zero_shot_continuous = {
        "QWEN-3b-Instruct": "src/FM3_MicRo/control_models/2025-01-20_12-12-47_llm_triton_qwen_3b_continuous_rewards_zero_shot_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "QWEN-7b-Instruct": "src/FM3_MicRo/control_models/2025-01-21_11-13-25_llm_triton_qwen_7b_continuous_rewards_zero_shot_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "QWEN-14b-Instruct": "src/FM3_MicRo/control_models/2025-01-23_21-46-11_llm_triton_qwen_14b_continuous_rewards_zero_shot_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "QWEN-32b-Instruct": "src/FM3_MicRo/control_models/2025-02-04_22-27-54_llm_triton_qwen_32b_continuous_rewards_zero_shot_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "sparse": "src/FM3_MicRo/control_models/2025-01-23_15-57-24_sparse_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "delta_r": "src/FM3_MicRo/control_models/2025-01-23_15-57-24_delta_r_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
    }

    npz_files_five_shot_binary = {
        "QWEN-3b-Instruct": "src/FM3_MicRo/control_models/2025-01-21_05-48-34_llm_triton_qwen_3b_binary_rewards_5_examples_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "QWEN-7b-Instruct": "src/FM3_MicRo/control_models/2025-01-23_09-59-26_llm_triton_qwen_7b_binary_rewards_5_examples_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "QWEN-14b-Instruct": "src/FM3_MicRo/control_models/2025-01-28_08-31-10_llm_triton_qwen_14b_binary_rewards_5_examples_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "QWEN-32b-Instruct": "src/FM3_MicRo/control_models/2025-02-01_23-47-49_llm_triton_qwen_32b_binary_rewards_5_examples_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "sparse": "src/FM3_MicRo/control_models/2025-01-23_15-57-24_sparse_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "delta_r": "src/FM3_MicRo/control_models/2025-01-23_15-57-24_delta_r_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
    }

    npz_files_five_shot_continuous = {
        "QWEN-3b-Instruct": "src/FM3_MicRo/control_models/2025-01-20_12-12-49_llm_triton_qwen_3b_continuous_rewards_5_examples_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "QWEN-7b-Instruct": "src/FM3_MicRo/control_models/2025-01-21_11-13-25_llm_triton_qwen_7b_continuous_rewards_5_examples_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "QWEN-14b-Instruct": "src/FM3_MicRo/control_models/2025-01-24_14-12-14_llm_triton_qwen_14b_continuous_rewards_5_examples_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "QWEN-32b-Instruct": "src/FM3_MicRo/control_models/2025-01-31_07-47-37_llm_triton_qwen_32b_continuous_rewards_5_examples_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "sparse": "src/FM3_MicRo/control_models/2025-01-23_15-57-24_sparse_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "delta_r": "src/FM3_MicRo/control_models/2025-01-23_15-57-24_delta_r_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
    }

    npz_files_one_shot_with_example_binary = {
        "QWEN-3b-Instruct": "src/FM3_MicRo/control_models/2025-01-21_06-30-08_llm_triton_qwen_3b_binary_rewards_1_example_explanation_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        #"QWEN-7b-Instruct": "src/FM3_MicRo/control_models/2025-02-08_17-41-25_llm_triton_qwen_7b_binary_rewards_1_example_explanation_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "QWEN-14b-Instruct": "src/FM3_MicRo/control_models/2025-01-28_09-47-00_llm_triton_qwen_14b_binary_rewards_1_example_explanation_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "QWEN-32b-Instruct": "src/FM3_MicRo/control_models/2025-02-02_00-58-18_llm_triton_qwen_32b_binary_rewards_1_example_explanation_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "sparse": "src/FM3_MicRo/control_models/2025-01-23_15-57-24_sparse_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "delta_r": "src/FM3_MicRo/control_models/2025-01-23_15-57-24_delta_r_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
    }

    npz_files_one_shot_with_example_continuous = {
        "QWEN-3b-Instruct": "src/FM3_MicRo/control_models/2025-01-20_12-13-39_llm_triton_qwen_3b_continuous_rewards_1_example_explanation_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "QWEN-7b-Instruct": "src/FM3_MicRo/control_models/2025-01-21_22-07-43_llm_triton_qwen_7b_continuous_rewards_1_example_explanation_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "QWEN-14b-Instruct": "src/FM3_MicRo/control_models/2025-01-24_14-26-37_llm_triton_qwen_14b_continuous_rewards_1_example_explanation_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "QWEN-32b-Instruct": "src/FM3_MicRo/control_models/2025-01-31_11-20-05_llm_triton_qwen_32b_continuous_rewards_1_example_explanation_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "sparse": "src/FM3_MicRo/control_models/2025-01-23_15-57-24_sparse_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "delta_r": "src/FM3_MicRo/control_models/2025-01-23_15-57-24_delta_r_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
    }

    npz_files_3b = {
        "continuous zero-shot": "src/FM3_MicRo/control_models/2025-01-20_12-12-47_llm_triton_qwen_3b_continuous_rewards_zero_shot_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "continuous five-shot": "src/FM3_MicRo/control_models/2025-01-20_12-12-49_llm_triton_qwen_3b_continuous_rewards_5_examples_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "continuous one-shot with example": "src/FM3_MicRo/control_models/2025-01-20_12-13-39_llm_triton_qwen_3b_continuous_rewards_1_example_explanation_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "binary zero-shot": "src/FM3_MicRo/control_models/2025-01-20_13-49-52_llm_triton_qwen_3b_binary_rewards_zero_shot_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "binary five-shot": "src/FM3_MicRo/control_models/2025-01-21_05-48-34_llm_triton_qwen_3b_binary_rewards_5_examples_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "binary one-shot with example": "src/FM3_MicRo/control_models/2025-01-21_06-30-08_llm_triton_qwen_3b_binary_rewards_1_example_explanation_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "sparse": "src/FM3_MicRo/control_models/2025-01-23_15-57-24_sparse_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "delta_r": "src/FM3_MicRo/control_models/2025-01-23_15-57-24_delta_r_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
    }

    npz_files_7b = {
        "continuous zero-shot": "src/FM3_MicRo/control_models/2025-01-21_11-13-25_llm_triton_qwen_7b_continuous_rewards_zero_shot_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "continuous five-shot": "src/FM3_MicRo/control_models/2025-01-21_11-13-25_llm_triton_qwen_7b_continuous_rewards_5_examples_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "continuous one-shot with example": "src/FM3_MicRo/control_models/2025-01-21_22-07-43_llm_triton_qwen_7b_continuous_rewards_1_example_explanation_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "binary zero-shot": "src/FM3_MicRo/control_models/2025-01-22_04-50-18_llm_triton_qwen_7b_binary_rewards_zero_shot_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "binary five-shot": "src/FM3_MicRo/control_models/2025-01-23_09-59-26_llm_triton_qwen_7b_binary_rewards_5_examples_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        #"binary one-shot with example": "src/FM3_MicRo/control_models/2025-02-08_17-41-25_llm_triton_qwen_7b_binary_rewards_1_example_explanation_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "sparse": "src/FM3_MicRo/control_models/2025-01-23_15-57-24_sparse_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "delta_r": "src/FM3_MicRo/control_models/2025-01-23_15-57-24_delta_r_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
    }

    npz_files_14b = {
        "continuous zero-shot": "src/FM3_MicRo/control_models/2025-01-23_21-46-11_llm_triton_qwen_14b_continuous_rewards_zero_shot_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "continuous five-shot": "src/FM3_MicRo/control_models/2025-01-24_14-12-14_llm_triton_qwen_14b_continuous_rewards_5_examples_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "continuous one-shot with example": "src/FM3_MicRo/control_models/2025-01-24_14-26-37_llm_triton_qwen_14b_continuous_rewards_1_example_explanation_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "binary zero-shot": "src/FM3_MicRo/control_models/2025-01-24_14-30-15_llm_triton_qwen_14b_binary_rewards_zero_shot_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "binary five-shot": "src/FM3_MicRo/control_models/2025-01-28_08-31-10_llm_triton_qwen_14b_binary_rewards_5_examples_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "binary one-shot with example": "src/FM3_MicRo/control_models/2025-01-28_09-47-00_llm_triton_qwen_14b_binary_rewards_1_example_explanation_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "sparse": "src/FM3_MicRo/control_models/2025-01-23_15-57-24_sparse_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "delta_r": "src/FM3_MicRo/control_models/2025-01-23_15-57-24_delta_r_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
    }

    npz_files_32b = {
        "continuous zero-shot": "src/FM3_MicRo/control_models/2025-02-04_22-27-54_llm_triton_qwen_32b_continuous_rewards_zero_shot_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "continuous five-shot": "src/FM3_MicRo/control_models/2025-01-31_07-47-37_llm_triton_qwen_32b_continuous_rewards_5_examples_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "continuous one-shot with example": "src/FM3_MicRo/control_models/2025-01-31_11-20-05_llm_triton_qwen_32b_continuous_rewards_1_example_explanation_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "binary zero-shot": "src/FM3_MicRo/control_models/2025-01-31_15-25-46_llm_triton_qwen_32b_binary_rewards_zero_shot_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "binary five-shot": "src/FM3_MicRo/control_models/2025-02-01_23-47-49_llm_triton_qwen_32b_binary_rewards_5_examples_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "binary one-shot with example": "src/FM3_MicRo/control_models/2025-02-02_00-58-18_llm_triton_qwen_32b_binary_rewards_1_example_explanation_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "sparse": "src/FM3_MicRo/control_models/2025-01-23_15-57-24_sparse_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
        "delta_r": "src/FM3_MicRo/control_models/2025-01-23_15-57-24_delta_r_1000000-steps_5-obs_ep-time-360.0/evaluations.npz",
    }

    npz_file = "src/FM3_MicRo/control_models/2025-02-04_22-27-54_llm_triton_qwen_32b_continuous_rewards_zero_shot_1000000-steps_5-obs_ep-time-360.0/evaluations.npz"

    plotter = NPZPlotter(npz_files=npz_files_five_shot_continuous, text_verbosity=True)
    plotter.print()
    #plotter.save()
    plotter.plot()