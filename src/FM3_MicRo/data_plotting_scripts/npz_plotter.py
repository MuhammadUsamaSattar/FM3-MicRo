import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class NPZPlotter:
    def __init__(self, npz_file, text_verbosity=False):
        """Initializes the NPZPlotter class and loads data from the specified .npz file.

        Args:
            npz_file (str): The path to the .npz file containing evaluation data.
            text_verbosity (bool, optional): Verbosity of textual outputs. Defaults to False.
        """        
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        # Store the path to the .npz file
        self.npz_file = project_root / npz_file

        # Load the evaluations.npz file
        data = np.load(self.npz_file)

        # Extract data
        self.results = data["results"]  # Mean rewards for each evaluation
        self.timesteps = data["timesteps"]  # Timesteps when evaluations were performed
        self.ep_lengths = (
            data["ep_lengths"] if "ep_lengths" in data else None
        )  # Episode lengths (optional)
        self.time_elapsed = (
            data["time_elapsed"] if "time_elapsed" in data else None
        )  # Time elapsed (optional)
        self.text_verbosity = text_verbosity

    def print(self):
        """
        Prints the contents of the .npz file in a readable format,
        including timesteps, results, episode lengths, and time elapsed (if available).
        """
        # Print extracted data in a more detailed format
        print("\nDetailed evaluation data:")
        print(f"Timesteps (shape={self.timesteps.shape}): {self.timesteps}")
        print(f"Results (shape={self.results.shape}):\n{self.results}")
        if self.ep_lengths is not None:
            print(
                f"Episode Lengths (shape={self.ep_lengths.shape}):\n{self.ep_lengths}"
            )
        else:
            print("Episode Lengths: Not available")

        if self.time_elapsed is not None:
            print(
                f"Time Elapsed (shape={self.time_elapsed.shape}):\n{self.time_elapsed}"
            )
        else:
            print("Time Elapsed: Not available")

    def save(self):
        """
        Saves the plot.
        """
        self._gen_plot()

        # Save the plot in the same directory as the .npz file
        output_dir = os.path.dirname(self.npz_file)
        output_path = os.path.join(output_dir, "rewards_plot.png")
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
        """Generates the plot figure using the data from the .npz file."""
        # Example: Plot mean rewards over timesteps
        plt.close("all")
        mean_rewards = np.mean(
            self.results, axis=1
        )  # Mean reward across episodes for each evaluation
        plt.plot(self.timesteps, mean_rewards, label="Mean Reward")
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Reward")
        plt.title("Evaluation Results")
        plt.legend()
        plt.grid()
        plt.ylim(bottom=min(self.results,0))


if __name__ == "__main__":
    plotter = NPZPlotter(
        npz_file="src/FM3_MicRo/control_models/2024-12-19_18-02-20_delta_r_5000000-steps_5-obs/evaluations.npz",
        text_verbosity=True,
    )
    plotter.print()
    plotter.save()
    plotter.plot()
