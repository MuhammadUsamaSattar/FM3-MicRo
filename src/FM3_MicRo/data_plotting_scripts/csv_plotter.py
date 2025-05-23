import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class CSVPlotter:
    def __init__(self, file_path: str) -> None:
        """Initialize the CSVPlotter with the path to the CSV file.

        Args:
            file_path (str): Path to the CSV file.
        """
        self.file_path = file_path

    def plot(self) -> None:
        """
        Reads the CSV file, cleans the data, computes a trendline, and plots
        Episode vs. Distance Ratio with the trendline.
        """
        # Load the CSV file
        data = pd.read_csv(self.file_path)

        # Extract the Episode and Ratio columns
        episodes = data["Episode"]
        ratios = data["Ratio"]

        # Ensure that Ratio column is numeric (convert if necessary)
        ratios = pd.to_numeric(ratios, errors="coerce")

        # Drop rows with NaN values in Episode or Ratio
        data_cleaned = data.dropna(subset=["Episode", "Ratio"])
        episodes = data_cleaned["Episode"]
        ratios = data_cleaned["Ratio"]

        # Calculate the trendline (linear fit)
        z = np.polyfit(episodes, ratios, 1)  # Degree 1 polynomial (linear)
        p = np.poly1d(z)

        # Generate trendline values
        trendline = p(episodes)

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, ratios, label="Data Points", color="blue", alpha=0.7)
        plt.plot(episodes, trendline, label="Trend Line", color="black")

        # Add labels, title, and legend
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Ratio", fontsize=12)
        plt.title("Episode vs. Ratio with Trend Line", fontsize=14)
        plt.legend()

        # Show the plot
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    plotter = CSVPlotter(
        "src/FM3_MicRo/control_models/2024-12-18_17-11-50_delta_r_10000-steps_10-obs/log.csv"
    )
    plotter.plot()
