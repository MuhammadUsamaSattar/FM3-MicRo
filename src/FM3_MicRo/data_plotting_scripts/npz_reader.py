import numpy as np

# Load the evaluations.npz file
data = np.load("src/FM3_MicRo/control_models/2024-12-18_16-58-03_delta_r_10000-steps_10-obs/evals/results/evaluations.npz")

# Access data
results = data["results"]  # Mean rewards for each evaluation
timesteps = data["timesteps"]  # Timesteps when evaluations were performed
ep_lengths = data["ep_lengths"] if "ep_lengths" in data else None  # Episode lengths (optional)
time_elapsed = data["time_elapsed"] if "time_elapsed" in data else None  # Time elapsed (optional)

# Example: Plot mean rewards over timesteps
import matplotlib.pyplot as plt

mean_rewards = np.mean(results, axis=1)  # Mean reward across episodes for each evaluation
plt.plot(timesteps, mean_rewards, label="Mean Reward")
plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.title("Evaluation Results")
plt.legend()
plt.grid()
plt.show()