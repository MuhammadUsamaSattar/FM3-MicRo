import gymnasium as gym
import gymnasium_env
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv


class DistancePlotCallback(BaseCallback):
    """
    A custom callback to track and plot the distance value in each episode
    over training iterations and save the plot as a .png file and data as a .csv file.
    """

    def __init__(self, log_dir):
        super(DistancePlotCallback, self).__init__()
        self.distances = []  # Store distances
        self.episode_distances = []  # Store distances for the current episode
        self.episodes = []  # Store episode indices for plotting
        self.log_dir = log_dir  # Directory to save logs
        self.csv_file = os.path.join(self.log_dir, "log.csv")  # CSV file path

    def _on_training_start(self) -> None:
        # Create the log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize the CSV file
        with open(self.csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Average Distance"])  # Header row

    def _on_step(self) -> bool:
        # Access the environment's info dictionary
        infos = self.locals["infos"]
        for info in infos:
            if "distance" in info:
                self.episode_distances.append(info["distance"])

        # Check if the episode has ended
        dones = self.locals["dones"]
        for done in dones:
            if done:
                # Append the mean distance of the episode
                if len(self.episode_distances) > 0:
                    avg_distance = sum(self.episode_distances) / len(self.episode_distances)
                    self.distances.append(avg_distance)
                    self.episodes.append(len(self.distances))  # Current episode index

                    # Log the data into the CSV file
                    with open(self.csv_file, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([len(self.distances), avg_distance])

                self.episode_distances = []  # Reset for the next episode

        return True

    def _on_training_end(self) -> None:
        # Plot distances over episodes
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.episodes, self.distances, label="Distance per Episode", marker="o"
        )

        # Compute linear trendline
        z = np.polyfit(self.episodes, self.distances, 1)  # 1 for linear fit
        p = np.poly1d(z)  # Create polynomial object from coefficients
        plt.plot(
            self.episodes,
            p(self.episodes),
            linestyle="--",
            color="red",
            label="Trendline"
        )

        plt.xlabel("Episode")
        plt.ylabel("Average Distance")
        plt.title("Average Distance Over Episodes During Training")
        plt.legend()
        plt.grid()

        # Save the plot
        plot_file = os.path.join(self.log_dir, "plot.png")
        plt.savefig(plot_file)
        print(f"Plot saved as {plot_file}")
        plt.show()


if __name__ == "__main__":
    total_timesteps = 163_840  # Change to your desired total timesteps
    render_fps = 3
    episode_time_limit = 10

    # Define the directory for saving model, plot, and logs
    save_dir = f"src/FM3_MicRo/control_models/ppo_default_parameters_{total_timesteps}_steps_fm_rewards"
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Single environment
    vec_env = DummyVecEnv(
        [
            lambda: gym.make(
                "gymnasium_env/SingleParticleNoCargo-v0",
                **{
                    "render_mode": "rgb_array",
                    "render_fps": render_fps,
                    "episode_time_limit": episode_time_limit,
                    # "model_id": "PATH_QWEN_14B",
                    # "model_type": "llm",
                    # "model_quant": "4b",
                    # "context_prompt_file": "llm_prompt_zero_shot.yaml",
                    # "verbose": True,
                },
            )
        ]
    )

    n_steps = 2048
    if total_timesteps < 2048:
        n_steps = total_timesteps

    # Define the model
    model = PPO("MultiInputPolicy", vec_env, verbose=1, device="cpu", n_steps=n_steps)

    # Create an instance of the custom callback
    distance_callback = DistancePlotCallback(log_dir=save_dir)

    # Train the model with the callback
    model.learn(
        total_timesteps=total_timesteps, callback=distance_callback, progress_bar=True
    )

    # Save the model
    model_file = os.path.join(save_dir, f"ppo_default_parameters_{total_timesteps}_steps_fm_rewards")
    model.save(model_file)
    print(f"Model saved as {model_file}")

    # Uncomment the following to load and reuse the model:
    # model = PPO.load(
    #    "control_models/ppo_default_parameters_2_500_000_steps_fm_rewards", env=vec_env
    # )

    # Test the trained model
    vec_env.env_method(
        "set_reward_params",
        model_id=None,
        model_type=None,
        model_quant=None,
        context_prompt_file=None,
    )

    obs = vec_env.reset()

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
