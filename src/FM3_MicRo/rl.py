import os
import time
from datetime import datetime

import gymnasium as gym
import gymnasium_env
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


class DistancePlotCallback(BaseCallback):
    """
    A custom callback to track and plot the ratio of average distance
    to starting distance for each episode during training. Uses Stable-Baselines3
    logger's CSVOutputFormat for logging.
    """

    def __init__(self, log_dir):
        super(DistancePlotCallback, self).__init__()
        """Initializes class with attributes.
        """
        
        self.distances = []  # Store distances
        self.starting_distances = []  # Store starting distances for each episode
        self.episode_distances = []  # Store distances for the current episode
        self.episodes = []  # Store episode indices for plotting
        self.ratios = []  # Store ratios (average distance / starting distance)
        self.log_dir = log_dir  # Directory to save logs
        self.start_time = None  # Track the start time of training
        self.csv_file = None  # Path to the CSV file for episode logs
        self.diagnostics_file = None  # Path to the CSV file for rollout logs
        self.iteration_counter = 0  # Counter to track iterations (rollouts)
        self.prev_total_time_steps_rollout = 0
        self.prev_total_time_steps_episode = 0

    def _on_training_start(self) -> None:
        # Create the log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize the start time
        self.start_time = time.time()

        # Initialize the CSV file paths for logging
        self.csv_file = os.path.join(self.log_dir, "log.csv")
        self.diagnostics_file = os.path.join(self.log_dir, "diagnostics.csv")

        # Write headers for log.csv (episode-level log)
        with open(self.csv_file, mode="w", newline="") as file:
            file.write(
                "Episode,Iteration,FPS,Time Elapsed (s),Total Timesteps,Average Distance,Starting Distance,Ratio,"
                "Approx KL,Clip Fraction,Entropy Loss,Explained Variance,Learning Rate,Loss,Policy Gradient Loss,Std,Value Loss\n"
            )

        # Write headers for diagnostics.csv (rollout-level log)
        with open(self.diagnostics_file, mode="w", newline="") as file:
            file.write(
                "Iteration,Episode,FPS,Time Elapsed (s),Total Timesteps,Approx KL,Clip Fraction,Entropy Loss,"
                "Explained Variance,Learning Rate,Loss,Policy Gradient Loss,Std,Value Loss\n"
            )

        self.delta_t_from_last_episode = time.time()

    def _write_to_csv(self, data: dict, file_path: str):
        """Write a row of data to the specified CSV file and close the file handle immediately.

        Args:
            data (dict): Dictionary to write to the csv file.
            file_path (str): Path of the csv file.
        """

        try:
            with open(file_path, mode="a", newline="") as file:
                file.write(",".join(map(str, data.values())) + "\n")
        except PermissionError as e:
            print(f"PermissionError while writing to CSV: {e}")
            print("Retrying in 1 second...")
            time.sleep(1)  # Wait and retry
            with open(file_path, mode="a", newline="") as file:
                file.write(",".join(map(str, data.values())) + "\n")

    def _on_rollout_end(self) -> None:
        """
        Increment the iteration counter at the end of each rollout (step within an episode).
        This tracks individual rollouts.
        """
        self.iteration_counter += 1

        # Retrieve time and step metrics from Monitor
        monitor_env = self.training_env.envs[0]  # Access the wrapped Monitor env
        total_steps = monitor_env.get_total_steps()
        episode_times = monitor_env.get_episode_times()

        # Calculate FPS based on episode steps and time elapsed
        prev_episode_time = 0
        if len(episode_times) > 1:
            prev_episode_time = episode_times[-2]

        time_elapsed = time.time() - self.delta_t_from_last_episode
        if len(episode_times) > 0:
            time_elapsed = episode_times[-1] + time_elapsed

        fps = int(
            (total_steps - self.prev_total_time_steps_rollout)
            / ((time_elapsed - prev_episode_time))
        )

        self.prev_total_time_steps_rollout = total_steps

        # Retrieve additional metrics from logger
        data = {
            "Iteration": self.iteration_counter,
            "Episode": len(self.distances) + 1,
            "FPS": fps,
            "Time Elapsed (s)": time_elapsed,
            "Total Timesteps": total_steps,
            "Approx KL": self.logger.name_to_value.get("train/approx_kl", "N/A"),
            "Clip Fraction": self.logger.name_to_value.get(
                "train/clip_fraction", "N/A"
            ),
            "Entropy Loss": self.logger.name_to_value.get("train/entropy_loss", "N/A"),
            "Explained Variance": self.logger.name_to_value.get(
                "train/explained_variance", "N/A"
            ),
            "Learning Rate": self.logger.name_to_value.get(
                "train/learning_rate", "N/A"
            ),
            "Loss": self.logger.name_to_value.get("train/loss", "N/A"),
            "Policy Gradient Loss": self.logger.name_to_value.get(
                "train/policy_gradient_loss", "N/A"
            ),
            "Std": self.logger.name_to_value.get("train/std", "N/A"),
            "Value Loss": self.logger.name_to_value.get("train/value_loss", "N/A"),
        }

        # Write the data to diagnostics.csv
        self._write_to_csv(data, self.diagnostics_file)

    def _on_step(self) -> bool:
        """Log data after each episode ends (episode-level logging).

        Returns:
            bool: False value terminates training early.
        """        

        # Access the environment's info dictionary to track distances
        infos = self.locals["infos"]
        for info in infos:
            if "distance" in info:
                self.episode_distances.append(info["distance"])

        dones = self.locals["dones"]
        for done in dones:
            if done:
                # After each episode ends, log episode data
                if len(self.episode_distances) > 0:
                    avg_distance = sum(self.episode_distances) / len(
                        self.episode_distances
                    )
                    starting_distance = self.episode_distances[0]
                    ratio = (
                        avg_distance / starting_distance
                        if starting_distance != 0
                        else 0
                    )

                    self.distances.append(avg_distance)
                    self.starting_distances.append(starting_distance)
                    self.ratios.append(ratio)

                    self.episodes.append(
                        len(self.distances)
                    )  # Episode index (used for plotting)

                    # Retrieve time and step metrics from Monitor
                    monitor_env = self.training_env.envs[
                        0
                    ]  # Access the wrapped Monitor env
                    total_steps = monitor_env.get_total_steps()
                    episode_times = monitor_env.get_episode_times()

                    # Calculate FPS based on episode steps and time elapsed
                    prev_episode_time = 0
                    if len(episode_times) > 1:
                        prev_episode_time = episode_times[-2]

                    # if len(episode_times) > 0:
                    time_elapsed = episode_times[-1]

                    fps = int(
                        (total_steps - self.prev_total_time_steps_episode)
                        / ((time_elapsed - prev_episode_time))
                    )

                    self.prev_total_time_steps_episode = total_steps

                    # Prepare data for episode log
                    data = {
                        "Episode": len(self.distances),
                        "Iteration": self.iteration_counter,
                        "FPS": fps,
                        "Time Elapsed (s)": time_elapsed,
                        "Total Timesteps": total_steps,
                        "Average Distance": avg_distance,
                        "Starting Distance": starting_distance,
                        "Ratio": ratio,
                        "Approx KL": self.logger.name_to_value.get(
                            "train/approx_kl", "N/A"
                        ),
                        "Clip Fraction": self.logger.name_to_value.get(
                            "train/clip_fraction", "N/A"
                        ),
                        "Entropy Loss": self.logger.name_to_value.get(
                            "train/entropy_loss", "N/A"
                        ),
                        "Explained Variance": self.logger.name_to_value.get(
                            "train/explained_variance", "N/A"
                        ),
                        "Learning Rate": self.logger.name_to_value.get(
                            "train/learning_rate", "N/A"
                        ),
                        "Loss": self.logger.name_to_value.get("train/loss", "N/A"),
                        "Policy Gradient Loss": self.logger.name_to_value.get(
                            "train/policy_gradient_loss", "N/A"
                        ),
                        "Std": self.logger.name_to_value.get("train/std", "N/A"),
                        "Value Loss": self.logger.name_to_value.get(
                            "train/value_loss", "N/A"
                        ),
                    }

                    # Write the data to log.csv
                    self._write_to_csv(data, self.csv_file)

                # Reset for the next episode
                self.episode_distances = []
                self.delta_t_from_last_episode = time.time()

        return True

    def _on_training_end(self) -> None:
        # Plot ratios over episodes
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.episodes,
            self.ratios,
            label="Ratio (Avg Distance / Starting Distance)",
            linestyle="-",
        )

        # Compute linear trendline
        z = np.polyfit(self.episodes, self.ratios, 1)  # 1 for linear fit
        p = np.poly1d(z)  # Create polynomial object from coefficients
        plt.plot(
            self.episodes,
            p(self.episodes),
            linestyle="--",
            color="red",
            label="Trendline",
        )

        plt.xlabel("Episode")
        plt.ylabel("Ratio (Avg Distance / Starting Distance)")
        plt.title("Ratio of Average Distance to Starting Distance Over Episodes")
        plt.ylim(0.0, 1.2 * (max(max(self.ratios), 1.0)))
        plt.legend()
        plt.grid()

        # Save the plot
        plot_file = os.path.join(self.log_dir, "plot.png")
        plt.savefig(plot_file)
        print(f"Plot saved as {plot_file}")
        plt.show()


if __name__ == "__main__":
    exc_options = {"train": "train", "test": "test"}
    reward_options = {
        "foundation_model": "foundation_model",
        "euclidean_distance": "euclidean_distance",
        "test_euclidean_distance": "test_euclidean_distance",
    }

    total_timesteps = 204_80  # Change to your desired total timesteps

    exc = exc_options["train"]
    reward_type = reward_options["euclidean_distance"]
    test_after_train = True

    train_render_fps = None
    train_episode_time_limit = 5

    test_render_fps = 120
    test_episode_time_limit = 5

    kwarg_options = {
        "foundation_model": {
            "render_mode": "rgb_array",
            "render_fps": train_render_fps,
            "episode_time_limit": train_episode_time_limit,
            "model_id": "PATH_QWEN_14B",
            "model_type": "llm",
            "model_quant": "4b",
            "context_prompt_file": "llm_prompt_zero_shot.yaml",
            "verbose": False,
        },
        "euclidean_distance": {
            "render_mode": "rgb_array",
            "render_fps": train_render_fps,
            "episode_time_limit": train_episode_time_limit,
        },
        "test_euclidean_distance": {
            "render_mode": "human",
            "render_fps": test_render_fps,
            "episode_time_limit": test_episode_time_limit,
        },
    }

    if exc == "train":
        kwargs = kwarg_options[reward_type]

        # Define the directory for saving model, plot, and logs
        save_dir = f"src/FM3_MicRo/control_models/ppo_default_parameters_{total_timesteps}_steps_fm_rewards_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

        # Single environment
        vec_env = gym.make("gymnasium_env/SingleParticleNoCargo-v0", **kwargs)
        monitored_env = Monitor(
            vec_env, filename=None
        )  # Use Monitor to log episode data
        vec_env = DummyVecEnv([lambda: monitored_env])

        n_steps = 2048
        if total_timesteps < 2048:
            n_steps = total_timesteps

        # Define the model
        model = PPO(
            "MultiInputPolicy", vec_env, verbose=1, device="cpu", n_steps=n_steps
        )
        # Create an instance of the custom callback
        distance_callback = DistancePlotCallback(log_dir=save_dir)

        # Train the model with the callback
        model.learn(
            total_timesteps=total_timesteps,
            callback=distance_callback,
            progress_bar=True,
        )

        # Save the model
        model_file = os.path.join(save_dir, f"model")
        model.save(model_file)
        print(f"Model saved as {model_file}")

    if test_after_train or exc == "test":
        kwargs = kwarg_options["test_euclidean_distance"]

        vec_env = gym.make("gymnasium_env/SingleParticleNoCargo-v0", **kwargs)
        monitored_env = Monitor(
            vec_env, filename=None
        )  # Use Monitor to log episode data
        vec_env = DummyVecEnv([lambda: monitored_env])

        model = PPO("MultiInputPolicy", vec_env, verbose=1, device="cpu")

        # Load and reuse the model:
        model = PPO.load(
            "src/FM3_MicRo/control_models/ppo_default_parameters_307200_steps_2024-12-12_00-00-00/model.zip",
            env=vec_env,
        )

        obs = vec_env.reset()

        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = vec_env.step(action)
            vec_env.render("human")
