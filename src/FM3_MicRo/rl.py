import argparse
import os
import time
from datetime import datetime

import gymnasium as gym
import gymnasium_env
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from data_plotting_scripts.npz_plotter import NPZPlotter


class DistancePlotCallback(BaseCallback):
    """
    A custom callback to track and plot the ratio of average distance
    to starting distance for each episode during training. Uses Stable-Baselines3
    logger's CSVOutputFormat for logging.
    """

    def __init__(self, log_dir, text_verbosity):
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
        self.text_verbosity = text_verbosity

    def _on_training_start(self) -> None:
        # Create the log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize the start time
        self.start_time = time.time()

        # Initialize the CSV file paths for logging
        self.csv_file = os.path.join(self.log_dir, "per_episode_log.csv")
        self.diagnostics_file = os.path.join(self.log_dir, "per_iteration_log.csv")

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
        plt.close("all")
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
        plot_file = os.path.join(self.log_dir, "r_ratio_plot.png")
        plt.savefig(plot_file)

        if self.text_verbosity:
            print(f"Plot saved as {plot_file}")

        # plt.show()


def parse_arguments():
    # Define options as lists
    exc_options = ["train", "test"]
    reward_options = ["llm", "euclidean", "delta_r"]
    train_render_mode_options = ["rgb_array", "human"]

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Train or test a PPO model in a magnetic manipulation environment."
    )

    # Add optional arguments
    parser.add_argument(
        "--exc",
        type=str,
        choices=exc_options,
        default="train",
        help="Execution mode: 'train' or 'test'. Default: 'train'.",
    )
    parser.add_argument(
        "--goal-reset",
        type=bool,
        default=False,
        help="Whether to reset goals. Default: False.",
    )
    parser.add_argument(
        "--llm-id",
        type=str,
        help="ID of llm in the .env folder. The variable value in the .env folder must point to local path to the checkpoint of the model.",
    )
    parser.add_argument(
        "--num-eval",
        type=int,
        default=10,
        help="Number of evaluations to run to track performance of model over time. Default: 10.",
    )
    parser.add_argument(
        "--num-obs",
        type=int,
        default=5,
        help="Number of observations. Default: 5.",
    )
    parser.add_argument(
        "--particle-reset",
        type=bool,
        default=True,
        help="Whether to reset particles. Default: True.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="llm_prompt_continuous_rewards_zero_shot.yaml",
        help="Prompt file for foundation model training. Default: 'llm_prompt_continuous_rewards_zero_shot.yaml'.",
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        choices=reward_options,
        default="delta_r",
        help="Reward type: 'llm', 'euclidean', or 'delta_r'. Default: 'delta_r'.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=2048,
        help="Rollout steps per update. Default: 2048.",
    )
    parser.add_argument(
        "--show-plots",
        type=bool,
        default=False,
        help="Shows r_ratio and reward plots. Default: False.",
    )
    parser.add_argument(
        "--test-after-train",
        type=bool,
        default=False,
        help="Whether to test the model after training. Default: False.",
    )
    parser.add_argument(
        "--test-episode-time-limit",
        type=float,
        default=5,
        help="Time limit for episodes during testing. Default: 5 seconds.",
    )
    parser.add_argument(
        "--test-model-path",
        type=str,
        default="src/FM3_MicRo/control_models/2024-12-19_18-02-20_delta_r_5000000-steps_5-obs/model",
        help="Path to the model used during testing. Default: [predefined path].",
    )
    parser.add_argument(
        "--test-render-fps",
        type=int,
        default=120,
        help="Frames per second during testing rendering. Default: 120.",
    )
    parser.add_argument(
        "--text-verbosity",
        type=bool,
        default=False,
        help="Verbosity of textual outputs. Default: False.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100000,
        help="Total time steps for PPO training.",
    )
    parser.add_argument(
        "--train-episode-time-limit",
        type=float,
        default=2.5,
        help="Time limit for episodes during training. Default: 2.5 seconds.",
    )
    parser.add_argument(
        "--train-render-fps",
        type=int,
        default=None,
        help="Frames per second during training rendering. Default: None.",
    )
    parser.add_argument(
        "--train-render-mode",
        type=str,
        choices=train_render_mode_options,
        default="rgb_array",
        help="Rendering mode during training: 'rgb_array' or 'human'. Default: 'rgb_array'.",
    )
    parser.add_argument(
        "--train-verbosity",
        type=bool,
        default=True,
        help="Verbosity of training parameters. Default: True.",
    )
    args = parser.parse_args()

    ## Parse known arguments and capture trailing positional arguments
    # args, unknown = parser.parse_known_args()
    #
    ## Extract the positional `total_timesteps`
    # if len(unknown) != 1:
    #    parser.error(
    #        "You must provide a single positional argument for total_timesteps."
    #    )
    # try:
    #    args.total_timesteps = int(unknown[0])
    # except ValueError:
    #    parser.error("total_timesteps must be an integer.")

    # Validate arguments
    if args.total_timesteps < args.rollout_steps:
        raise ValueError(
            "Total timesteps must be greater than or equal to rollout_steps."
        )

    return args


if __name__ == "__main__":
    args = parse_arguments()

    ########################################################################################################################
    # Extract parsed arguments
    exc = args.exc
    goal_reset = args.goal_reset
    llm_id = args.llm_id
    num_eval = args.num_eval
    num_obs = args.num_obs
    particle_reset = args.particle_reset
    prompt_file = args.prompt_file
    reward_type = args.reward_type
    rollout_steps = args.rollout_steps
    show_plots = args.show_plots
    test_after_train = args.test_after_train
    test_episode_time_limit = args.test_episode_time_limit
    test_model_path = args.test_model_path
    test_render_fps = args.test_render_fps * num_obs
    text_verbosity = args.text_verbosity
    total_timesteps = args.total_timesteps
    train_episode_time_limit = args.train_episode_time_limit
    train_render_fps = args.train_render_fps
    train_render_mode = args.train_render_mode
    train_verbosity = args.train_verbosity
    ########################################################################################################################

    # Prepare environment keyword arguments
    kwarg_options = {
        "llm": {
            "render_mode": train_render_mode,
            "render_fps": train_render_fps,
            "episode_time_limit": train_episode_time_limit,
            "n_obs": num_obs,
            "model_id": llm_id,
            "reward_type": "llm",
            "model_quant": "4b",
            "context_prompt_file": prompt_file,
            "verbose": False,
            "particle_reset": particle_reset,
            "goal_reset": goal_reset,
        },
        "euclidean": {
            "render_mode": train_render_mode,
            "render_fps": train_render_fps,
            "episode_time_limit": train_episode_time_limit,
            "n_obs": num_obs,
            "reward_type": "euclidean",
            "particle_reset": particle_reset,
            "goal_reset": goal_reset,
        },
        "delta_r": {
            "render_mode": train_render_mode,
            "render_fps": train_render_fps,
            "episode_time_limit": train_episode_time_limit,
            "n_obs": num_obs,
            "reward_type": "delta_r",
            "particle_reset": particle_reset,
            "goal_reset": goal_reset,
        },
        "test_euclidean_distance": {
            "render_mode": "human",
            "render_fps": test_render_fps,
            "episode_time_limit": test_episode_time_limit,
            "n_obs": num_obs,
            "reward_type": "delta_r",
            "particle_reset": particle_reset,
            "goal_reset": goal_reset,
        },
    }

    if exc == "train":
        kwargs = kwarg_options[reward_type]

        # Define the directory for saving model, plot, and logs
        if reward_type == "llm":
            save_dir = f"src/FM3_MicRo/control_models/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{reward_type}_{prompt_file[11:-5]}_{total_timesteps}-steps_{num_obs}-obs"
        elif reward_type in ["euclidean", "delta_r"]:
            save_dir = f"src/FM3_MicRo/control_models/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{reward_type}_{total_timesteps}-steps_{num_obs}-obs"

        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

        # Single environment
        vec_env = gym.make("gymnasium_env/SingleParticleNoCargo-v0", **kwargs)
        monitored_env = Monitor(vec_env, filename=None)
        vec_env = DummyVecEnv([lambda: monitored_env])

        # Eval environment
        eval_env = gym.make(
            "gymnasium_env/SingleParticleNoCargo-v0", **kwarg_options["delta_r"]
        )
        monitored_eval_env = Monitor(eval_env, filename=None)
        eval_env = DummyVecEnv([lambda: monitored_eval_env])

        # Define the model
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            verbose=train_verbosity,
            device="cpu",
            n_steps=rollout_steps,
        )

        # Create callbacks
        distance_callback = DistancePlotCallback(
            log_dir=save_dir, text_verbosity=text_verbosity
        )
        eval_callback = EvalCallback(
            eval_env,
            log_path=save_dir,
            deterministic=True,
            eval_freq=total_timesteps // num_eval,
        )

        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=[distance_callback, eval_callback],
            progress_bar=True,
        )

        # Save evaluation results and plot
        npz_plotter = NPZPlotter(
            npz_file=os.path.join(save_dir, "evaluations.npz"),
            text_verbosity=text_verbosity,
        )

        if text_verbosity:
            npz_plotter.print()

        npz_plotter.save()

        # Save the model
        test_model_path = os.path.join(save_dir, "model")
        model.save(test_model_path)
        if text_verbosity:
            print(f"Model saved as {test_model_path}")

    if test_after_train or exc == "test":
        kwargs = kwarg_options["test_euclidean_distance"]

        vec_env = gym.make("gymnasium_env/SingleParticleNoCargo-v0", **kwargs)
        monitored_env = Monitor(vec_env, filename=None)
        vec_env = DummyVecEnv([lambda: monitored_env])

        # Load the model
        model = PPO.load(test_model_path, env=vec_env)

        obs = vec_env.reset()

        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = vec_env.step(action)
            vec_env.render("human")
