import sys
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

from gymnasium_env.envs.System.Library import functions
from gymnasium_env.envs.System.simulator import Simulator
from gymnasium_env.envs.System import initializations
from gymnasium_env.envs.Library.llm import LLM
from gymnasium_env.envs.Library.vlm import VLM


class SingleParticleNoCargo(gym.Env):
    """Custom Gymnasium Environment wrapping a Pygame-based Simulator."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, episode_time_limit=5, reward_type="default"):
        # Initialize the simulator
        self.simulator = Simulator(self.metadata["render_fps"])

        # Define observation space: 2D particle location (-r to r) and 2D goal location (-r to r)
        r = initializations.SIM_SOL_CIRCLE_RAD
        self.observation_space = spaces.Dict(
            {
                "particle_loc": spaces.Box(
                    low=-r, high=r, shape=(2,), dtype=np.float32
                ),
                "goal_loc": spaces.Box(low=-r, high=r, shape=(2,), dtype=np.float32),
            }
        )

        # Define action space: 8 coil currents, each between 0 and 1
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)

        self.running = True

        assert render_mode == None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.episode_time_limit = episode_time_limit

        self.reward_type = reward_type

        print(reward_type) if reward_type else print("no")

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.simulator.resetAtRandomLocs(seed)
        obs = self._get_obs()
        info = self._get_info()

        self.frame_time = time.time()
        self.episode_time = time.time()

        return obs, info

    def step(self, action):
        """Apply action and return the new observation, reward, done, and info."""
        render = str(self.render_mode) == "human"

        # Run one frame of the simulator
        self.running, self.frame_time = self.simulator.step(
            self.running, self.frame_time, action, render
        )

        # Extract necessary information from the state
        obs = self._get_obs()
        info = self._get_info()

        # Calculate the reward as the negative distance to the goal
        reward = -info["reward"]

        # Determine if the episode is done (if the particle is close to the goal)
        done = (
            info["reward"] < initializations.SIM_MULTIPLE_GOALS_ACCURACY
            or not self.running
        )  # Done if close to goal or simulator is closed

        # Determine if the time limit has been reached
        truncated = (time.time() - self.episode_time) > (self.episode_time_limit)

        if self.running == False:
            self._close()

        return obs, reward, done, truncated, info

    def render(self):
        """Render the environment."""
        # Returns an rgb array of the image for stable-baselines to be able render it on OpenCV when showing results of training
        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.simulator.canvas)),
                axes=(1, 0, 2),
            )

    def _close(self):
        """Close the environment and simulator."""
        self.simulator.close()
        sys.exit()

    def _get_obs(self):
        particle_loc, goal_loc, _, _ = self.simulator.getState()

        return {
            "particle_loc": np.array(particle_loc, dtype=np.float32),
            "goal_loc": np.array(goal_loc, dtype=np.float32),
        }

    def _reward_gen_init_(self):
        if self.reward_type == "llm":
            self.reward_gen = LLM()

        elif self.reward_type == "vlm":
            self.reward_gen = VLM()

    def _get_info(self):
        particle_loc, goal_loc, _, _ = self.simulator.getState()

        if self.reward_type == "default":
            return {
                "reward": functions.distance(
                    particle_loc[0], particle_loc[1], goal_loc[0], goal_loc[1]
                )
            }

        elif self.reward_type == "llm":
            return {"reward": self.reward_gen.get_reward("TO BE DONE")}

        elif self.reward_type == "vlm":
            return {"reward": self.reward_gen.get_reward("TO BE DONE")}
