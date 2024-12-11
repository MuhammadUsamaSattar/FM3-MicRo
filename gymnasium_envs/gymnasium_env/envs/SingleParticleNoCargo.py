import sys
import time
import yaml

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from gymnasium_env.envs.Library.llm import LLM
from gymnasium_env.envs.Library.vlm import VLM
from gymnasium_env.envs.System import initializations
from gymnasium_env.envs.System.Library import functions
from gymnasium_env.envs.System.simulator import Simulator


class SingleParticleNoCargo(gym.Env):
    """Custom Gymnasium Environment wrapping a Pygame-based Simulator."""

    metadata = {
        "render_modes": ["None", "human", "rgb_array"],
        "render_fps": 60,
        "model_types": [None, "llm", "vlm"],
    }

    def __init__(
        self,
        render_mode: str = "None",
        episode_time_limit: int = 5,
        model_id: str | None = None,
        model_type: str | None = None,
        model_quant: str | None = None,
        context_prompt_file: str | None = None,
    ):
        """Initializes the environment.

        Args:
            render_mode (str, optional): The mode in which to render the game. "human" and "rgb_array" are available. Defaults to None.
            episode_time_limit (int, optional): The number of episodes to train for. Defaults to 5.
            reward_type (str, optional): Type of reward function. "default", "llm" and "vlm" are available. "default" uses the euclidian distance between the particle and goal. Defaults to None.
            model_id (str, optional): ID of the model on hugging face repository or local path to a download model.model_id. Defaults to None.
            model_type (str, optional): Type of the model. Options are "llm" and "vlm". Defaults to None.
            model_quant (str, optional): The quantization level of the model. "fp16", "8b" and "4b" are implemented. Defaults to "fp16". Defaults to None.
            context_prompt_file (str, optional): Name of the prompt file in the "prompts" folder. Defaults to None.
        """
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

        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.episode_time_limit = episode_time_limit

        if (
            model_id != None
            or model_type != None
            or model_quant != None
            or context_prompt_file != None
        ):
            assert model_id != None
            assert model_type != None
            assert model_quant != None
            assert context_prompt_file != None

            self.model_id = model_id

            assert model_type in self.metadata["model_types"]
            self.model_type = model_type

            self.model_quant = model_quant

            self._foundation_model_init_(context_prompt_file)

    def reset(self, seed: int | None = None, options=None) -> tuple[dict, dict]:
        """Reset the environment to an initial state.

        Args:
            seed (int | None, optional): Seed value to pass to self.np_random. Defaults to None.
            options: Options for the step function. Required by gymnasium. Defaults to None.

        Returns:
            tuple[dict, dict]: A tuple of observations dict and info dict.
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.simulator.resetAtRandomLocs(seed)
        self.obs = self._get_obs()
        self.prev_obs = self.obs.copy()
        info = self._get_info()

        self.frame_time = time.time()
        self.episode_time = time.time()

        return self.obs, info

    def step(self, action: list[float]) -> tuple[dict, float, bool, bool, dict]:
        """Apply action, process the new state and return state data.

        Args:
            action (list[float]): A list of length 8 with normalized current values for solenoids starting from Northern-most and then in clockwise direction.

        Returns:
            tuple[dict, float, bool, bool, dict]: Tuple containing the new observation, reward, done (epsiode completed), truncated (episode terminated) and info
        """
        render = str(self.render_mode) == "human"

        # Run one frame of the simulator
        self.running, self.frame_time = self.simulator.step(
            self.running, self.frame_time, action, render
        )

        # Extract necessary information from the state
        self.prev_obs = self.obs.copy()
        self.obs = self._get_obs()
        info = self._get_info()

        # Calculate the reward as the negative distance to the goal
        reward = info["reward"]

        # Determine if the episode is done (if the particle is close to the goal)
        done = (
            functions.distance(
                self.obs["particle_loc"][0],
                self.obs["particle_loc"][1],
                self.obs["goal_loc"][0],
                self.obs["goal_loc"][1],
            )
            < initializations.SIM_MULTIPLE_GOALS_ACCURACY
            or not self.running
        )  # Done if close to goal or simulator is closed

        # Determine if the time limit has been reached
        truncated = (time.time() - self.episode_time) > (self.episode_time_limit)

        if self.running == False:
            self._close()

        return self.obs, reward, done, truncated, info

    def render(self):
        """Render the environment. Only handles "rgb_array" option. "human" option is handled through pygame."""
        # Returns an rgb array of the image for stable-baselines to be able render it on OpenCV when showing results of training
        if self.render_mode == "rgb_array":
            return self.get_rgb_array()

    def get_rgb_array(self):
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.simulator.canvas)),
            axes=(1, 0, 2),
        )

    def _close(self):
        """Close the environment and simulator. Only called when pygame window is closed by the user"""
        self.simulator.close()
        sys.exit()

    def _get_obs(self) -> dict:
        """Returns the observations of the current game state

        Returns:
            dict: Observation dictionary containing the particle location and goal location.
        """
        particle_loc, goal_loc, _, _ = self.simulator.getState()

        return {
            "particle_loc": np.array(particle_loc, dtype=np.float32),
            "goal_loc": np.array(goal_loc, dtype=np.float32),
        }

    def _foundation_model_init_(self, context_prompt_file):
        """Initialzies the foundation model for reward generation"""

        dir = "src/FM3_MicRo/prompts/rl_fm_rewards/"
        context_prompt_file = dir + context_prompt_file

        with open(context_prompt_file) as stream:
            try:
                self.context = yaml.safe_load(stream)["prompt"]
                print("Contextual prompt is:\n" + self.context)
            except yaml.YAMLError as exc:
                print(exc)

        if self.model_type == "llm":
            self.model = LLM(
                model_id=self.model_id,
                model_quant=self.model_quant,
                device="cuda",
                verbose=True,
            )
        elif self.model_type == "vlm":
            self.model = VLM(
                model_id=self.model_id,
                model_quant=self.model_quant,
                device="cuda",
                verbose=True,
            )

    def _get_info(self) -> dict:
        """Returns the info of the current state.

        Returns:
            dict: Dictionary containing the reward
        """
        particle_loc, goal_loc, _, _ = self.simulator.getState()

        if self.model_type == "default":
            return {
                "reward": -functions.distance(
                    particle_loc[0], particle_loc[1], goal_loc[0], goal_loc[1]
                )
            }

        elif self.model_type != None:
            try:
                # Get the previous particle location
                prev_particle_loc = self.prev_obs["particle_loc"].copy()

                # Construct the prompt
                txt = (
                    f"The particle was located at ({prev_particle_loc[0]:.2f}, {prev_particle_loc[1]:.2f}).\n"
                    f"The particle is currently located at ({particle_loc[0]:.2f}, {particle_loc[1]:.2f}).\n"
                    f"The goal is currently located at ({goal_loc[0]}, {goal_loc[1]}).\n\n"
                    f"What is the reward score?"
                )
                print(txt)

                # Use the model to calculate coil values
                if self.model_type == "llm":
                    output = self.model.get_response(
                        context=self.context,
                        txt=txt,
                        tokens=1000,
                    )

                elif self.model_type == "vlm":
                    output = self.model.get_response(
                        img=self.get_rgb_array(),
                        img_parameter_type="image",
                        txt=txt,
                        tokens=1000,
                    )

                output = int(output)

                return {"reward": output}

            except Exception as e:
                print(f"Error calculating coil values: {e}")
