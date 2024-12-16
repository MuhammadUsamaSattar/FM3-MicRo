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
        render_fps: int = 60,
        episode_time_limit: int = 5,
        model_id: str | None = None,
        model_type: str | None = None,
        model_quant: str | None = None,
        context_prompt_file: str | None = None,
        verbose: bool = False,
    ):
        """Initializes the environment.

        Args:
            render_mode (str, optional): The mode in which to render the game. "human" and "rgb_array" are available. Defaults to "None".
            render_fps (int, optional): Rendering framerate. Defaults to 60.
            episode_time_limit (int, optional): The number of episodes to train for. Defaults to 5.
            model_id (str | None, optional): ID of the model on hugging face repository or local path to a download model.model_id. Defaults to None.
            model_type (str | None, optional): Type of the model. Options are "llm", "vlm" and None. Defaults to None.
            model_quant (str | None, optional): The quantization level of the model. "fp16", "8b" and "4b" are implemented. Defaults to "fp16". Defaults to None.
            context_prompt_file (str | None, optional): Name of the prompt file in the "prompts" folder. Defaults to None.
            verbose (bool, optional): Sets verbosity mode. Defaults to False.
        """

        self.metadata["render_fps"] = render_fps

        # Initialize the simulator
        self.simulator = Simulator(render_fps)

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

        self.verbose = verbose
        self.time_logs = {
            "simulator_step()_total": None,
            "_get_obs()_total": None,
            "_get_info()_total": None,
            "self.step()_total": None,
        }

        self.record = {"correct": 0, "incorrect": 0}

        assert model_type in self.metadata["model_types"]
        self.model_type = model_type

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

            self.model_quant = model_quant

            self._foundation_model_init_(context_prompt_file)

    def reset(
        self,
        seed: int | None = None,
        options=None,
    ) -> tuple[dict, dict]:
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

    def step(
        self,
        action: list[float],
    ) -> tuple[dict, float, bool, bool, dict]:
        """Apply action, process the new state and return state data.

        Args:
            action (list[float]): A list of length 8 with normalized current values for solenoids starting from Northern-most and then in clockwise direction.

        Returns:
            tuple[dict, float, bool, bool, dict]: Tuple containing the new observation, reward, done (epsiode completed), truncated (episode terminated) and info
        """

        self.time_logs["self.step()_total"] = time.time()

        render = str(self.render_mode) == "human"

        # Run one frame of the simulator
        self.time_logs["simulator_step()_total"] = time.time()
        self.running, self.frame_time = self.simulator.step(
            self.running, self.frame_time, action, render
        )
        self.time_logs["simulator_step()_total"] = (
            time.time() - self.time_logs["simulator_step()_total"]
        )

        # Extract necessary information from the state
        self.prev_obs = self.obs.copy()

        self.time_logs["_get_obs()_total"] = time.time()
        self.obs = self._get_obs()
        self.time_logs["_get_obs()_total"] = (
            time.time() - self.time_logs["_get_obs()_total"]
        )

        self.time_logs["_get_info()_total"] = time.time()
        info = self._get_info()
        self.time_logs["_get_info()_total"] = (
            time.time() - self.time_logs["_get_info()_total"]
        )

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

        self.time_logs["self.step()_total"] = (
            time.time() - self.time_logs["self.step()_total"]
        )

        if self.verbose == True:
            print(
                f"{'Time elapsed [simulator_step()_total]':45} : {self.time_logs['simulator_step()_total']:10.5f}"
            )
            print(
                f"{'Time elapsed [_get_obs()_total]':45} : {self.time_logs['_get_obs()_total']:10.5f}"
            )
            print(
                f"{'Time elapsed [_get_info()_total]':45} : {self.time_logs['_get_info()_total']:10.5f}"
            )
            print(
                f"{'Time elapsed [self.step()_total]':45} : {self.time_logs['self.step()_total']:10.5f}\n"
            )

        return self.obs, reward, done, truncated, info

    def render(self):
        """Render the environment. Only handles "rgb_array" option. "human" option is handled through pygame."""

        # Returns an rgb array of the image for stable-baselines to be able render it on OpenCV when showing results of training
        if self.render_mode == "rgb_array":
            return self.get_rgb_array()

    def set_reward_params(
        self,
        render_fps: int = 60,
        model_id: str | None = None,
        model_type: str | None = None,
        model_quant: str | None = None,
        context_prompt_file: str | None = None,
    ):
        """Sets reward generator's parameters.

        Args:
            render_fps (int, optional): Rendering framerate. Defaults to 60.
            model_id (str | None, optional): ID of the model on hugging face repository or local path to a download model.model_id. Defaults to None.
            model_type (str | None, optional): Type of the model. Options are "llm", "vlm" and None. Defaults to None.
            model_quant (str | None, optional): The quantization level of the model. "fp16", "8b" and "4b" are implemented. Defaults to "fp16". Defaults to None.
            context_prompt_file (str | None, optional): Name of the prompt file in the "prompts" folder. Defaults to None.
        """
        self.metadata['render_fps'] = render_fps

        self.model_id = model_id
        self.model_type = model_type
        self.model_quant = model_quant

        if context_prompt_file != None:
            self._foundation_model_init_(context_prompt_file)

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

    def _foundation_model_init_(
        self,
        context_prompt_file,
    ):
        """Initialzies the foundation model for reward generation

        Args:
            context_prompt_file (str, optional): Name of the prompt file in the "prompts" folder.
        """

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
                verbose=False,
            )
        elif self.model_type == "vlm":
            self.model = VLM(
                model_id=self.model_id,
                model_quant=self.model_quant,
                device="cuda",
                verbose=False,
            )

    def _get_info(
        self,
    ) -> dict:
        """Returns the info of the current state.

        Returns:
            dict: Dictionary containing the reward
        """

        particle_loc, goal_loc, _, _ = self.simulator.getState()

        if self.model_type == None:
            info = {
                "reward": -functions.distance(
                    particle_loc[0], particle_loc[1], goal_loc[0], goal_loc[1]
                ),
                "distance": functions.distance(
                    particle_loc[0], particle_loc[1], goal_loc[0], goal_loc[1]
                ),
            }

        elif self.model_type != None:
            try:
                # Get the previous particle location
                prev_particle_loc = self.prev_obs["particle_loc"].copy()

                # Construct the prompt
                txt = (
                    f"\n"
                    f"The particle was located at ({prev_particle_loc[0]:.2f}, {prev_particle_loc[1]:.2f}).\n"
                    f"The particle is currently located at ({particle_loc[0]:.2f}, {particle_loc[1]:.2f}).\n"
                    f"The goal is currently located at ({goal_loc[0]:.2f}, {goal_loc[1]:.2f}).\n\n"
                    f"What is the reward score?"
                )

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

                if (
                    (
                        functions.distance(
                            prev_particle_loc[0],
                            prev_particle_loc[1],
                            goal_loc[0],
                            goal_loc[1],
                        )
                        - functions.distance(
                            particle_loc[0], particle_loc[1], goal_loc[0], goal_loc[1]
                        )
                    )
                    > 0
                    and output == 1
                ) or (
                    (
                        functions.distance(
                            prev_particle_loc[0],
                            prev_particle_loc[1],
                            goal_loc[0],
                            goal_loc[1],
                        )
                        - functions.distance(
                            particle_loc[0], particle_loc[1], goal_loc[0], goal_loc[1]
                        )
                    )
                    <= 0
                    and output == 0
                ):
                    self.record["correct"] += 1
                else:
                    self.record["incorrect"] += 1

                if self.verbose:
                    print(f"{txt}\n")
                    print(f"{output}\n")
                    print(f"{self.record}\n")
                    print(
                        f"Accuracy: {(self.record['correct'] / (self.record['correct']+self.record['incorrect'])):.2f}\n"
                    )

                info = {
                    "reward": output,
                    "distance": functions.distance(
                        particle_loc[0], particle_loc[1], goal_loc[0], goal_loc[1]
                    ),
                }

            except Exception as e:
                print(f"Error calculating coil values: {e}")

        return info
