import sys
import time
import importlib.util
import yaml
from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from PyQt5.QtWidgets import QApplication

from gymnasium_env.envs.Library.llm import LLM
from gymnasium_env.envs.Library.vlm import VLM
from gymnasium_env.envs.System import initializations
from gymnasium_env.envs.System.Library import functions
from gymnasium_env.envs.System.simulator import Simulator

package_spec = importlib.util.find_spec("PySpin")
if package_spec:
    from gymnasium_env.envs.System.gui import mywindow


class SingleParticleNoCargo(gym.Env):
    """Custom Gymnasium Environment wrapping a Pygame-based Simulator."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 120,
        "reward_types": ["euclidean", "delta_r", "sparse", "llm", "vlm"],
        "train_fps": None,
    }

    def __init__(
        self,
        render_mode: str = "rgb_array",
        render_fps: int = 120,
        train_fps: int | None = None,
        episode_time_limit: int = 5,
        n_obs: int = 10,
        model_id: str | None = None,
        reward_type: str | None = None,
        model_quant: str | None = None,
        context_prompt_file: str | None = None,
        verbose: bool = False,
        particle_reset: bool = True,
        goal_reset: bool = True,
        goal_time: float = 1.0,
        env: str = "simulator",
    ):
        """Initializes the environment.

        Args:
            render_mode (str, optional): The mode in which to render the game. "human" and "rgb_array" are available. Defaults to "rgb_array".
            render_fps (int, optional): Rendering framerate. This argument has no effect for env == 'gui'. Defaults to 60.
            train_fps (int | None, optional): Framerate of the training loop. Sets timesteps per second. Defaults to None.
            episode_time_limit (int, optional): The number of episodes to train for. Defaults to 5.
            n_obs (int, optional): Number of particle locations to get for input. Defaults to 10.
            model_id (str | None, optional): ID of the model on hugging face repository or local path to a download model.model_id. Defaults to None.
            reward_type (str | None, optional): Type of the model. Options are "llm", "vlm", "delta_r", "euclidean" and None. Defaults to None.
            model_quant (str | None, optional): The quantization level of the model. "fp16", "8b" and "4b" are implemented. Defaults to "fp16". Defaults to None.
            context_prompt_file (str | None, optional): Name of the prompt file in the "prompts" folder. Defaults to None.
            verbose (bool, optional): Sets verbosity mode. Defaults to False.
            particle_reset (bool, optional): Resets particle location on start of episode. Defaults to True.
            goal_reset (bool, optional): Resets goal location on start of episode. Defaults to True.
            goal_time (float, optional): Time after which the goal is said to be achieved if particle remains close enough to it. Defualts to 1.0.
            env (str, optional): Type of env. Options are 'simulator' and 'gui'. Defualts to 'simulator'.
        """

        self.metadata["render_fps"] = render_fps
        self.metadata["train_fps"] = train_fps

        # Initialize the env
        if env == "simulator":
            self.env = Simulator(render_fps)
        elif env == "gui":
            app = QApplication(sys.argv)  # Initialization of QT app
            self.env = mywindow()
            self.env.show()

        # Define observation space: 2D particle location (-r to r) and 2D goal location (-r to r)
        r = initializations.SIM_SOL_CIRCLE_RAD
        self.observation_space = spaces.Dict(
            {
                "particle_locs": spaces.Box(
                    low=-r,
                    high=r,
                    shape=(
                        n_obs,
                        2,
                    ),
                    dtype=np.float32,
                ),
                "goal_loc": spaces.Box(low=-r, high=r, shape=(2,), dtype=np.float32),
            }
        )

        # Define action space: 8 coil currents, each between 0 and 1
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)

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

        assert reward_type in self.metadata["reward_types"]
        self.reward_type = reward_type

        if (
            model_id != None
            or reward_type == "llm"
            or reward_type == "vlm"
            or model_quant != None
            or context_prompt_file != None
        ):
            assert model_id != None
            assert reward_type == "llm" or reward_type == "vlm"
            assert model_quant != None
            assert context_prompt_file != None

            self.model_id = model_id

            self.model_quant = model_quant

            self._foundation_model_init_(context_prompt_file)

        self.time = time.time()

        self.particle_reset = particle_reset
        self.goal_reset = goal_reset
        self.n_obs = n_obs

        self.inside_goal_bool = False
        self.inside_goal_timer = float('inf')
        self.done = False
        self.goal_time = goal_time

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

        self.env.resetAtRandomLocs(seed, self.particle_reset, self.goal_reset)
        self.obs = self._get_obs()
        info = self._get_info()

        self.episode_time = time.time()
        self.frame_time = time.time()

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
        self.env.step(action, render)
        self.time_logs["simulator_step()_total"] = (
            time.time() - self.time_logs["simulator_step()_total"]
        )

        # Extract necessary information from the state
        self.time_logs["_get_obs()_total"] = time.time()
        self.obs = self._get_obs()
        self.time_logs["_get_obs()_total"] = (
            time.time() - self.time_logs["_get_obs()_total"]
        )

        # Determine if the episode is done (if the particle is close to the goal)
        self.done = self._get_done()

        self.time_logs["_get_info()_total"] = time.time()
        info = self._get_info()
        self.time_logs["_get_info()_total"] = (
            time.time() - self.time_logs["_get_info()_total"]
        )

        # Calculate the reward
        reward = info["reward"]

        # Determine if the time limit has been reached
        truncated = (time.time() - self.episode_time) > (self.episode_time_limit)

        self.time_logs["self.step()_total"] = (
            time.time() - self.time_logs["self.step()_total"]
        )

        if self.verbose == True:
            print(
                f"{'Time elapsed [simulator_step()_total]':45} : {self.time_logs['simulator_step()_total']:10.10f}"
            )
            print(
                f"{'Time elapsed [_get_obs()_total]':45} : {self.time_logs['_get_obs()_total']:10.10f}"
            )
            print(
                f"{'Time elapsed [_get_info()_total]':45} : {self.time_logs['_get_info()_total']:10.10f}"
            )
            print(
                f"{'Time elapsed [self.step()_total]':45} : {self.time_logs['self.step()_total']:10.10f}\n"
            )

        if self.metadata["train_fps"] != None:
            functions.sleep(max(0, (1/self.metadata["train_fps"]) - (time.time() - self.frame_time)))
        self.frame_time = time.time()

        return self.obs, reward, self.done, truncated, info

    def render(self):
        """Render the environment. Only handles "rgb_array" option. "human" option is handled through pygame."""

        # Returns an rgb array of the image for stable-baselines to be able render it on OpenCV when showing results of training
        if self.render_mode == "rgb_array":
            return self.get_rgb_array()

    def set_reward_params(
        self,
        render_fps: int = 60,
        train_fps: int | None = None,
        model_id: str | None = None,
        reward_type: str | None = None,
        model_quant: str | None = None,
        context_prompt_file: str | None = None,
    ):
        """Sets reward generator's parameters.

        Args:
            render_fps (int, optional): Rendering framerate. Defaults to 60.
            train_fps (int | None, optional): Framerate of the training loop. Sets timesteps per second. Defaults to None.
            model_id (str | None, optional): ID of the model on hugging face repository or local path to a download model.model_id. Defaults to None.
            model_type (str | None, optional): Type of the model. Options are "llm", "vlm" and None. Defaults to None.
            model_quant (str | None, optional): The quantization level of the model. "fp16", "8b" and "4b" are implemented. Defaults to "fp16". Defaults to None.
            context_prompt_file (str | None, optional): Name of the prompt file in the "prompts" folder. Defaults to None.
        """
        self.metadata["render_fps"] = render_fps
        self.metadata["train_fps"] = train_fps

        self.model_id = model_id
        self.reward_type = reward_type
        self.model_quant = model_quant

        if context_prompt_file != None:
            self._foundation_model_init_(context_prompt_file)

    def get_rgb_array(self):
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.env.canvas)),
            axes=(1, 0, 2),
        )

    def _close(self):
        """Close the environment and simulator. Only called when pygame window is closed by the user"""

        self.env.close()
        sys.exit()

    def _get_obs(self) -> dict:
        """Returns the observations of the current game state

        Returns:
            dict: Observation dictionary containing the particle location and goal location.
        """

        particle_locs, goal_loc, _, _ = self.env.getState(
            self.render_mode=="human",
            self.n_obs,
        )

        return {
            "particle_locs": np.array(particle_locs, dtype=np.float32),
            "goal_loc": np.array(goal_loc, dtype=np.float32),
        }
    
    def _get_done(self) -> bool:
        """Returns the done status of step.

        Returns:
            bool: Done boolen value.
        """
        done = False
        if (
            functions.distance(
                self.obs["particle_locs"][-1][0],
                self.obs["particle_locs"][-1][1],
                self.obs["goal_loc"][0],
                self.obs["goal_loc"][1],
            )
            < initializations.SIM_MULTIPLE_GOALS_ACCURACY
        ):
            if not self.inside_goal_bool:
                self.inside_goal_bool = True
                self.inside_goal_timer = time.time()
                self.prev_inside_goal_reward_time = time.time()

            else:
                if (time.time() - self.inside_goal_timer) > self.goal_time:
                    done = True

        else:
            if self.inside_goal_bool:
                self.inside_goal_bool = False
                self.inside_goal_timer = float('inf')
                self.prev_inside_goal_reward_time = float('inf')

        return done

    def _foundation_model_init_(
        self,
        context_prompt_file,
    ):
        """Initialzies the foundation model for reward generation

        Args:
            context_prompt_file (str, optional): Name of the prompt file in the "prompts" folder.
        """


        dir = Path(__file__).parent.parent.parent.parent / "src/FM3_MicRo/prompts/rl_fm_rewards/"
        context_prompt_file = dir / context_prompt_file

        with open(context_prompt_file) as stream:
            try:
                content = yaml.safe_load(stream)
                self.context = content["prompt"]
                self.range = content["range"]
                print("Contextual prompt is:\n" + self.context)
            except yaml.YAMLError as exc:
                print(exc)

        if self.reward_type == "llm":
            self.model = LLM(
                model_id=self.model_id,
                model_quant=self.model_quant,
                device="cuda",
                verbose=False,
            )
        elif self.reward_type == "vlm":
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

        # Movement based reward = [0, 1]
        info = {
            "reward": 0,
            "distance": functions.distance(
                self.obs["particle_locs"][-1][0],
                self.obs["particle_locs"][-1][1],
                self.obs["goal_loc"][0],
                self.obs["goal_loc"][1],
            ),
        }

        if self.reward_type == "euclidean":
            info = {
                "reward": -functions.distance(
                    self.obs["particle_locs"][-1][0],
                    self.obs["particle_locs"][-1][1],
                    self.obs["goal_loc"][0],
                    self.obs["goal_loc"][1],
                ) / initializations.SIM_SOL_CIRCLE_RAD,
                "distance": functions.distance(
                    self.obs["particle_locs"][-1][0],
                    self.obs["particle_locs"][-1][1],
                    self.obs["goal_loc"][0],
                    self.obs["goal_loc"][1],
                ),
            }

        elif self.reward_type == "delta_r":
            info = {
                "reward": (
                    functions.distance(
                        self.obs["particle_locs"][0][0],
                        self.obs["particle_locs"][0][1],
                        self.obs["goal_loc"][0],
                        self.obs["goal_loc"][1],
                    )
                    - functions.distance(
                        self.obs["particle_locs"][-1][0],
                        self.obs["particle_locs"][-1][1],
                        self.obs["goal_loc"][0],
                        self.obs["goal_loc"][1],
                    )
                ) / initializations.SIM_SOL_CIRCLE_RAD,
                "distance": functions.distance(
                    self.obs["particle_locs"][-1][0],
                    self.obs["particle_locs"][-1][1],
                    self.obs["goal_loc"][0],
                    self.obs["goal_loc"][1],
                ),
            }

        elif self.reward_type == "llm" or self.reward_type == "vlm":
            try:
                # Construct the prompt
                txt = (
                    f"\n"
                    f"The particle was located at ({self.obs['particle_locs'][0][0]:.2f}, {self.obs['particle_locs'][0][1]:.2f}).\n"
                    f"The particle is currently located at ({self.obs['particle_locs'][-1][0]:.2f}, {self.obs['particle_locs'][-1][1]:.2f}).\n"
                    f"The goal is currently located at ({self.obs['goal_loc'][0]:.2f}, {self.obs['goal_loc'][1]:.2f}).\n\n"
                    f"What is the reward score?"
                )

                # Use the model to calculate coil values
                if self.reward_type == "llm":
                    output = self.model.get_response(
                        batches=1,
                        contexts=[self.context],
                        txts=[txt],
                        tokens=1000,
                    )

                elif self.reward_type == "vlm":
                    output = self.model.get_response(
                        img=self.get_rgb_array(),
                        img_parameter_type="image",
                        txt=txt,
                        tokens=1000,
                    )

                try:
                    output = float(output[0])
                except ValueError:
                    output = float(output[0][: output[0].find("\n")])

                delta_r = functions.distance(
                    self.obs["particle_locs"][0][0],
                    self.obs["particle_locs"][0][1],
                    self.obs["goal_loc"][0],
                    self.obs["goal_loc"][1],
                ) - functions.distance(
                    self.obs["particle_locs"][-1][0],
                    self.obs["particle_locs"][-1][1],
                    self.obs["goal_loc"][0],
                    self.obs["goal_loc"][1],
                )

                if (delta_r > 0 and output > 0) or (delta_r <= 0 and output <= 0):
                    self.record["correct"] += 1
                else:
                    self.record["incorrect"] += 1

                if self.verbose:
                    print(f"{txt}\n")
                    print(f"{output}\n")
                    print(f"detla r: {delta_r}")
                    print(f"{self.record}\n")
                    print(
                        f"Accuracy: {(self.record['correct'] / (self.record['correct']+self.record['incorrect'])):.2f}\n"
                    )

                info = {
                    "reward": output/self.range[1],
                    "distance": functions.distance(
                        self.obs["particle_locs"][-1][0],
                        self.obs["particle_locs"][-1][1],
                        self.obs["goal_loc"][0],
                        self.obs["goal_loc"][1],
                    ),
                }

            except Exception as e:
                print(f"Error calculating coil values: {e}")       

        # Goal vicinity based reward = [00, 10]
        if (
            functions.distance(
                self.obs["particle_locs"][-1][0],
                self.obs["particle_locs"][-1][1],
                self.obs["goal_loc"][0],
                self.obs["goal_loc"][1],
            )
            < initializations.SIM_MULTIPLE_GOALS_ACCURACY
        ):
            if self.inside_goal_bool:
                new_time = time.time()
                info["reward"] += (((new_time - self.prev_inside_goal_reward_time)/(self.goal_time))*10)
                self.prev_inside_goal_reward_time = new_time

        # Time based reward penalty = [-100, 000]
        new_time = time.time()
        info["reward"] += (((self.time-new_time)/(self.episode_time_limit))*100)
        self.time = new_time

        # Goal based reward = [000, 1000]
        if self.done:
            info["reward"] += 1000

        if self.verbose:
            print(f"Total reward: {info['reward']}\n")

        return info
