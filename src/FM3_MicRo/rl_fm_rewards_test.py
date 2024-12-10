import json
import os
import queue
import threading
import yaml

import gymnasium as gym
import gymnasium_env
from PIL import Image

from gymnasium_env.envs.Library.llm import LLM
from gymnasium_env.envs.Library.vlm import VLM


class ZeroShotRaw:

    def __init__(
        self,
        env_name: str,
        model_id: str,
        model_type: str,
        context_prompt_file: str,
        model_quant: str = "fp16",
    ):
        """Initializes the foundation model and gym environment.

        Args:
            env_name (str): Name of the gymnasium environment that can be passed to gym.make().
            model_id (str): ID of the model on hugging face repository or local path to a download model.model_id.
            model_type (str): Type of the model. Options are "llm" and "vlm".
            context_prompt_file (str): Name of the prompt file in the "prompts" folder.
            model_quant (str, optional): The quantization level of the model. "fp16", "8b" and "4b" are implemented. Defaults to "fp16".
        """

        self.model_type = model_type
        self.env = gym.make(env_name, render_mode="human")

        dir = os.getcwd() + "/prompts/zero_shot/"
        context_prompt_file = dir + context_prompt_file
        with open(context_prompt_file) as stream:
            try:
                self.context = yaml.safe_load(stream)["prompt"]
                print("Contextual prompt is:\n" + self.context)
            except yaml.YAMLError as exc:
                print(exc)

        if self.model_type == "llm":
            self.model = LLM(
                model_id=model_id, model_quant=model_quant, device="cuda", verbose=True
            )
        elif self.model_type == "vlm":
            self.model = VLM(
                model_id=model_id, model_quant=model_quant, device="cuda", verbose=True
            )

        # Create a thread-safe queue to share coil values between threads
        self.coil_vals_queue = queue.Queue()
        self.obs_lock = threading.Lock()

    def run(self):
        """Continously run the pygame frames and get data from the environment."""
        self.obs, info = self.env.reset()
        self.prev_obs = self.obs
        coil_vals = [0, 0, 0, 0, 0, 0, 0, 0]

        # Start a separate thread for calculating coil values
        calc_thread = threading.Thread(target=self.calculate_coil_vals, daemon=True)
        calc_thread.start()

        # Main loop to update the environment
        while True:
            # Try to get the latest coil_vals from the queue (non-blocking)
            try:
                coil_vals = self.coil_vals_queue.get_nowait()
            except queue.Empty:
                pass

            # If new coil values are available, pass them to the step function
            obs, reward, done, truncated, info = self.env.step(coil_vals)
            with self.obs_lock:
                self.prev_obs = self.obs
                self.obs = obs

            if done:
                obs, info = self.env.reset()

                with self.obs_lock:
                    self.prev_obs = self.obs
                    self.obs = obs

    def calculate_coil_vals(self):
        """Continuously calculate coil values in a separate thread."""
        while True:
            try:
                # Get the current particle location and goal location
                with self.obs_lock:
                    prev_particle_loc = self.prev_obs["particle_loc"]
                    particle_loc = self.obs["particle_loc"]
                    goal_loc = self.obs["goal_loc"]

                # Construct the prompt
                txt = (
                    f"The particle was located at ({round(prev_particle_loc[0], 1)}, {round(prev_particle_loc[1], 1)}).\n"
                    f"The particle is currently located at ({round(particle_loc[0], 1)}, {round(particle_loc[1], 1)}).\n"
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
                        img=self.env.unwrapped.get_rgb_array(),
                        img_parameter_type="image",
                        txt=txt,
                        tokens=1000,
                    )

                coil_vals = json.loads(output)["currents"]
                coil_vals = [0.0 if i < 0 else i for i in coil_vals]

                # Push the new coil values to the queue
                self.coil_vals_queue.put(coil_vals)

            except Exception as e:
                print(f"Error calculating coil values: {e}", flush=True)


if __name__ == "__main__":
    env = ZeroShotRaw(
        env_name="gymnasium_env/SingleParticleNoCargo-v0",
        model_id="PATH_LLAMA",
        model_type="llm",
        context_prompt_file="llm_prompt_zero_shot.yaml",
        model_quant="fp16",
    )
    env.run()
