import json
import queue
import threading
import yaml

import gymnasium as gym
import gymnasium_env
from PIL import Image

from gymnasium_env.envs.Library.llm import LLM
from gymnasium_env.envs.Library.vlm import VLM


class ZeroShotRaw:
    def __init__(self, env_name, model_type, model_quant="fp16"):
        self.model_type = model_type
        self.env = gym.make(env_name, render_mode="human")

        dir = "src/FM3-MicRo/prompts/"
        context_prompt_file = (
            dir + self.model_type + "_prompt_guide+steps+output_steps.yaml"
        )
        with open(context_prompt_file) as stream:
            try:
                self.context = yaml.safe_load(stream)["prompt"]
                print("Contextual prompt is:\n" + self.context)
            except yaml.YAMLError as exc:
                print(exc)

        if self.model_type == "llm":
            self.model = LLM(model_quant=model_quant, device="cuda", verbose=True)
        elif self.model_type == "vlm":
            self.model = VLM(model_quant=model_quant, device="cuda", verbose=True)

        # Create a thread-safe queue to share coil values between threads
        self.coil_vals_queue = queue.Queue()
        self.obs_lock = threading.Lock()

    def run(self):
        self.obs, info = self.env.reset()
        coil_vals = [0, 0, 0, 0, 0, 0, 0, 0]

        # Start a separate thread for calculating coil values
        calc_thread = threading.Thread(
            target=self.calculate_coil_vals, daemon=True
        )  ############ Does this disable gpu? Compare to 15 T/s #######
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
                self.obs = obs

            if done:
                obs, info = self.env.reset()

                with self.obs_lock:
                    self.obs = obs

    def calculate_coil_vals(self):
        """
        Continuously calculate coil values in a separate thread.
        """
        while True:
            try:
                # Get the current particle location and goal location
                with self.obs_lock:
                    particle_loc = self.obs["particle_loc"]
                    goal_loc = self.obs["goal_loc"]

                # Construct the prompt
                txt = (
                    f"The particle is currently located at ({round(particle_loc[0], 1)}, {round(particle_loc[1], 1)}).\n"
                    f"The goal is currently located at ({goal_loc[0]}, {goal_loc[1]}).\n\n"
                    f"What should the current values be to reach the goal position?"
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
        "gymnasium_env/SingleParticleNoCargo-v0",
        model_type="llm",
        model_quant="4b",
    )
    env.run()
