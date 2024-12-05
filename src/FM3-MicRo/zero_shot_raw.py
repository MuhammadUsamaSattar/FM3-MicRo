import gymnasium as gym
import gymnasium_env
import threading
import yaml

from gymnasium_env.envs.Library.llm import LLM
from gymnasium_env.envs.Library.vlm import VLM


class ZeroShotRaw:
    def __init__(self, env_name, model_type):
        self.env = gym.make(env_name, render_mode="human")

        context_prompt_file = "src/FM3-MicRo/detailed_prompt.yaml"
        with open(context_prompt_file) as stream:
            try:
                self.context = yaml.safe_load(stream)["prompt"]
                print("Contextual prompt is:\n" + self.context)
            except yaml.YAMLError as exc:
                print(exc)

        if model_type == "llm":
            self.model = LLM(model_quant="fp16", device="cuda", verbose=True)

        elif model_type == "vlm":
            self.model = VLM(model_quant="fp16", device="cuda", verbose=True)

    def run(self):
        obs, info = self.env.reset()
        while True:
            obs, reward, done, truncated, info = self.step(obs)

            dones = done or truncated
            if dones:
                obs, info = self.env.reset()

    def step(self, obs):
        # particle_loc = [0, 0]

        # particle_loc[0] = input("Enter particle x-location: ")
        # particle_loc[1] = input("Enter particle y-location: ")
        particle_loc = obs["particle_loc"]
        goal_loc = obs["goal_loc"]

        txt = (
            """The particle is currently located at (""",
            particle_loc[0],
            """, """,
            particle_loc[1],
            """).\n
            The goal is currently located at (""",
            goal_loc[0],
            """, """,
            goal_loc[1],
            """).\n
            \n
            What should the current values be to reach the goal position?
            """,
        )
        output = self.model.get_response(context=self.context, txt=txt, tokens=1000)

        key = "currents: "
        index = output.find(key)

        if index > -1:
            coil_vals = eval(output[index + len(key) :])
        else:
            raise Exception("Current values not found in generated prompt")

        obs, reward, done, truncated, info = self.env.step(coil_vals)

        return obs, reward, done, truncated, info


if __name__ == "__main__":
    env = ZeroShotRaw("gymnasium_env/SingleParticleNoCargo-v0", model_type="llm")
    env.run()
