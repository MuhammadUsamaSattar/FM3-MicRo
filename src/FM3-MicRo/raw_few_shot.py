import gymnasium as gym
import gymnasium_env
import yaml

from gymnasium_env.envs.Library.llm import LLM
from gymnasium_env.envs.Library.vlm import VLM

class ZeroShotRaw:
    def __init__(self, env_name, model_type):
        self.env = gym.make(env_name, render_mode="human")

        context_prompt_file = "src/FM3-MicRo/detailed_prompt.yaml"
        with open(context_prompt_file) as stream:
            try:
                self.context = yaml.safe_load(stream)['prompt']
                print("Contextual prompt is:\n" + self.context)
            except yaml.YAMLError as exc:
                print(exc)
        
        if model_type == "llm":
            self.model = LLM(model_quant="fp16", device="cuda", verbose=True)

        elif model_type == "vlm":
            self.model = VLM(model_quant="fp16", device="cuda", verbose=True)
    
        self.env.reset()

    def run(self):
        while True:
            self.step()
            

    def step(self):
        particle_loc = [0, 0]
        while True:
            #particle_loc[0] = input("Enter particle x-location: ")
            #particle_loc[1] = input("Enter particle y-location: ")

            txt = (
                """The particle is currently located at (""",
                particle_loc[0],
                """, """,
                particle_loc[1],
                """)\n
                The goal position is (150, 150).\n
                What should the current values be to reach the goal position?
                """,
            )
            self.model.get_response(context=self.context, txt=txt, tokens=1000)


if __name__ == "__main__":
    env = ZeroShotRaw("gymnasium_env/SingleParticleNoCargo-v0", model_type='llm')
    env.run()
    
    

