from gymnasium_env.envs.Library.llm import LLM
import gymnasium as gym
import gymnasium_env

from gymnasium_envs.gymnasium_env.envs.Library.llm import LLM


class ZeroShotRaw:
    def __init__(self, env, model_type):
        env = gym.make("gymnasium_env/SingleParticleNoCargo-v0", render_mode="human")


if __name__ == "__main__":
    foundation_model = 'llm'

    if foundation_model == 'llm':
        model = LLM(model_quant="fp16", device="cuda")

    env = gym.make("gymnasium_env/SingleParticleNoCargo-v0", render_mode="human")
    env.reset()
    dones = False

    while True:
        env.step([0, 0, 0, 0, 0, 0, 0, 0])
