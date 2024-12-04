import gymnasium as gym
import gymnasium_env

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__ == "__main__":
    # Parallel environments
    vec_env = make_vec_env(
        "gymnasium_env/SingleParticleNoCargo-v0",
        n_envs=8,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"render_mode": "rgb_array", "reward_type": "default"},
    )

    model = PPO("MultiInputPolicy", vec_env, verbose=1, device="cpu")
    model.learn(total_timesteps=2_500_000)

    model.save("control_models/ppo_default_parameters_2_500_000_steps")

    obs = vec_env.reset()

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
