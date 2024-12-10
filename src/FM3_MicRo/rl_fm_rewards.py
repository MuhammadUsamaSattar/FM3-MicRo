import gymnasium as gym
import gymnasium_env

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__ == "__main__":
    # Parallel environments
    vec_env = make_vec_env(
        "gymnasium_env/SingleParticleNoCargo-v0",
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "render_mode": "rgb_array",
            "model_id": "PATH_LLAMA",
            "model_type": "llm",
            "model_quant": "fp16",
            "context_prompt_file": "llm_prompt_zero_shot.yaml",
        },
    )

    model = PPO("MultiInputPolicy", vec_env, verbose=1, device="cpu")
    model.learn(total_timesteps=2_50)

    model.save("control_models/ppo_default_parameters_2_500_000_steps_fm_rewards")

    obs = vec_env.reset()

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
