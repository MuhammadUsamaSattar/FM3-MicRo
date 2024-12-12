import gymnasium as gym
import gymnasium_env

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

if __name__ == "__main__":
    # Single environment
    #vec_env = DummyVecEnv(
    #   [
    #       lambda: gym.make(
    #           "gymnasium_env/SingleParticleNoCargo-v0",
    #           **{
    #               "render_mode": "rgb_array",
    #               "verbose": False,
    #           },
    #       )
    #   ]
    #)

    vec_env = DummyVecEnv(
        [
            lambda: gym.make(
                "gymnasium_env/SingleParticleNoCargo-v0",
                **{
                    "render_mode": "rgb_array",
                    "model_id": "PATH_QWEN_14B",
                    "model_type": "llm",
                    "model_quant": "4b",
                    "context_prompt_file": "llm_prompt_zero_shot.yaml",
                    "verbose": True,
                },
            )
        ]
    )

    # Parallel environments
    # vec_env = gym.make(
    #    "gymnasium_env/SingleParticleNoCargo-v0",
    #    #n_envs=1,
    #    #vec_env_cls=SubprocVecEnv,
    #    env_kwargs={
    #        "render_mode": "rgb_array",
    #        "model_id": "PATH_QWEN_14B",
    #        "model_type": "llm",
    #        "model_quant": "4b",
    #        "context_prompt_file": "llm_prompt_zero_shot.yaml",
    #        "verbose": True,
    #    },
    # )

    model = PPO("MultiInputPolicy", vec_env, verbose=1, device="cpu", n_steps=2_5)
    model.learn(total_timesteps=2_5)

    model.save("control_models/ppo_default_parameters_2_500_000_steps_fm_rewards")

    # model = PPO.load(
    #    "control_models/ppo_default_parameters_2_500_000_steps_fm_rewards", env=vec_env
    # )

    vec_env.env_method(
        "set_reward_params",
        model_id=None,
        model_type=None,
        model_quant=None,
        context_prompt_file=None,
    )

    obs = vec_env.reset()

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
