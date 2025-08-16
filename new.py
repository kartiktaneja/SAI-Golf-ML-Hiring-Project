import os
import gymnasium as gym
from sai_rl import SAIClient
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import torch

def make_env():
    def _init():
        client = SAIClient(comp_id="franka-ml-hiring")
        env = client.make_env(render_mode=None)
        return env
    return _init
    
num_envs = 8

if __name__ == "__main__":

    # torch.set_num_threads(1)  # Prevent thread contention    
    env_fns = [make_env() for _ in range(num_envs)]
    env = SubprocVecEnv(env_fns)

    model = SAC("MlpPolicy", env, verbose=1, batch_size=1024, learning_rate=3e-4,
        buffer_size=1_000_000, device = "cuda")

    model.learn(total_timesteps=200_000)
    model.save("sac_franka_model")

    env.close()

# if __name__ == "__main__":
#     import multiprocessing
#     multiprocessing.freeze_support()  # Required for Windows
#     train_sac()