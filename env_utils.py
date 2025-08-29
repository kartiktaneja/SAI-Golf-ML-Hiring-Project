from sai_rl import SAIClient
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import os

def make_env_fn():
    def _init():
        client = SAIClient(comp_id="franka-ml-hiring")
        env = client.make_env(render_mode=None, deterministic_reset=False)
        return env
    return _init

def create_env(num_envs: int, stats_path: str):
    env_fns = [make_env_fn() for _ in range(num_envs)]
    env = SubprocVecEnv(env_fns)

    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    return env
