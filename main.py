from sai_rl import SAIClient
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
import os

# ---------- ENV CREATION ----------
def make_env_fn():
    def _init():
        client = SAIClient(comp_id="franka-ml-hiring")
        env = client.make_env(render_mode=None, deterministic_reset=False)
        return env
    return _init
    
num_envs = 12
if __name__ == "__main__":

    torch.set_num_threads(1)

    # SubprocVecEnv with multiple processes
    env_fns = [make_env_fn() for _ in range(num_envs)]
    env = SubprocVecEnv(env_fns)
    
    stats_path = "vecnormalize_stats.pkl"

    if os.path.exists(stats_path):
        # Load normalization stats and wrap the new env
        env = VecNormalize.load(stats_path, env)
        env.training = True   # continue updating stats during training
        env.norm_reward = True
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "sac_franka_model.zip"

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = SAC.load(model_path, env=env, device=device)
    else:
        print("Creating new model")
        policy_kwargs = dict(
            net_arch=dict(pi=[512, 512, 256], qf=[512, 512, 256]),
            activation_fn=torch.nn.ReLU
        )
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            batch_size=512,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            gradient_steps=2,
            train_freq=(8, "step"),
            gamma=0.99,
            tau=0.005,
            ent_coef='auto',
            policy_kwargs=policy_kwargs,
            device=device
        )

    os.makedirs("./checkpoints", exist_ok=True)
    save_every = 100_000 // num_envs
    checkpoint_callback = CheckpointCallback(save_freq=save_every, save_path="./checkpoints/",
                                             name_prefix="sac_franka")

    total_timesteps = 20_000_000
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    model.save("sac_franka_model")
    env.save("vecnormalize_stats.pkl")
    env.close()