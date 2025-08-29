from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
import os

def create_or_load_model(env, model_path: str, device: str):
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
    return model

def setup_checkpoint(save_every: int, num_envs: int):
    os.makedirs("./checkpoints", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_every // num_envs,
        save_path="./checkpoints/",
        name_prefix="sac_franka"
    )
    return checkpoint_callback
