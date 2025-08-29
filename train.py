import torch
from env_utils import create_env
from model_utils import create_or_load_model, setup_checkpoint

if __name__ == "__main__":
    torch.set_num_threads(1)

    num_envs = 12
    stats_path = "vecnormalize_stats.pkl"
    model_path = "sac_franka_model.zip"

    env = create_env(num_envs, stats_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_or_load_model(env, model_path, device)

    save_every = 100_000
    checkpoint_callback = setup_checkpoint(save_every, num_envs)

    total_timesteps = 20_000_000
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    model.save("sac_franka_model")
    env.save("vecnormalize_stats.pkl")
    env.close()
