from sai_rl import SAIClient
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import os


def make_env():
    def _init():
        sai = SAIClient(comp_id="franka-ml-hiring")
        env = sai.make_env(render_mode="human")
        return env
    return _init


print("Hello")
env = DummyVecEnv([make_env()])
env.training = False
env.norm_reward = False

model = SAC.load("./checkpoints/sac_franka_1199952_steps", env=env)



num_episodes = 10
episode_rewards = []

for ep in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
    episode_rewards.append(total_reward)
    print(f"Episode {ep+1}: Reward = {total_reward}")

print("\n   Evaluation Complete")
print(f"Average reward over {num_episodes} episodes: {np.mean(episode_rewards)}")
print(f"Reward standard deviation: {np.std(episode_rewards)}")
env.close()



# Episode 1: Reward = [444.8326]
# Episode 2: Reward = [1200.1604]
# Episode 3: Reward = [1196.4216]
# Episode 4: Reward = [1365.2655]
# Episode 5: Reward = [1232.5862]
# Episode 6: Reward = [1217.2817]
# Episode 7: Reward = [1725.4745]
# Episode 8: Reward = [1525.43]
# Episode 9: Reward = [1513.2985]
# Episode 10: Reward = [1203.5219]

#    Evaluation Complete
# Average reward over 10 episodes: 1262.4273681640625
# Reward standard deviation: 322.5195617675781