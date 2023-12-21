import gymnasium as gym
import time
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Create the environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Initialize the model using the MlpPolicy
model = PPO("MlpPolicy", env, verbose=1)

start = time.process_time()
start_2 = time.time()

# Train the model
model.learn(total_timesteps=100_000)

end = time.process_time()
end_2 = time.time()

print("Process time: " + str(end - start) + "s")
print("Wall-clock time: " + str(end_2 - start_2) + "s")

# Evaluate the trained agent
rewards, lengths = evaluate_policy(
    model, model.get_env(), n_eval_episodes=100, return_episode_rewards=True
)

# Calculate performance metrics
avg_reward = np.mean(rewards)
std_reward = np.std(rewards)
min_reward = np.min(rewards)
max_reward = np.max(rewards)

# Print performance metrics
print(f"Mean Episode Reward: {avg_reward:.2f} +/- {std_reward:.2f}")
print(f"Minimum Episode Reward: {min_reward}")
print(f"Maximum Episode Reward: {max_reward}")

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(10_000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    if done:
        obs = vec_env.reset()
        print("Reset")
