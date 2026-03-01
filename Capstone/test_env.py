from carla_env import CarlaEnv
import numpy as np
import time

env = CarlaEnv()

obs, _ = env.reset()

done = False
total_reward = 0

while not done:
    # Random action for testing
    action = env.action_space.sample()

    obs, reward, done, truncated, info = env.step(action)

    total_reward += reward
    print(f"Step Reward: {reward:.2f}")

    time.sleep(0.05)

print("Episode finished")
print("Total Reward:", total_reward)

env.close()
