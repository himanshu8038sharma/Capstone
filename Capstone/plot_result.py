import pandas as pd
import matplotlib.pyplot as plt


def load_data(path):
    data = pd.read_csv(path + "/monitor.csv", skiprows=1)

    rewards = data["r"]
    timesteps = data["l"].cumsum()

    # Smooth reward curve
    smooth = rewards.rolling(20).mean()

    return timesteps, rewards, smooth


ppo_t, ppo_r, ppo_s = load_data("logs_ppo")
td3_t, td3_r, td3_s = load_data("logs_td3")
sac_t, sac_r, sac_s = load_data("logs_sac")


plt.figure(figsize=(10,6))

plt.plot(ppo_t, ppo_s, label="PPO", linewidth=2)
plt.plot(td3_t, td3_s, label="TD3", linewidth=2)
plt.plot(sac_t, sac_s, label="SAC", linewidth=2)

plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.title("RL Algorithm Comparison (CARLA)")

plt.legend()
plt.grid(True)

plt.show()