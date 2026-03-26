import os
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from carla_env import CarlaEnv
def train():
    os.makedirs("logs_td3", exist_ok=True)
    # ✅ STABLE TD3 HYPERPARAMETERS
    # ==============================
    HP = {
        "learning_rate": 1e-4,  # TD3 works better with higher LR than PPO
        "batch_size": 256,  # 1024 was too large → unstable updates
        "gamma": 0.99,  # stable discount
        "buffer_size": 100000,  # large replay buffer is critical
        "learning_starts": 5000,  # collect enough data first
        "train_freq": (1, "step"),  # update every step
        "gradient_steps": 1,  # stable incremental updates
        "tau": 0.005,  # soft update coefficient
        "policy_delay": 2,  # TD3 core feature
        "total_timesteps": 700000,
        "noise_sigma": 0.1  # good starting value for TD3
    }
    # ==============================
    # ENV
    # ==============================
    env = DummyVecEnv([lambda: Monitor(CarlaEnv(), "logs_td3/")])
    env = VecTransposeImage(env)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=HP["noise_sigma"] * np.ones(n_actions)
    )
    # ==============================
    # LOAD OR CREATE MODEL
    # ==============================
    if os.path.exists("td3_carla_model.zip"):

        print("Loading existing TD3 model...")

        model = TD3.load(
            "td3_carla_model",
            env=env,
            device="cuda"
        )
        # ==============================
        # 🔁 OVERRIDE HYPERPARAMETERS
        # ==============================
        model.learning_rate = HP["learning_rate"]
        model.batch_size = HP["batch_size"]
        model.gamma = HP["gamma"]
        from stable_baselines3.common.utils import TrainFreq, TrainFrequencyUnit
        model.train_freq = TrainFreq(1, TrainFrequencyUnit.STEP)
        model.gradient_steps = HP["gradient_steps"]
        model.learning_starts = HP["learning_starts"]
        # Update exploration noise
        model.action_noise = action_noise
        # ⚠ Replay buffer resize (only if increasing)
        """if model.replay_buffer.buffer_size < HP["buffer_size"]:
            print("Increasing replay buffer size...")
            model.replay_buffer.buffer_size = HP["buffer_size"]"""
        print("Hyperparameters updated for loaded model")
    else:
        print("Creating new TD3 model...")
        model = TD3(
            "CnnPolicy",
            env,
            learning_rate=HP["learning_rate"],
            batch_size=HP["batch_size"],
            gamma=HP["gamma"],
            buffer_size=HP["buffer_size"],
            learning_starts=HP["learning_starts"],
            train_freq=HP["train_freq"],
            gradient_steps=HP["gradient_steps"],
            action_noise=action_noise,
            tensorboard_log="./td3_tensorboard/",
            verbose=1,
            device="cuda"
        )
    # ==============================
    # TRAIN (continue if loaded)
    # ==============================
    
    try:
        model.learn(
            total_timesteps=HP["total_timesteps"],
            reset_num_timesteps=False  # ALWAYS continue training
        )
    except KeyboardInterrupt:
        print("Training paused... saving model")
    finally:
        model.save("td3_carla_model")
        print("TD3 model saved successfully")
        env.close()
    """model.learn(
        total_timesteps=HP["total_timesteps"],
        reset_num_timesteps=not os.path.exists("td3_carla_model.zip")
    )"""

    # ==============================
    # SAVE
    # ==============================
    model.save("td3_carla_model")
    print("TD3 model saved successfully")
    env.close()
if __name__ == "__main__":
    train()