import os
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from carla_env import CarlaEnv


TOTAL_TIMESTEPS = 1000000


def make_env(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    env = DummyVecEnv([lambda: Monitor(CarlaEnv(), log_dir)])
    env = VecTransposeImage(env)
    return env


def train_ppo():
    env = make_env("logs_ppo/")

    if os.path.exists("ppo_carla_model.zip"):
        print("Loading PPO model...")
        model = PPO.load("ppo_carla_model", env=env, device="cuda")
    else:
        print("Training PPO...")
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=1e-5,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log="./ppo_tensorboard/",
            verbose=1,
            device="cuda"
        )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("ppo_carla_model")
    env.close()


def train_td3():
    env = make_env("logs_td3/")

    print("Training TD3...")
    model = TD3(
        "CnnPolicy",
        env,
        learning_rate=1e-5,
        batch_size=64,
        gamma=0.99,
        tensorboard_log="./td3_tensorboard/",
        verbose=1,
        device="cuda"
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("td3_carla_model")
    env.close()


def train_sac():
    env = make_env("logs_sac/")

    print("Training SAC...")
    model = SAC(
        "CnnPolicy",
        env,
        learning_rate=1e-5,
        batch_size=64,
        gamma=0.99,
        tensorboard_log="./sac_tensorboard/",
        verbose=1,
        device="cuda"
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("sac_carla_model")
    env.close()


if __name__ == "__main__":
    train_ppo()
    train_td3()
    train_sac()