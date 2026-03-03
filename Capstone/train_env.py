import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from carla_env import CarlaEnv

def train():

    env = DummyVecEnv([lambda: CarlaEnv()])
    env = VecTransposeImage(env)

    if os.path.exists("ppo_carla_model.zip"):
        print("Loading existing model...")
        model = PPO.load("ppo_carla_model", env=env, device="cuda")
    else:
        print("Creating new model...")
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=5e-5,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            verbose=1,
            device="cuda"
        )

    model.learn(total_timesteps=100000)
    model.save("ppo_carla_model")

    env.close()

if __name__ == "__main__":
    train()
