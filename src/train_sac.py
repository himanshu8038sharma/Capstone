import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import TrainFreq, TrainFrequencyUnit
from carla_env import CarlaEnv
def train():
    os.makedirs("logs_sac", exist_ok=True)
    HP = {
        "learning_rate": 3e-4,
        "batch_size": 1024,
        "gamma": 0.9995,
        "buffer_size": 80000,
        "learning_starts": 1000,
        "train_freq": (1, "step"),
        "gradient_steps": 2,
        "tau": 0.005,
        "ent_coef": "auto",
        "total_timesteps": 200000
    }
    env = DummyVecEnv([lambda: Monitor(CarlaEnv(), "logs_sac/")])
    env = VecTransposeImage(env)
    n_actions = env.action_space.shape[-1]
    if os.path.exists("sac_carla_model1.zip"):
        print("Loading existing SAC model...")
        model = SAC.load(
            "sac_carla_model1",
            env=env,
            device="cuda",
            print_system_info=True
        )
        model.learning_rate = HP["learning_rate"]
        model.batch_size = HP["batch_size"]
        model.gamma = HP["gamma"]
        model.train_freq = TrainFreq(1, TrainFrequencyUnit.STEP)
        model.gradient_steps = HP["gradient_steps"]
        model.learning_starts = HP["learning_starts"]
        print("Hyperparameters updated for loaded model")
    else:
        print("Creating new SAC model...")
        model = SAC(
            "CnnPolicy",
            env,
            learning_rate=HP["learning_rate"],
            batch_size=HP["batch_size"],
            gamma=HP["gamma"],
            buffer_size=HP["buffer_size"],
            learning_starts=HP["learning_starts"],
            train_freq=HP["train_freq"],
            gradient_steps=HP["gradient_steps"],
            tau=HP["tau"],
            ent_coef=HP["ent_coef"],
            tensorboard_log="./sac_tensorboard1/",
            verbose=1,
            device="cuda"
        )
    try:
        model.learn(
            total_timesteps=HP["total_timesteps"],
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("Training paused... saving model")
    finally:
        model.save("sac_carla_model1")
        print("SAC model saved successfully")
        env.close()
    model.save("sac_carla_model1")
    print("SAC model saved successfully")
    env.close()
if __name__ == "__main__":
    train()
