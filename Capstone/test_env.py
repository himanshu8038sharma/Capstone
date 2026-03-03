from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from carla_env import CarlaEnv

def test():

    env = DummyVecEnv([lambda: CarlaEnv()])
    env = VecTransposeImage(env)

    model = PPO.load("ppo_carla_model")

    obs = env.reset()

    while True:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

        if done:
            obs = env.reset()

if __name__ == "__main__":
    test()
