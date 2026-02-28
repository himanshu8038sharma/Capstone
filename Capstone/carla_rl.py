import carla
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time
import cv2
from stable_baselines3 import PPO
import torch
import torch.nn as nn
# =========================================================
# 1️⃣ CARLA RL ENVIRONMENT
# =========================================================
class CarlaEnv(gym.Env):
    def __init__(self):
        super(CarlaEnv, self).__init__()
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        # Load optimized town for performance
        self.world = self.client.get_world()
        # Remove unwanted layers (clean environment)
        self.world.unload_map_layer(carla.MapLayer.Buildings)
        self.world.unload_map_layer(carla.MapLayer.Foliage)
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        self.world.unload_map_layer(carla.MapLayer.Decals)
        self.world.unload_map_layer(carla.MapLayer.Props)
        self.blueprints = self.world.get_blueprint_library()
        # Action: steer [-1,1], throttle [0,1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        # Observation: 84x84 grayscale
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,84, 84),
            dtype=np.float32
        )
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.state = None
        self.collision = False
    # -----------------------------
    def reset(self, seed=None, options=None):
        self.destroy()
        self.collision = False
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[0]  # Fixed starting location
        vehicle_bp = self.blueprints.find("vehicle.tesla.model3")
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        # Camera
        camera_bp = self.blueprints.find("sensor.camera.rgb")
        camera_transform = carla.Transform(
            carla.Location(x=2.5, z=0.7)
        )
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle
        )
        self.camera.listen(lambda image: self.process_image(image))
        # Collision
        collision_bp = self.blueprints.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.collision_sensor.listen(
            lambda event: self.set_collision()
        )
        time.sleep(1)
        while self.state is None:
            time.sleep(0.01)
        return self.state, {}
    # -----------------------------
    def process_image(self, image):
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((image.height, image.width, 4))
        img = img[:, :, :3]
        h = img.shape[0]
        img = img[int(h*0.35):int(h*0.85), :]
        '''
        img = cv2.resize(img, (84, 84))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("CARLA Camera", img)
        cv2.waitKey(1)
        img = img.astype(np.float32) / 255.0
        self.state = img[np.newaxis, :, :]'''
        # ----------------------------
        # 🔹 1️⃣ MODEL INPUT (84x84 grayscale)
        # ----------------------------
        small = cv2.resize(img, (84, 84))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
        self.state = gray[np.newaxis, :, :]

        # ----------------------------
        # 🔹 2️⃣ PRESENTATION WINDOW (BIG & COLOR)
        # ----------------------------
        display = cv2.resize(img, (960, 540))  # HD size
        cv2.imshow("CARLA Autonomous Driving Demo", display)
        cv2.waitKey(1)
    # -----------------------------
    def step(self, action):
        steer, throttle = action
        self.vehicle.apply_control(
            carla.VehicleControl(
                steer=float(steer),
                throttle=float(throttle)
            )
        )
        time.sleep(0.05)
        reward = self.get_reward()
        done = self.collision
        return self.state, reward, done, False, {}
    # -----------------------------
    def get_reward(self):
        vel = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        reward = speed * 0.05
        if self.collision:
            reward -= 50
        return reward
    # -----------------------------
    def set_collision(self):
        self.collision = True
    # -----------------------------
    def destroy(self):
        actors = [self.camera, self.collision_sensor, self.vehicle]
        for actor in actors:
            if actor is not None:
                try:
                    actor.destroy()
                except:
                    pass
# =========================================================
# 2️⃣ TRAINING
# =========================================================
def train():
    env = CarlaEnv()
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        policy_kwargs=dict(normalize_images=False),
        device="cuda"  # 👈 ADD THIS
    )
    model.learn(total_timesteps=500)
    model.save("carla_ppo_model")
    env.destroy()
# =========================================================
# 3️⃣ TESTING
# =========================================================
def test():
    env = CarlaEnv()
    model = PPO.load("carla_ppo_model")
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()
# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    mode = input("Enter mode (train/test): ")
    if mode == "train":
        train()
    else:
        test()
