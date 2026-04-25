import carla
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import cv2
import math
class CarlaEnv(gym.Env):
    def __init__(self):
        super(CarlaEnv, self).__init__()
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(30.0)
        town_train=["Town01_Opt","Town02_Opt","Town03_Opt","Town04_Opt","Town05_Opt"]
        selected_town = random.choice(town_train)
        town_test="Town10HD_Opt"
        self.world = self.client.load_world(selected_town)
        self.blueprint_library = self.world.get_blueprint_library()
        self.destination = None
        self.prev_distance = None
        self.success_threshold = 5.0
        self.traffic_violation=False
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)
        self.action_space = spaces.Box(
            low=np.array([-0.5, 0],dtype=np.float32),
            high=np.array([0.5, 0.5],dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, 1),
            dtype=np.uint8
        )
        self.actor_list = []
        self.collision_hist = []
        self.lane_invasion = False
        self.image = None
        self.max_steps = 1000
        self.step_count = 0
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._cleanup()
        self.step_count = 0
        self.collision_hist = []
        self.lane_invasion = False
        self.traffic_violation=False
        self.image = None
        weather = carla.WeatherParameters(
            cloudiness=random.uniform(0, 100),
            precipitation=random.uniform(0, 50),
            sun_altitude_angle=random.uniform(10, 90),
            fog_density=random.uniform(0, 50),
            fog_distance=random.uniform(0, 50),
            wind_intensity=random.uniform(0, 50),
        )
        self.world.set_weather(weather)
        vehicle_bp = random.choice(
            self.blueprint_library.filter("vehicle.tesla.model3")
        )
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        self.vehicle = None
        for spawn_point in spawn_points:
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if self.vehicle is not None:
                break
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle after trying all spawn points.")
        self.actor_list.append(self.vehicle)
        destination_point = random.choice(spawn_points)
        attempts=0
        while not (
                (destination_point.location.x - spawn_point.location.x) > 100 and
                (destination_point.location.y - spawn_point.location.y) > 50
        ) and attempts < 50:
            destination_point = random.choice(spawn_points)
            attempts += 1
        if attempts >= 50:
            destination_point = random.choice(spawn_points)
        self.destination = destination_point.location
        self.prev_distance = self._get_distance_to_destination()
        camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "84")
        camera_bp.set_attribute("image_size_y", "84")
        camera_bp.set_attribute("fov", "100")
        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4)
        )
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle
        )
        self.actor_list.append(self.camera)
        self.camera.listen(lambda image: self._process_image(image))
        collision_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(
            lambda event: self.collision_hist.append(event)
        )
        lane_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(
            lane_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.actor_list.append(self.lane_sensor)
        self.lane_sensor.listen(lambda event: self._lane_event())
        self._spawn_traffic()
        while self.image is None:
            self.world.tick()
        return self.image, {}
    def step(self, action):
        self.step_count += 1
        steer, throttle = action
        control = carla.VehicleControl(
            steer=float(steer),
            throttle=float(throttle),
            hand_brake = False,
            reverse=False,
        )
        self.vehicle.apply_control(control)
        self.world.tick()
        self._check_traffic_violation()
        self.update_spectator()
        reward = self._compute_reward()
        done = self._check_done()
        self.lane_invasion = False
        return self.image, reward, done, False, {}
    def _compute_reward(self):
        reward = 0
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2)
        reward += 0.1 * speed
        yaw_rate= abs(self.vehicle.get_angular_velocity().z)
        reward -= 0.2*yaw_rate
        current_distance = self._get_distance_to_destination()
        distance_reward = self.prev_distance - current_distance
        reward += 0.1 * distance_reward
        self.prev_distance = current_distance
        waypoint = self.world.get_map().get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True
        )
        lane_distance = self.vehicle.get_location().distance(
            waypoint.transform.location
        )
        reward -= 1.0 * lane_distance
        if speed<0.1:
            reward -= 10
        if current_distance < self.success_threshold:
            reward += 20
        if len(self.collision_hist) > 0:
            reward -= 10
        if self.lane_invasion:
            reward -= 5
        return reward
    def _get_distance_to_destination(self):
        vehicle_location = self.vehicle.get_location()
        return vehicle_location.distance(self.destination)
    def _check_done(self):
        if len(self.collision_hist) > 0:
            return True
        if self.traffic_violation:
            return True
        if self.step_count >= self.max_steps:
            return True
        if self._get_distance_to_destination() < self.success_threshold:
            return True
        waypoint = self.world.get_map().get_waypoint(
            self.vehicle.get_location(),
            project_to_road=False
        )
        if waypoint is None or waypoint.lane_type != carla.LaneType.Driving:
            return True
        return False
    def update_spectator(self):
        spectator = self.world.get_spectator()
        transform = self.vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation
        offset_x = -8 * math.cos(math.radians(rotation.yaw))
        offset_y = -8 * math.sin(math.radians(rotation.yaw))
        spectator_location = carla.Location(
            x=location.x + offset_x,
            y=location.y + offset_y,
            z=location.z + 4
        )
        spectator_rotation = carla.Rotation(
            pitch=-15,
            yaw=rotation.yaw,
            roll=0
        )
        spectator.set_transform(
            carla.Transform(spectator_location, spectator_rotation)
        )
    def _check_traffic_violation(self):
        if self.vehicle.is_at_traffic_light():
            traffic_light = self.vehicle.get_traffic_light()
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                velocity = self.vehicle.get_velocity()
                speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2)
                if speed > 1.0:
                    self.traffic_violation = True
    def _process_image(self, image):
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((84, 84, 4))
        img = img[:, :, :3]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.expand_dims(gray, axis=2)
        self.image = gray
    def _lane_event(self):
        self.lane_invasion = True
    def _spawn_traffic(self):
        spawn_points = self.world.get_map().get_spawn_points()
        for _ in range(20):
            vehicle_bp = random.choice(
                self.blueprint_library.filter("vehicle.*")
            )
            spawn_point = random.choice(spawn_points)
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle is not None:
                vehicle.set_autopilot(True, self.traffic_manager.get_port())
                self.actor_list.append(vehicle)
    def _cleanup(self):
        for actor in self.actor_list:
            if actor.is_alive:
                try:
                    actor.destroy()
                except:
                    pass
        self.actor_list = []
    def close(self):
        self._cleanup()
        self.world.tick()
