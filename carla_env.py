import carla
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import cv2


class CarlaEnv(gym.Env):

    def __init__(self):
        super(CarlaEnv, self).__init__()

        # =========================
        # 1️⃣ Connect to CARLA
        # =========================
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        # Destination variables
        self.destination = None
        self.prev_distance = None
        self.success_threshold = 5.0  # meters

        # =========================
        # 2️⃣ Synchronous Mode
        # =========================
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)

        # =========================
        # 3️⃣ Action Space
        # =========================
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # =========================
        # 4️⃣ Observation Space
        # =========================
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

    # =====================================================
    # RESET FUNCTION
    # =====================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._cleanup()

        self.step_count = 0
        self.collision_hist = []
        self.lane_invasion = False
        self.image = None

        # Random weather
        weather = carla.WeatherParameters(
            cloudiness=random.uniform(0, 100),
            precipitation=random.uniform(0, 50),
            sun_altitude_angle=random.uniform(10, 90)
        )
        self.world.set_weather(weather)

        # Spawn vehicle
        vehicle_bp = random.choice(
            self.blueprint_library.filter("vehicle.tesla.model3")
        )

        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)

        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.actor_list.append(self.vehicle)

        # =========================
        # Select Safe Destination
        # =========================
        destination_point = random.choice(spawn_points)

        while destination_point.location.distance(spawn_point.location) < 20:
            destination_point = random.choice(spawn_points)

        self.destination = destination_point.location
        self.prev_distance = self._get_distance_to_destination()

        # =========================
        # Attach Camera
        # =========================
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

        # =========================
        # Collision Sensor
        # =========================
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

        # =========================
        # Lane Invasion Sensor
        # =========================
        lane_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(
            lane_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.actor_list.append(self.lane_sensor)
        self.lane_sensor.listen(lambda event: self._lane_event())

        # Spawn Traffic
        self._spawn_traffic()

        # Wait for first frame
        while self.image is None:
            self.world.tick()

        return self.image, {}

    # =====================================================
    # STEP FUNCTION
    # =====================================================
    def step(self, action):
        self.step_count += 1

        steer, throttle, brake = action

        control = carla.VehicleControl(
            steer=float(steer),
            throttle=float(throttle),
            brake=float(brake)
        )

        self.vehicle.apply_control(control)
        self.world.tick()

        reward = self._compute_reward()
        done = self._check_done()

        # Reset lane invasion flag
        self.lane_invasion = False

        return self.image, reward, done, False, {}

    # =====================================================
    # REWARD FUNCTION
    # =====================================================
    def _compute_reward(self):
        reward = 0

        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2)

        # Encourage forward motion
        reward += 0.5 * speed

        # Distance-based reward
        current_distance = self._get_distance_to_destination()
        distance_reward = self.prev_distance - current_distance
        reward += 2.0 * distance_reward

        self.prev_distance = current_distance

        # Destination reached
        if current_distance < self.success_threshold:
            reward += 200

        # Penalties
        if len(self.collision_hist) > 0:
            reward -= 100

        if self.lane_invasion:
            reward -= 10

        return reward

    # =====================================================
    # DISTANCE CALCULATION
    # =====================================================
    def _get_distance_to_destination(self):
        vehicle_location = self.vehicle.get_location()
        return vehicle_location.distance(self.destination)

    # =====================================================
    # DONE CONDITIONS
    # =====================================================
    def _check_done(self):
        if len(self.collision_hist) > 0:
            return True

        if self.step_count >= self.max_steps:
            return True

        if self._get_distance_to_destination() < self.success_threshold:
            return True

        return False

    # =====================================================
    # IMAGE PROCESSING
    # =====================================================
    def _process_image(self, image):
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((84, 84, 4))
        img = img[:, :, :3]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.expand_dims(gray, axis=2)

        self.image = gray

    # =====================================================
    # LANE EVENT
    # =====================================================
    def _lane_event(self):
        self.lane_invasion = True

    # =====================================================
    # TRAFFIC SPAWN
    # =====================================================
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

    # =====================================================
    # CLEANUP
    # =====================================================
    def _cleanup(self):
        for actor in self.actor_list:
            if actor.is_alive:
                actor.destroy()
        self.actor_list = []

    def close(self):
        self._cleanup()
