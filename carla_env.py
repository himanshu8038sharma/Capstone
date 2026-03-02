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

        # Connect to CARLA
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        # Episode variables
        self.destination = None
        self.prev_distance = None
        self.success_threshold = 5.0

        self.collision_hist = []
        self.lane_invasion = False
        self.traffic_violation = False
        self.pedestrian_collision = False
        self.image = None

        self.max_steps = 1000
        self.step_count = 0

        # Synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)

        # Action space
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Observation space
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, 1),
            dtype=np.uint8
        )

        self.actor_list = []

    # =====================================================
    # RESET
    # =====================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._cleanup()

        self.step_count = 0
        self.collision_hist = []
        self.lane_invasion = False
        self.traffic_violation = False
        self.pedestrian_collision = False
        self.image = None

        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)

        vehicle_bp = random.choice(
            self.blueprint_library.filter("vehicle.tesla.model3")
        )

        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.actor_list.append(self.vehicle)

        # Destination selection
        destination_point = random.choice(spawn_points)
        attempts = 0
        while (
            destination_point.location.distance(spawn_point.location) < 20
            and attempts < 50
        ):
            destination_point = random.choice(spawn_points)
            attempts += 1

        self.destination = destination_point.location
        self.prev_distance = self._get_distance_to_destination()

        # ===============================
        # CAMERA SENSOR
        # ===============================
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

        # ===============================
        # COLLISION SENSOR
        # ===============================
        collision_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(
            lambda event: self._collision_event(event)
        )

        # ===============================
        # LANE SENSOR
        # ===============================
        lane_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(
            lane_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.actor_list.append(self.lane_sensor)
        self.lane_sensor.listen(lambda event: self._lane_event())

        # ===============================
        # PEDESTRIAN GENERATION
        # ===============================
        num_pedestrians = 10
        walker_blueprints = self.blueprint_library.filter("walker.pedestrian.*")
        controller_bp = self.blueprint_library.find("controller.ai.walker")

        for _ in range(num_pedestrians):
            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                continue

            walker_transform = carla.Transform(loc)
            walker_bp = random.choice(walker_blueprints)

            walker = self.world.spawn_actor(walker_bp, walker_transform)
            self.actor_list.append(walker)

            controller = self.world.spawn_actor(
                controller_bp,
                carla.Transform(),
                walker
            )
            self.actor_list.append(controller)

            controller.start()
            controller.go_to_location(
                self.world.get_random_location_from_navigation()
            )
            controller.set_max_speed(1.4)

        # Wait until first image is received
        while self.image is None:
            self.world.tick()

        return self.image, {}

    # =====================================================
    # STEP
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

        self._check_traffic_violation()

        reward = self._compute_reward()
        done = self._check_done()

        self.lane_invasion = False

        return self.image, reward, done, False, {}

    # =====================================================
    # COLLISION EVENT
    # =====================================================
    def _collision_event(self, event):
        other_actor = event.other_actor

        if "walker" in other_actor.type_id:
            self.pedestrian_collision = True
        else:
            self.collision_hist.append(event)

    # =====================================================
    # TRAFFIC VIOLATION CHECK
    # =====================================================
    def _check_traffic_violation(self):
        if self.vehicle.is_at_traffic_light():
            traffic_light = self.vehicle.get_traffic_light()

            if traffic_light.get_state() == carla.TrafficLightState.Red:
                velocity = self.vehicle.get_velocity()
                speed = math.sqrt(velocity.x**2 + velocity.y**2)

                if speed > 1.0:
                    self.traffic_violation = True

    # =====================================================
    # REWARD
    # =====================================================
    def _compute_reward(self):
        reward = 0

        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2)
        reward += 0.5 * speed

        current_distance = self._get_distance_to_destination()
        reward += 2.0 * (self.prev_distance - current_distance)
        self.prev_distance = current_distance

        if len(self.collision_hist) > 0:
            reward -= 100

        if self.pedestrian_collision:
            reward -= 300

        if self.traffic_violation:
            reward -= 150

        if self.lane_invasion:
            reward -= 10

        if current_distance < self.success_threshold:
            reward += 200

        return reward

    # =====================================================
    # DONE CONDITIONS
    # =====================================================
    def _check_done(self):

        if self.pedestrian_collision:
            return True

        if len(self.collision_hist) > 0:
            return True

        if self.traffic_violation:
            return True

        if self.step_count >= self.max_steps:
            return True

        if self._get_distance_to_destination() < self.success_threshold:
            return True

        return False

    # =====================================================
    # HELPERS
    # =====================================================
    def _get_distance_to_destination(self):
        return self.vehicle.get_location().distance(self.destination)
    def _process_image(self, image):
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((84, 84, 4))
        img = img[:, :, :3]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.image = np.expand_dims(gray, axis=2)

    def _lane_event(self):
        self.lane_invasion = True

    def _cleanup(self):
        for actor in self.actor_list:
            if actor.is_alive:
                try:
                    actor.stop()
                except:
                    pass
                actor.destroy()
        self.actor_list = []
    def close(self):
        self._cleanup()
