# =============================================
# BASIC END-TO-END CARLA RL DEMO (SIMPLIFIED)
# This single script covers:
# - Connect to CARLA
# - Spawn vehicle + sensors
# - Preprocess camera image
# - Very simple RL-style random policy loop
# - Reward calculation
# - Cleanup
# =============================================

import carla
import random
import numpy as np
import cv2
import time

# ===============================
# IMAGE PREPROCESSING
# ===============================

def preprocess(image):
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((image.height, image.width, 4))[:, :, :3]

    # crop road region
    h = img.shape[0]
    img = img[int(h * 0.35): int(h * 0.85), :]

    # resize
    img = cv2.resize(img, (84, 84))

    # grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # normalize
    img = img.astype(np.float32) / 255.0

    return img


# ===============================
# CONNECT TO CARLA
# ===============================

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()
blueprints = world.get_blueprint_library()

print("Connected to map:", world.get_map().name)

# ===============================
# SPAWN VEHICLE
# ===============================

vehicle_bp = random.choice(blueprints.filter("vehicle.*"))
spawn_point = random.choice(world.get_map().get_spawn_points())

vehicle = world.spawn_actor(vehicle_bp, spawn_point)
print("Vehicle spawned")

# ===============================
# CAMERA SENSOR
# ===============================

camera_bp = blueprints.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "640")
camera_bp.set_attribute("image_size_y", "480")

camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))

camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

state = None

def camera_callback(image):
    global state
    state = preprocess(image)

camera.listen(lambda image: camera_callback(image))

print("Camera attached")

# ===============================
# COLLISION SENSOR
# ===============================

collision_bp = blueprints.find("sensor.other.collision")
collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)

collision = False

def collision_callback(event):
    global collision
    collision = True
    print("Collision detected!")

collision_sensor.listen(lambda event: collision_callback(event))

print("Collision sensor attached")

# ===============================
# SPEED FUNCTION
# ===============================

def get_speed(vehicle):
    v = vehicle.get_velocity()
    return 3.6 * (v.x**2 + v.y**2 + v.z**2) ** 0.5


# ===============================
# SIMPLE RANDOM RL LOOP
# ===============================

try:
    print("Starting basic driving loop...")

    start = time.time()
    EPISODE_TIME = 30  # seconds

    while time.time() - start < EPISODE_TIME:

        if state is None:
            continue

        # RANDOM ACTION (placeholder for RL policy)
        steer = random.uniform(-1.0, 1.0)
        throttle = random.uniform(0.3, 0.7)
        brake = 0.0

        vehicle.apply_control(carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake
        ))

        # ===== Reward calculation (very basic) =====
        speed = get_speed(vehicle)

        reward = 0
        reward += speed * 0.05        # encourage movement

        if collision:
            reward -= 50

        print(f"Speed: {speed:.2f} km/h | Reward: {reward:.2f}")

        # visualize processed state
        cv2.imshow("State", state)
        cv2.waitKey(1)

        time.sleep(0.1)

finally:
    print("Cleaning up...")

    camera.stop()
    collision_sensor.stop()

    camera.destroy()
    collision_sensor.destroy()
    vehicle.destroy()

    cv2.destroyAllWindows()

    print("Finished.")
