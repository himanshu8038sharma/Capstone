import carla
import random
import numpy as np
import time

# ===============================
# Connect to CARLA
# ===============================
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()
blueprints = world.get_blueprint_library()

print("Connected to map:", world.get_map().name)

# ===============================
# Spawn vehicle
# ===============================
vehicle_bp = random.choice(blueprints.filter("vehicle.*"))
spawn_point = random.choice(world.get_map().get_spawn_points())

vehicle = world.spawn_actor(vehicle_bp, spawn_point)
print("Vehicle spawned")

# ===============================
# Attach RGB camera
# ===============================
camera_bp = blueprints.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "640")
camera_bp.set_attribute("image_size_y", "480")
camera_bp.set_attribute("fov", "110")

camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

print("Camera attached")

# ===============================
# Attach collision sensor
# ===============================
collision_bp = blueprints.find("sensor.other.collision")
collision_sensor = world.spawn_actor(
    collision_bp,
    carla.Transform(),
    attach_to=vehicle
)

collision_detected = False

def collision_callback(event):
    global collision_detected
    collision_detected = True
    print("Collision detected!")

collision_sensor.listen(lambda event: collision_callback(event))

# ===============================
# Camera image callback
# ===============================
def image_callback(image):
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((image.height, image.width, 4))
    img = img[:, :, :3]

    # PRINT IMAGE SHAPE → final verification
    print("Image shape:", img.shape)

camera.listen(lambda image: image_callback(image))

# ===============================
# Read vehicle speed
# ===============================
def get_speed(vehicle):
    vel = vehicle.get_velocity()
    speed = 3.6 * (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5
    return speed  # km/h

# ===============================
# Run loop
# ===============================
try:
    print("Running for 20 seconds...\n")
    start_time = time.time()

    while time.time() - start_time < 20:
        speed = get_speed(vehicle)
        print(f"Speed: {speed:.2f} km/h | Collision: {collision_detected}")
        time.sleep(1)

finally:
    print("\nCleaning up actors...")
    camera.stop()
    collision_sensor.stop()

    camera.destroy()
    collision_sensor.destroy()
    vehicle.destroy()

    print("Done. Setup verified ✅")