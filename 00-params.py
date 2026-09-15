import numpy as np
import math


# ==============================
# 1. LiDAR parameters
# ==============================

LIDAR_FOV_DEG = 270.0
LIDAR_ANGLE_RES_DEG = 0.25
LIDAR_HEIGHT_M = 2.0
LIDAR_MAX_RANGE_M = 25.0
LIDAR_NOISE_M = 0.02
LIDAR_SCAN_RATE_HZ = 20.0


# ==============================
# 2. Nozzle parameters
# ==============================

NOZZLE_COUNT = 6
NOZZLE_PITCH_M = 0.4
LOWEST_NOZZLE_HEIGHT_M = 0.5


# Calculate nozzle centre heights
nozzle_heights = np.array([
    LOWEST_NOZZLE_HEIGHT_M + i * NOZZLE_PITCH_M
    for i in range(NOZZLE_COUNT)
])


# Each nozzle gets a 0.4 m vertical zone
nozzle_zones = [
    (
        height - NOZZLE_PITCH_M / 2,
        height + NOZZLE_PITCH_M / 2
    )
    for height in nozzle_heights
]


# ==============================
# 3. Display our setup
# ==============================

print("LiDAR FOV:", LIDAR_FOV_DEG, "degrees")
print("LiDAR angular resolution:", LIDAR_ANGLE_RES_DEG, "degrees")
print("LiDAR height:", LIDAR_HEIGHT_M, "m")

print("\nNozzle heights:")
for i, height in enumerate(nozzle_heights, start=1):
    print(f"N{i}: {height:.2f} m")

print("\nNozzle zones:")
for i, (z_min, z_max) in enumerate(nozzle_zones, start=1):
    print(f"N{i}: {z_min:.2f} m to {z_max:.2f} m")