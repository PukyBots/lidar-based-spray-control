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


# ==============================
# 4. Generate LiDAR beam angles
# ==============================

half_fov = LIDAR_FOV_DEG / 2

beam_angles_deg = np.arange(
    -half_fov,
    half_fov + LIDAR_ANGLE_RES_DEG,
    LIDAR_ANGLE_RES_DEG
)

print("\nLiDAR beam information:")
print("Number of beams:", len(beam_angles_deg))
print("First angle:", beam_angles_deg[0], "degrees")
print("Last angle:", beam_angles_deg[-1], "degrees")
print("Angular spacing:", beam_angles_deg[1] - beam_angles_deg[0], "degrees")


# ==============================
# 5. Orchard / tree parameters
# ==============================

# ==============================
# 5. Orchard / tree parameters
# ==============================

trees = [
    {"x": 5.0,  "height": 2.4, "width": 1.2, "density": 0.85},
    {"x": 9.0,  "height": 3.1, "width": 1.5, "density": 0.90},
    {"x": 13.0, "height": 1.8, "width": 1.0, "density": 0.35},  # sparse
    {"x": 17.0, "height": 2.8, "width": 1.4, "density": 0.75},
    {"x": 21.0, "height": 3.4, "width": 1.6, "density": 0.95},
    # x = 25 m → missing tree
    {"x": 29.0, "height": 2.2, "width": 1.2, "density": 0.60},
    {"x": 33.0, "height": 3.0, "width": 1.5, "density": 0.80},
    # x = 37 m → missing tree
    {"x": 41.0, "height": 1.5, "width": 1.0, "density": 0.45},
    {"x": 45.0, "height": 3.5, "width": 1.7, "density": 0.90},
    {"x": 49.0, "height": 2.6, "width": 1.3, "density": 0.70},
]

print("\nOrchard:")
for i, tree in enumerate(trees, start=1):
    print(
        f"T{i}: "
        f"x={tree['x']:.1f} m, "
        f"height={tree['height']:.1f} m, "
        f"width={tree['width']:.1f} m,"
        f"density={tree['density']:.2f} ,"
        
    )