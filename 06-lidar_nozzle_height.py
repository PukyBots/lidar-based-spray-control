import numpy as np
import math
import matplotlib.pyplot as plt



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
    {"x": 13.0, "height": 1.8, "width": 1.0, "density": 0.35},
    {"x": 17.0, "height": 2.8, "width": 1.4, "density": 0.75},
    {"x": 21.0, "height": 3.4, "width": 1.6, "density": 0.95},

    # Missing tree position
    {"x": 25.0, "height": 0.0, "width": 0.0, "density": 0.0},

    {"x": 29.0, "height": 2.2, "width": 1.2, "density": 0.60},
    {"x": 33.0, "height": 3.0, "width": 1.5, "density": 0.80},

    # Missing tree position
    {"x": 37.0, "height": 0.0, "width": 0.0, "density": 0.0},

    {"x": 41.0, "height": 1.5, "width": 1.0, "density": 0.45},
    {"x": 45.0, "height": 3.5, "width": 1.7, "density": 0.90},
    {"x": 49.0, "height": 2.6, "width": 1.3, "density": 0.70},
]


def generate_canopy(tree):
    """Generate simulated canopy points for one tree."""

    x_center = tree["x"]
    height = tree["height"]
    width = tree["width"]

    # No points for a missing tree
    if height == 0 or width == 0:
        return np.empty((0, 3))

    # 12 vertical levels
    z_levels = np.linspace(1.0, height, 12)

    canopy_points = []

    for z in z_levels:

        # Canopy becomes narrower toward the top
        relative_height = (z - 1.0) / (height - 1.0)

        width_factor = 1.0 - 0.45 * relative_height

        current_width = width * width_factor

        # 8 points across the canopy width
        y_positions = np.linspace(
            0.5,
            0.5 + current_width,
            8
        )

        for y in y_positions:

            # Small random irregularity along X
            x_irregularity = np.random.uniform(-0.08, 0.08)

            canopy_points.append(
                (
                    x_center + x_irregularity,
                    y,
                    z
                )
            )

    return np.array(canopy_points)


# Generate canopy for all 12 positions

all_canopy_points = []

for i, tree in enumerate(trees, start=1):

    canopy = generate_canopy(tree)

    if len(canopy) > 0:
        all_canopy_points.append(canopy)

    print(
        f"Position {i}: "
        f"x={tree['x']:.1f} m, "
        f"height={tree['height']:.1f} m, "
        f"width={tree['width']:.1f} m, "
        f"density={tree['density']:.2f}, "
        f"canopy points={len(canopy)}"
    )


# Combine all trees into one canopy point cloud

all_canopy_points = np.vstack(all_canopy_points)

print("\nTotal canopy points:", len(all_canopy_points))

plt.figure(figsize=(14, 5))

plt.scatter(
    all_canopy_points[:, 0],
    all_canopy_points[:, 2],
    s=8
)

plt.xlabel("Distance along row X (m)")
plt.ylabel("Height Z (m)")
plt.title("Simulated Orchard Canopy")

plt.xlim(0, 55)
plt.ylim(0, 4)

plt.grid(True)
plt.show()

# Test one LiDAR beam

test_angle_deg = 30.0
test_range_m = 2.0

theta = np.deg2rad(test_angle_deg)

# LiDAR local coordinates
y = test_range_m * np.cos(theta)
z = LIDAR_HEIGHT_M + test_range_m * np.sin(theta)

print("\nTest LiDAR beam")
print("Angle:", test_angle_deg, "degrees")
print("Range:", test_range_m, "m")
print("Beam endpoint:")
print("Y =", y, "m")
print("Z =", z, "m")

# Check whether the test beam is close to the first tree canopy

tree_points = generate_canopy(trees[0])

beam_angle_rad = np.deg2rad(test_angle_deg)

# Direction of the beam in the Y-Z plane
beam_direction = np.array([
    np.cos(beam_angle_rad),
    np.sin(beam_angle_rad)
])

# Convert canopy points to coordinates relative to LiDAR
points_yz = tree_points[:, [1, 2]]
points_yz[:, 1] = points_yz[:, 1] - LIDAR_HEIGHT_M

# Distance of each canopy point along the beam
along_beam = points_yz @ beam_direction

print("\nBeam check:")
print("Number of canopy points:", len(tree_points))
print("Closest distance along beam:", np.min(along_beam))

# --------------------------------------------------
# LiDAR scan geometry for all angles
# --------------------------------------------------

scan_ranges = np.full(len(beam_angles_deg), LIDAR_MAX_RANGE_M)

angles_rad = np.deg2rad(beam_angles_deg)

scan_y = scan_ranges * np.cos(angles_rad)
scan_z = LIDAR_HEIGHT_M + scan_ranges * np.sin(angles_rad)

print("\nLiDAR full scan:")
print("Number of scan beams:", len(beam_angles_deg))
print("Y range:", scan_y.min(), "to", scan_y.max())
print("Z range:", scan_z.min(), "to", scan_z.max())

# Current LiDAR position along the travel direction
lidar_x = 5.0

print("\nCurrent LiDAR position:")
print("X =", lidar_x, "m")

def simulate_lidar_scan(lidar_x, canopy_points):

    ranges = np.full(len(beam_angles_deg), -1.0)

    for i, beam_angle_deg in enumerate(beam_angles_deg):

        # Select canopy points close to current LiDAR X position
        x_difference = np.abs(canopy_points[:, 0] - lidar_x)
        nearby_points = canopy_points[x_difference < 0.10]

        if len(nearby_points) == 0:
            continue

        # Coordinates relative to LiDAR
        y = nearby_points[:, 1]
        z = nearby_points[:, 2] - LIDAR_HEIGHT_M

        # Actual angle from LiDAR to each canopy point
        point_angles = np.rad2deg(np.arctan2(z, y))

        # Difference between point angle and current beam angle
        angle_difference = np.abs(
            point_angles - beam_angle_deg
        )

        # Points belonging to this beam
        beam_points = nearby_points[
            angle_difference <= LIDAR_ANGLE_RES_DEG / 2
        ]

        if len(beam_points) == 0:
            continue

        # Calculate range to these points
        y_beam = beam_points[:, 1]
        z_beam = beam_points[:, 2] - LIDAR_HEIGHT_M

        distances = np.sqrt(
            y_beam**2 + z_beam**2
        )

        # LiDAR returns closest point
        ranges[i] = np.min(distances)

    return ranges

lidar_ranges = simulate_lidar_scan(
    lidar_x,
    all_canopy_points
)

print("\nSimulated LiDAR scan:")
print("Total beams:", len(lidar_ranges))
print("Detected returns:", np.sum(lidar_ranges > 0))
print("No-return beams:", np.sum(lidar_ranges < 0))


# --------------------------------------------------
# Convert LiDAR returns to detected canopy heights
# --------------------------------------------------

detected_heights = []

for i, range_m in enumerate(lidar_ranges):

    # Ignore beams with no return
    if range_m < 0:
        continue

    angle_rad = np.deg2rad(beam_angles_deg[i])

    # Height of detected point
    z = LIDAR_HEIGHT_M + range_m * np.sin(angle_rad)

    detected_heights.append(z)

detected_heights = np.array(detected_heights)

print("\nDetected canopy heights:")
print("Number of detected points:", len(detected_heights))
print("Minimum height:", detected_heights.min())
print("Maximum height:", detected_heights.max())


# --------------------------------------------------
# Assign LiDAR returns to nozzle zones
# --------------------------------------------------

zone_counts = np.zeros(NOZZLE_COUNT, dtype=int)

for z in detected_heights:

    for i, (z_min, z_max) in enumerate(nozzle_zones):

        if z_min <= z < z_max:
            zone_counts[i] += 1
            break

print("\nLiDAR returns per nozzle zone:")

for i, count in enumerate(zone_counts, start=1):
    print(f"N{i}: {count} returns")