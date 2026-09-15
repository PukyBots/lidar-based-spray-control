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


# --------------------------------------------------
# Simulate LiDAR scanning while vehicle moves
# --------------------------------------------------

lidar_positions = np.arange(0.0, 55.0, 0.25)

scan_results = []

for lidar_x in lidar_positions:

    ranges = simulate_lidar_scan(
        lidar_x,
        all_canopy_points
    )

    scan_results.append({
        "x": lidar_x,
        "ranges": ranges
    })

print("\nScanning simulation:")
print("Number of LiDAR positions:", len(scan_results))
print("First LiDAR position:", scan_results[0]["x"], "m")
print("Last LiDAR position:", scan_results[-1]["x"], "m")

# --------------------------------------------------
# Visualize actual LiDAR returns
# --------------------------------------------------

lidar_return_x = []
lidar_return_z = []

for scan in scan_results:

    lidar_x = scan["x"]
    ranges = scan["ranges"]

    for i, range_m in enumerate(ranges):

        if range_m < 0:
            continue

        angle_rad = np.deg2rad(beam_angles_deg[i])

        # Height of detected return
        z = LIDAR_HEIGHT_M + range_m * np.sin(angle_rad)

        lidar_return_x.append(lidar_x)
        lidar_return_z.append(z)

LIDAR_NOZZLE_OFFSET_M = 3.0

def nozzle_action_position(lidar_x):
    return lidar_x + LIDAR_NOZZLE_OFFSET_M

test_lidar_x = 5.0

action_x = nozzle_action_position(test_lidar_x)

print("\nLiDAR-to-nozzle registration:")
print("LiDAR detected canopy at X =", test_lidar_x, "m")
print("Nozzle should act at X =", action_x, "m")

scan_index = np.argmin(np.abs(lidar_positions - 5.0))

ranges = scan_results[scan_index]["ranges"]

print("\nUsing LiDAR scan at X =", scan_results[scan_index]["x"], "m")

nozzle_returns = {f"N{i}": [] for i in range(1, NOZZLE_COUNT + 1)}

for i, range_m in enumerate(ranges):

    if range_m < 0:
        continue

    angle_rad = np.deg2rad(beam_angles_deg[i])

    y = range_m * np.cos(angle_rad)
    z = LIDAR_HEIGHT_M + range_m * np.sin(angle_rad)

    for nozzle_index, (z_min, z_max) in enumerate(nozzle_zones):

        if z_min <= z < z_max:

            nozzle_returns[f"N{nozzle_index + 1}"].append({
                "angle": beam_angles_deg[i],
                "range": range_m,
                "y": y,
                "z": z
            })

            break

registered_nozzle_data = {}

for nozzle, returns in nozzle_returns.items():

    registered_nozzle_data[nozzle] = {
        "lidar_x": test_lidar_x,
        "nozzle_x": test_lidar_x + LIDAR_NOZZLE_OFFSET_M,
        "return_count": len(returns)
    }


print("\nRegistered LiDAR → nozzle data:")

for nozzle, data in registered_nozzle_data.items():

    print(
        f"{nozzle}: "
        f"LiDAR X={data['lidar_x']:.2f} m, "
        f"Nozzle X={data['nozzle_x']:.2f} m, "
        f"Returns={data['return_count']}"
    )

registration_buffer = []

for scan in scan_results:

    lidar_x = scan["x"]

    registration_buffer.append({
        "lidar_x": lidar_x,
        "nozzle_x": lidar_x + LIDAR_NOZZLE_OFFSET_M,
        "ranges": scan["ranges"]
    })

print("\nRegistration buffer:")
print("Number of stored scans:", len(registration_buffer))

for item in registration_buffer[:5]:
    print(
        f"LiDAR X={item['lidar_x']:.2f} m "
        f"→ Nozzle X={item['nozzle_x']:.2f} m"
    )

WHEEL_DIAMETER_M = 0.8
ENCODER_PPR = 6

wheel_circumference = np.pi * WHEEL_DIAMETER_M
distance_per_pulse = wheel_circumference / ENCODER_PPR

print("\nEncoder:")
print("Wheel circumference:", wheel_circumference, "m")
print("Distance per pulse:", distance_per_pulse, "m")

LIDAR_NOZZLE_OFFSET_M = 3.0

required_pulses = LIDAR_NOZZLE_OFFSET_M / distance_per_pulse

print("Encoder pulses required for 3 m:",
      required_pulses)


def vehicle_speed(t):
    if t < 20:
        return 1.0          # normal speed
    elif t < 35:
        return 0.5          # slower speed
    elif t < 60:
        return 0.0          # stopped for 25 seconds
    else:
        return 1.0          # resume
    
time = 0.0
simulation_dt = 0.1
# Encoder state
encoder_pulses = 0
encoder_distance = 0.0

# Simulate vehicle movement
vehicle_x = 0.0

# Next encoder pulse occurs after this much true travel
next_pulse_distance = distance_per_pulse

encoder_history = []

while vehicle_x <= 55.0:

    speed = vehicle_speed(time)

    # True vehicle movement
    vehicle_x += speed * simulation_dt

    # Generate encoder pulses whenever enough distance has travelled
    while vehicle_x >= next_pulse_distance:

        encoder_pulses += 1

        encoder_distance = (
            encoder_pulses * distance_per_pulse
        )

        next_pulse_distance += distance_per_pulse

    encoder_history.append({
        "time": time,
        "true_x": vehicle_x,
        "encoder_pulses": encoder_pulses,
        "encoder_x": encoder_distance
    })

    time += simulation_dt

print("\nEncoder simulation:")

print("Total encoder pulses:", encoder_pulses)
print("Encoder distance:", round(encoder_distance, 3), "m")
print("True vehicle distance:", round(vehicle_x, 3), "m")


registration_events = []

for item in registration_buffer:

    lidar_x = item["lidar_x"]

    required_encoder_distance = lidar_x + LIDAR_NOZZLE_OFFSET_M

    for state in encoder_history:

        if state["encoder_x"] >= required_encoder_distance:

            registration_events.append({
                "lidar_x": lidar_x,
                "encoder_x": state["encoder_x"],
                "true_x": state["true_x"],
                "time": state["time"]
            })

            break


print("\nEncoder-based LiDAR → nozzle registration:")

for event in registration_events[:10]:

    print(
        f"LiDAR X={event['lidar_x']:.2f} m | "
        f"Encoder X={event['encoder_x']:.3f} m | "
        f"True X={event['true_x']:.3f} m | "
        f"Time={event['time']:.1f} s"
    )

for scan in scan_results:
    lidar_x = scan["x"]

    # Find the encoder state when LiDAR was at this position
    encoder_state = next(
        state for state in encoder_history
        if state["true_x"] >= lidar_x
    )

    scan["encoder_x_at_capture"] = encoder_state["encoder_x"]
    scan["encoder_pulses_at_capture"] = encoder_state["encoder_pulses"]

for scan in scan_results[:10]:
    print(
        f"LiDAR X = {scan['x']:.2f} m | "
        f"Encoder X = {scan['encoder_x_at_capture']:.3f} m | "
        f"Pulses = {scan['encoder_pulses_at_capture']}"
    )

registration_events = []

for scan in scan_results:
    capture_encoder_x = scan["encoder_x_at_capture"]

    # Find the first encoder state that is 3 m ahead
    for state in encoder_history:
        encoder_advance = state["encoder_x"] - capture_encoder_x

        if encoder_advance >= LIDAR_NOZZLE_OFFSET_M:
            registration_events.append({
                "lidar_x": scan["x"],
                "capture_encoder_x": capture_encoder_x,
                "release_encoder_x": state["encoder_x"],
                "true_x_at_release": state["true_x"],
                "release_time": state["time"],
                "encoder_advance": encoder_advance
            })
            break

for event in registration_events[:10]:
    print(
        f"LiDAR X={event['lidar_x']:.2f} m | "
        f"Capture encoder={event['capture_encoder_x']:.3f} m | "
        f"Release encoder={event['release_encoder_x']:.3f} m | "
        f"Advance={event['encoder_advance']:.3f} m | "
        f"True X={event['true_x_at_release']:.3f} m | "
        f"Time={event['release_time']:.1f} s"
    )

stop_samples = [
    state for state in encoder_history
    if 35.0 <= state["time"] <= 60.0
]

print("\nEncoder during 25-second stop:")

for state in stop_samples[::20]:
    print(
        f"Time={state['time']:.1f} s | "
        f"True X={state['true_x']:.3f} m | "
        f"Encoder X={state['encoder_x']:.3f} m | "
        f"Pulses={state['encoder_pulses']}"
    )

registration_queue = []
released_scans = []

for scan in scan_results:

    # Add newly captured LiDAR scan to the queue
    registration_queue.append({
        "lidar_x": scan["x"],
        "capture_encoder_x": scan["encoder_x_at_capture"],
        "ranges": scan["ranges"]
    })

    # Check encoder history for when this scan becomes due
    for state in encoder_history:

        encoder_advance = (
            state["encoder_x"]
            - scan["encoder_x_at_capture"]
        )

        if encoder_advance >= LIDAR_NOZZLE_OFFSET_M:

            released_scans.append({
                "lidar_x": scan["x"],
                "nozzle_x": state["true_x"],
                "release_time": state["time"],
                "encoder_advance": encoder_advance
            })

            break

for scan in released_scans[:10]:
    print(
        f"LiDAR X={scan['lidar_x']:.2f} m | "
        f"Nozzle X={scan['nozzle_x']:.3f} m | "
        f"Time={scan['release_time']:.1f} s | "
        f"Encoder advance={scan['encoder_advance']:.3f} m"
    )

registration_queue = []
released_scans = []

print("\nReal-time registration simulation:")

for state in encoder_history:

    current_encoder_x = state["encoder_x"]

    # Add LiDAR scans captured at this vehicle position
    for scan in scan_results:
        if (
            "queued" not in scan
            and scan["x"] <= state["true_x"]
        ):
            registration_queue.append(scan)
            scan["queued"] = True

    # Check the oldest queued scan
    while registration_queue:

        scan = registration_queue[0]

        encoder_advance = (
            current_encoder_x
            - scan["encoder_x_at_capture"]
        )

        if encoder_advance >= LIDAR_NOZZLE_OFFSET_M:

            released_scans.append({
                "lidar_x": scan["x"],
                "release_time": state["time"],
                "nozzle_x": state["true_x"],
                "encoder_advance": encoder_advance
            })

            registration_queue.pop(0)

        else:
            break

print(f"Total released scans: {len(released_scans)}")
print(f"Scans still waiting: {len(registration_queue)}")

registered_zone_returns = {
    f"N{i}": []
    for i in range(1, NOZZLE_COUNT + 1)
}

for i, range_m in enumerate(ranges):

    if range_m < 0:
        continue

    angle_rad = np.deg2rad(beam_angles_deg[i])

    y = range_m * np.cos(angle_rad)
    z = LIDAR_HEIGHT_M + range_m * np.sin(angle_rad)

    for nozzle_index, (z_min, z_max) in enumerate(nozzle_zones):

        if z_min <= z < z_max:

            registered_zone_returns[
                f"N{nozzle_index + 1}"
            ].append({
                "angle": beam_angles_deg[i],
                "range": range_m,
                "y": y,
                "z": z
            })

            break

for nozzle, returns in registered_zone_returns.items():
    print(
        f"{nozzle}: {len(returns)} valid canopy returns"
    )

zone_return_counts = {}

for nozzle, returns in registered_zone_returns.items():
    zone_return_counts[nozzle] = len(returns)

print("\nCanopy returns per nozzle zone:")

for nozzle, count in zone_return_counts.items():
    print(f"{nozzle}: {count}")

usable_beams = {
    f"N{i}": 0
    for i in range(1, NOZZLE_COUNT + 1)
}

for angle_deg in beam_angles_deg:

    angle_rad = np.deg2rad(angle_deg)

    # Ignore beams pointing below the LiDAR plane
    # for the initial observable-zone calculation.
    if np.sin(angle_rad) <= 0:
        continue

    # Vertical position reached at maximum LiDAR range
    z_at_max_range = (
        LIDAR_HEIGHT_M
        + LIDAR_MAX_RANGE_M * np.sin(angle_rad)
    )

    # Check whether the beam can pass through each nozzle zone
    for nozzle_index, (z_min, z_max) in enumerate(nozzle_zones):

        if z_at_max_range >= z_min:

            usable_beams[f"N{nozzle_index + 1}"] += 1

print("\nObservable beams per nozzle zone:")

for nozzle, count in usable_beams.items():
    print(f"{nozzle}: {count}")