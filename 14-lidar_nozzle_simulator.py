import csv
import math
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# LiDAR-Based Nozzle Control Simulator
# Technical Task — LiDAR-Based Nozzle Control
# One-side orchard sprayer model
# ============================================================

# -------------------------
# 1. LiDAR / geometry
# -------------------------
LIDAR_FOV_DEG = 270.0
LIDAR_RES_DEG = 0.25
LIDAR_RATE_HZ = 20.0
LIDAR_HEIGHT_M = 2.0
LIDAR_LATERAL_OFFSET_M = 1.0       # nominal design value; tune to actual machine
LIDAR_MAX_RANGE_M = 25.0
LIDAR_RANGE_NOISE_STD_M = 0.02     # ±2 cm accuracy model
LIDAR_X_SLICE_HALF_WIDTH_M = 0.12
LIDAR_NOISE_SEED = 7

# LiDAR is 3 m ahead of the nozzle plane along +X.
LIDAR_TO_NOZZLE_OFFSET_M = 3.0

# -------------------------
# 2. Nozzles
# -------------------------
NOZZLE_COUNT = 6
NOZZLE_PITCH_M = 0.40
LOWEST_NOZZLE_HEIGHT_M = 0.50

nozzle_heights = np.array([
    LOWEST_NOZZLE_HEIGHT_M + i * NOZZLE_PITCH_M
    for i in range(NOZZLE_COUNT)
])

nozzle_zones = [
    (h - NOZZLE_PITCH_M / 2.0, h + NOZZLE_PITCH_M / 2.0)
    for h in nozzle_heights
]

# -------------------------
# 3. Encoder
# -------------------------
WHEEL_DIAMETER_M = 0.80
ENCODER_PPR = 6
WHEEL_CIRCUMFERENCE_M = math.pi * WHEEL_DIAMETER_M
ENCODER_DISTANCE_PER_PULSE_M = WHEEL_CIRCUMFERENCE_M / ENCODER_PPR

# -------------------------
# 4. Spray-control assumptions
#    These are simulator assumptions, not machine specifications.
# -------------------------
TARGET_APPLICATION_L_PER_M3 = 0.085
CANOPY_DEPTH_M = 0.25
ZONE_HEIGHT_M = NOZZLE_PITCH_M
NOZZLE_MAX_FLOW_LPM = 0.80
MIN_DUTY_PERCENT = 15.0

# -------------------------
# 5. Simulation
# -------------------------
DT = 1.0 / LIDAR_RATE_HZ
END_X_M = 55.0
MAX_TIME_S = 120.0

# Scenario:
# 12 positions total, including 2 missing trees.
TREES = [
    {"name": "T1",  "x": 5.0,  "height": 2.4, "width": 1.2, "density": 0.85},
    {"name": "T2",  "x": 9.0,  "height": 3.1, "width": 1.5, "density": 0.90},
    {"name": "T3",  "x": 13.0, "height": 1.8, "width": 1.0, "density": 0.35},
    {"name": "T4",  "x": 17.0, "height": 2.8, "width": 1.4, "density": 0.75},
    {"name": "T5",  "x": 21.0, "height": 3.4, "width": 1.6, "density": 0.95},
    {"name": "GAP1", "x": 25.0, "height": 0.0, "width": 0.0, "density": 0.0},
    {"name": "T6",  "x": 29.0, "height": 2.2, "width": 1.2, "density": 0.60},
    {"name": "T7",  "x": 33.0, "height": 3.0, "width": 1.5, "density": 0.80},
    {"name": "GAP2", "x": 37.0, "height": 0.0, "width": 0.0, "density": 0.0},
    {"name": "T8",  "x": 41.0, "height": 1.5, "width": 1.0, "density": 0.45},
    {"name": "T9",  "x": 45.0, "height": 3.5, "width": 1.7, "density": 0.90},
    {"name": "T10", "x": 49.0, "height": 2.6, "width": 1.3, "density": 0.70},
]

# Scan angles: useful 270° sector, blind 90° sector directed away from canopy.
half_fov = LIDAR_FOV_DEG / 2.0
beam_angles_deg = np.arange(
    -half_fov,
    half_fov + LIDAR_RES_DEG * 0.5,
    LIDAR_RES_DEG,
)
beam_angles_rad = np.deg2rad(beam_angles_deg)

rng = np.random.default_rng(LIDAR_NOISE_SEED)


def generate_canopy_points():
    """Generate 96 base points/tree = 960 points for 10 real trees.

    Coordinates are LiDAR-relative in Y-Z for the scan calculation.
    X is the longitudinal orchard coordinate.
    """
    points = []
    for tree in TREES:
        if tree["height"] <= 0:
            continue

        z_levels = np.linspace(1.0, tree["height"], 12)
        for z in z_levels:
            rel_h = (z - 1.0) / max(tree["height"] - 1.0, 1e-6)
            width_factor = 1.0 - 0.45 * rel_h
            current_width = tree["width"] * width_factor

            y_positions = np.linspace(0.50, 0.50 + current_width, 8)
            for y in y_positions:
                x_jitter = rng.uniform(-0.08, 0.08)
                y_jitter = rng.uniform(-0.035, 0.035)
                z_jitter = rng.uniform(-0.025, 0.025)
                points.append([
                    tree["x"] + x_jitter,
                    y + y_jitter,
                    max(1.0, z + z_jitter),
                    tree["density"],
                    tree["name"],
                ])

    return points


CANOPY_POINTS = generate_canopy_points()


def simulate_lidar_scan(lidar_x):
    """Return one single-return scan.

    Each beam selects the closest visible canopy point in a thin X slice.
    Tree density controls probability of a foliage return. Range noise is
    added at the end. Intensity is simulated for valid returns.
    """
    ranges = np.full(len(beam_angles_deg), np.nan, dtype=float)
    intensities = np.full(len(beam_angles_deg), np.nan, dtype=float)

    candidates = []

    for p in CANOPY_POINTS:
        px, py, pz, density, tree_name = p
        if abs(px - lidar_x) > LIDAR_X_SLICE_HALF_WIDTH_M:
            continue

        dz = pz - LIDAR_HEIGHT_M
        if py <= 0:
            continue

        r = math.hypot(py, dz)
        if r <= 0 or r > LIDAR_MAX_RANGE_M:
            continue

        angle = math.degrees(math.atan2(dz, py))
        if angle < -half_fov or angle > half_fov:
            continue

        beam_idx = int(round((angle + half_fov) / LIDAR_RES_DEG))
        if beam_idx < 0 or beam_idx >= len(beam_angles_deg):
            continue

        angle_error = abs(beam_angles_deg[beam_idx] - angle)
        if angle_error > LIDAR_RES_DEG / 2.0:
            continue

        # Sparse canopy produces fewer returns.
        if rng.random() > density:
            continue

        candidates.append((beam_idx, r, density, tree_name))

    # Single-return LiDAR: nearest return wins for each beam.
    nearest = {}
    for beam_idx, r, density, tree_name in candidates:
        if beam_idx not in nearest or r < nearest[beam_idx][0]:
            nearest[beam_idx] = (r, density, tree_name)

    for beam_idx, (r, density, _) in nearest.items():
        noisy_r = max(0.05, r + rng.normal(0.0, LIDAR_RANGE_NOISE_STD_M))
        ranges[beam_idx] = noisy_r
        intensities[beam_idx] = np.clip(
            0.35 + 0.60 * density + rng.normal(0.0, 0.04), 0.0, 1.0
        )

    return ranges, intensities


def beam_point(range_m, angle_deg):
    """Convert LiDAR polar return to Y-Z point in scan plane."""
    a = math.radians(angle_deg)
    y = range_m * math.cos(a)
    z = LIDAR_HEIGHT_M + range_m * math.sin(a)
    return y, z


def representative_zone_beam_indices():
    """Approximate observable angular sector per nozzle zone.

    Uses a 2 m reference range. This gives a stable denominator for the
    density estimator and avoids counting the whole 270° FOV.
    """
    ref_range = 2.0
    result = {}

    for i, (zmin, zmax) in enumerate(nozzle_zones, start=1):
        theta1 = math.degrees(math.atan2(zmin - LIDAR_HEIGHT_M, ref_range))
        theta2 = math.degrees(math.atan2(zmax - LIDAR_HEIGHT_M, ref_range))
        low, high = sorted((theta1, theta2))

        idx = np.where(
            (beam_angles_deg >= low) & (beam_angles_deg <= high)
        )[0]
        result[i] = idx

    return result


ZONE_BEAM_INDICES = representative_zone_beam_indices()


def map_scan_to_zones(ranges, intensities):
    """Map valid returns to the six nozzle height zones."""
    zone_returns = {i: [] for i in range(1, NOZZLE_COUNT + 1)}
    upper_unassigned = 0

    for i, r in enumerate(ranges):
        if not np.isfinite(r):
            continue

        y, z = beam_point(r, beam_angles_deg[i])

        assigned = False
        for nozzle_idx, (zmin, zmax) in enumerate(nozzle_zones, start=1):
            if zmin <= z < zmax:
                zone_returns[nozzle_idx].append({
                    "range": float(r),
                    "angle": float(beam_angles_deg[i]),
                    "y": float(y),
                    "z": float(z),
                    "intensity": float(intensities[i]),
                })
                assigned = True
                break

        if not assigned and z >= nozzle_zones[-1][1]:
            upper_unassigned += 1

    return zone_returns, upper_unassigned


def calculate_densities(zone_returns):
    """Density = canopy returns / usable beams in each zone sector."""
    densities = {}
    for nozzle_idx in range(1, NOZZLE_COUNT + 1):
        denominator = max(len(ZONE_BEAM_INDICES[nozzle_idx]), 1)
        numerator = len(zone_returns[nozzle_idx])
        densities[nozzle_idx] = min(1.0, numerator / denominator)
    return densities


def spray_command(density, speed_mps):
    """Convert canopy density into flow and PWM."""
    if speed_mps <= 0 or density <= 0:
        return 0.0, 0.0

    travel_m_per_min = speed_mps * 60.0
    required_flow_lpm = (
        TARGET_APPLICATION_L_PER_M3
        * CANOPY_DEPTH_M
        * ZONE_HEIGHT_M
        * travel_m_per_min
        * density
    )

    raw_pwm = 100.0 * required_flow_lpm / NOZZLE_MAX_FLOW_LPM
    if raw_pwm <= 0:
        pwm = 0.0
    else:
        pwm = max(MIN_DUTY_PERCENT, raw_pwm)
        pwm = min(100.0, pwm)

    return required_flow_lpm, pwm


def vehicle_speed(t):
    """Speed profile: 1.0 m/s -> 0.5 m/s -> 25 s stop -> resume."""
    if t < 20.0:
        return 1.0
    if t < 35.0:
        return 0.5
    if t < 60.0:
        return 0.0
    return 1.0


def simulate_vehicle_and_control():
    """Run the complete time-based LiDAR/encoder/registration simulation."""
    t = 0.0
    true_x = 0.0
    encoder_pulses = 0
    encoder_x = 0.0

    # Registration buffer: scans wait 3 m before their data is used.
    scan_buffer = deque()
    latest_scan_at_x = None

    history = []
    released_scans = []
    capture_count = 0
    skipped_stationary_scans = 0

    while t <= MAX_TIME_S and true_x < END_X_M:
        speed = vehicle_speed(t)
        true_x += speed * DT

        # Encoder quantization.
        exact_pulses = true_x / ENCODER_DISTANCE_PER_PULSE_M
        new_pulses = int(math.floor(exact_pulses))
        encoder_pulses = max(encoder_pulses, new_pulses)
        encoder_x = encoder_pulses * ENCODER_DISTANCE_PER_PULSE_M

        # LiDAR scan at 20 Hz.
        ranges, intensities = simulate_lidar_scan(true_x)
        capture_count += 1

        # During a stop, scans contain no new spatial information. Keep the
        # latest scan for the same spatial position rather than filling the
        # registration FIFO with duplicates.
        if latest_scan_at_x is None or abs(true_x - latest_scan_at_x) >= 0.01:
            scan_buffer.append({
                "capture_time": t,
                "capture_true_x": true_x,
                "capture_encoder_x": encoder_x,
                "ranges": ranges,
                "intensities": intensities,
            })
            latest_scan_at_x = true_x
        else:
            skipped_stationary_scans += 1

        # Distance-based registration using encoder distance.
        # A scan captured at X is used when encoder has advanced >= 3 m.
        while scan_buffer:
            scan = scan_buffer[0]
            advance = encoder_x - scan["capture_encoder_x"]
            if advance + 1e-9 < LIDAR_TO_NOZZLE_OFFSET_M:
                break

            scan_buffer.popleft()

            zone_returns, upper_unassigned = map_scan_to_zones(
                scan["ranges"], scan["intensities"]
            )
            densities = calculate_densities(zone_returns)

            command = {}
            for nozzle_idx in range(1, NOZZLE_COUNT + 1):
                flow, pwm = spray_command(densities[nozzle_idx], speed)
                command[nozzle_idx] = {
                    "density": densities[nozzle_idx],
                    "flow_lpm": flow,
                    "pwm_percent": pwm,
                }

            released = {
                "capture_time": scan["capture_time"],
                "capture_x": scan["capture_true_x"],
                "release_time": t,
                "release_true_x": true_x,
                "release_encoder_x": encoder_x,
                "registration_advance": advance,
                "speed": speed,
                "densities": densities,
                "commands": command,
                "upper_unassigned": upper_unassigned,
            }
            released_scans.append(released)

            # Record one row per nozzle.
            for nozzle_idx in range(1, NOZZLE_COUNT + 1):
                row = {
                    "time_s": t,
                    "capture_x_m": scan["capture_true_x"],
                    "release_x_m": true_x,
                    "encoder_x_m": encoder_x,
                    "registration_m": advance,
                    "speed_mps": speed,
                    "nozzle": nozzle_idx,
                    "nozzle_height_m": nozzle_heights[nozzle_idx - 1],
                    "zone_min_m": nozzle_zones[nozzle_idx - 1][0],
                    "zone_max_m": nozzle_zones[nozzle_idx - 1][1],
                    "density": command[nozzle_idx]["density"],
                    "flow_lpm": command[nozzle_idx]["flow_lpm"],
                    "pwm_percent": command[nozzle_idx]["pwm_percent"],
                }
                history.append(row)

        t += DT

    return history, released_scans, {
        "capture_count": capture_count,
        "skipped_stationary_scans": skipped_stationary_scans,
        "remaining_buffer": len(scan_buffer),
        "final_true_x": true_x,
        "final_encoder_x": encoder_x,
        "final_encoder_pulses": encoder_pulses,
    }


def save_csv(history, filename="nozzle_commands.csv"):
    if not history:
        return
    fields = list(history[0].keys())
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def plot_results(history, released_scans):
    if not history:
        return

    # Convert history to arrays grouped by nozzle.
    x = np.array([r["release_x_m"] for r in history])
    nozzle = np.array([r["nozzle"] for r in history])
    density = np.array([r["density"] for r in history])
    pwm = np.array([r["pwm_percent"] for r in history])
    speed = np.array([r["speed_mps"] for r in history])
    time = np.array([r["time_s"] for r in history])

    # 1. Orchard profile.
    fig, ax = plt.subplots(figsize=(12, 5))
    for tree in TREES:
        if tree["height"] <= 0:
            ax.axvline(tree["x"], linestyle="--", alpha=0.35)
            continue
        x0 = tree["x"]
        w = tree["width"]
        h = tree["height"]
        ax.fill_between(
            [x0 - w / 2, x0 + w / 2],
            [1.0, 1.0],
            [h, h],
            alpha=0.25,
        )
        ax.text(x0, h + 0.08, tree["name"], ha="center")
    for i, (zmin, zmax) in enumerate(nozzle_zones, 1):
        ax.axhspan(zmin, zmax, alpha=0.08)
        ax.text(0.2, (zmin + zmax) / 2, f"N{i}", va="center")
    ax.set_title("Orchard profile and six nozzle height zones")
    ax.set_xlabel("Travel direction X (m)")
    ax.set_ylabel("Height Z (m)")
    ax.set_xlim(0, END_X_M)
    ax.set_ylim(0, 3.8)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    # 2. Density.
    fig, ax = plt.subplots(figsize=(12, 5))
    for n in range(1, NOZZLE_COUNT + 1):
        mask = nozzle == n
        ax.plot(x[mask], density[mask], label=f"N{n}")
    for tree in TREES:
        if tree["height"] <= 0:
            ax.axvspan(tree["x"] - 0.5, tree["x"] + 0.5, alpha=0.12)
    ax.set_title("Registered canopy density per nozzle")
    ax.set_xlabel("Nozzle action position X (m)")
    ax.set_ylabel("Density (0–1)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()

    # 3. PWM.
    fig, ax = plt.subplots(figsize=(12, 5))
    for n in range(1, NOZZLE_COUNT + 1):
        mask = nozzle == n
        ax.plot(x[mask], pwm[mask], label=f"N{n}")
    ax.set_title("Nozzle PWM command after 3 m LiDAR registration")
    ax.set_xlabel("Nozzle action position X (m)")
    ax.set_ylabel("PWM (%)")
    ax.set_ylim(-2, 105)
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()

    # 4. Vehicle speed and stop.
    fig, ax = plt.subplots(figsize=(12, 4))
    unique = {}
    for tt, ss in zip(time, speed):
        unique.setdefault(round(float(tt), 6), float(ss))
    tt = np.array(list(unique.keys()))
    ss = np.array(list(unique.values()))
    ax.step(tt, ss, where="post")
    ax.axvspan(35, 60, alpha=0.12, label="25 s stop")
    ax.set_title("Vehicle speed profile")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    # 5. Example LiDAR scan near first tree.
    example_x = 5.0
    ranges, intensities = simulate_lidar_scan(example_x)
    valid = np.isfinite(ranges)
    y = ranges[valid] * np.cos(np.deg2rad(beam_angles_deg[valid]))
    z = LIDAR_HEIGHT_M + ranges[valid] * np.sin(np.deg2rad(beam_angles_deg[valid]))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, z, s=12, alpha=0.7)
    ax.axhline(LIDAR_HEIGHT_M, linestyle="--", alpha=0.5, label="LiDAR height")
    for i, (zmin, zmax) in enumerate(nozzle_zones, 1):
        ax.axhspan(zmin, zmax, alpha=0.08)
        ax.text(0.05, (zmin + zmax) / 2, f"N{i}")
    ax.set_title("Example 2D LiDAR scan at X = 5 m")
    ax.set_xlabel("Lateral Y relative to LiDAR (m)")
    ax.set_ylabel("Height Z (m)")
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 3.8)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    # 6. Registration error.
    if released_scans:
        cap = np.array([r["capture_x"] for r in released_scans])
        rel = np.array([r["release_true_x"] for r in released_scans])
        err = rel - (cap + LIDAR_TO_NOZZLE_OFFSET_M)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(rel, err * 100.0, linewidth=1.2)
        ax.axhline(0, linestyle="--", alpha=0.6)
        ax.set_title("Distance-registration error caused by 6 PPR encoder")
        ax.set_xlabel("Nozzle action position X (m)")
        ax.set_ylabel("Registration error (cm)")
        ax.grid(alpha=0.25)
        fig.tight_layout()

    plt.show()


def print_summary(history, released_scans, stats):
    print("\n" + "=" * 72)
    print("LiDAR-BASED NOZZLE CONTROL SIMULATOR")
    print("=" * 72)
    print(f"LiDAR: {LIDAR_FOV_DEG:.0f} deg FOV, {LIDAR_RES_DEG:.2f} deg, {LIDAR_RATE_HZ:.0f} Hz")
    print(f"LiDAR height: {LIDAR_HEIGHT_M:.2f} m")
    print(f"Nominal lateral offset d: {LIDAR_LATERAL_OFFSET_M:.2f} m")
    print(f"LiDAR-to-nozzle longitudinal offset: {LIDAR_TO_NOZZLE_OFFSET_M:.2f} m")
    print(f"Nozzle heights: {', '.join(f'{h:.2f}' for h in nozzle_heights)} m")
    print(f"Encoder: {ENCODER_PPR} PPR, {ENCODER_DISTANCE_PER_PULSE_M:.4f} m/pulse")
    print(f"3 m registration = {LIDAR_TO_NOZZLE_OFFSET_M / ENCODER_DISTANCE_PER_PULSE_M:.2f} pulses")
    print("\nApproximate usable angular sectors (2 m reference range):")
    for i, idx in ZONE_BEAM_INDICES.items():
        if len(idx):
            print(
                f"  N{i}: {beam_angles_deg[idx[0]]:.2f} to "
                f"{beam_angles_deg[idx[-1]]:.2f} deg, {len(idx)} beams"
            )

    print("\nSimulation result:")
    print(f"  Raw LiDAR scans at 20 Hz: {stats['capture_count']}")
    print(f"  Stationary duplicate scans suppressed: {stats['skipped_stationary_scans']}")
    print(f"  Registered scan events: {len(released_scans)}")
    print(f"  Remaining registration buffer: {stats['remaining_buffer']}")
    print(f"  Final true position: {stats['final_true_x']:.3f} m")
    print(f"  Final encoder position: {stats['final_encoder_x']:.3f} m")

    if released_scans:
        first = released_scans[0]
        print("\nFirst registered event:")
        print(f"  Scan captured at X = {first['capture_x']:.3f} m")
        print(f"  Used at X = {first['release_true_x']:.3f} m")
        print(f"  Encoder advance = {first['registration_advance']:.3f} m")
        print(f"  Speed = {first['speed']:.2f} m/s")
        for n in range(1, NOZZLE_COUNT + 1):
            c = first["commands"][n]
            print(
                f"  N{n}: density={c['density']:.3f}, "
                f"flow={c['flow_lpm']:.3f} L/min, PWM={c['pwm_percent']:.1f}%"
            )


def main():
    print(f"Generated canopy points: {len(CANOPY_POINTS)}")
    history, released_scans, stats = simulate_vehicle_and_control()
    save_csv(history, "nozzle_commands.csv")
    print_summary(history, released_scans, stats)
    print("\nSaved: nozzle_commands.csv")
    plot_results(history, released_scans)


if __name__ == "__main__":
    main()
