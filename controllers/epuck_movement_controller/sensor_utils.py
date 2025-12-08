import config


def filter_wall_measurements(raw_measurements):
    if len(raw_measurements) < 3:
        return raw_measurements

    filtered = []
    MIN_ANGLE_DIFF = 0.2  # Minimum angle difference between landmarks (radians)
    MIN_DIST_DIFF = 0.1   # Minimum distance difference to be different objects (meters)
    MAX_WALL_DIST = config.WALL_MAX_DISTANCE    # Maximum distance from config

    # Group consecutive measurements that might be the same surface
    i = 0
    while i < len(raw_measurements):
        current_dist, current_angle, current_sig = raw_measurements[i]

        # Skip if this is likely a wall (very close and multiple consecutive hits)
        consecutive_count = 1
        j = i + 1

        # Check how many consecutive measurements are at similar distance/angle
        while j < len(raw_measurements):
            next_dist, next_angle, next_sig = raw_measurements[j]

            # Check if measurements are consecutive and similar (likely same wall)
            angle_diff = abs(next_angle - current_angle)
            dist_diff = abs(next_dist - current_dist)

            if angle_diff < MIN_ANGLE_DIFF and dist_diff < MIN_DIST_DIFF:
                consecutive_count += 1
                j += 1
            else:
                break

        # If we have 3+ consecutive similar measurements and they're close, it's likely a wall
        if consecutive_count >= 3 and current_dist < MAX_WALL_DIST:
            # Skip all these wall measurements
            i = j
        else:
            # Keep this measurement (likely a distinct landmark)
            filtered.append(raw_measurements[i])
            i += 1

    return filtered