import config


def create_stable_occupancy_grid(state_estimate, current_x, current_y):
    # Fixed grid size from config
    # This gives us visibility from -1.5m to +1.5m around the robot
    GRID_SIZE = config.PLANNING_GRID_SIZE
    CELL_SIZE = config.GRID_CELL_SIZE  # Cell size from config

    # Initialize grid with all cells free
    grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    # Robot is always at center of the stable grid
    robot_grid_x = GRID_SIZE // 2
    robot_grid_y = GRID_SIZE // 2

    # Mark robot position (for debugging - path planner will handle this)
    grid[robot_grid_y][robot_grid_x] = 0  # Keep as free for path planning

    # Process landmarks from state estimate
    if len(state_estimate) > 3:
        i = 3
        while i < len(state_estimate):
            try:
                # Get landmark position in world coordinates
                lm_x = state_estimate[i, 0] if state_estimate.ndim > 1 else state_estimate[i]
                lm_y = state_estimate[i + 1, 0] if state_estimate.ndim > 1 else state_estimate[i + 1]

                # Convert to robot-relative coordinates
                rel_x = lm_x - current_x
                rel_y = lm_y - current_y

                # Convert to grid coordinates (robot is at center)
                grid_x = int(robot_grid_x + rel_x / CELL_SIZE)
                grid_y = int(robot_grid_y - rel_y / CELL_SIZE)  # Y inverted for grid

                # Only add if within grid bounds
                if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
                    grid[grid_y][grid_x] = 1  # Mark as obstacle
            except:
                pass
            i += 3

    # Ensure the grid is the expected size (safety check)
    if len(grid) != GRID_SIZE or any(len(row) != GRID_SIZE for row in grid):
        print(f"Warning: Stable grid size mismatch! Expected {GRID_SIZE}x{GRID_SIZE}")
        grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    # Mark the center (robot position) for path planner
    grid[GRID_SIZE // 2][GRID_SIZE // 2] = config.GRID_ORIGIN_MARKER

    return grid