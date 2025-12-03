import config


def create_stable_occupancy_grid(state_estimate, current_x, current_y):
    GRID_SIZE = config.PLANNING_GRID_SIZE
    CELL_SIZE = config.GRID_CELL_SIZE

    # Initialize grid with all cells free
    grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    # Robot is always at center of the grid
    center = GRID_SIZE // 2

    # Process landmarks from state estimate
    # Landmarks start at index 3 in state_estimate: [x, y, theta, lm1_x, lm1_y, lm1_sig, lm2_x, ...]
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

                # Convert to grid coordinates
                # Grid X: center + rel_x/cell_size (positive rel_x = right = higher column)
                # Grid Y: center - rel_y/cell_size (positive rel_y = forward = LOWER row, Y inverted)
                grid_col = int(center + rel_x / CELL_SIZE)
                grid_row = int(center - rel_y / CELL_SIZE)

                # Only add if within grid bounds
                if 0 <= grid_col < GRID_SIZE and 0 <= grid_row < GRID_SIZE:
                    grid[grid_row][grid_col] = 1  # Mark as obstacle
                    # Note: Safety buffer is applied by A* planner's _create_buffered_grid()
            except Exception:
                pass
            i += 3

    # Mark the center (robot position) with origin marker
    # This tells the A* planner where (0,0) in relative coordinates is
    grid[center][center] = config.GRID_ORIGIN_MARKER

    return grid