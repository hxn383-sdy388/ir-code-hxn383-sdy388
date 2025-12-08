import numpy as np
import math
import config

class SmartDisplayGrid:

    def __init__(self, display_size=None, base_cell_size=None):
        self.display_size = display_size if display_size is not None else config.DISPLAY_GRID_SIZE
        self.base_cell_size = base_cell_size if base_cell_size is not None else config.GRID_CELL_SIZE

        # Zoom presets for different scenarios from config
        self.zoom_levels = {
            'detail': config.DISPLAY_ZOOM_DETAIL,    # High detail for nearby objects
            'normal': config.DISPLAY_ZOOM_NORMAL,     # Standard view
            'wide': config.DISPLAY_ZOOM_WIDE,       # Medium range view
            'overview': config.DISPLAY_ZOOM_OVERVIEW    # Maximum overview
        }

        # Visual encoding - Extended for complete visualization
        self.EMPTY = 0
        self.OBSTACLE = 1
        self.ORIGIN = 2
        self.ROBOT = 3
        self.PATH = 4
        self.LANDMARK_NEAR = 5
        self.LANDMARK_FAR = 6
        self.WALL = 7
        self.BUFFER_ZONE = 8         # Safety buffer around obstacles
        self.WAYPOINT = 9             # Path waypoints
        self.CURRENT_WAYPOINT = 10    # Current target waypoint
        self.DIAGONAL_BUFFER = 11     # Diagonal restriction zones
        self.ROBOT_HEADING = 12       # Robot heading indicator

    def create_adaptive_display(self, state_estimate, robot_x, robot_y,
                                origin_x=0, origin_y=0, path_waypoints=None,
                                current_waypoint_idx=None, robot_theta=0,
                                safety_buffer_cells=2, diagonal_buffer_cells=1):

        # Calculate key distances
        dist_to_origin = math.sqrt((robot_x - origin_x)**2 + (robot_y - origin_y)**2)

        # Determine appropriate zoom level
        zoom_level, cell_size = self._select_zoom_level(dist_to_origin, state_estimate,
                                                        robot_x, robot_y)

        # Determine view center based on zoom
        if zoom_level == 'detail':
            # High detail - center on robot
            center_x, center_y = robot_x, robot_y
        elif zoom_level in ['normal', 'wide']:
            # Medium zoom - weight toward robot but include origin
            weight = config.DISPLAY_CENTER_WEIGHT  # Weight from config
            center_x = robot_x * weight + origin_x * (1 - weight)
            center_y = robot_y * weight + origin_y * (1 - weight)
        else:  # overview
            # Wide view - center between robot and origin
            center_x = (robot_x + origin_x) / 2
            center_y = (robot_y + origin_y) / 2

        # Initialize grid
        grid = np.zeros((self.display_size, self.display_size), dtype=int)

        # Calculate world bounds for this view
        half_size = self.display_size // 2
        world_min_x = center_x - half_size * cell_size
        world_max_x = center_x + half_size * cell_size
        world_min_y = center_y - half_size * cell_size
        world_max_y = center_y + half_size * cell_size

        # Place origin (if in view)
        origin_grid_x, origin_grid_y = self._world_to_grid(
            origin_x, origin_y, world_min_x, world_min_y, cell_size
        )
        if self._in_bounds(origin_grid_x, origin_grid_y):
            grid[origin_grid_y, origin_grid_x] = self.ORIGIN

        # Place robot with heading indicator
        robot_grid_x, robot_grid_y = self._world_to_grid(
            robot_x, robot_y, world_min_x, world_min_y, cell_size
        )
        if self._in_bounds(robot_grid_x, robot_grid_y):
            grid[robot_grid_y, robot_grid_x] = self.ROBOT

        # Add landmarks with distance-based encoding
        obstacle_positions = self._add_landmarks(grid, state_estimate, robot_x, robot_y,
                                                 world_min_x, world_min_y, cell_size)

        # Add buffer zones around obstacles
        self._add_buffer_zones(grid, obstacle_positions, safety_buffer_cells,
                               diagonal_buffer_cells)

        # Add path if provided
        if path_waypoints:
            self._add_complete_path(grid, path_waypoints, current_waypoint_idx,
                                   world_min_x, world_min_y, cell_size)

        # Add robot heading indicator
        self._add_robot_heading(grid, robot_grid_x, robot_grid_y, robot_theta, cell_size)

        # Create metadata
        metadata = {
            'zoom_level': zoom_level,
            'cell_size': cell_size,
            'center': (center_x, center_y),
            'coverage': (world_max_x - world_min_x, world_max_y - world_min_y),
            'origin_visible': self._in_bounds(origin_grid_x, origin_grid_y),
            'robot_grid_pos': (robot_grid_x, robot_grid_y),
            'origin_grid_pos': (origin_grid_x, origin_grid_y)
        }

        return grid, metadata

    def _select_zoom_level(self, dist_to_origin, state_estimate, robot_x, robot_y):
        """
        Intelligently select zoom level based on context.
        """
        # Count nearby landmarks
        nearby_landmarks = 0
        if len(state_estimate) > 3:
            i = 3
            while i < len(state_estimate):
                try:
                    lm_x = state_estimate[i, 0] if state_estimate.ndim > 1 else state_estimate[i]
                    lm_y = state_estimate[i + 1, 0] if state_estimate.ndim > 1 else state_estimate[i + 1]
                    dist = math.sqrt((lm_x - robot_x)**2 + (lm_y - robot_y)**2)
                    if dist < config.DISPLAY_ZOOM_NORMAL_THRESHOLD:  # Within threshold from config
                        nearby_landmarks += 1
                except:
                    pass
                i += 3

        # Decision logic using config thresholds
        if nearby_landmarks > 3 and dist_to_origin < config.DISPLAY_ZOOM_NEAR_THRESHOLD:
            # Many nearby landmarks and close to origin - need detail
            return 'detail', self.zoom_levels['detail']
        elif dist_to_origin < config.DISPLAY_ZOOM_NORMAL_THRESHOLD:
            # Close to origin - normal view
            return 'normal', self.zoom_levels['normal']
        elif dist_to_origin < config.DISPLAY_ZOOM_WIDE_THRESHOLD:
            # Medium distance - wide view
            return 'wide', self.zoom_levels['wide']
        else:
            # Far from origin - overview
            return 'overview', self.zoom_levels['overview']

    def _world_to_grid(self, world_x, world_y, world_min_x, world_min_y, cell_size):
        """Convert world coordinates to grid indices."""
        grid_x = int((world_x - world_min_x) / cell_size)
        grid_y = self.display_size - 1 - int((world_y - world_min_y) / cell_size)
        return grid_x, grid_y

    def _in_bounds(self, grid_x, grid_y):
        """Check if grid coordinates are within bounds."""
        return 0 <= grid_x < self.display_size and 0 <= grid_y < self.display_size

    def _add_landmarks(self, grid, state_estimate, robot_x, robot_y,
                      world_min_x, world_min_y, cell_size):
        """Add landmarks with smart clustering and distance encoding."""
        obstacle_positions = []

        if len(state_estimate) <= 3:
            return obstacle_positions

        # Process landmarks
        i = 3
        while i < len(state_estimate):
            try:
                lm_x = state_estimate[i, 0] if state_estimate.ndim > 1 else state_estimate[i]
                lm_y = state_estimate[i + 1, 0] if state_estimate.ndim > 1 else state_estimate[i + 1]

                # Calculate distance from robot
                dist = math.sqrt((lm_x - robot_x)**2 + (lm_y - robot_y)**2)

                # Convert to grid
                grid_x, grid_y = self._world_to_grid(
                    lm_x, lm_y, world_min_x, world_min_y, cell_size
                )

                if self._in_bounds(grid_x, grid_y):
                    # Don't overwrite important markers
                    if grid[grid_y, grid_x] not in [self.ROBOT, self.ORIGIN]:
                        # Encode based on distance using config thresholds
                        if dist < config.DISPLAY_LANDMARK_NEAR_DIST:
                            grid[grid_y, grid_x] = self.LANDMARK_NEAR
                        elif dist < config.DISPLAY_LANDMARK_MID_DIST:
                            grid[grid_y, grid_x] = self.OBSTACLE
                        else:
                            grid[grid_y, grid_x] = self.LANDMARK_FAR

                        # Track obstacle positions for buffer zones
                        if grid[grid_y, grid_x] in [self.OBSTACLE, self.LANDMARK_NEAR]:
                            obstacle_positions.append((grid_x, grid_y))
            except:
                pass
            i += 3

        return obstacle_positions

    def _add_buffer_zones(self, grid, obstacle_positions, safety_buffer_cells, diagonal_buffer_cells):
        """Add buffer zones around obstacles for safety visualization."""
        # Add regular safety buffer
        for ox, oy in obstacle_positions:
            for dx in range(-safety_buffer_cells, safety_buffer_cells + 1):
                for dy in range(-safety_buffer_cells, safety_buffer_cells + 1):
                    if dx == 0 and dy == 0:
                        continue  # Skip the obstacle itself

                    bx, by = ox + dx, oy + dy
                    if self._in_bounds(bx, by):
                        # Only mark empty cells as buffer
                        if grid[by, bx] == self.EMPTY:
                            # Check if it's a diagonal buffer zone
                            if abs(dx) <= diagonal_buffer_cells and abs(dy) <= diagonal_buffer_cells and dx != 0 and dy != 0:
                                grid[by, bx] = self.DIAGONAL_BUFFER
                            else:
                                grid[by, bx] = self.BUFFER_ZONE

    def _add_complete_path(self, grid, path_waypoints, current_waypoint_idx,
                           world_min_x, world_min_y, cell_size):
        """Add complete path with waypoint highlighting."""
        if not path_waypoints:
            return

        # Draw path segments
        for i in range(len(path_waypoints) - 1):
            # Interpolate between waypoints for smoother path visualization
            start = path_waypoints[i]
            end = path_waypoints[i + 1]

            # Simple line interpolation
            steps = max(abs(int((end[0] - start[0]) / (cell_size * 0.5))),
                       abs(int((end[1] - start[1]) / (cell_size * 0.5)))) + 1

            for step in range(steps + 1):
                t = step / max(steps, 1)
                px = start[0] + t * (end[0] - start[0])
                py = start[1] + t * (end[1] - start[1])

                grid_x, grid_y = self._world_to_grid(
                    px, py, world_min_x, world_min_y, cell_size
                )

                if self._in_bounds(grid_x, grid_y):
                    if grid[grid_y, grid_x] == self.EMPTY:
                        grid[grid_y, grid_x] = self.PATH

        # Highlight waypoints
        for i, waypoint in enumerate(path_waypoints):
            wx, wy = waypoint
            grid_x, grid_y = self._world_to_grid(
                wx, wy, world_min_x, world_min_y, cell_size
            )

            if self._in_bounds(grid_x, grid_y):
                # Mark current waypoint specially
                if current_waypoint_idx is not None and i == current_waypoint_idx:
                    grid[grid_y, grid_x] = self.CURRENT_WAYPOINT
                # Mark other waypoints (don't overwrite robot/origin/current)
                elif grid[grid_y, grid_x] not in [self.ROBOT, self.ORIGIN, self.CURRENT_WAYPOINT]:
                    grid[grid_y, grid_x] = self.WAYPOINT

    def _add_robot_heading(self, grid, robot_grid_x, robot_grid_y, robot_theta, cell_size):
        """Add robot heading indicator."""
        if not self._in_bounds(robot_grid_x, robot_grid_y):
            return

        # Calculate heading indicator position (1-2 cells in front of robot)
        heading_length = 2  # cells
        dx = int(heading_length * math.cos(robot_theta))
        dy = -int(heading_length * math.sin(robot_theta))  # Negative because grid Y is inverted

        # Draw heading line
        for i in range(1, heading_length + 1):
            hx = robot_grid_x + int(i * math.cos(robot_theta))
            hy = robot_grid_y - int(i * math.sin(robot_theta))

            if self._in_bounds(hx, hy):
                if grid[hy, hx] == self.EMPTY:
                    grid[hy, hx] = self.ROBOT_HEADING

    def _add_path(self, grid, path_waypoints, world_min_x, world_min_y, cell_size):
        """Add planned path to the grid (deprecated - use _add_complete_path)."""
        for waypoint in path_waypoints:
            wx, wy = waypoint
            grid_x, grid_y = self._world_to_grid(
                wx, wy, world_min_x, world_min_y, cell_size
            )
            if self._in_bounds(grid_x, grid_y):
                # Only mark as path if cell is empty
                if grid[grid_y, grid_x] == self.EMPTY:
                    grid[grid_y, grid_x] = self.PATH

    def get_color_map(self):
        """
        Return color mapping for visualization.

        Returns:
            Dict mapping cell values to hex colors
        """
        return {
            self.EMPTY: 0x202020,                # Dark grey background
            self.OBSTACLE: 0xff4444,             # Bright red - confirmed obstacle
            self.ORIGIN: 0x0080ff,               # Bright blue - start position
            self.ROBOT: 0x00ff00,                # Bright green - current position
            self.PATH: 0xffff00,                 # Yellow - planned path
            self.LANDMARK_NEAR: 0xff8888,        # Light red - close danger
            self.LANDMARK_FAR: 0x884444,         # Dark red - far obstacle
            self.WALL: 0x666666,                 # Medium grey - wall
            self.BUFFER_ZONE: 0x4a4a00,          # Dark yellow - safety buffer
            self.WAYPOINT: 0xffaa00,             # Orange - path waypoints
            self.CURRENT_WAYPOINT: 0x00ffff,     # Cyan - current target
            self.DIAGONAL_BUFFER: 0x6a3a00,      # Dark orange - diagonal restriction
            self.ROBOT_HEADING: 0x00aa00         # Medium green - robot direction
        }

    def get_ascii_representation(self, grid):
        """
        Get ASCII representation for debugging.
        """
        symbols = {
            self.EMPTY: '.',
            self.OBSTACLE: '#',
            self.ORIGIN: 'O',
            self.ROBOT: 'R',
            self.PATH: '*',
            self.LANDMARK_NEAR: '!',
            self.LANDMARK_FAR: '?',
            self.WALL: '=',
            self.BUFFER_ZONE: '~',
            self.WAYPOINT: 'W',
            self.CURRENT_WAYPOINT: '@',
            self.DIAGONAL_BUFFER: '+',
            self.ROBOT_HEADING: '>'
        }

        lines = []
        for row in grid:
            line = ''.join(symbols.get(cell, ' ') for cell in row)
            lines.append(line)
        return '\n'.join(lines)