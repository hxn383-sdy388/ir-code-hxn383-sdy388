"""A* path planning algorithm for grid-based navigation with relative coordinates."""

import math
import heapq
from typing import List, Tuple, Optional, Union, Dict, Set

class Node:
    """Represents a node in the A* search."""

    def __init__(self, position: Tuple[int, int], parent=None):
        self.position = position
        self.parent = parent

        self.g = 0.0  # Cost from start
        self.h = 0.0  # Heuristic cost to goal
        self.f = 0.0  # Total cost (g + h)

    def __eq__(self, other):
        if other is None:
            return False
        return self.position == other.position

    def __lt__(self, other):
        # Tie-breaking: prefer nodes with lower h value when f is equal
        if abs(self.f - other.f) < 1e-9:
            return self.h < other.h
        return self.f < other.f

    def __hash__(self):
        return hash(self.position)


class AStarPlanner:
    # Movement directions: (dx, dy, cost)
    MOVES_4 = [
        (0, 1, 1.0),   # Up
        (1, 0, 1.0),   # Right
        (0, -1, 1.0),  # Down
        (-1, 0, 1.0),  # Left
    ]

    MOVES_8 = [
        (0, 1, 1.0),       # Up
        (1, 0, 1.0),       # Right
        (0, -1, 1.0),      # Down
        (-1, 0, 1.0),      # Left
        (1, 1, 1.414),     # Up-Right (diagonal)
        (1, -1, 1.414),    # Down-Right (diagonal)
        (-1, -1, 1.414),   # Down-Left (diagonal)
        (-1, 1, 1.414),    # Up-Left (diagonal)
    ]

    def __init__(self, grid_cell_size, origin_marker='/',
                 diagonal_cost=1.414, straight_cost=1.0,
                 allow_diagonal=True, goal_tolerance=0.05,
                 safety_buffer=2, diagonal_restriction_buffer=2,
                 aggressive_smoothing=True,
                 smooth_around_corner=True, smooth_corner_distance=0.10,
                 smooth_corner_arc_points=3, smooth_corner_clearance=0.06,
                 smooth_corner_min_angle=0.5):

        self.cell_size = grid_cell_size
        self.origin_marker = origin_marker

        self.diagonal_cost = diagonal_cost
        self.straight_cost = straight_cost
        self.allow_diagonal = allow_diagonal
        self.goal_tolerance = goal_tolerance
        self.safety_buffer = safety_buffer
        self.diagonal_restriction_buffer = diagonal_restriction_buffer
        self.aggressive_smoothing = aggressive_smoothing

        # Smooth corner parameters
        self.smooth_around_corner = smooth_around_corner
        self.smooth_corner_distance = smooth_corner_distance
        self.smooth_corner_arc_points = smooth_corner_arc_points
        self.smooth_corner_clearance = smooth_corner_clearance
        self.smooth_corner_min_angle = smooth_corner_min_angle

        # Grid storage using dictionaries for sparse representation
        self.occupancy_grid: Dict[Tuple[int, int], int] = {}
        self.buffered_grid: Dict[Tuple[int, int], int] = {}
        self.diagonal_restricted_grid: Set[Tuple[int, int]] = set()

        # Origin position in grid coordinates
        self.origin_grid_x = 0
        self.origin_grid_y = 0

        # Grid bounds for bounded search
        self.grid_min_x = 0
        self.grid_max_x = 0
        self.grid_min_y = 0
        self.grid_max_y = 0

        # Goal tracking
        self.exact_goal = None

        # Path state
        self.current_path: List[Tuple[float, float]] = []
        self.current_waypoint_index = 0

        # Statistics
        self.last_nodes_explored = 0
    
    def world_to_grid(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates.

        IMPORTANT: The grid from grid_utils.py is ROBOT-CENTRIC:
        - Robot is always at grid center (origin marker position)
        - Grid Y is inverted (positive world Y = negative grid Y offset from center)

        This method converts world coordinates to grid coordinates where:
        - (0, 0) in world = origin marker in grid
        - Positive world X = positive grid X
        - Positive world Y = NEGATIVE grid Y (Y is inverted!)
        """
        # Grid X: world X maps directly (scaled by cell size)
        grid_x = int(math.floor(world_x / self.cell_size))
        # Grid Y: INVERTED - positive world Y becomes negative grid Y
        grid_y = int(math.floor(-world_y / self.cell_size))
        return grid_x, grid_y

    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates (cell center).

        Inverse of world_to_grid - accounts for Y inversion.
        """
        world_x = (grid_x + 0.5) * self.cell_size
        # Invert Y back: negative grid Y = positive world Y
        world_y = -(grid_y + 0.5) * self.cell_size
        return world_x, world_y

    def _create_buffered_grid(self):
        """Create a buffered occupancy grid with safety margin around obstacles.

        This creates two separate buffers:
        1. Safety buffer: Marks cells around obstacles as blocked
        2. Diagonal restriction zone: Where diagonal moves are not allowed
        """
        # Start with a copy of the original grid
        self.buffered_grid = dict(self.occupancy_grid)
        self.diagonal_restricted_grid = set()

        # Find all occupied cells
        occupied_cells = [(pos, val) for pos, val in self.occupancy_grid.items() if val == 1]

        # Create safety buffer around obstacles
        for (occupied_x, occupied_y), _ in occupied_cells:
            for dx in range(-self.safety_buffer, self.safety_buffer + 1):
                for dy in range(-self.safety_buffer, self.safety_buffer + 1):
                    # Skip the center cell (already marked)
                    if dx == 0 and dy == 0:
                        continue

                    buffer_x = occupied_x + dx
                    buffer_y = occupied_y + dy

                    # Use Manhattan distance for buffer (creates diamond shape)
                    # or Chebyshev distance for square buffer
                    dist = max(abs(dx), abs(dy))  # Chebyshev distance
                    if dist <= self.safety_buffer:
                        # Only add if not already an obstacle
                        if self.occupancy_grid.get((buffer_x, buffer_y), 0) == 0:
                            self.buffered_grid[(buffer_x, buffer_y)] = 1

        # Create diagonal restriction zone (larger than safety buffer)
        total_restriction = self.safety_buffer + self.diagonal_restriction_buffer
        for (occupied_x, occupied_y), _ in occupied_cells:
            for dx in range(-total_restriction, total_restriction + 1):
                for dy in range(-total_restriction, total_restriction + 1):
                    restrict_x = occupied_x + dx
                    restrict_y = occupied_y + dy
                    self.diagonal_restricted_grid.add((restrict_x, restrict_y))

    def is_valid_cell(self, grid_x: int, grid_y: int, use_buffer: bool = True) -> bool:
        """Check if grid cell is valid (not occupied)."""
        # Check bounds
        if not (self.grid_min_x <= grid_x <= self.grid_max_x and
                self.grid_min_y <= grid_y <= self.grid_max_y):
            return False

        grid_to_check = self.buffered_grid if use_buffer else self.occupancy_grid
        return grid_to_check.get((grid_x, grid_y), 0) == 0

    def is_diagonal_allowed(self, grid_x: int, grid_y: int) -> bool:
        """Check if diagonal movement is allowed at this position."""
        return (grid_x, grid_y) not in self.diagonal_restricted_grid

    def _is_collision(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Check if moving from start to end would cause a collision.

        For diagonal moves, also checks the two adjacent cells to prevent
        cutting corners around obstacles."""
        # Check if end cell is blocked
        if not self.is_valid_cell(end[0], end[1], use_buffer=True):
            return True

        # For diagonal moves, check corner-cutting
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        if abs(dx) == 1 and abs(dy) == 1:
            # Diagonal move - check both adjacent cells
            cell1 = (start[0] + dx, start[1])  # Horizontal neighbor
            cell2 = (start[0], start[1] + dy)  # Vertical neighbor

            if not self.is_valid_cell(cell1[0], cell1[1], use_buffer=True):
                return True
            if not self.is_valid_cell(cell2[0], cell2[1], use_buffer=True):
                return True

        return False
    
    def load_occupancy_grid(self, grid_data: List[List[Union[int, str]]]):
        """Load occupancy grid from 2D list with origin marker."""
        self.occupancy_grid.clear()
        origin_found = False

        # Get grid dimensions
        grid_height = len(grid_data)
        grid_width = len(grid_data[0]) if grid_data else 0

        # FIRST PASS: Find the origin marker
        for row_idx, row in enumerate(grid_data):
            for col_idx, cell in enumerate(row):
                if cell == self.origin_marker:
                    self.origin_grid_x = col_idx
                    self.origin_grid_y = row_idx
                    origin_found = True
                    break
            if origin_found:
                break

        if not origin_found:
            # Default to center if no origin marker
            self.origin_grid_x = grid_width // 2
            self.origin_grid_y = grid_height // 2
            print(f"Warning: Origin marker '{self.origin_marker}' not found. Using center ({self.origin_grid_x}, {self.origin_grid_y})")

        # Set grid bounds (relative to origin)
        self.grid_min_x = -self.origin_grid_x
        self.grid_max_x = grid_width - 1 - self.origin_grid_x
        self.grid_min_y = -self.origin_grid_y
        self.grid_max_y = grid_height - 1 - self.origin_grid_y

        # SECOND PASS: Process all cells
        for row_idx, row in enumerate(grid_data):
            for col_idx, cell in enumerate(row):
                # Convert to relative coordinates (origin-centered)
                grid_x = col_idx - self.origin_grid_x
                grid_y = row_idx - self.origin_grid_y

                if cell == self.origin_marker:
                    self.occupancy_grid[(grid_x, grid_y)] = 0  # Origin is free
                elif cell == 1:
                    self.occupancy_grid[(grid_x, grid_y)] = 1  # Obstacle
                else:
                    self.occupancy_grid[(grid_x, grid_y)] = 0  # Free space

        # Create buffered grid with safety margins
        self._create_buffered_grid()

        # Debug output (reduced frequency)
        occupied = [pos for pos, val in self.occupancy_grid.items() if val == 1]
        buffered = [pos for pos, val in self.buffered_grid.items() if val == 1]

        if len(occupied) > 0:
            print(f"Grid loaded: {grid_width}x{grid_height}, origin at ({self.origin_grid_x}, {self.origin_grid_y})")
            print(f"Obstacles: {len(occupied)}, Buffered cells: {len(buffered)}")
    
    def set_cell_occupied(self, world_x: float, world_y: float, occupied: bool = True):
        """Mark a cell as occupied or free based on world coordinates."""
        grid_x, grid_y = self.world_to_grid(world_x, world_y)
        if occupied:
            self.occupancy_grid[(grid_x, grid_y)] = 1
        else:
            self.occupancy_grid.pop((grid_x, grid_y), None)
        
        self._create_buffered_grid()
    
    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate heuristic cost between two grid positions.

        Uses Euclidean distance for diagonal movement, Manhattan for 4-connected.
        The heuristic is admissible (never overestimates) for both cases.
        """
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])

        if self.allow_diagonal:
            # Octile distance: optimal for 8-connected grid
            # This is more accurate than Euclidean and still admissible
            return max(dx, dy) + (self.diagonal_cost - 1.0) * min(dx, dy)
        else:
            # Manhattan distance for 4-connected grid
            return float(dx + dy)

    def _get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """Get valid neighboring cells and their movement costs."""
        x, y = position
        neighbors = []

        # Determine which move set to use
        allow_diagonals_here = self.allow_diagonal and self.is_diagonal_allowed(x, y)
        moves = self.MOVES_8 if allow_diagonals_here else self.MOVES_4

        for dx, dy, base_cost in moves:
            new_x, new_y = x + dx, y + dy
            new_pos = (new_x, new_y)

            # Check if this move causes a collision
            if self._is_collision(position, new_pos):
                continue

            # Adjust cost based on movement type
            if abs(dx) + abs(dy) == 2:
                # Diagonal move
                cost = self.diagonal_cost
            else:
                # Straight move
                cost = self.straight_cost

            neighbors.append((new_pos, cost))

        return neighbors

    def _get_move_cost(self, start: Tuple[int, int], end: Tuple[int, int]) -> float:
        """Calculate the cost of moving from start to end."""
        if self._is_collision(start, end):
            return float('inf')

        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])

        if dx == 1 and dy == 1:
            return self.diagonal_cost
        elif dx + dy == 1:
            return self.straight_cost
        else:
            # Non-adjacent cells - use Euclidean distance
            return math.hypot(dx, dy)
    
    def _line_of_sight(self, pos1: Tuple[int, int], pos2: Tuple[int, int],
                        use_buffer: bool = False) -> bool:
        """Check if there's a clear line of sight between two grid positions.

        Uses Bresenham's line algorithm to check all cells along the path.
    """
        x0, y0 = pos1
        x1, y1 = pos2

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        x_step = 1 if x0 < x1 else -1
        y_step = 1 if y0 < y1 else -1

        # Bresenham's line algorithm
        if dx > dy:
            error = dx // 2
            y = y0
            for x in range(x0, x1 + x_step, x_step):
                if not self.is_valid_cell(x, y, use_buffer=use_buffer):
                    return False
                error -= dy
                if error < 0:
                    y += y_step
                    error += dx
        else:
            error = dy // 2
            x = x0
            for y in range(y0, y1 + y_step, y_step):
                if not self.is_valid_cell(x, y, use_buffer=use_buffer):
                    return False
                error -= dx
                if error < 0:
                    x += x_step
                    error += dy

        return True
    
    def _smooth_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Smooth path by removing unnecessary waypoints while maintaining safety.

        Uses line-of-sight checks to skip intermediate waypoints when possible.
        This creates more natural, direct paths while avoiding obstacles."""
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        current_idx = 0

        while current_idx < len(path) - 1:
            # Find the farthest visible waypoint from current position
            farthest_visible = current_idx + 1

            for test_idx in range(len(path) - 1, current_idx + 1, -1):
                current_grid = self.world_to_grid(*smoothed[-1])
                test_grid = self.world_to_grid(*path[test_idx])

                # Use buffered check for line of sight to respect safety margins
                if self._line_of_sight(current_grid, test_grid, use_buffer=True):
                    farthest_visible = test_idx
                    break

            smoothed.append(path[farthest_visible])
            current_idx = farthest_visible

        return smoothed
    
    def _optimize_path_segments(self, path: List[Tuple[float, float]],
                                min_segment_length: float = None) -> List[Tuple[float, float]]:
        """Optimize path by creating longer directional segments.

        Attempts to extend segments while maintaining clearance from obstacles.
        Prefers longer, straighter paths that are easier to follow."""
        if len(path) <= 2:
            return path

        if min_segment_length is None:
            min_segment_length = self.cell_size * 2.0

        optimized = [path[0]]
        current_idx = 0

        while current_idx < len(path) - 1:
            # Find the farthest reachable point with safe line of sight
            best_idx = current_idx + 1

            for test_idx in range(len(path) - 1, current_idx, -1):
                current_grid = self.world_to_grid(*optimized[-1])
                test_grid = self.world_to_grid(*path[test_idx])

                # Check line of sight using buffered grid for safety
                if self._line_of_sight(current_grid, test_grid, use_buffer=True):
                    best_idx = test_idx
                    break

            optimized.append(path[best_idx])
            current_idx = best_idx

        return optimized

    def _is_near_obstacle(self, world_x: float, world_y: float) -> bool:
        """Check if a world position is near an obstacle."""
        grid_x, grid_y = self.world_to_grid(world_x, world_y)

        # Check nearby cells for obstacles
        check_radius = max(2, int(self.smooth_corner_distance / self.cell_size))
        for dx in range(-check_radius, check_radius + 1):
            for dy in range(-check_radius, check_radius + 1):
                check_pos = (grid_x + dx, grid_y + dy)
                if self.occupancy_grid.get(check_pos, 0) == 1:
                    return True
        return False

    def _find_nearest_obstacle_direction(self, world_x: float, world_y: float) -> Optional[float]:
        """Find the direction to the nearest obstacle from a world position.

        Returns the angle in radians pointing TOWARDS the nearest obstacle,
        or None if no obstacle is nearby."""
        grid_x, grid_y = self.world_to_grid(world_x, world_y)

        check_radius = max(3, int(self.smooth_corner_distance / self.cell_size) + 1)
        min_dist = float('inf')
        nearest_obstacle = None

        for dx in range(-check_radius, check_radius + 1):
            for dy in range(-check_radius, check_radius + 1):
                check_pos = (grid_x + dx, grid_y + dy)
                if self.occupancy_grid.get(check_pos, 0) == 1:
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_obstacle = (dx, dy)

        if nearest_obstacle is None:
            return None

        # Convert grid direction to world angle
        # Remember Y is inverted in grid coordinates
        obs_dx, obs_dy = nearest_obstacle
        return math.atan2(-obs_dy, obs_dx)  # Negate dy due to Y inversion

    def _smooth_corners(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Add smooth arc waypoints around corners near obstacles.

        This method detects sharp turns in the path that are near obstacles
        and inserts intermediate waypoints to create a smooth curve around them.
        """
        if len(path) < 3 or not self.smooth_around_corner:
            return path

        smoothed = [path[0]]

        for i in range(1, len(path) - 1):
            prev_wp = path[i - 1]
            curr_wp = path[i]
            next_wp = path[i + 1]

            # Calculate incoming and outgoing directions
            in_dx = curr_wp[0] - prev_wp[0]
            in_dy = curr_wp[1] - prev_wp[1]
            out_dx = next_wp[0] - curr_wp[0]
            out_dy = next_wp[1] - curr_wp[1]

            in_len = math.sqrt(in_dx * in_dx + in_dy * in_dy)
            out_len = math.sqrt(out_dx * out_dx + out_dy * out_dy)

            if in_len < 0.001 or out_len < 0.001:
                smoothed.append(curr_wp)
                continue

            # Normalize directions
            in_dx /= in_len
            in_dy /= in_len
            out_dx /= out_len
            out_dy /= out_len

            # Calculate turn angle using dot product
            dot = in_dx * out_dx + in_dy * out_dy
            dot = max(-1.0, min(1.0, dot))  # Clamp for numerical stability
            turn_angle = math.acos(dot)

            # Check if this is a significant turn near an obstacle
            if turn_angle >= self.smooth_corner_min_angle and self._is_near_obstacle(*curr_wp):
                # Find direction away from obstacle
                obstacle_dir = self._find_nearest_obstacle_direction(*curr_wp)

                if obstacle_dir is not None:
                    # Direction away from obstacle
                    away_dir = obstacle_dir + math.pi

                    # Calculate the bisector of the turn (average of in and out directions)
                    in_angle = math.atan2(in_dy, in_dx)
                    out_angle = math.atan2(out_dy, out_dx)

                    # Use cross product to determine turn direction (left or right)
                    cross = in_dx * out_dy - in_dy * out_dx

                    # Generate arc waypoints
                    arc_points = []
                    for j in range(self.smooth_corner_arc_points):
                        # Parameter from 0 to 1 along the arc
                        t = (j + 1) / (self.smooth_corner_arc_points + 1)

                        # Interpolate position along the path
                        # Blend between approaching curr_wp and leaving curr_wp
                        blend_in = 1.0 - t
                        blend_out = t

                        # Base position: interpolate between before and after corner
                        base_x = prev_wp[0] + (curr_wp[0] - prev_wp[0]) * (0.5 + t * 0.5)
                        base_y = prev_wp[1] + (curr_wp[1] - prev_wp[1]) * (0.5 + t * 0.5)

                        if t > 0.5:
                            # Past the midpoint, blend towards next waypoint
                            t2 = (t - 0.5) * 2
                            base_x = curr_wp[0] + (next_wp[0] - curr_wp[0]) * t2 * 0.5
                            base_y = curr_wp[1] + (next_wp[1] - curr_wp[1]) * t2 * 0.5

                        # Add offset away from obstacle (curved path)
                        # Maximum offset at the middle of the arc
                        offset_factor = math.sin(t * math.pi) * self.smooth_corner_clearance

                        arc_x = base_x + math.cos(away_dir) * offset_factor
                        arc_y = base_y + math.sin(away_dir) * offset_factor

                        # Verify the arc point is valid (respects safety buffer)
                        arc_grid = self.world_to_grid(arc_x, arc_y)
                        if self.is_valid_cell(*arc_grid, use_buffer=True):
                            arc_points.append((arc_x, arc_y))

                    # Add arc points if we generated valid ones
                    if arc_points:
                        smoothed.extend(arc_points)
                    else:
                        smoothed.append(curr_wp)
                else:
                    smoothed.append(curr_wp)
            else:
                smoothed.append(curr_wp)

        smoothed.append(path[-1])
        return smoothed

    def plan_path(self, start_x: float, start_y: float,
                  goal_x: float, goal_y: float) -> Optional[List[Tuple[float, float]]]:
        """Plan a path from start to goal using A* algorithm.

        This is the main path planning method. It:
        1. Validates start and goal positions
        2. Runs A* search to find optimal path
        3. Smooths the path to remove unnecessary waypoints
        4. Optionally optimizes path segments"""
        self.exact_goal = (goal_x, goal_y)
        self.last_nodes_explored = 0

        # Convert to grid coordinates
        start_grid = self.world_to_grid(start_x, start_y)
        goal_grid = self.world_to_grid(goal_x, goal_y)

        print(f"A* Planning: ({start_x:.3f}, {start_y:.3f}) -> ({goal_x:.3f}, {goal_y:.3f})")
        print(f"Grid coords: {start_grid} -> {goal_grid}")

        # Validate start position (don't use buffer - robot might be near obstacle)
        if not self.is_valid_cell(*start_grid, use_buffer=False):
            print(f"Warning: Start position {start_grid} is blocked!")
            # Try to find nearest valid cell
            start_grid = self._find_nearest_free_cell(start_grid)
            if start_grid is None:
                print("Error: Could not find valid start position")
                return None
            print(f"Using nearest free cell: {start_grid}")

        # Validate goal position (don't use buffer - allow goals near obstacles)
        if not self.is_valid_cell(*goal_grid, use_buffer=False):
            print(f"Warning: Goal position {goal_grid} is blocked!")
            goal_grid = self._find_nearest_free_cell(goal_grid)
            if goal_grid is None:
                print("Error: Could not find valid goal position")
                return None
            print(f"Using nearest free cell for goal: {goal_grid}")

        # Check if already at goal
        if start_grid == goal_grid:
            print("Already at goal!")
            self.current_path = [(goal_x, goal_y)]
            self.current_waypoint_index = 0
            return self.current_path

        # Initialize A* data structures
        # Using dictionaries for O(1) lookups
        g_score: Dict[Tuple[int, int], float] = {start_grid: 0.0}
        f_score: Dict[Tuple[int, int], float] = {}
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {start_grid: start_grid}

        # Priority queue: (f_score, counter, position)
        # Counter ensures stable ordering for equal f_scores
        counter = 0
        open_heap = []
        start_h = self._heuristic(start_grid, goal_grid)
        f_score[start_grid] = start_h
        heapq.heappush(open_heap, (start_h, counter, start_grid))

        # Set for O(1) membership testing
        open_set = {start_grid}
        closed_set: Set[Tuple[int, int]] = set()

        # A* main loop
        while open_heap:
            # Get node with lowest f_score
            _, _, current = heapq.heappop(open_heap)
            open_set.discard(current)

            self.last_nodes_explored += 1

            # Goal reached?
            if current == goal_grid:
                print(f"Path found! Explored {self.last_nodes_explored} nodes")
                raw_path = self._reconstruct_path_from_parents(parent, start_grid, goal_grid)

                # Convert to world coordinates
                world_path = [self.grid_to_world(*pos) for pos in raw_path]

                # Set exact start and goal positions
                world_path[0] = (start_x, start_y)
                world_path[-1] = (goal_x, goal_y)

                print(f"Raw path: {len(world_path)} waypoints")

                # Smooth the path
                smoothed = self._smooth_path(world_path)
                if smoothed:
                    smoothed[-1] = (goal_x, goal_y)
                print(f"Smoothed path: {len(smoothed)} waypoints")

                # Optional: additional optimization
                if self.aggressive_smoothing and len(smoothed) > 2:
                    optimized = self._optimize_path_segments(smoothed)
                    if optimized:
                        optimized[-1] = (goal_x, goal_y)
                    print(f"Optimized path: {len(optimized)} waypoints")
                    self.current_path = optimized
                else:
                    self.current_path = smoothed

                # Apply smooth corner processing if enabled
                if self.smooth_around_corner and len(self.current_path) >= 3:
                    corner_smoothed = self._smooth_corners(self.current_path)
                    if corner_smoothed:
                        corner_smoothed[-1] = (goal_x, goal_y)
                        if len(corner_smoothed) != len(self.current_path):
                            print(f"Corner-smoothed path: {len(corner_smoothed)} waypoints")
                        self.current_path = corner_smoothed

                self.current_waypoint_index = 0

                # Print final path
                for i, wp in enumerate(self.current_path):
                    print(f"  WP{i}: ({wp[0]:.3f}, {wp[1]:.3f})")

                return self.current_path

            closed_set.add(current)

            # Explore neighbors
            for neighbor, move_cost in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue

                # Calculate tentative g_score
                tentative_g = g_score[current] + move_cost

                # Check if this path is better
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # Update path
                    parent[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = self._heuristic(neighbor, goal_grid)
                    f = tentative_g + h
                    f_score[neighbor] = f

                    if neighbor not in open_set:
                        counter += 1
                        heapq.heappush(open_heap, (f, counter, neighbor))
                        open_set.add(neighbor)

            # Safety limit to prevent infinite loops
            if self.last_nodes_explored > 50000:
                print("Warning: A* search exceeded node limit!")
                break

        print(f"No path found from {start_grid} to {goal_grid}")
        print(f"Explored {self.last_nodes_explored} nodes")
        return None

    def _find_nearest_free_cell(self, pos: Tuple[int, int],
                                 max_radius: int = 5) -> Optional[Tuple[int, int]]:
        """Find the nearest free cell to the given position.

        Uses BFS-like expansion to find closest unblocked cell."""
        x, y = pos

        for radius in range(1, max_radius + 1):
            # Check cells at this radius (Manhattan distance)
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) != radius:
                        continue
                    check_x, check_y = x + dx, y + dy
                    if self.is_valid_cell(check_x, check_y, use_buffer=False):
                        return (check_x, check_y)

        return None

    def _reconstruct_path_from_parents(self, parent: Dict[Tuple[int, int], Tuple[int, int]],
                                        start: Tuple[int, int],
                                        goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from parent dictionary."""
        path = [goal]
        current = goal

        while current != start:
            current = parent[current]
            path.append(current)

        path.reverse()
        return path
    
    def _reconstruct_path(self, goal_node: Node) -> List[Tuple[float, float]]:
        """Reconstruct path from goal node by following parent links."""
        path = []
        current = goal_node
        
        while current is not None:
            world_pos = self.grid_to_world(*current.position)
            path.append(world_pos)
            current = current.parent
        
        path.reverse()
        
        return path
    
    def get_next_waypoint(self, current_x: float, current_y: float,
                          final_waypoint_tolerance: float = 0.03) -> Optional[Tuple[float, float]]:
        """Get the next waypoint to navigate to.

        Automatically advances to next waypoint when current one is reached."""
        if not self.current_path or self.current_waypoint_index >= len(self.current_path):
            return None

        waypoint = self.current_path[self.current_waypoint_index]

        # Calculate distance to current waypoint
        distance = math.sqrt((current_x - waypoint[0])**2 + (current_y - waypoint[1])**2)

        # Use different tolerances for intermediate and final waypoints
        is_final_waypoint = (self.current_waypoint_index == len(self.current_path) - 1)

        if is_final_waypoint:
            tolerance = final_waypoint_tolerance
        else:
            # For intermediate waypoints, use larger tolerance to avoid stopping
            tolerance = self.goal_tolerance

        # Check if we've reached this waypoint
        if distance < tolerance:
            self.current_waypoint_index += 1

            if self.current_waypoint_index >= len(self.current_path):
                return None

            waypoint = self.current_path[self.current_waypoint_index]

        return waypoint

    def is_goal_reached(self) -> bool:
        """Check if the final goal has been reached."""
        return not self.current_path or self.current_waypoint_index >= len(self.current_path)

    def get_path_progress(self) -> Tuple[int, int]:
        """Get current progress along the path."""
        return self.current_waypoint_index, len(self.current_path)

    def clear_path(self):
        """Clear the current path and reset state."""
        self.current_path = []
        self.current_waypoint_index = 0
        self.exact_goal = None

    def get_remaining_distance(self, current_x: float, current_y: float) -> float:
        """Calculate remaining distance along the path."""
        if not self.current_path or self.current_waypoint_index >= len(self.current_path):
            return 0.0

        # Distance to current waypoint
        waypoint = self.current_path[self.current_waypoint_index]
        total_dist = math.sqrt((current_x - waypoint[0])**2 + (current_y - waypoint[1])**2)

        # Add distances between remaining waypoints
        for i in range(self.current_waypoint_index, len(self.current_path) - 1):
            wp1 = self.current_path[i]
            wp2 = self.current_path[i + 1]
            total_dist += math.sqrt((wp2[0] - wp1[0])**2 + (wp2[1] - wp1[1])**2)

        return total_dist

    def replan_if_blocked(self, current_x: float, current_y: float) -> bool:
        """Check if current path is blocked and replan if necessary."""
        if not self.current_path or not self.exact_goal:
            return False

        # Check if next waypoint is now blocked
        if self.current_waypoint_index < len(self.current_path):
            waypoint = self.current_path[self.current_waypoint_index]
            waypoint_grid = self.world_to_grid(*waypoint)

            if not self.is_valid_cell(*waypoint_grid, use_buffer=True):
                print("Waypoint blocked! Replanning...")
                goal_x, goal_y = self.exact_goal
                new_path = self.plan_path(current_x, current_y, goal_x, goal_y)
                return new_path is not None

        return False