"""A* path planning algorithm for grid-based navigation with relative coordinates."""

import math
import heapq
from typing import List, Tuple, Optional, Union

class Node:
    """Represents a node in the A* search."""
    
    def __init__(self, position: Tuple[int, int], parent=None):
        self.position = position
        self.parent = parent
        
        self.g = 0
        self.h = 0
        self.f = 0
    
    def __eq__(self, other):
        return self.position == other.position
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __hash__(self):
        return hash(self.position)


class AStarPlanner:
    """A* path planning on unbounded occupancy grid with relative coordinates."""
    
    def __init__(self, grid_cell_size, origin_marker='/',
                 diagonal_cost=1.414, straight_cost=1.0,
                 allow_diagonal=True, goal_tolerance=0.05,
                 safety_buffer=2, diagonal_restriction_buffer=2,
                 aggressive_smoothing=True):
        """Initialize A* path planner with unbounded grid support."""
        self.cell_size = grid_cell_size
        self.origin_marker = origin_marker
        
        self.diagonal_cost = diagonal_cost
        self.straight_cost = straight_cost
        self.allow_diagonal = allow_diagonal
        self.goal_tolerance = goal_tolerance
        self.safety_buffer = safety_buffer
        self.diagonal_restriction_buffer = diagonal_restriction_buffer
        self.aggressive_smoothing = aggressive_smoothing
        
        self.occupancy_grid = {}
        self.buffered_grid = {}
        self.diagonal_restricted_grid = {}
        
        self.origin_grid_x = 0
        self.origin_grid_y = 0
        
        self.exact_goal = None
        
        self.current_path = []
        self.current_waypoint_index = 0
    
    def world_to_grid(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates relative to origin."""
        grid_x = int(math.floor(world_x / self.cell_size))
        grid_y = int(math.floor(world_y / self.cell_size))
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates (cell center)."""
        world_x = (grid_x + 0.5) * self.cell_size
        world_y = (grid_y + 0.5) * self.cell_size
        return world_x, world_y
    
    def _create_buffered_grid(self):
        """Create a buffered occupancy grid with safety margin around obstacles."""
        self.buffered_grid = self.occupancy_grid.copy()
        self.diagonal_restricted_grid = {}
        
        occupied_cells = [pos for pos, val in self.occupancy_grid.items() if val == 1]
        
        for occupied_x, occupied_y in occupied_cells:
            for dx in range(-self.safety_buffer, self.safety_buffer + 1):
                for dy in range(-self.safety_buffer, self.safety_buffer + 1):
                    buffer_x = occupied_x + dx
                    buffer_y = occupied_y + dy
                    if (buffer_x, buffer_y) not in self.occupancy_grid or self.occupancy_grid[(buffer_x, buffer_y)] == 0:
                        self.buffered_grid[(buffer_x, buffer_y)] = 1
        
        for occupied_x, occupied_y in occupied_cells:
            for dx in range(-self.diagonal_restriction_buffer, self.diagonal_restriction_buffer + 1):
                for dy in range(-self.diagonal_restriction_buffer, self.diagonal_restriction_buffer + 1):
                    restrict_x = occupied_x + dx
                    restrict_y = occupied_y + dy
                    self.diagonal_restricted_grid[(restrict_x, restrict_y)] = True
    
    def is_valid_cell(self, grid_x: int, grid_y: int, use_buffer: bool = True) -> bool:
        """Check if grid cell is valid (not occupied)."""
        grid_to_check = self.buffered_grid if use_buffer else self.occupancy_grid
        return grid_to_check.get((grid_x, grid_y), 0) == 0
    
    def is_diagonal_allowed(self, grid_x: int, grid_y: int) -> bool:
        """Check if diagonal movement is allowed at this position."""
        return (grid_x, grid_y) not in self.diagonal_restricted_grid
    
    def load_occupancy_grid(self, grid_data: List[List[Union[int, str]]]):
        """Load occupancy grid from 2D list with origin marker."""
        self.occupancy_grid.clear()
        origin_found = False
        
        # FIRST PASS: Find the origin
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
            raise ValueError(f"Origin marker '{self.origin_marker}' not found in grid data")
        
        # SECOND PASS: Process all cells with correct origin
        for row_idx, row in enumerate(grid_data):
            for col_idx, cell in enumerate(row):
                grid_x = col_idx - self.origin_grid_x
                grid_y = row_idx - self.origin_grid_y
                
                if cell == self.origin_marker:
                    self.occupancy_grid[(grid_x, grid_y)] = 0
                elif cell == 1:
                    self.occupancy_grid[(grid_x, grid_y)] = 1
        
        self._create_buffered_grid()
        
        # DEBUG: Print obstacle and buffer info
        print(f"Occupancy grid loaded. Origin at grid indices ({self.origin_grid_x}, {self.origin_grid_y})")
        print(f"Safety buffer: {self.safety_buffer} cells ({self.safety_buffer * self.cell_size:.2f}m)")
        print(f"Diagonal restriction: {self.diagonal_restriction_buffer} cells ({self.diagonal_restriction_buffer * self.cell_size:.2f}m)")
        
        occupied = [pos for pos, val in self.occupancy_grid.items() if val == 1]
        print(f"Occupied cells: {len(occupied)}")
        print(f"First 10 occupied: {sorted(occupied)[:10]}")
        
        buffered = [pos for pos, val in self.buffered_grid.items() if val == 1]
        print(f"Buffered cells (including safety margin): {len(buffered)}")
    
    def set_cell_occupied(self, world_x: float, world_y: float, occupied: bool = True):
        """Mark a cell as occupied or free based on world coordinates."""
        grid_x, grid_y = self.world_to_grid(world_x, world_y)
        if occupied:
            self.occupancy_grid[(grid_x, grid_y)] = 1
        else:
            self.occupancy_grid.pop((grid_x, grid_y), None)
        
        self._create_buffered_grid()
    
    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate heuristic cost between two grid positions."""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        
        if self.allow_diagonal:
            return math.sqrt(dx*dx + dy*dy)
        else:
            return dx + dy
    
    def _get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """Get valid neighboring cells and their costs."""
        x, y = position
        neighbors = []
        
        allow_diagonals_here = self.allow_diagonal and self.is_diagonal_allowed(x, y)
        
        if allow_diagonals_here:
            moves = [
                (0, 1, self.straight_cost),
                (1, 0, self.straight_cost),
                (0, -1, self.straight_cost),
                (-1, 0, self.straight_cost),
                (1, 1, self.diagonal_cost),
                (1, -1, self.diagonal_cost),
                (-1, -1, self.diagonal_cost),
                (-1, 1, self.diagonal_cost),
            ]
        else:
            moves = [
                (0, 1, self.straight_cost),
                (1, 0, self.straight_cost),
                (0, -1, self.straight_cost),
                (-1, 0, self.straight_cost),
            ]
        
        for dx, dy, cost in moves:
            new_x, new_y = x + dx, y + dy
            
            if self.is_valid_cell(new_x, new_y, use_buffer=True):
                if abs(dx) + abs(dy) == 2:
                    if not (self.is_valid_cell(x + dx, y, use_buffer=True) and 
                           self.is_valid_cell(x, y + dy, use_buffer=True)):
                        continue
                
                neighbors.append(((new_x, new_y), cost))
        
        return neighbors
    
    def _line_of_sight(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if there's a clear line of sight between two grid positions."""
        x0, y0 = pos1
        x1, y1 = pos2
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        x_step = 1 if x0 < x1 else -1
        y_step = 1 if y0 < y1 else -1
        
        error = dx - dy
        x, y = x0, y0
        
        while True:
            if not self.is_valid_cell(x, y, use_buffer=False):
                return False
            
            if x == x1 and y == y1:
                break
            
            error2 = 2 * error
            
            if error2 > -dy:
                error -= dy
                x += x_step
            
            if error2 < dx:
                error += dx
                y += y_step
        
        return True
    
    def _smooth_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Smooth path by removing unnecessary waypoints while maintaining safety."""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        current_idx = 0
        
        if len(path) > 2:
            dist_to_first = math.sqrt((path[1][0] - path[0][0])**2 + 
                                      (path[1][1] - path[0][1])**2)
            if dist_to_first < self.cell_size * 1.5:
                current_idx = 1
        
        while current_idx < len(path) - 1:
            farthest_idx = current_idx + 1
            
            for test_idx in range(current_idx + 2, len(path)):
                current_grid = self.world_to_grid(*smoothed[-1])
                test_grid = self.world_to_grid(*path[test_idx])
                
                if self._line_of_sight(current_grid, test_grid):
                    farthest_idx = test_idx
                else:
                    break
            
            if farthest_idx < len(path):
                smoothed.append(path[farthest_idx])
                current_idx = farthest_idx
            else:
                break
        
        if smoothed[-1] != path[-1]:
            smoothed.append(path[-1])
        
        return smoothed
    
    def _optimize_path_segments(self, path: List[Tuple[float, float]], 
                                min_segment_length: float = None) -> List[Tuple[float, float]]:
        """Optimize path by creating longer directional segments while respecting safety buffers."""
        if len(path) <= 2:
            return path
        
        if min_segment_length is None:
            min_segment_length = self.cell_size * 2.0
        
        optimized = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            best_idx = current_idx + 1
            best_score = 0.0
            
            for test_idx in range(current_idx + 1, len(path)):
                current_pos = optimized[-1]
                test_pos = path[test_idx]
                
                segment_length = math.sqrt(
                    (test_pos[0] - current_pos[0])**2 + 
                    (test_pos[1] - current_pos[1])**2
                )
                
                current_grid = self.world_to_grid(*current_pos)
                test_grid = self.world_to_grid(*test_pos)
                
                if not self._line_of_sight_safe(current_grid, test_grid):
                    break
                
                direction_score = 1.0
                if len(optimized) >= 2:
                    prev_pos = optimized[-2]
                    
                    prev_dx = current_pos[0] - prev_pos[0]
                    prev_dy = current_pos[1] - prev_pos[1]
                    prev_len = math.sqrt(prev_dx**2 + prev_dy**2)
                    
                    curr_dx = test_pos[0] - current_pos[0]
                    curr_dy = test_pos[1] - current_pos[1]
                    curr_len = math.sqrt(curr_dx**2 + curr_dy**2)
                    
                    if prev_len > 0 and curr_len > 0:
                        dot_product = (prev_dx * curr_dx + prev_dy * curr_dy) / (prev_len * curr_len)
                        direction_score = (dot_product + 1.0) / 2.0
                
                length_score = min(segment_length / (self.cell_size * 10.0), 1.0)
                score = length_score * 0.7 + direction_score * 0.3
                
                if score > best_score and segment_length >= min_segment_length:
                    best_score = score
                    best_idx = test_idx
                elif segment_length >= min_segment_length:
                    best_idx = test_idx
            
            optimized.append(path[best_idx])
            current_idx = best_idx
            
            if best_idx >= len(path) - 1:
                break
        
        if optimized[-1] != path[-1]:
            optimized.append(path[-1])
        
        return optimized
    
    def _line_of_sight_safe(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if there's a safe line of sight between two grid positions."""
        x0, y0 = pos1
        x1, y1 = pos2
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        x_step = 1 if x0 < x1 else -1
        y_step = 1 if y0 < y1 else -1
        
        error = dx - dy
        x, y = x0, y0
        
        while True:
            if not self._is_cell_safe_with_buffer(x, y):
                return False
            
            if x == x1 and y == y1:
                break
            
            error2 = 2 * error
            
            if error2 > -dy:
                error -= dy
                x += x_step
            
            if error2 < dx:
                error += dx
                y += y_step
        
        return True
    
    def _is_cell_safe_with_buffer(self, grid_x: int, grid_y: int) -> bool:
        """Check if a cell is safe considering the safety buffer."""
        return self.buffered_grid.get((grid_x, grid_y), 0) == 0
    
    def plan_path(self, start_x: float, start_y: float, 
                  goal_x: float, goal_y: float) -> Optional[List[Tuple[float, float]]]:
        """Plan a path from start to goal using A* algorithm with path smoothing and optimization."""
        self.exact_goal = (goal_x, goal_y)
        
        start_grid = self.world_to_grid(start_x, start_y)
        goal_grid = self.world_to_grid(goal_x, goal_y)
        
        if not self.is_valid_cell(*start_grid, use_buffer=False):
            print(f"Warning: Start position {start_grid} is not valid")
            return None
        
        if not self.is_valid_cell(*goal_grid, use_buffer=False):
            print(f"Warning: Goal position {goal_grid} is not valid")
            return None
        
        start_node = Node(start_grid)
        goal_node = Node(goal_grid)
        
        open_list = []
        heapq.heappush(open_list, start_node)
        closed_set = set()
        open_dict = {start_node.position: start_node}
        
        while open_list:
            current_node = heapq.heappop(open_list)
            
            if current_node.position in open_dict:
                del open_dict[current_node.position]
            
            closed_set.add(current_node.position)
            
            if current_node == goal_node:
                raw_path = self._reconstruct_path(current_node)
                
                if raw_path:
                    raw_path[-1] = self.exact_goal
                
                print(f"Raw path: {len(raw_path)} waypoints")
                
                smoothed_path = self._smooth_path(raw_path)
                
                if smoothed_path:
                    smoothed_path[-1] = self.exact_goal
                
                print(f"Smoothed path: {len(smoothed_path)} waypoints")
                
                if self.aggressive_smoothing:
                    optimized_path = self._optimize_path_segments(smoothed_path)
                    
                    if optimized_path:
                        optimized_path[-1] = self.exact_goal
                    
                    print(f"Optimized path: {len(optimized_path)} waypoints")
                    
                    self.current_path = optimized_path
                    self.current_waypoint_index = 0
                    
                    return optimized_path
                else:
                    self.current_path = smoothed_path
                    self.current_waypoint_index = 0
                    
                    return smoothed_path
            
            for neighbor_pos, move_cost in self._get_neighbors(current_node.position):
                if neighbor_pos in closed_set:
                    continue
                
                neighbor_node = Node(neighbor_pos, current_node)
                neighbor_node.g = current_node.g + move_cost
                neighbor_node.h = self._heuristic(neighbor_pos, goal_node.position)
                neighbor_node.f = neighbor_node.g + neighbor_node.h
                
                if neighbor_pos in open_dict:
                    existing_node = open_dict[neighbor_pos]
                    if neighbor_node.g < existing_node.g:
                        existing_node.g = neighbor_node.g
                        existing_node.f = neighbor_node.f
                        existing_node.parent = current_node
                        heapq.heapify(open_list)
                else:
                    heapq.heappush(open_list, neighbor_node)
                    open_dict[neighbor_pos] = neighbor_node
        
        print(f"No path found from {start_grid} to {goal_grid}")
        return None
    
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
        """Get the next waypoint to navigate to."""
        if not self.current_path or self.current_waypoint_index >= len(self.current_path):
            return None
        
        waypoint = self.current_path[self.current_waypoint_index]
        
        distance = math.sqrt((current_x - waypoint[0])**2 + (current_y - waypoint[1])**2)
        
        is_final_waypoint = (self.current_waypoint_index == len(self.current_path) - 1)
        
        if is_final_waypoint:
            tolerance = final_waypoint_tolerance
        else:
            tolerance = self.goal_tolerance
        
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
        """Clear the current path."""
        self.current_path = []
        self.current_waypoint_index = 0
        self.exact_goal = None