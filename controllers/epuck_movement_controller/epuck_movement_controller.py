"""Main e-puck robot controller with collision avoidance and path planning."""
from controller import Robot, Keyboard
import math
import numpy as np
from odometry import Odometry
from motion import EPuckMotionController
from collision_avoidance import CollisionAvoidance
from astar_planner import AStarPlanner
from navigation_controller import NavigationController
from ekf_slam import EkfSlamController
from displays import DisplayController
from measurements import MeasurementController
from smart_display_grid import SmartDisplayGrid
import config
import slam_utils
import grid_utils
import sensor_utils

# Create robot instance
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Initialize hardware devices
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_encoder = robot.getDevice('left wheel sensor')
right_encoder = robot.getDevice('right wheel sensor')
keyboard = robot.getKeyboard()

# Enable sensors
keyboard.enable(timestep)
left_encoder.enable(timestep)
right_encoder.enable(timestep)

# Initialize subsystems
odometry = Odometry(config.WHEEL_RADIUS, config.AXLE_LENGTH)

motion = EPuckMotionController(
    left_motor, 
    right_motor, 
    timestep,
    max_acceleration=config.MAX_ACCELERATION,
    max_deceleration=config.MAX_DECELERATION,
    turning_factor=config.TURNING_FACTOR
)

collision_avoidance = CollisionAvoidance(
    robot,
    timestep,
    sensor_positions=config.SENSOR_POSITIONS,
    sensor_groups=config.SENSOR_GROUPS,
    obstacle_threshold=config.OBSTACLE_THRESHOLD,
    danger_threshold=config.DANGER_THRESHOLD,
    critical_threshold=config.CRITICAL_THRESHOLD,
    danger_speed_factor=config.DANGER_ZONE_SPEED_FACTOR,
    obstacle_speed_reduction=config.OBSTACLE_SPEED_REDUCTION,
    approach_angle_threshold=config.APPROACH_ANGLE_THRESHOLD,
    min_velocity_threshold=config.MIN_VELOCITY_THRESHOLD
)

path_planner = AStarPlanner(
    grid_cell_size=config.GRID_CELL_SIZE,
    origin_marker=config.GRID_ORIGIN_MARKER,
    diagonal_cost=config.ASTAR_DIAGONAL_COST,
    straight_cost=config.ASTAR_STRAIGHT_COST,
    allow_diagonal=config.ASTAR_ALLOW_DIAGONAL,
    goal_tolerance=config.ASTAR_GOAL_TOLERANCE,
    safety_buffer=config.GRID_SAFETY_BUFFER,
    diagonal_restriction_buffer=config.GRID_DIAGONAL_RESTRICTION_BUFFER,
    aggressive_smoothing=config.ASTAR_AGGRESSIVE_SMOOTHING
)

navigator = NavigationController(
    motion_controller=motion,
    heading_tolerance=config.HEADING_TOLERANCE,
    waypoint_tolerance=config.WAYPOINT_DISTANCE_TOLERANCE,
    turn_speed=config.NAVIGATION_TURN_SPEED,
    forward_speed=config.NAVIGATION_FORWARD_SPEED
)

# Initialize EKF-SLAM controller
ekf_slam_controller = EkfSlamController(timestep)

# Initialize measurement controller with lidar (verify lidar exists)
try:
    measurement_controller = MeasurementController(robot, timestep)
    measurements_available = True
except Exception as e:
    measurements_available = False

# Initialize display controller (optional - check if displays exist)
try:
    display_controller = DisplayController(robot)
    displays_available = True
    # Initialize smart display grid system
    smart_display = SmartDisplayGrid(display_size=20, base_cell_size=0.05)
except:
    displays_available = False
    smart_display = None

# Initialize with fixed-size occupancy grid
# Three-tier grid system:
# 1. EKF-SLAM Grid: Dynamic, can grow with landmarks (internal use only)
# 2. Planning Grid: 60x60 = 3m x 3m at 5cm resolution (for path planning)
# 3. Display Grid: 20x20 with dynamic resolution to show origin (blue) and robot (green)
GRID_SIZE = config.PLANNING_GRID_SIZE  # For path planning (from config)
DISPLAY_GRID_SIZE = config.DISPLAY_GRID_SIZE  # For display only (from config)
initial_occupancy_grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
# Place origin marker at center
initial_occupancy_grid[GRID_SIZE // 2][GRID_SIZE // 2] = config.GRID_ORIGIN_MARKER
path_planner.load_occupancy_grid(initial_occupancy_grid)

# Control state
mode = config.MODE_MANUAL
start_position = None

# SLAM state variables
x_t = [0, 0, 0]  # Pose at current time t: [x, y, theta]
u_t = [0, 0]     # Control at time t: [linear_velocity, angular_velocity]
z_t = []         # Landmark measurements: [(distance, bearing, correspondence), ...]

# Stable grid for path planning (initialized with empty 60x60 grid)
# Use list comprehension to ensure deep copy
stable_grid = [row[:] for row in initial_occupancy_grid]

# Main control loop
iteration = 0
while robot.step(timestep) != -1:
    iteration += 1

    # Update subsystems
    collision_avoidance.update()
    odometry.update(left_encoder.getValue(), right_encoder.getValue())
    current_x, current_y, current_theta = odometry.get_pose()

    # Update SLAM state from odometry and motion controller
    slam_utils.update_pose_from_odometry(odometry, x_t)  # Updates x_t from odometry
    u_t = slam_utils.calculate_control_vector(motion)  # Calculate control vector from wheel velocities

    # Collect landmark measurements
    if measurements_available:
        raw_measurements = measurement_controller.lidar_measure_landmarks()
        z_t = sensor_utils.filter_wall_measurements(raw_measurements)
    else:
        z_t = []

    # Run EKF-SLAM processing
    # Note: EKF-SLAM internally updates its state_estimate and covariance
    state_estimate, covariance = ekf_slam_controller.run_steps(u_t, z_t)

    # Create stable grid from SLAM state (replaces the unstable EKF grid)
    # This grid has fixed size (60x60) and won't grow infinitely
    stable_grid = grid_utils.create_stable_occupancy_grid(state_estimate, current_x, current_y)

    # Note: We do NOT call ekf_slam_controller.construct_occupancy_grid() anymore
    # because it can grow infinitely large and cause memory errors.
    # The stable grid serves all our needs for display and path planning.

    # Update path planner less frequently (every 10 iterations to avoid overhead)
    if iteration % 10 == 0:
        try:
            path_planner.load_occupancy_grid(stable_grid)
            # Debug: Verify grid size
            if iteration % 100 == 0:
                print(f"Path planner grid updated: {len(stable_grid)}x{len(stable_grid[0])} cells")
                # Stable grid size is fixed, no need to check for growth
        except Exception as e:
            print(f"Warning: Failed to update path planner grid: {e}")
            print(f"Stable grid dimensions: {len(stable_grid)}x{len(stable_grid[0]) if stable_grid else 0}")
            # DO NOT fall back to EKF grid - keep using last known stable grid

    # Update displays if available
    if displays_available and smart_display:
        display_controller.clean_displays()
        display_controller.draw_true_pose(x_t)  # Draw odometry-based pose
        display_controller.draw_state_estimate(state_estimate)  # Draw SLAM estimate
        display_controller.temp_draw_landmarks()  # Draw known landmark positions

        # Use smart display grid system
        origin_x = start_position[0] if start_position else 0
        origin_y = start_position[1] if start_position else 0

        # Get current path and waypoint info if in autonomous mode
        current_path = None
        current_waypoint_idx = None
        if mode == config.MODE_AUTONOMOUS:
            if hasattr(path_planner, 'current_path') and path_planner.current_path:
                current_path = path_planner.current_path
                # Get current waypoint index
                if hasattr(path_planner, 'current_waypoint_index'):
                    current_waypoint_idx = path_planner.current_waypoint_index
                else:
                    # Try to calculate it based on progress
                    current_wp, total_wp = path_planner.get_path_progress()
                    if current_wp > 0:
                        current_waypoint_idx = current_wp - 1

        # Get robot heading (theta)
        robot_theta = current_theta

        # Create adaptive display with ALL smart features including buffer zones
        display_grid, display_metadata = smart_display.create_adaptive_display(
            state_estimate, current_x, current_y,
            origin_x, origin_y,
            path_waypoints=current_path,
            current_waypoint_idx=current_waypoint_idx,
            robot_theta=robot_theta,
            safety_buffer_cells=config.GRID_SAFETY_BUFFER,      # Safety buffer from config
            diagonal_buffer_cells=config.GRID_DIAGONAL_RESTRICTION_BUFFER     # Diagonal buffer from config
        )

        # Find special marker positions for display controller
        origin_pos = display_metadata.get('origin_grid_pos', (10, 10))
        robot_pos = display_metadata.get('robot_grid_pos', (10, 10))

        # Draw the smart grid
        display_controller.draw_occupancy_grid(origin_pos, robot_pos, display_grid)

    # Store start position on first iteration
    if start_position is None:
        start_position = (current_x, current_y)
        print(f"Start position recorded: ({start_position[0]:.3f}, {start_position[1]:.3f})")
    
    # Handle keyboard input
    key = keyboard.getKey()
    
    if key == ord('R'):  # Return to start
        if mode != config.MODE_AUTONOMOUS:
            print(f"\n{'='*60}")
            print(f"RETURN TO START ACTIVATED - ENHANCED VISUALIZATION ENABLED")
            print(f"Current: ({current_x:.3f}, {current_y:.3f})")
            print(f"Target:  ({start_position[0]:.3f}, {start_position[1]:.3f})")
            
            # Plan path using stable grid coordinates (robot-centric)
            # Convert start position to relative coordinates for path planning
            rel_start_x = start_position[0] - current_x
            rel_start_y = start_position[1] - current_y

            # Check if start position is within our stable grid window
            if abs(rel_start_x) > config.PATH_PLANNING_MAX_RANGE or abs(rel_start_y) > config.PATH_PLANNING_MAX_RANGE:  # Outside planning range
                print(f"Warning: Start position is {math.sqrt(rel_start_x**2 + rel_start_y**2):.2f}m away")
                print(f"Path planning may fail as target is outside stable grid window (3m x 3m)")

            path = path_planner.plan_path(
                current_x, current_y,
                start_position[0], start_position[1]
            )
            
            if path:
                mode = config.MODE_AUTONOMOUS
                navigator.reset()
                print(f"Final path: {len(path)} waypoints")
                for i, wp in enumerate(path):
                    print(f"  Waypoint {i}: ({wp[0]:.3f}, {wp[1]:.3f})")
            else:
                print("Failed to plan path to start position!")
    
    elif key == ord('M'):  # Manual mode
        if mode == config.MODE_AUTONOMOUS:
            print("\nSwitching to MANUAL mode")
            mode = config.MODE_MANUAL
            navigator.reset()
            path_planner.clear_path()
    
    # Control based on mode
    if mode == config.MODE_AUTONOMOUS:
        waypoint = path_planner.get_next_waypoint(
            current_x, current_y,
            final_waypoint_tolerance=config.WAYPOINT_DISTANCE_TOLERANCE
        )
        
        if waypoint is None:
            # Reached final goal
            if navigator.get_state() != 'idle':
                print("\n" + "="*60)
                print("DESTINATION REACHED!")
                final_x, final_y, final_theta = odometry.get_pose()
                error_x = final_x - start_position[0]
                error_y = final_y - start_position[1]
                error_distance = math.sqrt(error_x**2 + error_y**2)
                print(f"Final position: ({final_x:.3f}, {final_y:.3f})")
                print(f"Target position: ({start_position[0]:.3f}, {start_position[1]:.3f})")
                print(f"Position error: {error_distance*100:.1f} cm")
                print("="*60 + "\n")
            
            mode = config.MODE_MANUAL
            navigator.reset()
        else:
            # Navigate to waypoint
            target_x, target_y = waypoint
            current_wp, total_wp = path_planner.get_path_progress()
            is_final = (current_wp == total_wp - 1)
            
            navigator.navigate_to_waypoint(
                current_x, current_y, current_theta,
                target_x, target_y,
                is_final_waypoint=is_final
            )
            
    elif mode == config.MODE_MANUAL:
        if key == Keyboard.UP:
            motion.forward(config.NORMAL_SPEED)
        elif key == Keyboard.DOWN:
            motion.backward(config.NORMAL_SPEED)
        elif key == Keyboard.LEFT:
            motion.turn_left(config.NORMAL_SPEED)
        elif key == Keyboard.RIGHT:
            motion.turn_right(config.NORMAL_SPEED)
        elif key == ord(' '):
            motion.stop()
        elif key == ord('E'):
            motion.emergency_stop()
    
    # Apply collision avoidance safety layer
    desired_left, desired_right = motion.get_current_velocities()
    safe_left, safe_right = collision_avoidance.get_safe_velocities(
        desired_left, desired_right
    )
    
    if (abs(safe_left - desired_left) > 0.001 or 
        abs(safe_right - desired_right) > 0.001):
        motion.set_target_velocities(safe_left, safe_right)
    
    # Update motion controller
    motion.update()
    
    # Display status (every 10 iterations)
    if iteration % 10 == 0:
        left_vel, right_vel = motion.get_current_velocities()
        status = collision_avoidance.get_status()
        status_str = ""
        if status['critical']:
            status_str = " [CRITICAL OBSTACLE - STOPPED]"
        elif status['danger_zone']:
            status_str = " [DANGER ZONE]"
        elif status['obstacle_detected']:
            status_str = " [OBSTACLE DETECTED]"
        
        if mode == config.MODE_AUTONOMOUS:
            current_wp, total_wp = path_planner.get_path_progress()
            waypoint = path_planner.get_next_waypoint(
                current_x, current_y,
                final_waypoint_tolerance=config.WAYPOINT_DISTANCE_TOLERANCE
            )
            
            if waypoint:
                distance = math.sqrt((waypoint[0] - current_x)**2 + (waypoint[1] - current_y)**2)
                dx = waypoint[0] - current_x
                dy = waypoint[1] - current_y
                required_heading = math.atan2(dy, dx)
                heading_error = navigator.calculate_heading_error(current_theta, required_heading)
                
                nav_state = navigator.get_state().upper()
                if navigator.get_state() == 'stabilizing':
                    nav_state += f":{navigator.get_stabilize_counter()}"
                
                mode_str = (f"AUTO [{nav_state}] "
                           f"WP:{current_wp}/{total_wp} Dist:{distance:.2f}m "
                           f"HdgErr:{math.degrees(heading_error):.1f}°")
            else:
                mode_str = "AUTO [COMPLETE]"
        else:
            mode_str = "MANUAL"
        
        print(f"[{mode_str}] Pose: ({current_x:.3f}, {current_y:.3f}, {math.degrees(current_theta):.1f}°) | "
              f"Vel: L={left_vel:.2f}, R={right_vel:.2f}{status_str}")