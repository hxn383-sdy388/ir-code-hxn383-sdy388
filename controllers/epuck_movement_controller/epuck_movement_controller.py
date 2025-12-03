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
odometry = Odometry(
    config.WHEEL_RADIUS,
    config.AXLE_LENGTH,
    linear_scale=config.ODOM_LINEAR_SCALE,
    angular_scale=config.ODOM_ANGULAR_SCALE,
    wheel_left_scale=config.ODOM_WHEEL_LEFT_SCALE,
    wheel_right_scale=config.ODOM_WHEEL_RIGHT_SCALE
)

motion = EPuckMotionController(
    left_motor,
    right_motor,
    timestep,
    max_acceleration=config.MAX_ACCELERATION,
    max_linear_deceleration=config.MAX_LINEAR_DECELERATION,
    max_angular_deceleration=config.MAX_ANGULAR_DECELERATION,
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
goal_position = None  # Track the final goal for replanning

# SLAM state variables
x_t = [0, 0, 0]  # Pose at current time t: [x, y, theta]
u_t = [0, 0]     # Control at time t: [linear_velocity, angular_velocity]
z_t = []         # Landmark measurements: [(distance, bearing, correspondence), ...]

# Stable grid for path planning (initialized with empty 60x60 grid)
# Use list comprehension to ensure deep copy
stable_grid = [row[:] for row in initial_occupancy_grid]

# Dynamic replanning state
last_replan_iteration = 0
stuck_counter = 0
last_position = (0.0, 0.0)
stuck_check_start_position = (0.0, 0.0)  # Position when stuck counter started incrementing
replan_attempts = 0
MAX_REPLAN_ATTEMPTS = 5  # Maximum consecutive replan attempts before giving up
obstacle_avoidance_active = False  # Track if we're in obstacle avoidance mode
consecutive_obstacle_detections = 0  # Track how long obstacle has been detected

# Critical obstacle recovery state
backup_mode_active = False
backup_iterations = 0
BACKUP_DURATION = 30  # Number of iterations to backup
BACKUP_SPEED = 1.5    # Speed for backing up (rad/s)


def plan_path_to_goal(planner, current_x, current_y, goal_x, goal_y):
    """Helper function to plan path in robot-relative coordinates."""
    # Convert goal to robot-relative coordinates
    goal_rel_x = goal_x - current_x
    goal_rel_y = goal_y - current_y

    # Store current position for coordinate conversion
    planner.robot_world_x = current_x
    planner.robot_world_y = current_y

    # Plan in relative coordinates
    path = planner.plan_path(0.0, 0.0, goal_rel_x, goal_rel_y)

    if path:
        # Convert to world coordinates
        world_path = []
        for wp_rel_x, wp_rel_y in path:
            wp_world_x = current_x + wp_rel_x
            wp_world_y = current_y + wp_rel_y
            world_path.append((wp_world_x, wp_world_y))
        planner.current_path = world_path
        return world_path
    return None


def find_avoidance_direction(collision_status, current_theta, goal_x, goal_y, current_x, current_y):
    """Determine best direction to avoid obstacle based on sensor readings and goal direction."""
    if not collision_status['obstacle_detected']:
        return None

    obstacle_dir = collision_status.get('obstacle_direction', 0)
    if obstacle_dir is None:
        return None

    # Calculate perpendicular directions to obstacle (in world frame)
    # obstacle_dir is relative to robot, convert to world frame
    obstacle_world_dir = current_theta + obstacle_dir

    perp_left = obstacle_world_dir + math.pi / 2
    perp_right = obstacle_world_dir - math.pi / 2

    # Normalize angles
    perp_left = math.atan2(math.sin(perp_left), math.cos(perp_left))
    perp_right = math.atan2(math.sin(perp_right), math.cos(perp_right))

    # Calculate direction to goal
    goal_dir = math.atan2(goal_y - current_y, goal_x - current_x)

    # Choose direction that gets us closer to the goal direction
    diff_left = abs(math.atan2(math.sin(perp_left - goal_dir),
                               math.cos(perp_left - goal_dir)))
    diff_right = abs(math.atan2(math.sin(perp_right - goal_dir),
                                math.cos(perp_right - goal_dir)))

    return perp_left if diff_left < diff_right else perp_right


def create_local_avoidance_waypoint(current_x, current_y, avoidance_direction, distance):
    """Create a temporary waypoint to avoid local obstacle."""
    avoid_x = current_x + distance * math.cos(avoidance_direction)
    avoid_y = current_y + distance * math.sin(avoidance_direction)
    return (avoid_x, avoid_y)


def execute_obstacle_avoidance(motion_controller, collision_status, current_theta, turn_speed):
    """Execute reactive obstacle avoidance - turn away from obstacle."""
    obstacle_dir = collision_status.get('obstacle_direction', 0)
    if obstacle_dir is None:
        return False

    # Determine which way to turn based on obstacle location
    # obstacle_dir is in robot frame: positive = left, negative = right
    if obstacle_dir > 0:
        # Obstacle on left, turn right
        motion_controller.rotate_right(turn_speed * 0.5)
        return True
    else:
        # Obstacle on right, turn left
        motion_controller.rotate_left(turn_speed * 0.5)
        return True

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
        z_t = measurement_controller.lidar_measure_landmarks()
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

            # Store goal for potential replanning
            goal_position = start_position

            # Check if goal is within our stable grid window
            goal_rel_x = goal_position[0] - current_x
            goal_rel_y = goal_position[1] - current_y
            if abs(goal_rel_x) > config.PATH_PLANNING_MAX_RANGE or abs(goal_rel_y) > config.PATH_PLANNING_MAX_RANGE:
                print(f"Warning: Goal is {math.sqrt(goal_rel_x**2 + goal_rel_y**2):.2f}m away")
                print(f"Path planning may fail as target is outside stable grid window (3m x 3m)")

            # Plan path using helper function
            world_path = plan_path_to_goal(path_planner, current_x, current_y,
                                           goal_position[0], goal_position[1])

            if world_path:
                mode = config.MODE_AUTONOMOUS
                navigator.reset()
                replan_attempts = 0  # Reset replan counter on successful initial plan
                last_replan_iteration = iteration
                print(f"Final path: {len(world_path)} waypoints")
                for i, wp in enumerate(world_path):
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
        # Get collision avoidance status for dynamic replanning
        ca_status = collision_avoidance.get_status()
        max_sensor = ca_status['max_sensor_value']

        # Check if we're stuck (not making progress)
        # Only check for stuck when actually moving forward, not during turns
        nav_state = navigator.get_state()

        # Calculate distance moved since last iteration
        distance_moved = math.sqrt((current_x - last_position[0])**2 +
                                   (current_y - last_position[1])**2)

        # Calculate total distance moved since stuck check started
        total_distance_from_start = math.sqrt(
            (current_x - stuck_check_start_position[0])**2 +
            (current_y - stuck_check_start_position[1])**2
        )

        # Only count as stuck when in 'moving' state - turning in place is expected to not move
        if nav_state == 'moving':
            if distance_moved < config.REPLAN_STUCK_DISTANCE:
                if stuck_counter == 0:
                    # Just started potential stuck period - record starting position
                    stuck_check_start_position = (current_x, current_y)
                stuck_counter += 1
            else:
                # Made progress this iteration
                stuck_counter = 0
                # Only reset replan attempts if we've moved significantly
                if distance_moved > config.REPLAN_STUCK_DISTANCE * 10:
                    replan_attempts = 0
                    obstacle_avoidance_active = False
        elif nav_state in ('turning', 'stabilizing'):
            # Don't increment stuck counter during turning, but don't reset either
            # unless we've been turning for a very long time
            if stuck_counter > config.REPLAN_STUCK_ITERATIONS * 2:
                # Very long turning - might actually be stuck
                pass  # Keep the counter
            # Otherwise just don't increment
        else:
            # Idle or other state - reset if we have moved overall
            if total_distance_from_start > config.REPLAN_STUCK_DISTANCE_TOTAL:
                stuck_counter = 0
                replan_attempts = 0

        last_position = (current_x, current_y)

        # Track consecutive obstacle detections
        if ca_status['danger_zone'] or ca_status['critical']:
            consecutive_obstacle_detections += 1
        else:
            consecutive_obstacle_detections = 0
            obstacle_avoidance_active = False

        # REACTIVE OBSTACLE AVOIDANCE
        # If we detect an obstacle while moving, immediately start turning away
        if ca_status['danger_zone'] and navigator.get_state() == 'moving' and goal_position is not None:
            if consecutive_obstacle_detections > 3:  # Confirm it's not a glitch
                if not obstacle_avoidance_active:
                    print(f"\n[OBSTACLE AVOIDANCE] Danger zone detected - turning away")
                    obstacle_avoidance_active = True

                # Execute reactive avoidance - turn away from obstacle
                execute_obstacle_avoidance(motion, ca_status, current_theta, config.NAVIGATION_TURN_SPEED)

                # After turning for a bit, insert an avoidance waypoint
                if consecutive_obstacle_detections > 15 and consecutive_obstacle_detections % 20 == 0:
                    avoidance_dir = find_avoidance_direction(
                        ca_status, current_theta,
                        goal_position[0], goal_position[1],
                        current_x, current_y
                    )
                    if avoidance_dir is not None:
                        avoid_wp = create_local_avoidance_waypoint(
                            current_x, current_y, avoidance_dir,
                            config.LOCAL_AVOIDANCE_DISTANCE * 1.5
                        )
                        print(f"[OBSTACLE AVOIDANCE] Inserting detour waypoint: ({avoid_wp[0]:.3f}, {avoid_wp[1]:.3f})")

                        current_wp_idx = path_planner.current_waypoint_index
                        if current_wp_idx < len(path_planner.current_path):
                            path_planner.current_path.insert(current_wp_idx, avoid_wp)
                            navigator.reset()
                            stuck_counter = 0
                            replan_attempts += 1

        # CRITICAL OBSTACLE BACKUP BEHAVIOR
        # When critically close to obstacle, back up first before replanning
        if backup_mode_active:
            # Continue backing up
            motion.backward(BACKUP_SPEED)
            backup_iterations += 1

            # Check if we should stop backing up
            if backup_iterations >= BACKUP_DURATION or not ca_status['critical']:
                print(f"[BACKUP] Completed - backed up for {backup_iterations} iterations")
                backup_mode_active = False
                backup_iterations = 0
                motion.stop()
                navigator.reset()

                # Now do a full replan from current position
                print(f"[BACKUP] Replanning from new position ({current_x:.3f}, {current_y:.3f})")
                new_path = plan_path_to_goal(
                    path_planner, current_x, current_y,
                    goal_position[0], goal_position[1]
                )
                if new_path:
                    print(f"[BACKUP] New path found: {len(new_path)} waypoints")
                    stuck_counter = 0
                    stuck_check_start_position = (current_x, current_y)
                    replan_attempts = 0  # Reset attempts after successful backup+replan
                    consecutive_obstacle_detections = 0
                else:
                    print("[BACKUP] Replanning failed - no path found")

            # Skip normal navigation while backing up
            # (handled by the elif below)

        elif ca_status['critical'] and consecutive_obstacle_detections > 5 and goal_position is not None:
            # Trigger backup mode when critically close to obstacle
            if not backup_mode_active:
                print(f"\n[BACKUP] Critical obstacle detected - initiating backup maneuver")
                backup_mode_active = True
                backup_iterations = 0
                navigator.reset()  # Stop current navigation
                motion.backward(BACKUP_SPEED)

        # REPLANNING LOGIC
        # Determine if we need to replan
        needs_replan = False
        replan_reason = ""

        # Check if stuck for too long (and not already in avoidance mode or backup mode)
        # Must meet BOTH criteria: high stuck counter AND low total movement
        if stuck_counter > config.REPLAN_STUCK_ITERATIONS and not obstacle_avoidance_active and not backup_mode_active:
            # Also verify we haven't actually moved much overall
            if total_distance_from_start < config.REPLAN_STUCK_DISTANCE_TOTAL:
                if iteration - last_replan_iteration > config.REPLAN_COOLDOWN_ITERATIONS:
                    needs_replan = True
                    replan_reason = f"Stuck for {stuck_counter} iterations (moved only {total_distance_from_start:.3f}m)"
                    replan_attempts += 1  # Count this as an attempt
            else:
                # We've actually moved - reset stuck counter
                stuck_counter = 0
                stuck_check_start_position = (current_x, current_y)

        # Note: Critical obstacle handling is now done via backup_mode above
        # Only trigger traditional replan if backup mode is not active
        elif ca_status['critical'] and consecutive_obstacle_detections > 10 and not backup_mode_active:
            # This branch handles cases where backup didn't trigger (e.g., no goal)
            if iteration - last_replan_iteration > config.REPLAN_COOLDOWN_ITERATIONS:
                needs_replan = True
                replan_reason = "CRITICAL obstacle - need alternate path"
                replan_attempts += 1

        # Perform replanning if needed
        if needs_replan and goal_position is not None and replan_attempts < MAX_REPLAN_ATTEMPTS:
            print(f"\n{'='*40}")
            print(f"REPLANNING - Reason: {replan_reason}")
            print(f"Attempt {replan_attempts}/{MAX_REPLAN_ATTEMPTS}")

            # Try local avoidance first - insert waypoint perpendicular to obstacle
            avoidance_dir = find_avoidance_direction(
                ca_status, current_theta,
                goal_position[0], goal_position[1],
                current_x, current_y
            )

            if avoidance_dir is not None:
                # Create avoidance waypoint further out
                avoid_distance = config.LOCAL_AVOIDANCE_DISTANCE * (1 + replan_attempts * 0.5)
                avoid_wp = create_local_avoidance_waypoint(
                    current_x, current_y, avoidance_dir, avoid_distance
                )
                print(f"Local avoidance waypoint: ({avoid_wp[0]:.3f}, {avoid_wp[1]:.3f})")

                # Insert at beginning of remaining path
                current_wp_idx = path_planner.current_waypoint_index
                if current_wp_idx < len(path_planner.current_path):
                    path_planner.current_path.insert(current_wp_idx, avoid_wp)
                    print(f"Inserted avoidance waypoint")
                    navigator.reset()
                    stuck_counter = 0
            else:
                # No clear avoidance direction, try full replan
                print(f"Full replan from ({current_x:.3f}, {current_y:.3f}) to goal")
                path_planner.load_occupancy_grid(stable_grid)

                new_path = plan_path_to_goal(path_planner, current_x, current_y,
                                             goal_position[0], goal_position[1])

                if new_path:
                    print(f"New path found: {len(new_path)} waypoints")
                    navigator.reset()
                    stuck_counter = 0
                else:
                    print("Replanning failed - no valid path found")

            last_replan_iteration = iteration
            print(f"{'='*40}\n")

        # Check if we've exceeded replan attempts
        if replan_attempts >= MAX_REPLAN_ATTEMPTS:
            print("\nMax replan attempts reached - switching to manual mode")
            mode = config.MODE_MANUAL
            navigator.reset()
            path_planner.clear_path()
            replan_attempts = 0
            obstacle_avoidance_active = False

        # Normal waypoint navigation (only if not in active avoidance or backup mode)
        if (not obstacle_avoidance_active or not ca_status['danger_zone']) and not backup_mode_active:
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
                    error_x = final_x - goal_position[0]
                    error_y = final_y - goal_position[1]
                    error_distance = math.sqrt(error_x**2 + error_y**2)
                    print(f"Final position: ({final_x:.3f}, {final_y:.3f})")
                    print(f"Target position: ({goal_position[0]:.3f}, {goal_position[1]:.3f})")
                    print(f"Position error: {error_distance*100:.1f} cm")
                    print("="*60 + "\n")

                mode = config.MODE_MANUAL
                navigator.reset()
                goal_position = None
                obstacle_avoidance_active = False
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
            status_str = " [CRITICAL!]"
        elif status['danger_zone']:
            status_str = " [DANGER]"
        elif status['obstacle_detected']:
            status_str = " [OBSTACLE]"

        # Add stuck indicator
        if stuck_counter > 10:
            status_str += f" [STUCK:{stuck_counter}]"

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