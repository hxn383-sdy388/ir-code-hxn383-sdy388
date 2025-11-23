"""Main e-puck robot controller with collision avoidance and path planning."""
from controller import Robot, Keyboard
import math
from odometry import Odometry
from motion import EPuckMotionController
from collision_avoidance import CollisionAvoidance
from astar_planner import AStarPlanner
from navigation_controller import NavigationController
import config

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

# Initialize occupancy grid - 20x20 grid for 1m x 1m world (0.05m = 5cm cell resolution)
# Origin at physical position (0.5m, 0.5m) which maps to grid center (10, 10)
occupancy_grid_data = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 1
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 2
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 3
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 4
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 5
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 6
    [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 7
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 9
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '/', 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 10 - ORIGIN
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 11
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 12
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 13
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 14
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 15
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 16
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 17
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 18
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Row 19
]
path_planner.load_occupancy_grid(occupancy_grid_data)

# Control state
mode = config.MODE_MANUAL
start_position = None

# Main control loop
iteration = 0
while robot.step(timestep) != -1:
    iteration += 1
    
    # Update subsystems
    collision_avoidance.update()
    odometry.update(left_encoder.getValue(), right_encoder.getValue())
    current_x, current_y, current_theta = odometry.get_pose()
    
    # Store start position on first iteration
    if start_position is None:
        start_position = (current_x, current_y)
        print(f"Start position recorded: ({start_position[0]:.3f}, {start_position[1]:.3f})")
    
    # Handle keyboard input
    key = keyboard.getKey()
    
    if key == ord('R'):  # Return to start
        if mode != config.MODE_AUTONOMOUS:
            print(f"\n{'='*60}")
            print(f"RETURN TO START ACTIVATED")
            print(f"Current: ({current_x:.3f}, {current_y:.3f})")
            print(f"Target:  ({start_position[0]:.3f}, {start_position[1]:.3f})")
            print(f"{'='*60}\n")
            
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