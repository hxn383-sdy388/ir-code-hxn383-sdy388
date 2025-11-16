"""Main e-puck robot controller with collision avoidance."""

from controller import Robot, Keyboard
import math

from odometry import Odometry
from motion import EPuckMotionController
from collision_avoidance import CollisionAvoidance
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

# Initialize collision avoidance - TAKES ABSOLUTE PRECEDENCE
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

# Control mode
mode = config.MODE_MANUAL

# Main control loop
while robot.step(timestep) != -1:
    # Update collision avoidance FIRST
    collision_avoidance.update()
    
    # Update odometry
    odometry.update(left_encoder.getValue(), right_encoder.getValue())
    
    # Get collision status
    status = collision_avoidance.get_status()
    
    # Manual control mode
    if mode == config.MODE_MANUAL:
        key = keyboard.getKey()
        
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
        elif key == ord('E'):  # Emergency stop
            motion.emergency_stop()
    
    # Get desired velocities from motion controller
    desired_left, desired_right = motion.get_current_velocities()
    
    # CRITICAL: Filter through collision avoidance
    safe_left, safe_right = collision_avoidance.get_safe_velocities(
        desired_left, 
        desired_right
    )
    
    # Override motion controller if collision avoidance modified velocities
    if (abs(safe_left - desired_left) > 0.001 or 
        abs(safe_right - desired_right) > 0.001):
        motion.set_target_velocities(safe_left, safe_right)
    
    # Update motion controller (applies acceleration/deceleration)
    motion.update()
    
    # Display status
    x, y, theta = odometry.get_pose()
    left_vel, right_vel = motion.get_current_velocities()
    
    status_str = ""
    if status['critical']:
        status_str = " [CRITICAL OBSTACLE - STOPPED]"
    elif status['danger_zone']:
        status_str = " [DANGER ZONE]"
    elif status['obstacle_detected']:
        status_str = " [OBSTACLE DETECTED]"
    
    print(f"Pose: ({x:.3f}, {y:.3f}, {math.degrees(theta):.1f}°) | "
          f"Vel: L={left_vel:.2f}, R={right_vel:.2f}{status_str}")