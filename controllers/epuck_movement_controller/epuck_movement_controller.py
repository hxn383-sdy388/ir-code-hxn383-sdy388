"""Main e-puck robot controller."""

from controller import Robot, Keyboard
import math

from odometry import Odometry
from motion import EPuckMotionController
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

# Control mode
mode = config.MODE_MANUAL

# Main control loop
while robot.step(timestep) != -1:
    # Update subsystems
    odometry.update(left_encoder.getValue(), right_encoder.getValue())
    motion.update()  # Apply acceleration/deceleration
    
    # Get and display pose
    x, y, theta = odometry.get_pose()
    left_vel, right_vel = motion.get_current_velocities()
    print(f"Pose: ({x:.3f}, {y:.3f}, {math.degrees(theta):.1f}°) | "
          f"Vel: L={left_vel:.2f}, R={right_vel:.2f}")
    
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
        elif key == ord('e'):  # Emergency stop
            motion.emergency_stop()