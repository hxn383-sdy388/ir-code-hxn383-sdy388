"""Main e-puck robot controller."""

from controller import Robot, Keyboard
import math

# Import our modules
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
motion = EPuckMotionController(left_motor, right_motor, config.TURNING_FACTOR)

# Control mode
mode = config.MODE_MANUAL

# Main control loop
while robot.step(timestep) != -1:
    # Update odometry
    odometry.update(left_encoder.getValue(), right_encoder.getValue())
    
    # Get and display pose
    x, y, theta = odometry.get_pose()
    print(f"Position: ({x:.3f}, {y:.3f}) - Orientation: {math.degrees(theta):.1f}°")
    
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