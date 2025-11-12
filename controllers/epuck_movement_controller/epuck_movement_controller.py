"""epuck_movement_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Motor, Keyboard, PositionSensor
import math

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# variables for each motor...
right_motor = robot.getDevice('right wheel motor')
left_motor = robot.getDevice('left wheel motor')
left_encoder = robot.getDevice('left wheel sensor')
right_encoder = robot.getDevice('right wheel sensor')

# initialise the keybaord
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# initialise the encoders
left_encoder.enable(timestep)
right_encoder.enable(timestep)

# Set motors to velocity control mode
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Stop all movement
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Odometry constants
WHEEL_RADIUS = 0.02
AXLE_LENGTH = 0.05
ENCODER_RESOLUTION = 160

# Constant Global Variables...
MODE = 'manual' # Determines how to navigate the robot
MAXIMUM_SPEED = 5 # rad/s
NORMAL_SPEED = MAXIMUM_SPEED / 2
TURNING_FACTOR = 0.1 # Assigned to half for now

# Odometry state variables
x = 0.0 # x position in meters
y = 0.0 # y position in meters
theta = 0.0 # orientation in radians

# Previous encoder values
prev_left_encoder = 0.0
prev_right_encoder = 0.0

# Movement Functions - Follow prefixes
# Move (m)
# Turn (t)
# Rotate (r)

def stop():
    left_motor.setVelocity(0)
    right_motor.setVelocity(0)
    
def m_forward(s):
    left_motor.setVelocity(s)
    right_motor.setVelocity(s)

def m_backward(s):
    left_motor.setVelocity(-s)
    right_motor.setVelocity(-s)

def r_left(s): 
    left_motor.setVelocity(-s)
    right_motor.setVelocity(s)

def r_right(s):
    left_motor.setVelocity(s)
    right_motor.setVelocity(-s)
    
def t_left(s):
    left_motor.setVelocity(s * TURNING_FACTOR)
    right_motor.setVelocity(s)

def t_right(s):
    left_motor.setVelocity(s)
    right_motor.setVelocity(s * TURNING_FACTOR)

def odometry_update():
    global x, y, theta, prev_left_encoder, prev_right_encoder
    
    # get the encoder values - in radians
    l_encoder_val = left_encoder.getValue()
    r_encoder_val = right_encoder.getValue()
    
    # calculate the shift between encoder values
    d_left = l_encoder_val - prev_left_encoder
    d_right = r_encoder_val - prev_right_encoder
    
    # Convert these into linear distance
    l_distance = d_left * WHEEL_RADIUS
    r_distance = d_right * WHEEL_RADIUS
    
    # Calculate center displacement and orientation change
    center_distance = (l_distance + r_distance) / 2.0
    d_theta = (r_distance - l_distance) / AXLE_LENGTH
    
    # Update orientation
    theta += d_theta
    
    # normalize theta to a value between -pi, pi
    theta = math.atan2(math.sin(theta), math.cos(theta))
    
    # Update position
    x += center_distance * math.cos(theta)
    y += center_distance * math.sin(theta)
    
    # store current encoder values for next iteration
    prev_left_encoder = l_encoder_val
    prev_right_encoder = r_encoder_val

def get_pose():
    # Neatly returns the epuck position
    return x, y, theta

# Main loop: Testing Movement Functions
while robot.step(timestep) != -1:
    # update odometry every timestep
    odometry_update()
    
    # Get current pose
    current_x, current_y, current_theta = get_pose()
    
    # Print current pose
    print(f"Position: ({current_x:.3f}, {current_y:.3f}) - Orientation: {math.degrees(current_theta):.1f}°")
    
    if MODE == 'manual':
        # Get the current key pressed
        key = keyboard.getKey()
        
        # Move according to the key press
        if key == Keyboard.UP:
            m_forward(NORMAL_SPEED)
            
        elif key == Keyboard.DOWN:
            m_backward(NORMAL_SPEED)
            
        elif key == Keyboard.LEFT:
            t_left(NORMAL_SPEED)
            
        elif key == Keyboard.RIGHT:
            t_right(NORMAL_SPEED)
            
        elif key == ord(' '):  # Spacebar
            stop()
            
        elif key == -1:
            # No key pressed - maintain current velocity
            pass
    
# Enter here exit cleanup code.
