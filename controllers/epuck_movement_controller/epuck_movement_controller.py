"""epuck_movement_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Motor, Keyboard

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# variables for each motor...
right_motor = robot.getDevice('right wheel motor')
left_motor = robot.getDevice('left wheel motor')

# initialise the keybaord
keyboard = robot.getKeyboard()
keyboard.enable(timestamp)

# Set motors to velocity control mode
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Stop all movement
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Constant Global Variables...
MODE = 'manual' # Determines how to navigate the robot
MAXIMUM_SPEED = 5 # rad/s
NORMAL_SPEED = MAXIMUM_SPEED / 2
TURNING_FACTOR = 0.5 # Assigned to half for now

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
    
# Fun extra movement functions for smoother navigation
# DO NOT USE IN PRODUCTION OUTSIDE OF MANUAL MOVEMENTS --
def add

# Main loop: Testing Movement Functions
while robot.step(timestep) != -1:
    if mode is 'manual':
        """
        In this mode we want to keep a record of all of the keys being pressed in order to replicate
        normal video game like navigation. The aim is to have the velocity decay as the button is no 
        longer pressed. First let's get all of the keys pressed into a set
        """
        
      


# Enter here exit cleanup code.
