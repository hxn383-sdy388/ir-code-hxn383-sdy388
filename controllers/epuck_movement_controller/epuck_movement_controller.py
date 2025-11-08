"""epuck_movement_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Motor

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# variables for each motor...
right_motor = robot.getDevice('right wheel motor')
left_motor = robot.getDevice('left wheel motor')

# Set motors to velocity control mode
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Stop all movement
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Constant Global Variables...
MAXIMUM_SPEED = 5 # rad/s
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
    
# Main loop: Testing Movement Functions
count = 0
while robot.step(timestep) != -1:
    if count < 2000 / timestep:
            m_forward(MAXIMUM_SPEED * 0.7)
    elif count < 4500 / timestep:
            t_right(MAXIMUM_SPEED * 0.8)
    elif count < 7000 / timestep:
            t_left(MAXIMUM_SPEED * 0.9)
    elif count < 10000 / timestep:
            m_backward(MAXIMUM_SPEED * 0.8)
    elif count < 12000 / timestep:
            r_right(MAXIMUM_SPEED * 0.7)
    elif count < 14000 / timestep:
            r_left(MAXIMUM_SPEED * 0.5)
    else:
        stop()
    count += 1

# Enter here exit cleanup code.
