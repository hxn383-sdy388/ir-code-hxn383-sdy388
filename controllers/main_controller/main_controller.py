"""main_controller controller."""
import numpy as np

# IMPORTS
from controller import Robot
from controller import Supervisor
from controller import Keyboard # for driving in the "explore" phase


# CONSTANTS
SPEED_UNIT = 0.00628
MAX_SPEED = 200 # can actually go to a maximum of 1000
SPEED_INCREMENT = 4
LEFT = 0 # used to refer to the left wheel's motor in the speed
RIGHT = 1


# VARIABLES & DATA STRUCTURES
speed = [0,0] # list for controlling the speed - done this way control speed using encoder steps rather than target velocity directly
x_k = [0,0,0] # list for storing pose at current time $ k $, in the format [$ x $, $ y $, $ \theta $]
u_k = [0,0] # list for storing the control applied at time $ k - 1 $ to drive the epuck to pose $ \vec{x}_k $ at time $ k $. Elements: linear velocity, angular velocity


# SETUP
# create the Robot instance
# robot = Robot()
robot = Supervisor() # for my stage of work, epuck is a supervisor to give me access to accurate positional data


# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())


# get motors for moving the epuck
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_motor.setPosition(float('inf')) # target position for motors is infinity
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0.0) # initial velocities 0
right_motor.setVelocity(0.0)


# get keyboard for manual driving for the "explore" phase
kb = Keyboard()
kb.enable(timestep)


# use supervisor capabilities to get fields required for polling the epuck's pose
epuck_node = robot.getSelf()
translation = epuck_node.getField('translation')
rotation = epuck_node.getField('rotation')


# FUNCTIONS
def set_speed():
    # update motors with value in speed list
    left_motor.setVelocity(SPEED_UNIT * speed[LEFT])
    right_motor.setVelocity(SPEED_UNIT * speed[RIGHT])
    return

def radToDeg(rads):
    # convenience function
    degs = rads * (180.0 / np.pi)
    return degs

def degToRad(degs):
    # convenience function
    rads = degs * (np.pi / 180.0)
    return rads

def drive_logic():
    key = kb.getKey()

    # manipulate speed list based on what keyboard action is
    if key == Keyboard.UP:
        if speed[LEFT] < MAX_SPEED:
            speed[LEFT] += SPEED_INCREMENT
        if speed[RIGHT] < MAX_SPEED:
            speed[RIGHT] += SPEED_INCREMENT
    if key == Keyboard.DOWN:
        if speed[LEFT] > -MAX_SPEED:
            speed[LEFT] -= SPEED_INCREMENT
        if speed[RIGHT] > -MAX_SPEED:
            speed[RIGHT] -= SPEED_INCREMENT

    if key == Keyboard.LEFT:
        if speed[LEFT] > -MAX_SPEED:
            speed[LEFT] -= SPEED_INCREMENT
        if speed[RIGHT] < MAX_SPEED:
            speed[RIGHT] += SPEED_INCREMENT
    if key == Keyboard.RIGHT:
        if speed[LEFT] < MAX_SPEED:
            speed[LEFT] += SPEED_INCREMENT
        if speed[RIGHT] > -MAX_SPEED:
            speed[RIGHT] -= SPEED_INCREMENT

    # send updated speed values to motors
    set_speed()
    return

def get_pose():
    # goal:
    #   - return a vector $ x_k $ which represents the e-puck pose at the current time step $ k $
    #   - the pose consists of an $ x $ coordinate, a $ z $ coordinate, and a heading $ \theta $
    #       - $ \theta $ is taken as the angle in degrees above the $ x $ axis
    #   - the $ z $ coordinate has been omitted from the pose, as it has been decided to constrain the e-puck to flat surfaces (i.e. $ z = 0 $) for scope management

    # get absolute position
    x , y , z = translation.getSFVec3f()
    x_k[0] = x
    x_k[1] = y

    # get orientation
    # rot_axis_x and rot_axis_y will be close to 0, and rot_axis_z will be close to 1, as the epuck will be rotating about the vertical axis
    rot_axis_x, rot_axis_y, rot_axis_z, theta = rotation.getSFRotation()

    # heading (relative to $ x $ axis) is in radians, and want it in degrees
    x_k[2] = radToDeg(theta)
    return

def get_control():
    # goal:
    #   - populate the control vector $ \vec{u}_k $
    #   - this should be the control applied to the epuck at time $ k - 1 $ to yield the epuck being at state $ \vec{x}_k $ at current time $ k $

    # get the velocity from the epuck
    velocity = epuck_node.getVelocity()

    # unpack the above vector for convenience
    vx, vy, vz, wx, wy, wz = velocity

    # linear (ie. straight line) velocity is the norm of the vector formed of x and y velocities
    # linear velocity is in metres/second
    u_k[0] = np.sqrt(vx ** 2 + vy ** 2)

    # angular velocity ($ \omega $) is the velocity of rotation about the z (vertical) axis
    # therefore, angular velocity is in radians/second
    u_k[1] = vz
    return


# MAIN LOOP
# Perform simulation steps until controller is stopped
while robot.step(timestep) != -1:
    # Poll sensors
    get_pose()
    get_control()

    # Process sensor data
    # Actuate

    drive_logic()

    pass

# Enter here exit cleanup code.
