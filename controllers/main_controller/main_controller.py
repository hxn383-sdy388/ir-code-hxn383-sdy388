"""main_controller controller."""

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

speed = [0,0] # tuple for controlling the speed - done this way control speed using encoder steps rather than target velocity directly


# get keyboard for manual driving for the "explore" phase
kb = Keyboard()
kb.enable(timestep)


# FUNCTIONS
def set_speed():
    # update motors with value in speed tuple
    left_motor.setVelocity(SPEED_UNIT * speed[LEFT])
    right_motor.setVelocity(SPEED_UNIT * speed[RIGHT])
    return

def drive_logic():
    key = kb.getKey()

    # manipulate speed tuple based on what keyboard action is
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


# MAIN LOOP
# Perform simulation steps until controller is stopped
while robot.step(timestep) != -1:
    # Poll sensors
    # Process sensor data
    # Actuate

    drive_logic()

    pass

# Enter here exit cleanup code.
