"""main_controller controller."""
# ---------- IMPORTS ----------
import numpy as np

from controller import Robot
from controller import Supervisor
from controller import Keyboard # for driving in the "explore" phase

from ekf_slam import EkfSlamController
from displays import DisplayController
from measurements import MeasurementController

# ---------- CONSTANTS ----------
SPEED_UNIT = 0.00628
LEFT = 0 # used to refer to the left wheel's motor in the speed
RIGHT = 1


# ---------- PARAMETERS ----------
MAX_SPEED = 200 # can go to a maximum of 1000. Speed limit helps keep motion of epuck smooth
SPEED_INCREMENT = 4


# ---------- VARIABLES & DATA STRUCTURES ----------
speed = [0,0] # list for controlling the speed - done this way control speed using encoder steps rather than target velocity directly
x_t = [0,0,0] # list for storing pose at current time t, in the format [x, y, theta]
u_t = [0,0] # list for storing the control applied at time t - 1 to drive the epuck to pose x_t at time t. Elements: linear velocity, angular velocity
z_t = [] # list for landmark measurements, where each element is of the form (distance, bearing from epuck, correspondence)


# ---------- SETUP ----------
# create the Robot instance
# robot = Robot()
robot = Supervisor() # for my stage of work, epuck is a supervisor to give me access to accurate positional data


# get the time step of the current world
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


# initialise the ekf-slam controller object
ekf_slam_controller = EkfSlamController(timestep)

# initialise the display controller object
display_controller = DisplayController(robot)

# initialise the measurement controller object
measurement_controller = MeasurementController(robot, timestep)


# ---------- FUNCTIONS ----------
def get_pose():
    # goal:
    #   - return a vector x_t which represents the e-puck pose at the current time step t
    #   - the pose consists of an x coordinate, a z coordinate, and a heading theta
    #       - theta is taken as the angle in degrees above the x-axis
    #   - the z coordinate has been omitted from the pose, as it has been decided to constrain the e-puck to flat
    #           surfaces (i.e. z = 0) for scope management

    # get absolute position
    x , y , z = translation.getSFVec3f()
    x_t[0] = x
    x_t[1] = y

    # get orientation
    # rot_axis_x and rot_axis_y will be close to 0, and rot_axis_z will be close to 1, as the epuck will be rotating
    # about the vertical axis
    rot_axis_x, rot_axis_y, rot_axis_z, theta = rotation.getSFRotation()

    # heading (relative to x-axis) is in radians
    theta = np.arctan2(np.sin(theta), np.cos(theta)) # bound it

    # check if the axis of rotation around z has become negative
    # it does this when changing the direction of rotating
    # hence to get an accurate sign on theta, need to multiply by -1 if axis of rotation has become negative
    if rot_axis_z >= 0:
        x_t[2] = theta
    else:
        x_t[2] = -1 * theta

    return

def get_control():
    # goal:
    #   - populate the control vector u_t
    #   - this should be the control applied to the epuck at time t - 1 to yield the epuck being at state x_t at
    #           current time t

    # get the velocity from the epuck
    velocity = epuck_node.getVelocity()

    # unpack the above vector for convenience
    vx, vy, vz, wx, wy, wz = velocity

    # linear (ie. straight line) velocity is the norm of the vector formed of x and y velocities
    # linear velocity is in metres/second
    u_t[0] = vx * np.cos(x_t[2]) + vy * np.sin(x_t[2]) # use with signed velocity to cope with the epuck reversing

    # angular velocity (omega) is the velocity of rotation about the z (vertical) axis
    # therefore, angular velocity is in radians/second
    u_t[1] = wz
    return

# Robot actuation
def set_speed():
    # update motors with value in speed list
    left_motor.setVelocity(SPEED_UNIT * speed[LEFT])
    right_motor.setVelocity(SPEED_UNIT * speed[RIGHT])
    return

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


# ---------- MAIN LOOP ----------
# Perform simulation steps until controller is stopped
while robot.step(timestep) != -1:
    # Poll sensors
    get_pose()
    get_control()
    # z_t = measurement_controller.cam_recog_measure_landmarks()
    z_t = measurement_controller.lidar_measure_landmarks()


    # Process sensor data
    state_estimate, covariance = ekf_slam_controller.run_steps(u_t, z_t)

    origin_cell, epuck_cell, occupancy_grid = ekf_slam_controller.construct_occupancy_grid()

    print(f"state_estimate: {state_estimate}")
    print("\n")
    print(f"occupancy grid:\n {occupancy_grid}")
    print("\n\n\n")


    # Actuate
    drive_logic()

    # ---------------------------------

    # Update displays
    display_controller.clean_displays()
    display_controller.draw_true_pose(x_t)
    display_controller.temp_draw_landmarks()
    display_controller.draw_state_estimate(state_estimate)
    display_controller.draw_occupancy_grid(origin_cell, epuck_cell, occupancy_grid)

    pass

# Enter here exit cleanup code.
