"""main_controller controller."""
import numpy as np

# ---------- IMPORTS ----------
from controller import Robot
from controller import Supervisor
from controller import Keyboard # for driving in the "explore" phase


# ---------- CONSTANTS ----------
SPEED_UNIT = 0.00628
MAX_SPEED = 200 # can go to a maximum of 1000. Speed limit helps keep motion of epuck smooth
SPEED_INCREMENT = 4
LEFT = 0 # used to refer to the left wheel's motor in the speed
RIGHT = 1
PIXELS_PER_METRE = 100 # ratio of pixels per metre - i.e. 1 pixels corresponds to 1cm


# ---------- VARIABLES & DATA STRUCTURES ----------
speed = [0,0] # list for controlling the speed - done this way control speed using encoder steps rather than target velocity directly
x_t = [0,0,0] # list for storing pose at current time $ t $, in the format [$ x $, $ y $, $ \theta $]
u_t = [0,0] # list for storing the control applied at time $ t - 1 $ to drive the epuck to pose $ \vec{x}_t $ at time $ t $. Elements: linear velocity, angular velocity
landmarks = [(-0.25,0.25), (0.25,0.25), (-0.25,-0.25), (0.25,-0.25)]

'''
Probabilistic Robotics defines the combined state vector as a the vector of the robot's pose and (x,y) coordinates of all landmarks.
The coordinates of each of these landmarks is assumed to remain constant throughout simulation.

The purpose of the matrix f_x is for the state estimate to be updated, manipulating only the entries corresponding to the robot's pose, leaving entries corresponding to landmarks unchanged.

The matrix is initialised as a horizontally stacked (3 x 3) identity matrix, and a (3 x 2n) matrix, where n is the number of landmarks. 
'''
f_x = np.hstack((np.eye(3), np.zeros((3, 2 * len(landmarks)))))

'''
The state estimate vector, notated in Probabilistic Robotics as $ u_t $, is a vector containing first elements of the epuck's pose, and then elements for all the (x,y) coordinates of all landmarks.

The state estimate corresponds to the mean of the multivariate gaussian being used to model the uncertainty in our belief over pose and landmark positions - ie. the best guess to where the robot is (based on its pose and surrounding landmarks).

The vector is initialised as a column vector with (3 + 2n) elements, where n is the number of landmarks.

The values of the vector are initialised with zero, as the initial pose is arbitrarily taken as the origin, and none of the landmark locations are known initially.
'''
state_estimate = np.zeros((3 + 2 * len(landmarks), 1))


# ---------- SETUP ----------
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


# get display
display = robot.getDevice('display')
display_width = display.getWidth()
display_height = display.getHeight()

# visualise the epuck's x-y origin (0,0) in the centre of the display
display_origin_x = display_width / 2
display_origin_y = display_height / 2


# ---------- FUNCTIONS ----------
# EKF-SLAM
def get_pose():
    # goal:
    #   - return a vector $ x_t $ which represents the e-puck pose at the current time step $ t $
    #   - the pose consists of an $ x $ coordinate, a $ z $ coordinate, and a heading $ \theta $
    #       - $ \theta $ is taken as the angle in degrees above the $ x $ axis
    #   - the $ z $ coordinate has been omitted from the pose, as it has been decided to constrain the e-puck to flat surfaces (i.e. $ z = 0 $) for scope management

    # get absolute position
    x , y , z = translation.getSFVec3f()
    x_t[0] = x
    x_t[1] = y

    # get orientation
    # rot_axis_x and rot_axis_y will be close to 0, and rot_axis_z will be close to 1, as the epuck will be rotating about the vertical axis
    rot_axis_x, rot_axis_y, rot_axis_z, theta = rotation.getSFRotation()

    # heading (relative to $ x $ axis) is in radians
    x_t[2] = theta
    return

def get_control():
    # goal:
    #   - populate the control vector $ \vec{u}_t $
    #   - this should be the control applied to the epuck at time $ t - 1 $ to yield the epuck being at state $ \vec{x}_t $ at current time $ t $

    # get the velocity from the epuck
    velocity = epuck_node.getVelocity()

    # unpack the above vector for convenience
    vx, vy, vz, wx, wy, wz = velocity

    # linear (ie. straight line) velocity is the norm of the vector formed of x and y velocities
    # linear velocity is in metres/second
    u_t[0] = np.sqrt(vx ** 2 + vy ** 2)

    # angular velocity ($ \omega $) is the velocity of rotation about the z (vertical) axis
    # therefore, angular velocity is in radians/second
    u_t[1] = vz
    return

def time_update():
    # TODO
    # line 3 prob robotics (ekf slam known correspondences)
    # update the state estimate: $ \bar{u}_t $
    # intuition - take the best state estimate from the previous time step, and based on the control $ \vec{u}_t $, update the state estimate for this time step

    # need to calculate how the epuck's pose will have changed between the last time step $ t - 1 $ and now $ t $
    # the motion model for the epuck will be based on that it's moving with both a value for linear velocity and angular velocity
    # hence, for the time step, it has moved on a circular arc of radius R = linear velocity / angular velocity, centered about the ICC (Instantaneous Centre of Curvature)
    # following through the trig (using a diagram on differential drive robots from Dudek and Jenkin, Computational Principles of Mobile Robotics) this gives that:
    #   the location of the ICC relative to the centre of the robot (x,y) is $ x_{t-1} - R \sin \theta $ for the x coord, and $ y_{t-1} + R \cos \theta $ for the y coord
    #   therefore, we can get the updated pose of the epuck with:
    #       $ x_t = x_{t-1} - R \sin(\theta_{t-1}) + R \sin(\theta_{t-1} + \delta \theta) $  ie. the x coord is the x coord of the ICC sum the x distance between the ICC and centre of the epuck after it's heading has changed by $ \delta \theta $
    #       $ y_t = y_{t-1} + R \cos(\theta_{t-1}) - R \cost(\theta_{t-1} + \delta \theta) $  ie. the y coord is the y coord of the ICC sum the y distance between the ICC and the centre of the epuck after it's heading has changed by $ \delta \theta $
    #       $ \theta_t = \theta_{t-1} + \delta \theta $  ie. the heading of the epuck is it's old heading sum the change in heading
    #   substituting in that R = linear velocity / angular velocity, gives us the matrix below:

    # helper variables
    dt = timestep / 1000 # timestep is in miliseconds, want in seconds
    old_theta = state_estimate[2,0] # theta is 3'rd element of vector - need to get it from column zero, as it's implemented as a matrix
    delta_theta = u_t[1] * dt # ie. change in angle is angular velocity multiplied by change in time

    # matrix components
    # because of numerical stability issues, need to handle case where angular velocity is near zero
    if u_t[1] > 0.0001:
        r = (u_t[0] / u_t[1])  # linear velocity / angular velocity

        motion_model_x = (-1 * r * (np.sin(old_theta))) + (r * np.sin(old_theta + delta_theta))
        motion_model_y = (r * (np.cos(old_theta))) - (r * np.cos(old_theta + delta_theta))
        motion_model_theta = delta_theta
    else:
        motion_model_x = u_t[0] * np.cos(old_theta) * dt
        motion_model_y = u_t[0] * np.sin(old_theta) * dt
        motion_model_theta = 0.0 # angular velocity tends to zero, so heading assumed to not change

    # matrix construction
    motion_model_matrix = np.array([[float(motion_model_x)], [float(motion_model_y)], [float(motion_model_theta)]]) # 3x1 matrix

    # apply the motion model to update the state estimate
    updated_state_est = state_estimate + np.dot(f_x.T, motion_model_matrix)

    print(f"actual pose = {x_t}")
    print(f"actual control = {u_t}")
    print(f"old state est: {state_estimate} \n new state est: {updated_state_est} \n")
    print(f"motion model x: {motion_model_x}")
    print(f"motion model y: {motion_model_y}")
    print(f"motion model theta: {motion_model_theta} \n\n\n")

    return updated_state_est

def observation_update():
    # TODO
    return

# Convenience
def rad_to_deg(rads):
    # convenience function
    degs = rads * (180.0 / np.pi)
    return degs

def deg_to_rad(degs):
    # convenience function
    rads = degs * (np.pi / 180.0)
    return rads

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

# Display
def world_coords_to_display_cords(x, y):
    # convert world coordinates to display coordinates
    # in the environment, x is right, y is up, whereas on the display, x is right, y is down
    display_x = int(display_origin_x + x * PIXELS_PER_METRE)
    display_y = int(display_origin_y - y * PIXELS_PER_METRE) # subtract because of display y direction being opposite to world y direction

    return display_x, display_y

def draw_pose():
    # pose is stored in x_k = [x_coord, y_coord, theta]

    # draw epuck body
    display.setColor(0x1026cc)
    epuck_radius = 0.037 * PIXELS_PER_METRE # epuck has 7.4cm diameter => 0.037m radius
    display_x, display_y = world_coords_to_display_cords(x_t[0], x_t[1]) # convert world coordinates to display coordinates
    display.fillOval(display_x, display_y, epuck_radius, epuck_radius)

    # draw heading of epuck
    heading_x = display_x + (int(epuck_radius * np.cos(x_t[2])))
    heading_y = display_y - (int(epuck_radius * np.sin(x_t[2])))
    display.setColor(0xe01fb3)
    display.drawLine(int(display_x), int(display_y), int(heading_x), int(heading_y))
    return

def temp_draw_landmarks():
    # temporary function, until landmarks implemented in environment

    for x, y in landmarks:
        display.setColor(0x3d3527)
        display_x, display_y = world_coords_to_display_cords(x, y)  # convert world coordinates to display coordinates
        display.fillOval(display_x, display_y, 1, 1)
    return

def clean_display():
    # removes anything not explicitly drawn this time-step on the display
    display.setColor(0xffffff)  # make display background white
    display.fillRectangle(0, 0, display_width, display_height)
    return


# ---------- MAIN LOOP ----------
# Perform simulation steps until controller is stopped
while robot.step(timestep) != -1:
    # Poll sensors
    get_pose()
    get_control()

    # Process sensor data
    state_estimate = time_update()


    # get_control()

    # Actuate
    drive_logic()

    # ---------------------------------

    # Update display
    clean_display()
    draw_pose()
    temp_draw_landmarks()

    pass

# Enter here exit cleanup code.
