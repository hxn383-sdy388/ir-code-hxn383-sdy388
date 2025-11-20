"""main_controller controller."""
# ---------- IMPORTS ----------
import numpy as np

from controller import Robot
from controller import Supervisor
from controller import Keyboard # for driving in the "explore" phase

from ekf_slam import EkfSlamController


# ---------- CONSTANTS ----------
SPEED_UNIT = 0.00628
LEFT = 0 # used to refer to the left wheel's motor in the speed
RIGHT = 1
PIXELS_PER_METRE = 100 # ratio of pixels per metre - i.e. 1 pixels corresponds to 1cm




# ---------- PARAMETERS ----------
MAX_SPEED = 200 # can go to a maximum of 1000. Speed limit helps keep motion of epuck smooth
SPEED_INCREMENT = 4


# ---------- VARIABLES & DATA STRUCTURES ----------
speed = [0,0] # list for controlling the speed - done this way control speed using encoder steps rather than target velocity directly
x_t = [0,0,0] # list for storing pose at current time $ t $, in the format [$ x $, $ y $, $ \theta $]
u_t = [0,0] # list for storing the control applied at time $ t - 1 $ to drive the epuck to pose $ \vec{x}_t $ at time $ t $. Elements: linear velocity, angular velocity
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


# get the camera and enable the recognition node
camera = robot.getDevice('recognitioncamera')
camera.enable(timestep)
camera.recognitionEnable(timestep)


# get display (the standard one that shows the pose and landmarks - not the occupancy grid one)
display = robot.getDevice('display')
display_width = display.getWidth()
display_height = display.getHeight()

# visualise the epuck's x-y origin (0,0) in the centre of the display
display_origin_x = display_width / 2
display_origin_y = display_height / 2


# get occupancy grid display
occ_grid_disp = robot.getDevice('occupancy-grid')
occ_grid_disp_width = occ_grid_disp.getWidth()
occ_grid_disp_height = occ_grid_disp.getHeight()


# initalise the ekf-slam controller object
ekf_slam_controller = EkfSlamController(timestep)


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
    #   - populate the control vector $ \vec{u}_t $
    #   - this should be the control applied to the epuck at time $ t - 1 $ to yield the epuck being at state $ \vec{x}_t $ at current time $ t $

    # get the velocity from the epuck
    velocity = epuck_node.getVelocity()

    # unpack the above vector for convenience
    vx, vy, vz, wx, wy, wz = velocity

    # linear (ie. straight line) velocity is the norm of the vector formed of x and y velocities
    # linear velocity is in metres/second
    u_t[0] = vx * np.cos(x_t[2]) + vy * np.sin(x_t[2]) # use with signed velocity to cope with the epuck reversing

    # angular velocity ($ \omega $) is the velocity of rotation about the z (vertical) axis
    # therefore, angular velocity is in radians/second
    u_t[1] = wz
    return

def cam_recog_measure_landmarks():
    # using the camera's recognition feature, measure the landmark positions
    # landmarks have been given model tags that represent the correspondences
    z = [] # measurements

    recognised_objects = camera.getRecognitionObjects()
    for object in recognised_objects:
        index = int(object.getModel()) # this is the correspondence of the landmark

        rel_x, rel_y, rel_z = object.getPosition() # position is relative to the epuck

        distance = np.hypot(float(rel_y), float(rel_x)) # calculate straight line distance between epuck and recognised landmark

        alpha = np.arctan2(rel_y, rel_x) # calculate the relative bearing between the epuck and landmark
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha)) # bound the bearing to be between -pi and +pi

        z.append(((float(distance), float(alpha), 0), index))

    return z

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

def draw_state_estimate():
    # state_estimate = [x_coord, y_coord, theta, landmarks...]

    # draw epuck body
    display.setColor(0x24fc03)
    epuck_radius = 0.037 * PIXELS_PER_METRE # epuck has 7.4cm diameter => 0.037m radius
    display_x, display_y = world_coords_to_display_cords(state_estimate[0], state_estimate[1]) # convert world coordinates to display coordinates
    display.fillOval(display_x, display_y, epuck_radius, epuck_radius)

    # draw heading of epuck
    heading_x = display_x + (int(epuck_radius * np.cos(state_estimate[2])))
    heading_y = display_y - (int(epuck_radius * np.sin(state_estimate[2])))
    display.setColor(0x2a4226)
    display.drawLine(int(display_x), int(display_y), int(heading_x), int(heading_y))

    # draw landmarks
    i = 3
    while i < len(state_estimate):
        landmark_x = state_estimate[i]
        landmark_y = state_estimate[i + 1]

        if not(np.isnan(landmark_x)) and not(np.isnan(landmark_y)):
            display.setColor(0xbf22bd)
            disp_x, disp_y = world_coords_to_display_cords(landmark_x, landmark_y)  # convert world coordinates to display coordinates
            display.fillOval(disp_x, disp_y, 1, 1)

        i += 3

    return

def temp_draw_landmarks():
    # temporary function - draws on display the true location of landmarks so that it can be seen how well the EKF-SLAM algorithm is mapping landmarks
    landmarks = [(-0.25, 0.25, 0), (0.25, 0.25, 0), (-0.25, -0.25, 0), (0.25, -0.25, 0)] # coords of the landmarks actually in the environment

    for x, y, signature in landmarks:
        display.setColor(0x3d3527)
        display_x, display_y = world_coords_to_display_cords(x, y)  # convert world coordinates to display coordinates
        radius = 0.03 * PIXELS_PER_METRE # environment cylinder objects currently have 0.03 radius
        display.fillOval(display_x, display_y, radius, radius)
    return

def draw_occupancy_grid(origin_cell, epuck_cell, occupancy_grid):
    # Visualises the occupancy grid on a display
    # Unoccupied cells are drawn as alternating shades of grey (for visual differentiation)
    # Occupied cells are drawn in red
    # The cell containing the origin is drawn in pink (to make it easier to visually landmark cells in the occupancy grid)
    # The cell currently occupied by the centre of the epuck is drawn in green
    #
    # Throughout epuck movement, the drawn grid may appear to become distorted
    # This is a consequence of the limited size of the display, and the occupancy grid growing unbalanced in either its number of rows or columns
    # Despite this visual behaviour, each cell in the occupancy actually represents a square area at all times
    #
    # Another visual anomaly may be white lines appearing between grid cells on the display
    # This attributable to precision issues mapping points from within the occupancy grid space to the display space, where the former's size is dynamic and unlimited, and the latter is static and (hence) limited

    # ------------------------------

    # take the number of rows/cols, divide display width by num of rows/cols, that's cell width/height
    row_count = np.shape(occupancy_grid)[0]
    col_count = np.shape(occupancy_grid)[1]

    cell_width = display_width / col_count
    cell_height = display_height / row_count


    # iterate through rows, and then through columns, to draw cells one at a time
    row_alternator = False # first of the two alternators - two are required to ensure that the grey used for empty cells alternates in both row and column directions
    for row in range(np.shape(occupancy_grid)[0]):
        col_alternator = row_alternator
        for col in range(np.shape(occupancy_grid)[1]):
            # if the cell contains the epuck, draw it in green
            if (row, col) == epuck_cell:
                occ_grid_disp.setColor(0x24fc03)
                occ_grid_disp.fillRectangle(col * cell_width, row * cell_height, cell_width, cell_height)
            # if the cell contains the origin, draw it in pink
            elif (row, col) == origin_cell:
                occ_grid_disp.setColor(0xbf22bd)
                occ_grid_disp.fillRectangle(col * cell_width, row * cell_height, cell_width, cell_height)
            # if the cell is believed to be occupied, draw it in red
            elif occupancy_grid[row, col] == 1:
                occ_grid_disp.setColor(0xd1022e)
                occ_grid_disp.fillRectangle(col * cell_width, row * cell_height, cell_width, cell_height)
            # draw in cell as grey first for case where cell is free
            elif col_alternator:
                occ_grid_disp.setColor(0xbbbdbf)
                occ_grid_disp.fillRectangle(col * cell_width, row * cell_height, cell_width, cell_height)
            else:
                occ_grid_disp.setColor(0x9b9d9e)
                occ_grid_disp.fillRectangle(col * cell_width, row * cell_height, cell_width, cell_height)

            col_alternator = not col_alternator # flip the alternator before drawing the cell in the next column
        row_alternator = not row_alternator # flip the alternator before drawing the cell in the next row

    return

def clean_displays():
    # removes anything not explicitly drawn this time-step on the display
    display.setColor(0xffffff)  # make display background white
    display.fillRectangle(0, 0, display_width, display_height)

    # do the same for the occupancy grid display
    occ_grid_disp.setColor(0xffffff)
    occ_grid_disp.fillRectangle(0, 0, occ_grid_disp_width, occ_grid_disp_height)
    return


# ---------- MAIN LOOP ----------
# Perform simulation steps until controller is stopped
while robot.step(timestep) != -1:
    # Poll sensors
    get_pose()
    get_control()
    z_t = cam_recog_measure_landmarks()


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
    clean_displays()
    draw_pose()
    temp_draw_landmarks()
    draw_state_estimate()
    draw_occupancy_grid(origin_cell, epuck_cell, occupancy_grid)

    pass

# Enter here exit cleanup code.
