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

CELL_SIZE = 0.05 # (5cm) - the cell side length measurement for the occupancy grid encoding of the state estimate


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

def calculate_occupancy_grid():
    # The occupancy grid is a matrix where each entry corresponds to a cell in the grid
    # If the cell contains a 0, the cell is believed to be unoccupied by landmarks - ie. the epuck can enter it
    # If the cell contains a 1, the cell is believed to be occupied by landmarks 0 ie. the epuck should not attempt to enter it
    #
    # Dimensions: m is the number of cells on the horizontal axis of the occupancy grid, n is the number on the vertical axis
    #
    # The occupancy grid is initialised as a 3x3 grid, with the origin taken as the centre of the middle cell
    #
    # Because the occupancy grid conceptually operates in the Cartesian plane (with the origin taken as the epuck's original position), but is encoded as a matrix without support for negative rows or columns, an origin cell variable is constructed to enable occupancy in negative rows and columns to be represented as well as in positive ones
    #
    # The matrix is initially constructed where direction of row growth corresponds to an increase in the Cartesian y coordinate of occupancy values
    # This behaviour is the same for columns and Cartesian x coordinates
    # However, for providing a more glanceable grid on printing/display, the matrix has been flipped about the horizontal direction
    # This means that in the top-left (0,0) of the matrix is the negative,positive quadrant of the Cartesian plane, and in the bottom-right (n,m) is the positive,negative quadrant of the Cartesian plane
    # Of course, this means that the cell containing the origin lies (initially) centrally in this matrix encoding
    #
    # The occupancy grid is completely re-calculated every time step, as it relies directly on the combined state estimate vector
    # The combined state estimator vector's values are calculated under the Markov assumption
    # Hence, at each timestep, the prior state of the occupancy grid should not influence the current state of the occupancy grid
    # Practically speaking, this is required so as to not hold prior occupancy values as truth when their estimates may have been updated in the EKF-SLAM process
    #
    # This means that the occupancy grid may shrink out unoccupied space over time
    # For example, in development, when driving the epuck around a 1x1 metre grid with only four landmarks, in many timesteps the epuck wouldn't perceive a landmark, and hence would be driving in open space
    # This would be reflected in the occupancy grid with 0's being filled in for the epuck's pose
    # However, when the epuck left the empty space, and it didn't fall within the minimum bounds set for the grid, these cells would be removed from the grid
    # This behaviour is intended, and stems from my decision to carry the Markov assumption through to the occupancy grid based on it's dependency on the state estimate

    # ------------------------------

    # iterate through the state estimate and find the bounding coordinates
    # also set a minimum size, so that cells are drawn initially
    # 2 cell lengths in all directions, plus the one cell in the middle for the origin, means that the grid will initially be 3xe
    min_m = -(2 * CELL_SIZE) # steps later on require this to be a negative value
    max_m = 2 * CELL_SIZE
    min_n = -(2 * CELL_SIZE) # steps later on require this to be a negative value
    max_n = 2 * CELL_SIZE

    i = 0
    while i < len(state_estimate):
        if state_estimate[i] < min_m:
            min_m = state_estimate[i]
        if state_estimate[i] > max_m:
            max_m = state_estimate[i]

        if state_estimate[i + 1] < min_n:
            min_n = state_estimate[i + 1]
        if state_estimate[i + 1] > max_n:
            max_n = state_estimate[i+1]

        i += 3 # 3 components per state estimate vector entry for epuck pose and landmarks


    # take the number of rows and columns either side of the origin as the minimum values to over-shoot the bounding coordinates, so that no landmark cannot be encoded within a grid cell
    # half a cell side length is taken off to account for the cell that's explicitly added for the origin coord to lie in
    # (ie. half of the origin cell lies in each of the Cartesian quadrants)
    left_of_origin_cols = int(np.ceil((np.abs(min_m) - (0.5 * CELL_SIZE)) / CELL_SIZE)) # abs so this quantity is positive
    right_of_origin_cols = int(np.ceil((max_m - (0.5 * CELL_SIZE)) / CELL_SIZE))

    below_origin_rows = int(np.ceil((np.abs(min_n) - (0.5 * CELL_SIZE)) / CELL_SIZE))  # abs so this quantity is positive
    above_origin_rows = int(np.ceil((max_n - (0.5 * CELL_SIZE)) / CELL_SIZE))

    # take the total row and column counts as one more than the respective counts on either side of the origin, so that there's explicitly a cell for the origin coord to lie in
    row_count = below_origin_rows + above_origin_rows + 1
    col_count = left_of_origin_cols + right_of_origin_cols + 1


    # initialise the matrix with zeros - ie. all cells unoccupied unless explicitly set to be occupied
    occupancy_grid = np.full((row_count, col_count), 0)


    # populate cell that epuck centre is in as free - epuck has been able to drive into it, so probably doesn't contain a landmark
    epuck_x = state_estimate[0]
    epuck_y = state_estimate[1]

    # calculate what proportion of the way it falls between the distance representable in the number of columns and rows determined required
    x_dist = col_count * CELL_SIZE
    epuck_x_dist = epuck_x - (-1 * (left_of_origin_cols + 0.5) * CELL_SIZE) # 0.5 here to account for the half a cell side length held in the negative x quadrant by the origin cell
    x_proportion = epuck_x_dist / x_dist
    col_pos = int(np.floor(x_proportion * col_count)) # take the floor as if it's 4.3/10 cols for example, want to declare it as in a cell in column 4 - this also takes care of converting this from a count to an index

    y_dist = row_count * CELL_SIZE
    epuck_y_dist = epuck_y - (-1 * (below_origin_rows + 0.5) * CELL_SIZE) # 0.5 here to account for the half a cell side length held in the negative y quadrant by the origin cell
    y_proportion = epuck_y_dist / y_dist
    row_pos = int(np.floor(y_proportion * row_count))

    epuck_cell = (row_count - row_pos - 1, col_pos) # need to augment the row pos because the occupancy matrix will be later flipped vertically for intuition wrt. the Cartesian plane when printed/displayed

    occupancy_grid[row_pos, col_pos] = 0 # 0 for free


    # iterate through the landmarks and apply the same logic as above for the epuck, but filling in 1 for each landmark instead of 0, to indicate cell not free
    i = 3 # first three state estimate vector components correspond to epuck pose - fourth element is where landmarks (if any) start
    while i < len(state_estimate):
        landmark_x = state_estimate[i]
        landmark_y = state_estimate[i+1]

        x_dist = col_count * CELL_SIZE
        landmark_x_dist = landmark_x - (-1 * (left_of_origin_cols + 0.5) * CELL_SIZE) # 0.5 here to account for the half a cell side length held in the negative x quadrant by the origin cell
        x_proportion = landmark_x_dist / x_dist
        col_pos = int(np.floor(x_proportion * col_count))

        y_dist = row_count * CELL_SIZE
        landmark_y_dist = landmark_y - (-1 * (below_origin_rows + 0.5) * CELL_SIZE) # 0.5 here to account for the half a cell side length held in the negative y quadrant by the origin cell
        y_proportion = landmark_y_dist / y_dist
        row_pos = int(np.floor(y_proportion * row_count))

        occupancy_grid[row_pos,col_pos] = 1

        i += 3


    # flip in the up/down direction to get negative columns at the bottom rather than at the top for when printing out - less disturbing to look at
    occupancy_grid = np.flipud(occupancy_grid)

    # the origin cell row index is now given by the number of above origin rows - because it's one more than this count, hence taking it without +1 is fine, and it's above origin row count rather than below because of the flip done to the matrix to make it more intuitive for when printed out
    origin_cell = (above_origin_rows, left_of_origin_cols)

    return origin_cell, epuck_cell, occupancy_grid

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

    origin_cell, epuck_cell, occupancy_grid = calculate_occupancy_grid()

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
