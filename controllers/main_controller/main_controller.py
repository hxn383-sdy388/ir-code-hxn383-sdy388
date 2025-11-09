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
landmarks = [(-0.25,0.25,0), (0.25,0.25,0), (-0.25,-0.25,0), (0.25,-0.25,0)] # signatures added as 0
z_t = [] # list for landmark measurements, where each element is of the form (distance, bearing from epuck, correspondence)

'''
Probabilistic Robotics defines the combined state vector as a the vector of the robot's pose and (x,y) coordinates of all landmarks.
The coordinates of each of these landmarks is assumed to remain constant throughout simulation.

The purpose of the matrix f_x is for the state estimate to be updated, manipulating only the entries corresponding to the robot's pose, leaving entries corresponding to landmarks unchanged.

The matrix is initialised as a horizontally stacked (3 x 3) identity matrix, and a (3 x 3n) matrix, where n is the number of landmarks. 
'''
f_x = np.hstack((np.eye(3), np.zeros((3, 3 * len(landmarks)))))

'''
The state estimate vector, notated in Probabilistic Robotics as $ u_t $, is a vector containing first elements of the epuck's pose, and then elements for all the (x,y) coordinates of all landmarks.

The state estimate corresponds to the mean of the multivariate gaussian being used to model the uncertainty in our belief over pose and landmark positions - ie. the best guess to where the robot is (based on its pose and surrounding landmarks).

The vector is initialised as a column vector with (3 + 3n) elements, where n is the number of landmarks.

# The values of the vector are initialised with zero, as the initial pose is arbitrarily taken as the origin, and none of the landmark locations are known initially.

The landmark location elements have been initialised as nan's so that it can be ascertained whether they've been seen before. (nan indicates landmark has not been seen yet).
'''
state_estimate = np.full((3 + 3 * len(landmarks), 1), np.nan)
state_estimate[0:3] = [0] # correct the pose elements back to way described above

'''
The covariance matrix is a square matrix of size (3 + 3n) x (3 + 3n), on which the diagonal elements are initialised to a large value, as the initial positions of the landmarks are unknown.
Note that in Probabilistic Robots, infinity is used for these values, but instead a relatively large finite value is used here due to computational issues encountered when using infinity.

The diagonal elements correspond to the variances of the uncertainty in x, y, theta, and positions of the landmarks. The elements beyond the first 3 diagonal entries are set to relatively large finite values, as explained above, as initial landmark positions are unknown. 
The off-diagonal elements correspond to the correlations, which are initialised as zero so as to not assume correlation between variables, where these elements are updated by the EKF in the prediction and correction steps.

Zeros are filled in on the first three diagonal elements for the epuck's pose, consistent with the initialisation outlined in Probabilistic Robotics.
'''
covariance = np.zeros((3 + 3 * len(landmarks), 3 + 3 * len(landmarks)))
np.fill_diagonal(covariance, 100)
covariance[:3, :3] = 0 # state uncertainty for epuck - assume known position initially

'''
The noise matrix is a diagonal matrix on which the elements are the covariance of random noise added to the state uncertainty at every prediction step.

ie. if the first diagonal element was set to 1, this would correspond to an addition of 1 metre's worth of uncertainty in the x coordinate of the epuck's pose per timestep.
'''
noise = np.diag([0.000000001,0.000000001,0.000000001]) # experimenting with some simulated noise

Q = np.diag([0.000000001,0.000000001,0.000000001]) # units for the first two elements are metres - ie. each of these are just distances - ie. for given range and relative bearing measurements, how far off are the actual range and bearing (and signature - but that's expected to be zero in the current configuration) measurements expected to be. Currently experimenting with small noise values. None of these values can be zero or the inverse matrix operation later falls apart


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
    u_t[1] = wz
    return

def time_update():
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
    if np.abs(u_t[1]) > 0.0001:
        r = (u_t[0] / u_t[1])  # linear velocity / angular velocity

        motion_model_x = (-1 * r * (np.sin(old_theta))) + (r * np.sin(old_theta + delta_theta))
        motion_model_y = (r * (np.cos(old_theta))) - (r * np.cos(old_theta + delta_theta))
        motion_model_theta = delta_theta
    else:
        # angle near zero, model as only linear movement
        motion_model_x = u_t[0] * np.cos(old_theta) * dt
        motion_model_y = u_t[0] * np.sin(old_theta) * dt
        motion_model_theta = 0.0 # angular velocity tends to zero, so heading assumed to not change

    # matrix construction
    motion_model_matrix = np.array([[float(motion_model_x)], [float(motion_model_y)], [float(motion_model_theta)]]) # 3x1 matrix

    # apply the motion model to update the state estimate
    updated_state_est = state_estimate + np.dot(f_x.T, motion_model_matrix)

    # print(f"actual pose = {x_t}")
    # print(f"actual control = {u_t}")
    # print(f"old state est: {state_estimate} \n new state est: {updated_state_est} \n")
    # print(f"motion model x: {motion_model_x}")
    # print(f"motion model y: {motion_model_y}")
    # print(f"motion model theta: {motion_model_theta} \n\n\n")


    # Note that Probabilistic robotics termed $ y_t $ as the combined state vector - ie. the pose and landmarks
    # Lines 4 and 5 of the textbook algorithm are responsible for updating the state uncertainty with respect to the motion model and random noise

    # Line 4 defines $ G_t $ - an auxiliary matrix constructed by taking the Jacobian of the state model, where the only non-zero elements are the first derivatives of the x and y motion model components w.r.t theta
    # ie. The non-zero elements model how the x and y coords of the epuck are changing
    # There is a differentiation error in the version of the book that I had access to, where both elements are off by a factor of -1, which has been corrected in this implementation
    # Again, as working with dividing by the angular velocity again, need to handle the case where the angular velocity tends to 0
    if np.abs(u_t[1]) > 0.0001:
        r = (u_t[0] / u_t[1])  # linear velocity / angular velocity

        first_deriv_x = (-1 * r * (np.cos(old_theta))) + (r * np.cos(old_theta + delta_theta)) # derivative w.r.t theta
        first_deriv_y = (-1 * r * (np.sin(old_theta))) + (r * np.sin(old_theta + delta_theta)) # derivative w.r.t theta
    else:
        # angle near zero, model as only linear movement
        first_deriv_x = -1 * u_t[0] * np.sin(old_theta) * dt
        first_deriv_y = u_t[0] * np.cos(old_theta) * dt

    # construct the matrix with derivatives of the motion model w.r.t the epuck's pose
    jacobian_wrt_pose = np.zeros((3,3))
    jacobian_wrt_pose[0,2] = first_deriv_x
    jacobian_wrt_pose[1,2] = first_deriv_y

    # construct the matrix that embeds this in the space represented by the combined state vector, rather than just the pose
    jacobian_wrt_combined_state_vector = np.eye((f_x.shape[1])) + np.dot(np.dot(f_x.transpose(), jacobian_wrt_pose), f_x)

    # line 5
    # Use the auxiliary matrix to update the covariance matrix from the previous covariance, plus some noise from a random variable that's added to the state each prediction update
    updated_covariance = np.dot(np.dot(jacobian_wrt_combined_state_vector, covariance), jacobian_wrt_combined_state_vector.transpose()) + np.dot(np.dot(f_x.transpose(), noise), f_x)


    return updated_state_est, updated_covariance

def observation_update(state_estimate_bar, covariance_bar):
    # Goal is to use the measurements to inform the state estimate (of both epuck pose and landmark positions) and inform uncertainties around the epuck's pose and landmark positions

    # line 7 and beyond prob robotics (ekf slam known correspondences)
    # define lists that accumulate relevant matrices from within the for-loop, that are required outside of it
    measurement_deltas = []
    measurement_jacobians = []
    kalman_gains = []

    # iterate through all the landmark measurements
    # Probabilistic Robotics takes the relative bearing between the robot's heading theta and the landmark as phi - when calculating it, I denoted it as alpha
    # Probabilistic robotics refers to the iterator variable j as the index of the landmark in the list of landmarks - I denoted it as "correspondence". This version of the algorithm assumes this to be known for each measurement
    # distance is the range between the landmark observed and the epuck
    # signature is treated as 0 at the moment
    #   - Probabilistic Robotics explains that a feature extractor may generate a signature, which is assumed to be a numerical value - the example they give is average colour
    #   - the signature is treated as 0 at the moment as it has not been implemented to any significance
    for ((distance, alpha, signature), correspondence) in z_t:
        measurement = np.array([[distance], [alpha], [signature]]) # re-pack so vector can be used later for getting the delta between the actual measurement and the expected measurement

        landmark_estimate = state_estimate_bar[3 + (3 * correspondence) : 3 + (3 * correspondence) + 3] # estimated x and y coords of landmark

        # lines 9 & 10
        # landmark x,y coordinates were initialised to nan to distinguish whether landmark has been seen before
        # ie. nan -> not seen before
        if np.isnan(landmark_estimate[0]):
            # landmark hasn't been seen before
            # hence, take its position relative to the current estimated pose of the epuck

            # third element of the state estimate is theta, which isn't relevant here - instead the third element should be the signature (taken as 0 for now), hence why the first term slices only [0:2] of state_estimate
            landmark_estimate = np.array([state_estimate_bar[0], state_estimate_bar[1], [0]]) + (distance *  np.array([[np.cos(alpha + state_estimate_bar[2,0])], [np.sin(alpha + state_estimate_bar[2,0])], [signature]]))
            state_estimate_bar[3 + (3 * correspondence): 3 + (3 * correspondence) + 3] = landmark_estimate # write back into the state estimate vector (the version passed in to this function from the time update step, not the global one)


        # lines 12 and 13
        # helper variables for the x and y displacement between the epuck and current landmark
        delta_x = landmark_estimate[0,0] - state_estimate_bar[0,0]
        delta_y = landmark_estimate[1,0] - state_estimate_bar[1,0]
        delta = np.array([delta_x, delta_y])

        q = np.dot(np.transpose(delta), delta) # squared distance between epuck and current landmark

        # line 14
        # estimate the measurement using the measurement model
        # ie. what is the expected value of the measurement of the landmark, which is compared to the actual measured value in section that implements line 19
        # the estimated measurement is constructed of the distance (sqrt q), the relative heading, and the signature variable
        estimated_measurement = np.array([[np.sqrt(q)], [np.atan2(delta_y, delta_x) - state_estimate_bar[2,0]], landmark_estimate[2]])

        # line 15
        # matrix used to apply the jacobian of the measurement model w.r.t the combined state vector to only the elements that correspond to the epuck pose and current landmark
        f_xj = np.zeros((6, 3 + 3 * len(landmarks)))
        f_xj[:3,:3] = np.eye(3)
        f_xj[3:,3 + (2 * correspondence) - 2:3 + (2 * correspondence) - 2+3] = np.eye(3)

        # line 16
        # note that in Table 10.1 of Probabilistic Robotics - at least in the version of the book that I had access to - there's several elements of the matrix that are off by a factor of -1. This has been corrected in this implementation
        # where h is the measurement model:
        h_jacobian_r_line = np.array([[-1 * delta_x * np.sqrt(q)], [-1 * delta_y * np.sqrt(q)], [0], [delta_x * np.sqrt(q)], [delta_y * np.sqrt(q)], [0]])
        h_jacobian_phi_line = np.array([[delta_y], [-1 * delta_x], [-1], [-1 * delta_y], [delta_x], [0]])
        h_jacobian_signature_line = np.array([[0], [0], [0], [0], [0], [1]])

        h_jacobian = np.vstack([
            np.transpose(h_jacobian_r_line),
            np.transpose(h_jacobian_phi_line),
            np.transpose(h_jacobian_signature_line)
        ])

        h_jacobian = (1 / q) * np.dot(h_jacobian, f_xj)

        # line 17
        # Q - noise parameters - for a given range & bearing (& signature) measurement, how far off is it expected to be from the true measurement
        k_t = np.dot(np.dot(covariance_bar, np.transpose(h_jacobian)),  np.linalg.inv(np.dot(h_jacobian, np.dot(covariance_bar, h_jacobian.transpose())) + Q))

        # append relevant matrices to the lists outside the loop so that the updated state estimate and updated covariance can be calculated and returned
        measurement_deltas.append(measurement - estimated_measurement)
        measurement_jacobians.append(h_jacobian)
        kalman_gains.append(k_t)

    # (out of loop)
    updated_covariance = covariance_bar
    updated_state_estimate = state_estimate_bar
    if len(measurement_deltas) > 0:
        intermediate_var = np.zeros(np.shape(np.dot(kalman_gains[0], measurement_jacobians[0])))
        for i in range(len(measurement_deltas)):
            updated_state_estimate += np.dot(kalman_gains[i], measurement_deltas[i]) # implements functionality on line 19 - updates state estimate
            intermediate_var += np.dot(kalman_gains[i], measurement_jacobians[i])

        updated_covariance = np.dot((np.eye(intermediate_var.shape[0]) - intermediate_var), covariance_bar) # implements functionality on line 20 - updates state uncertainty

    return updated_state_estimate, updated_covariance

def temp_measure_landmarks():
    z = [] # measurements
    index = 0 # iterator index so correspondences are known

    # consider each landmark
    for landmark in landmarks:
        # get distance between epuck (ie. current pose) and landmark
        distance = np.sqrt(np.square((landmark[0] - x_t[0])) + np.square((landmark[1] - x_t[1])))
        # check whether it's within the epuck's fov - 0.84 radians
        alpha = np.arctan2(landmark[1] - x_t[1], landmark[0] - x_t[0]) - x_t[2] # in this case, opposite is delta y, and adjacent is delta x
        if np.abs(alpha) < 0.42 and distance < 0.06: # Wks 1-4 lab handout says epuck camera can see about 5.5cm in front of it
            # each measurement takes the form of: distance to landmark, relative angle from epuck heading to landmark, correspondence of landmark (as per Probabilistic Robotics Table 10.1)
            z.append(((distance, alpha, 0), index)) # add 0 as the signature
            # print(f"landmark measured: distance: {distance}, alpha: {rad_to_deg(alpha)}, index: {index}")

        index += 1 # increment index to keep correspondences correct

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
    # temporary function, until landmarks implemented in environment

    for x, y, signature in landmarks:
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
    z_t = temp_measure_landmarks()

    # Process sensor data
    state_estimate_prime, covariance_prime = time_update()
    state_estimate, covariance = observation_update(state_estimate_prime, covariance_prime)

    print(f"state_estimate: {state_estimate}")
    print(f"covariance: {covariance}")
    print("\n\n")


    # Actuate
    drive_logic()

    # ---------------------------------

    # Update display
    clean_display()
    draw_pose()
    temp_draw_landmarks()
    draw_state_estimate()

    pass

# Enter here exit cleanup code.
