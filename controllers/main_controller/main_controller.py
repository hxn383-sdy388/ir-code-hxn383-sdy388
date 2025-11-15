"""main_controller controller."""
import numpy as np

# ---------- IMPORTS ----------
from controller import Robot
from controller import Supervisor
from controller import Keyboard # for driving in the "explore" phase


# ---------- CONSTANTS ----------
SPEED_UNIT = 0.00628
LEFT = 0 # used to refer to the left wheel's motor in the speed
RIGHT = 1
PIXELS_PER_METRE = 100 # ratio of pixels per metre - i.e. 1 pixels corresponds to 1cm


# ---------- PARAMETERS ----------
MAX_SPEED = 200 # can go to a maximum of 1000. Speed limit helps keep motion of epuck smooth
SPEED_INCREMENT = 4

ALPHA_THRESHOLD = 10

'''
The noise matrix is a diagonal matrix on which the elements are the covariance of random noise added to the state uncertainty at every prediction step.

ie. if the first diagonal element was set to 1, this would correspond to an addition of 1 metre's worth of uncertainty in the x coordinate of the epuck's pose per timestep.
'''
NOISE = np.diag([0.000000001, 0.000000001, 0.000000001]) # experimenting with some simulated noise

'''
Units for the first two elements are metres - ie. each of these are just distances.
ie. for given range and relative bearing measurements, how far off are the actual range and bearing (and signature - but that's expected to be zero in the current configuration) measurements expected to be.
None of these values can be zero or the inverse matrix operation later falls apart.

Q distance uncertainties currently set to half the radius of the landmark objects.
'''
Q = np.diag([0.015,0.015,0.000000001])

'''
Diagonal elements on the covariance matrix corresponding to landmarks are initialised to a large value, to model that the initial positions of the landmarks are unknown.
In Probabilistic Robots, infinity is used for these values, but instead a relatively large finite value is used here due to computational issues encountered when using infinity.
'''
LANDMARK_COVARIANCE_INIT = 100


# ---------- VARIABLES & DATA STRUCTURES ----------
speed = [0,0] # list for controlling the speed - done this way control speed using encoder steps rather than target velocity directly
x_t = [0,0,0] # list for storing pose at current time $ t $, in the format [$ x $, $ y $, $ \theta $]
u_t = [0,0] # list for storing the control applied at time $ t - 1 $ to drive the epuck to pose $ \vec{x}_t $ at time $ t $. Elements: linear velocity, angular velocity
z_t = [] # list for landmark measurements, where each element is of the form (distance, bearing from epuck, correspondence)
landmark_counter = 0 # counter for how many distinct landmarks the system believes there to be at the current time $ t $

'''
The state estimate vector, notated in Probabilistic Robotics as $ u_t $, is a vector containing first elements of the epuck's pose, and then elements for all the (x,y) coordinates of all landmarks.

The state estimate corresponds to the mean of the multivariate gaussian being used to model the uncertainty in our belief over pose and landmark positions - ie. the best guess to where the robot is (based on its pose and surrounding landmarks).

The vector is initialised as a column vector with 3 elements - one for each of the epuck's pose variables, and zero for landmarks (as there are no initially assumed landmarks).

# The values of the vector are initialised with zero, as the initial pose is arbitrarily taken as the origin.
'''
state_estimate = np.zeros((3,1))

'''
The covariance matrix is a square matrix of size (3 + 3n) x (3 + 3n), where n corresponds to the number of landmarks at time t.
The first (3 x 3) elements correspond to the epuck pose, and the rest correspond to landmarks.
The matrix is initialised as a (3 x 3) matrix of zeros, as there are no initially assumed landmarks, hence the covariance matrix only needs to account for the epuck's pose at t = 0.
Zeros are filled in on the first three diagonal elements for the epuck's pose, consistent with the initialisation outlined in Probabilistic Robotics.

The diagonal elements correspond to the variances of the uncertainty in x, y, theta, and positions of the landmarks. 
The off-diagonal elements correspond to the correlations, which are initialised as zero so as to not assume correlation between variables, where these elements are updated by the EKF in the prediction and correction steps.
'''
covariance = np.zeros((3,3)) # state uncertainty for epuck - assume known position initially


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

def time_update(input_state_estimate):
    # line 3 prob robotics (ekf slam full)

    # Probabilistic Robotics defines the combined state vector as the vector of the robot's pose and (x,y) coordinates of all landmarks.
    # The coordinates of each of these landmarks is assumed to remain constant throughout simulation.
    #
    # The purpose of the selector matrix f_x is for the state estimate to be updated, manipulating only the entries corresponding to the robot's pose, leaving entries corresponding to landmarks unchanged.
    #
    # The matrix is initialised as a horizontally stacked (3 x 3) identity matrix, and a (3 x 3n) matrix, where n is the number of landmarks.
    f_x = np.hstack((np.eye(3), np.zeros((3, (np.shape(state_estimate)[0] - 3)))))


    # line 4
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
    dt = timestep / 1000 # timestep is in milliseconds, want in seconds
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
    updated_state_est = input_state_estimate + np.dot(f_x.T, motion_model_matrix)


    # Note that Probabilistic robotics termed $ y_t $ as the combined state vector - ie. the pose and landmarks
    # Lines 5 and 6 of the textbook algorithm are responsible for updating the state uncertainty with respect to the motion model and random noise

    # Line 5 defines $ G_t $ - an auxiliary matrix constructed by taking the Jacobian of the state model, where the only non-zero elements are the first derivatives of the x and y motion model components w.r.t theta
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


    # line 6
    # Use the auxiliary matrix to update the covariance matrix from the previous covariance, plus some noise from a random variable that's added to the state each prediction update
    updated_covariance = np.dot(np.dot(jacobian_wrt_combined_state_vector, covariance), jacobian_wrt_combined_state_vector.transpose()) + np.dot(np.dot(f_x.transpose(), NOISE), f_x)


    return updated_state_est, updated_covariance

def observation_update(state_estimate_bar, covariance_bar):
    # Goal is to use the measurements to inform the state estimate (of both epuck pose and landmark positions) and inform uncertainties around the epuck's pose and landmark positions
    # Need to decide whether each measurement corresponds to a new or existing landmark


    # line 8 and beyond prob robotics (ekf slam known correspondences)
    # define lists that accumulate relevant matrices from within the for-loops, that are required outside of it
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

    # define a locally-scoped counter used to track how many new landmarks are created
    n_t = landmark_counter

    # outer loop - iterate through the measurements
    for ((distance, alpha, signature), correspondence) in z_t:
        measurement = np.array([[distance], [alpha], [signature]]) # re-pack so vector can be used later for getting the delta between the actual measurement and the expected measurement


        # line 9
        # speculate that the measurement corresponds to an unseen landmark, and take its position relative to the current estimated pose of the epuck
        landmark_estimate = np.array([state_estimate_bar[0], state_estimate_bar[1], [0]]) + (distance * np.array([[np.cos(alpha + state_estimate_bar[2, 0])], [np.sin(alpha + state_estimate_bar[2, 0])], [signature]]))

        # define lists that accumulate relevant data from within the inner for-loop, which are required for deciding whether to create a new landmark or update and existing one
        intermediate_deltas = []
        intermediate_landmark_measurement_covs = []
        intermediate_manahalobis_dists = []
        intermediate_h_jacobians = []

        # provisionally augment the state estimate vector to be one larger
        # ie. model the observation as being a new landmark for the scope of this inner for-loop
        state_estimate_speculation = np.vstack((state_estimate_bar, landmark_estimate))

        # provisionally augment the covariance matrix accordingly
        n = covariance_bar.shape[0]
        covariance_speculation = np.pad(covariance_bar, ((0, 3), (0, 3)), mode='constant', constant_values=0)
        covariance_speculation[n, n] = LANDMARK_COVARIANCE_INIT # set the diagonal elements
        covariance_speculation[n + 1, n + 1] = LANDMARK_COVARIANCE_INIT
        covariance_speculation[n + 2, n + 2] = LANDMARK_COVARIANCE_INIT


        # line 10
        # inner for-loop - needs to be one more than n_t - explanation is for when there are zero landmarks, this still needs to run, hence needs to be n_t + 1
        for k in range(0, n_t + 1):

            # lines 11 and 12
            # helper variables for the x and y displacement between the epuck and current landmark k
            delta_kx = state_estimate_speculation[3 + (3 * k), 0] - state_estimate_speculation[0,0]
            delta_ky = state_estimate_speculation[3 + (3 * k) + 1, 0] - state_estimate_speculation[1,0]
            delta_k = np.array([delta_kx, delta_ky])

            q_k = np.dot(np.transpose(delta_k), delta_k) # squared distance between epuck and current landmark k


            # line 13
            # estimate the measurement using the measurement model
            # ie. what is the expected value of the measurement of the landmark k, which is compared to the actual measured value later on, out of both of the for-loops
            # the estimated measurement is constructed of the distance (sqrt q), the relative heading, and the signature variable
            heading_est_k = np.atan2(delta_ky, delta_kx) - state_estimate_speculation[2,0]
            estimated_measurement_k = np.array([[np.sqrt(q_k)], [np.arctan2(np.sin(heading_est_k), np.cos(heading_est_k))], state_estimate_speculation[3 + k + 2]])


            # line 14
            # selector matrix used to apply the jacobian of the measurement model w.r.t the combined state vector to only the elements that correspond to the epuck pose and current landmark k
            f_xk = np.zeros((6, 3 + 3 * (n_t + 1)))
            f_xk[:3,:3] = np.eye(3) # select elements corresponding to epuck pose
            f_xk[3:,3 + (2 * k) - 2:3 + (2 * k) - 2+3] = np.eye(3) # select elements corresponding to current landmark k


            # line 15
            # note that in Table 10.2 of Probabilistic Robotics - at least in the version of the book that I had access to - there's several elements of the matrix that are off by a factor of -1. This has been corrected in this implementation
            # where h is the measurement model:
            h_k_jacobian_r_line = np.array([[-1 * delta_kx * np.sqrt(q_k)], [-1 * delta_ky * np.sqrt(q_k)], [0], [delta_kx * np.sqrt(q_k)], [delta_ky * np.sqrt(q_k)], [0]])
            h_k_jacobian_phi_line = np.array([[delta_ky], [-1 * delta_kx], [-1], [-1 * delta_ky], [delta_kx], [0]])
            h_k_jacobian_signature_line = np.array([[0], [0], [0], [0], [0], [1]])

            h_k_jacobian = np.vstack([
                np.transpose(h_k_jacobian_r_line),
                np.transpose(h_k_jacobian_phi_line),
                np.transpose(h_k_jacobian_signature_line)
            ])

            h_k_jacobian = (1 / q_k) * np.dot(h_k_jacobian, f_xk)


            # line 16
            # calculate covariance of the estimated measurement of landmark k, with respect to current epuck pose
            # a quantity based on:
            #   - the jacobian of the measurement model, with respect to the current state
            #   - the covariance, with respect to the full state estimate
            #   - measurement noise covariance
            # which combined, represents the uncertainty of measuring landmark k from the current epuck pose
            landmark_measurement_covariance = np.dot(np.dot(h_k_jacobian, covariance_speculation), np.transpose(h_k_jacobian)) + Q


            # line 17
            # calculate the Mahalanobis distance between the measurement z, and the estimated measurement of landmark k with respect to the current epuck pose
            # want this quantity to be as low as possible, as want to interpret the measurement as measuring the landmark that makes the most sense from the current epuck pose
            # if no existing landmark exists that makes more sense than categorising the measurement has measuring a new landmark, a new landmark is made - this threshold is set and evaluated later
            mahalanobis_dist = np.dot(np.dot(np.transpose(measurement - estimated_measurement_k), np.linalg.inv(landmark_measurement_covariance)), (measurement - estimated_measurement_k))

            # add all these structures to the intermediate lists so that it can be decided whether the landmark is new or not, and then update the state estimate and covariance accordingly
            intermediate_landmark_measurement_covs.append(landmark_measurement_covariance)
            intermediate_manahalobis_dists.append(mahalanobis_dist)
            intermediate_h_jacobians.append(h_k_jacobian)
            intermediate_deltas.append((measurement - estimated_measurement_k))
            # end inner for-loop

        # (out of the inner loop)

        # line 19
        # set the threshold for creating a new landmark
        # ie. the measurement is interpreted as measuring a new landmark if the Mahalanobis distance to all existing landmarks exceeds this threshold
        # setting this to a larger value increases the likelihood of rejecting the measurement as being a new landmark, as it's more likely on the next line to prefer an existing landmark that has a lower value
        # setting this to a lower value increases the likelihood of accepting this measurement as being a new landmark, where it's more likely that this value will be lower than the corresponding values for the other landmarks
        #
        # intuition for why this needs to be set at all:
        #   this quantity is populated by the inner for-loop above, but needs to be augmented, as the mahalanobis distance between the estimated measurement of landmark k if k is actually just this measurement modelled as a new landmark, will always be trivially low
        #   hence, to stop always preferring creating new landmarks in the state estimate, the mahalanobis distance between the estimated measurement of this landmark if it's new, and the actual measurement, needs to be upped to a threshold to regulate the association rate with existing landmarks vs creating new ones
        intermediate_manahalobis_dists[len(intermediate_manahalobis_dists) - 1] = np.array([[ALPHA_THRESHOLD]])


        # line 20
        # select the landmark index that minimises the mahalanobis distance
        j_i = np.argmin(intermediate_manahalobis_dists)


        # line 21
        # if the measurement is a new landmark
        if j_i + 1 > n_t:
            n_t += 1 # increase the landmark counter

            # augment state estimate vector and covariance matrix, assigning the speculated versions that have the dimension(s) added to account for the new landmark
            state_estimate_bar = state_estimate_speculation
            covariance_bar = covariance_speculation


        # line 22
        # use the kalman gain to regulate how much weight to give this measurement in updating the state estimate and covariance later
        # ie. it regulates belief in the measurement vs belief in the prior state estimate
        # when the prior state estimate is more uncertain, the kalman gain assumes a larger value to put more weight on the measurement
        # when the measurement is noisier, the kalman gain assumes a smaller value to put more weight on the prior state estimate
        kalman_gain = np.dot(np.dot(covariance_speculation, np.transpose(intermediate_h_jacobians[j_i])), np.linalg.inv(intermediate_landmark_measurement_covs[j_i]))

        kalman_gains.append(kalman_gain)
        measurement_deltas.append(intermediate_deltas[j_i])
        measurement_jacobians.append(intermediate_h_jacobians[j_i])


    # (out of outer for-loop)
    # preparation for lines 24 and 25
    if n_t > landmark_counter:
        # ie. at least one measurement has been interpreted as a new landmark

        # therefore:
        #   - need to step up the dimensions of the kalman gain matrices to have the correct number of rows to have entries for all new landmarks, so that matrix calculations can proceed correctly
        #   - need to step up the dimensions of the measurement jacobian matrices to have the correct number of columns to have entries for all new landmarks, so that matrix calculations can proceed correctly

        for i in range(0,len(kalman_gains)): # can use the kalman gains for the index, as each of these lists will have the same number of elements (one per measurement)
            # cols need expanding to num of cols in state est
            while np.shape(kalman_gains[i])[0] < np.shape(state_estimate_bar)[0]:
                kalman_gains[i] = np.vstack([kalman_gains[i], np.zeros((1, kalman_gains[i].shape[1]))])

            # step up measurement jacobians - rows need expanding to num of cols in state est:
            extra_cols = np.shape(state_estimate_bar)[0] - np.shape(measurement_jacobians[i])[1]
            if extra_cols > 0:
                measurement_jacobians[i] = np.hstack([
                    measurement_jacobians[i],
                    np.zeros((measurement_jacobians[i].shape[0], extra_cols))
                ])
    else:
        # ie. no measurements were interpreted as a new landmark

        # therefore:
        #   - need to trim the number of rows on the kalman gain matrices to remove the last three rows which correspond to how the new measurement (for the iteration this matrix belongs to) was speculated to be a new landmark, so that matrix calculations can proceed correctly under the assumption that this measurement corresponds to an existing landmark
        #   - need to trim the number of columns on the measurement jacobian matrices to remove the last three rows which correspond to how the new measurement (for the iteration this matrix belongs to) was speculated to be a new landmark, so that matrix calculations can proceed correctly under the assumption that this measurement corresponds to an existing landmark

        for i in range(0,len(measurement_jacobians)):
            if np.shape(kalman_gains[i])[0] > np.shape(state_estimate_bar)[0]:
                kalman_gains[i] = kalman_gains[i][:-3,:] # remove the last three rows

            measurement_jacobians[i] = measurement_jacobians[i][:, :-3] # remove last three columns


    # lines 24 and 25
    # initialise the return values
    updated_covariance = covariance_bar
    updated_state_estimate = state_estimate_bar

    if len(measurement_deltas) > 0:
        intermediate_var = np.zeros(np.shape(np.dot(kalman_gains[0], measurement_jacobians[0])))
        for i in range(0, len(measurement_deltas)):
            updated_state_estimate += np.dot(kalman_gains[i], measurement_deltas[i]) # implements functionality on line 24 - updates state estimate
            intermediate_var += np.dot(kalman_gains[i], measurement_jacobians[i])

        updated_covariance = np.dot((np.eye(intermediate_var.shape[0]) - intermediate_var), covariance_bar) # implements functionality on line 25 - updates state uncertainty

    updated_state_estimate[2,0] = np.arctan2(np.sin(updated_state_estimate[2,0]), np.cos(updated_state_estimate[2,0]))
    return updated_state_estimate, updated_covariance, n_t

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

# Occupancy Grid
def calculate_occupancy_grid():
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
    # z_t = temp_measure_landmarks()
    z_t = cam_recog_measure_landmarks()


    # Process sensor data
    state_estimate_prime, covariance_prime = time_update(state_estimate)
    state_estimate, covariance, landmark_counter = observation_update(state_estimate_prime, covariance_prime)

    print(f"state_estimate: {state_estimate}")
    print("\n\n")


    # Actuate
    drive_logic()

    # ---------------------------------

    # Update displays
    clean_displays()
    draw_pose()
    temp_draw_landmarks()
    draw_state_estimate()

    pass

# Enter here exit cleanup code.
