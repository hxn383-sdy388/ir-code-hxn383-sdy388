# ---------- IMPORTS ----------
import numpy as np


# ---------- CLASSES ----------
class EkfSlamController:
    # ---------- CONSTANTS ----------
    TIMESTEP = 0

    CELL_SIZE = 0.05  # (5cm) - the cell side length measurement for the occupancy grid encoding of the state estimate


    # ---------- PARAMETERS ----------
    ALPHA_THRESHOLD = 10


    # The noise matrix is a diagonal matrix on which the elements are the covariance of random noise added to the state
    # uncertainty at every prediction step.
    #
    # ie. if the first diagonal element was set to 1, this would correspond to an addition of 1 metre's worth of
    # uncertainty in the x coordinate of the epuck's pose per timestep.
    #
    NOISE = np.diag([0.000000001, 0.000000001, 0.000000001])  # very low values due to high confidence in pose data


    # Units for the first two elements are metres - ie. each of these are just distances.
    #
    # ie. for given range and relative bearing measurements, how far off are the actual range and bearing (and
    # signature - but that's expected to be zero in the current configuration) measurements expected to be.
    #
    # None of these values can be zero or the matrix inversion operation later falls apart.
    #
    Q = np.diag([0.015, 0.015, 0.000000001]) # currently set to half the radius of the landmark objects.
    # Q = np.diag([0.000000001, 0.000000001, 0.000000001])


    # Diagonal elements on the covariance matrix corresponding to landmarks are initialised to a large value, to model
    # that the initial positions of the landmarks are unknown.
    #
    # In Probabilistic Robots, infinity is used for these values, but instead a (relatively) large finite value is used
    # here due to computational issues encountered when using infinity.
    #
    LANDMARK_COVARIANCE_INIT = 100


    # ---------- VARIABLES & DATA STRUCTURES ----------
    # counter for how many distinct landmarks the system believes there to be at the current time t
    landmark_counter = 0


    # The state estimate vector, notated in Probabilistic Robotics as u_t, is a vector containing first elements of
    # the epuck's pose, and then elements for all the (x,y) coordinates of all landmarks.
    #
    # The state estimate corresponds to the mean of the gaussian distribution being used to model the uncertainty in
    # our belief over pose and landmark positions - ie. the best guess to where the robot is (based on its pose and
    # surrounding landmarks).
    #
    # The vector is initialised as a column vector with 3 elements - one for each of the epuck's pose variables, and
    # zero for landmarks (as there are no initially assumed landmarks).
    #
    # The values of the vector are initialised with zero, as the initial pose is arbitrarily taken as the origin.
    state_estimate = np.zeros((3, 1))


    # The covariance matrix is a square matrix of size (3 + 3n) x (3 + 3n), where n corresponds to the number of
    # landmarks at time t.
    #
    # The first (3 x 3) elements correspond to the epuck pose, and the rest correspond to landmarks.
    #
    # The matrix is initialised as a (3 x 3) matrix of zeros, as there are no initially assumed landmarks, hence the
    # covariance matrix only needs to account for the epuck's pose at t = 0.
    #
    # The diagonal elements correspond to the variances of the uncertainty in x, y, theta, and positions of landmarks.
    #
    # The off-diagonal elements correspond to the correlations, which are initialised as zero so as to not assume
    # correlation between variables, where these elements are updated by the EKF in the prediction and correction steps.
    covariance = np.zeros((3, 3))  # state uncertainty for epuck - assume known position initially


    # ---------- CONSTRUCTOR ----------
    def __init__(self, timestep):
        self.TIMESTEP = timestep


    # ---------- METHODS ----------
    def __time_update(self, u_t):
        # line 3 prob robotics (ekf slam full)

        # Probabilistic Robotics defines the combined state vector as the vector of the robot's pose and (x,y)
        # coordinates of all landmarks.
        #
        # The coordinates of each of these landmarks is assumed to remain constant throughout simulation.
        #
        # The purpose of the selector matrix f_x is for the state estimate to be updated, manipulating only the entries
        # corresponding to the robot's pose, leaving entries corresponding to landmarks unchanged.
        #
        # The matrix is initialised as a horizontally stacked (3 x 3) identity matrix, and a (3 x 3n) matrix, where n
        # is the number of landmarks.
        f_x = np.hstack((np.eye(3), np.zeros((3, (np.shape(self.state_estimate)[0] - 3)))))


        # line 4
        # update the state estimate: u_t
        # intuition - take the best state estimate from the previous time step, and based on the control u_t, update
        # the state estimate for this time step

        # need to calculate how the epuck's pose will have changed between the last time step (t - 1) and now (t)
        #
        # the motion model for the epuck will be based on that it's moving with both a value for linear velocity and
        # angular velocity
        #
        # hence, for the time step, it has moved on a circular arc of radius R = linear velocity / angular velocity,
        # centered about the ICC (Instantaneous Centre of Curvature)
        #
        # following through the trig (using a diagram on differential drive robots from Dudek and Jenkin, Computational
        # Principles of Mobile Robotics) this gives that:
        #   the location of the ICC relative to the centre of the robot (x,y) is (x_{t-1} - R(sin theta)) for the x
        #   coord, and (y_{t-1} + R(cos theta)) for the y coord
        #
        #   therefore, we can get the updated pose of the epuck with:
        #       x_t = x_{t-1} - R(sin(theta_{t-1})) + R(sin(theta_{t-1} + delta * theta))  ie. the x-coord
        #           is the x-coord of the ICC sum the x-distance between the ICC and centre of the epuck after it's
        #           heading has changed by (delta * theta)
        #       y_t = y_{t-1} + R(cos(theta_{t-1})) - R(cos(theta_{t-1} + delta * theta))  ie. the y-coord
        #           is the y-coord of the ICC sum the y-distance between the ICC and the centre of the epuck after it's
        #           heading has changed by (delta * theta)
        #       theta_t = theta_{t-1} + delta * theta  ie. the heading of the epuck is it's old heading
        #           sum the change in heading
        #
        #   substituting in that R = linear velocity / angular velocity, gives us the motion model matrix below:

        # helper variables
        dt = self.TIMESTEP / 1000  # timestep is in milliseconds, want in seconds
        old_theta = self.state_estimate[
            2, 0]  # theta is 3rd element of vector - need to get it from column zero, as it's implemented as a matrix
        delta_theta = u_t[1] * dt  # ie. change in angle is angular velocity multiplied by change in time

        # matrix components
        # because of numerical stability issues, need to handle case where angular velocity is near zero
        if np.abs(u_t[1]) > 0.0001:
            r = (u_t[0] / u_t[1])  # linear velocity / angular velocity

            motion_model_x = (-1 * r * (np.sin(old_theta))) + (r * np.sin(old_theta + delta_theta))
            motion_model_y = (r * (np.cos(old_theta))) - (r * np.cos(old_theta + delta_theta))
            motion_model_theta = delta_theta
        else:
            # angle near zero, model as strictly linear movement
            motion_model_x = u_t[0] * np.cos(old_theta) * dt
            motion_model_y = u_t[0] * np.sin(old_theta) * dt
            motion_model_theta = 0.0  # angular velocity tends to zero, so heading assumed to not change

        # matrix construction
        motion_model_matrix = np.array(
            [[float(motion_model_x)], [float(motion_model_y)], [float(motion_model_theta)]])  # 3x1 matrix

        # apply the motion model to update the state estimate
        updated_state_est = self.state_estimate + np.dot(f_x.T, motion_model_matrix)


        # Note that Probabilistic Robotics termed y_t as the combined state vector - ie. the pose and landmarks
        #
        # Lines 5 and 6 of the textbook algorithm are responsible for updating the state uncertainty with respect to
        # the motion model and random noise


        # Line 5 defines G_t - an auxiliary matrix constructed by taking the Jacobian (matrix of all first-order
        # partial derivatives) of the state model, where the only non-zero elements are the first derivatives of
        # the x and y motion model components w.r.t theta
        #
        # ie. The non-zero elements model how the x and y coords of the epuck are changing
        #
        # There is a differentiation error in the version of the book that I had access to, where both elements are off
        # by a factor of -1, which has been corrected in this implementation
        #
        # Again, as working with dividing by the angular velocity again, need to handle the case where the angular
        # velocity tends to 0
        if np.abs(u_t[1]) > 0.0001:
            r = (u_t[0] / u_t[1])  # linear velocity / angular velocity

            # differentiate with respect to theta
            first_deriv_x = (-1 * r * (np.cos(old_theta))) + (r * np.cos(old_theta + delta_theta))
            first_deriv_y = (-1 * r * (np.sin(old_theta))) + (r * np.sin(old_theta + delta_theta))
        else:
            # angle near zero, model as strictly linear movement

            # differentiate with respect to theta
            first_deriv_x = -1 * u_t[0] * np.sin(old_theta) * dt
            first_deriv_y = u_t[0] * np.cos(old_theta) * dt

        # construct the jacobian matrix with derivatives of the motion model w.r.t the epuck's pose
        jacobian_wrt_pose = np.zeros((3, 3))
        jacobian_wrt_pose[0, 2] = first_deriv_x
        jacobian_wrt_pose[1, 2] = first_deriv_y

        # construct matrix that embeds this in the space represented by the combined state vector, rather than just pose
        jacobian_wrt_combined_state_vector = np.eye((f_x.shape[1])) + np.dot(np.dot(f_x.transpose(), jacobian_wrt_pose),
                                                                             f_x)

        # line 6
        # Use the auxiliary matrix to update the covariance matrix from the prior covariance, plus some noise from a
        # random variable that's added to the state each prediction update
        updated_covariance = np.dot(np.dot(jacobian_wrt_combined_state_vector, self.covariance),
                                    jacobian_wrt_combined_state_vector.transpose()) + np.dot(
            np.dot(f_x.transpose(), self.NOISE), f_x)


        # results are NOT written to the attributes of the object, as they are intermediate results
        return updated_state_est, updated_covariance

    def __observation_update(self, z_t, state_estimate_bar, covariance_bar):
        # Goal is to use the measurements to inform the state estimate (of both epuck pose and landmark positions) and
        # inform uncertainties around the epuck's pose and landmark positions
        #
        # Need to decide whether each measurement corresponds to a new or existing landmark


        # line 8 and beyond prob robotics (ekf slam full)
        # define lists that accumulate relevant matrices from within the inner for-loop, that are required outside of it
        measurement_deltas = []
        measurement_jacobians = []
        kalman_gains = []


        # iterate through all the landmark measurements
        #
        # Note that Probabilistic Robotics takes the relative bearing between the robot's heading theta and the
        # landmark as phi - when calculating it, I denoted it as alpha
        #
        # distance is the range between the landmark observed and the epuck
        #
        # signature is treated as 0 at the moment
        #   - Probabilistic Robotics explains that a feature extractor may generate a signature, which is assumed to
        #       be a numerical value - the example they give is average colour
        #   - the signature is treated as 0 at the moment as it has not been implemented to any significance


        # define a locally-scoped counter used to track how many new landmarks are created
        n_t = self.landmark_counter

        # outer loop - iterate through the measurements
        for (distance, alpha, signature) in z_t:
            # re-pack so vector can be used later for getting delta between the actual and expected measurement
            measurement = np.array([[distance], [alpha], [signature]])

            # line 9
            # speculate that the measurement corresponds to an unseen landmark, and take its position relative to the
            # current estimated pose of the epuck
            landmark_estimate = np.array([state_estimate_bar[0], state_estimate_bar[1], [0]]) + (distance * np.array(
                [[np.cos(alpha + state_estimate_bar[2, 0])], [np.sin(alpha + state_estimate_bar[2, 0])], [signature]]))

            # define lists that accumulate relevant data from within the inner for-loop, which are required for
            # deciding whether to create a new landmark or update and existing one
            intermediate_deltas = []
            intermediate_landmark_measurement_covs = []
            intermediate_mahalanobis_dists = []
            intermediate_h_jacobians = []

            # provisionally augment the state estimate vector to be one larger
            # ie. model the observation as being a new landmark for the scope of this inner for-loop
            state_estimate_speculation = np.vstack((state_estimate_bar, landmark_estimate))

            # provisionally augment the covariance matrix accordingly
            n = covariance_bar.shape[0]
            # ((0,3), (0,3)): first pair -> pad 0 rows on top, 3 on the bottom, second pair -> 0 on left, 3 on right
            covariance_speculation = np.pad(covariance_bar, ((0, 3), (0, 3)), mode = 'constant', constant_values = 0)
            covariance_speculation[n, n] = self.LANDMARK_COVARIANCE_INIT  # set the diagonal elements
            covariance_speculation[n + 1, n + 1] = self.LANDMARK_COVARIANCE_INIT
            covariance_speculation[n + 2, n + 2] = self.LANDMARK_COVARIANCE_INIT

            # line 10
            # inner for-loop - needs to be one more than n_t - explanation is for when there are zero landmarks, this
            # still needs to run, hence needs to be n_t + 1
            for k in range(0, n_t + 1):
                # lines 11 and 12
                # helper variables for the x and y displacement between the epuck and current landmark k
                delta_kx = state_estimate_speculation[3 + (3 * k), 0] - state_estimate_speculation[0, 0]
                delta_ky = state_estimate_speculation[3 + (3 * k) + 1, 0] - state_estimate_speculation[1, 0]
                delta_k = np.array([delta_kx, delta_ky])

                q_k = np.dot(np.transpose(delta_k), delta_k)  # squared distance between epuck and current landmark k

                # line 13
                # estimate the measurement using the measurement model
                #
                # ie. what is the expected value of the measurement of the landmark k, which is compared to the actual
                # measured value later
                #
                # the estimated measurement is constructed of the distance (sqrt q), the relative heading, and the
                # signature variable
                heading_est_k = np.arctan2(delta_ky, delta_kx) - state_estimate_speculation[2, 0]
                estimated_measurement_k = np.array(
                    [[np.sqrt(q_k)], [np.arctan2(np.sin(heading_est_k), np.cos(heading_est_k))],
                     state_estimate_speculation[3 + (3 * k) + 2]])

                # line 14
                # selector matrix used to apply the jacobian of the measurement model w.r.t the combined state vector
                # to only the elements that correspond to the epuck pose and current landmark k
                f_xk = np.zeros((6, 3 + 3 * (n_t + 1)))
                f_xk[:3, :3] = np.eye(3)  # select elements corresponding to epuck pose
                f_xk[3:, 3 + (3 * k):3 + (3 * k) + 3] = np.eye(
                    3)  # select elements corresponding to current landmark k

                # line 15
                #
                # note that in Table 10.2 of Probabilistic Robotics - at least in the version of the book that
                # I had access to - there's several elements of the matrix that are off by a factor of -1. This has
                # been corrected in this implementation
                #
                # where h is the measurement model:
                h_k_jacobian_r_line = np.array(
                    [[-1 * delta_kx * np.sqrt(q_k)], [-1 * delta_ky * np.sqrt(q_k)], [0], [delta_kx * np.sqrt(q_k)],
                     [delta_ky * np.sqrt(q_k)], [0]])
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
                landmark_measurement_covariance = np.dot(np.dot(h_k_jacobian, covariance_speculation),
                                                         np.transpose(h_k_jacobian)) + self.Q

                # line 17
                # calculate the Mahalanobis distance (dist between vector and distribution) between the measurement z,
                # and the estimated measurement of landmark k with respect to the current epuck pose
                #
                # want this quantity to be as low as possible, as want to interpret the measurement as measuring the
                # landmark that makes the most sense from the current epuck pose
                #
                # if no existing landmark exists that makes more sense than categorising the measurement has measuring
                # a new landmark, a new landmark is made - this threshold is set and evaluated later
                mahalanobis_dist = np.dot(np.dot(np.transpose(measurement - estimated_measurement_k),
                                                 np.linalg.inv(landmark_measurement_covariance)),
                                          (measurement - estimated_measurement_k))

                # add all these structures to the intermediate lists so that it can be decided whether the landmark is
                # new or not, and then update the state estimate and covariance accordingly
                intermediate_landmark_measurement_covs.append(landmark_measurement_covariance)
                intermediate_mahalanobis_dists.append(mahalanobis_dist)
                intermediate_h_jacobians.append(h_k_jacobian)
                intermediate_deltas.append((measurement - estimated_measurement_k))
                # end inner for-loop

            # (out of the inner loop)

            # line 19
            # set the threshold for creating a new landmark
            #
            # ie. the measurement is interpreted as measuring a new landmark if the Mahalanobis distance to all
            # existing landmarks exceeds this threshold
            #
            # setting this to a larger value increases the likelihood of rejecting the measurement as being a new
            # landmark, as it's more likely on the next line to prefer an existing landmark that has a lower value
            #
            # setting this to a lower value increases the likelihood of accepting this measurement as being a new
            # landmark, where it's more likely that this value will be lower than the corresponding values for the
            # other landmarks
            #
            # intuition for why this needs to be set at all:
            #   this quantity is populated by the inner for-loop above, but needs to be augmented, as the mahalanobis
            #       distance between the estimated measurement of landmark k if k is actually just this measurement
            #       modelled as a new landmark, will always be trivially low. ie. Can always trivially create a new
            #       landmark
            #   hence, to stop always preferring creating new landmarks in the state estimate, the mahalanobis distance
            #       between the estimated measurement of this landmark if it's new, and the actual measurement, needs
            #       to be upped to a threshold to regulate the association rate with existing landmarks vs creating
            #       new ones
            intermediate_mahalanobis_dists[len(intermediate_mahalanobis_dists) - 1] = np.array([[self.ALPHA_THRESHOLD]])

            # line 20
            # select the landmark index that minimises the mahalanobis distance
            j_i = np.argmin(intermediate_mahalanobis_dists)

            # line 21
            # if the measurement is a new landmark
            if j_i + 1 > n_t:
                n_t += 1  # increase the landmark counter

                # augment state estimate vector and covariance matrix, assigning the speculated versions that have the
                # dimension(s) added to account for the new landmark
                state_estimate_bar = state_estimate_speculation
                covariance_bar = covariance_speculation

            # line 22
            # use the kalman gain to regulate how much weight to give this measurement in updating the state estimate
            # and covariance later
            #
            # ie. it regulates belief in the measurement vs belief in the prior state estimate
            #
            # when the prior state estimate is more uncertain, the kalman gain assumes a larger value to put more
            # weight on the measurement
            #
            # when the measurement is noisier, the kalman gain assumes a smaller value to put more weight on the
            # prior state estimate
            kalman_gain = np.dot(np.dot(covariance_speculation, np.transpose(intermediate_h_jacobians[j_i])),
                                 np.linalg.inv(intermediate_landmark_measurement_covs[j_i]))

            kalman_gains.append(kalman_gain)
            measurement_deltas.append(intermediate_deltas[j_i])
            measurement_jacobians.append(intermediate_h_jacobians[j_i])


            # (still in the outer for-loop)

            # preparation for lines 24 and 25
            #
            # in the textbook, logic beyond here lies outside the outer for-loop
            #
            # however, I've moved it inside so that each measurement is processed after the correction step taken by
            # processing the previous measurement
            #
            # previously, when implemented exactly as set out in Table 10.2 of Probabilistic Robotics, the system was
            # behaving erratically, with landmark estimates being highly unstable over time
            #
            # with this change, each measurement that is processed now benefits from the uncertainty improvement made
            # from the previous, as the state estimate and covariance are updated inline, rather than in a batch
            # fashion at the end, which means that the kalman gain is correctly informed when processing each measurement


            # can use the kalman gains for the index, as these lists have same number of elements (one per measurement)
            for i in range(0, len(kalman_gains)):
                # kalman gains:
                # regulate the number of columns to be the same as the number in the state estimate
                while np.shape(kalman_gains[i])[0] < np.shape(state_estimate_bar)[0]:
                    kalman_gains[i] = np.vstack([kalman_gains[i], np.zeros((1, kalman_gains[i].shape[1]))])
                while np.shape(kalman_gains[i])[0] > np.shape(state_estimate_bar)[0]:
                    kalman_gains[i] = kalman_gains[i][:-3, :]  # remove the last three rows

                # jacobians:
                # regulate the number of rows to be the same as the number of columns in the state estimate
                extra_cols = np.shape(state_estimate_bar)[0] - np.shape(measurement_jacobians[i])[1]
                if extra_cols > 0:
                    measurement_jacobians[i] = np.hstack([
                        measurement_jacobians[i],
                        np.zeros((measurement_jacobians[i].shape[0], extra_cols))
                    ])
                while np.shape(measurement_jacobians[i])[1] > np.shape(state_estimate_bar)[0]:
                    measurement_jacobians[i] = measurement_jacobians[i][:, :-3]  # remove last three columns


            # logic for lines 24 and 25
            if len(measurement_deltas) > 0:
                intermediate_var = np.zeros(np.shape(np.dot(kalman_gains[0], measurement_jacobians[0])))
                for i in range(0, len(measurement_deltas)):
                    # implements functionality on line 24 - updates state estimate
                    state_estimate_bar += np.dot(kalman_gains[i], measurement_deltas[i])
                    intermediate_var += np.dot(kalman_gains[i], measurement_jacobians[i])

                # implements functionality on line 25 - updates state uncertainty
                covariance_bar = np.dot((np.eye(intermediate_var.shape[0]) - intermediate_var), covariance_bar)

            state_estimate_bar[2, 0] = np.arctan2(np.sin(state_estimate_bar[2, 0]), np.cos(state_estimate_bar[2, 0]))

        # (out of outer for-loop)

        # update the state of the object
        self.state_estimate = state_estimate_bar
        self.covariance = covariance_bar
        self.landmark_counter = n_t

        return

    def run_steps(self, u_t, z_t):
        # hide the steps from the caller, running both time update and observation update steps in one
        # (both time_update and observation_update are private - signified by the leading double-underscores)

        state_estimate_bar, covariance_bar = self.__time_update(u_t)
        self.__observation_update(z_t, state_estimate_bar, covariance_bar)

        return self.state_estimate, self.covariance

    def construct_occupancy_grid(self):
        # The occupancy grid is a matrix where each entry corresponds to a cell in the grid
        #
        # If the cell contains a 0, the cell is believed to be unoccupied by landmarks - ie. the epuck can enter it
        #
        # If the cell contains a 1, the cell is believed to be occupied by landmarks 0 ie. the epuck should not
        # attempt to enter it
        #
        #
        # Dimensions: m is the number of cells on the horizontal axis of the occupancy grid, n is the number on the
        # vertical axis
        #
        #
        # The occupancy grid is initialised as a 3x3 grid, with the origin taken as the centre of the middle cell
        #
        #
        # Because the occupancy grid conceptually operates in the Cartesian plane (with the origin taken as the epuck's
        # original position), but is encoded as a matrix without support for negative rows or columns, an origin cell
        # variable is constructed to enable occupancy in negative rows and columns to be represented as well as in
        # positive ones
        #
        #
        # The matrix is initially constructed where direction of row growth corresponds to an increase in the Cartesian
        # y coordinate of occupancy values
        #
        # This behaviour is the same for columns and Cartesian x coordinates
        #
        # However, for providing a more glanceable grid on printing/display, the matrix has been flipped about the
        # horizontal direction
        #
        # This means that in the top-left (0,0) of the matrix is the negative,positive quadrant of the Cartesian plane,
        # and in the bottom-right (n,m) is the positive,negative quadrant of the Cartesian plane
        #
        # Of course, this means that the cell containing the origin lies (initially) centrally in this matrix encoding
        #
        #
        # The occupancy grid is completely re-calculated every time step, as it relies directly on the combined state
        # estimate vector
        #
        # The combined state estimator vector's values are calculated under the Markov assumption
        #
        # Hence, at each timestep, the prior state of the occupancy grid should not influence the current state of the
        # occupancy grid
        #
        # Practically speaking, this is required so as to not hold prior occupancy values as truth when their estimates
        # may have been updated in the EKF-SLAM process
        #
        #
        # This means that the occupancy grid may shrink out unoccupied space over time
        #
        # For example, in development, when driving the epuck around a 1x1 metre grid with only four landmarks, in many
        # timesteps the epuck wouldn't perceive a landmark, and hence would be driving in open space
        #
        # This would be reflected in the occupancy grid with 0's being filled in for the epuck's pose
        #
        # However, when the epuck left the empty space, and it didn't fall within the minimum bounds set for the grid,
        # these cells would be removed from the grid
        #
        # This behaviour is intended, and stems from my decision to carry the Markov assumption through to the occupancy
        # grid based on it's dependency on the state estimate

        # ------------------------------

        # iterate through the state estimate and find the bounding coordinates
        # also set a minimum size, so that cells are drawn initially
        # 2 cell lengths in all directions, plus one middle cell for the origin, means the grid will initially be 3x3
        min_m = -(2 * self.CELL_SIZE)  # steps later on require this to be a negative value
        max_m = 2 * self.CELL_SIZE
        min_n = -(2 * self.CELL_SIZE)  # steps later on require this to be a negative value
        max_n = 2 * self.CELL_SIZE

        i = 0
        while i < len(self.state_estimate):
            if self.state_estimate[i] < min_m:
                min_m = self.state_estimate[i]
            if self.state_estimate[i] > max_m:
                max_m = self.state_estimate[i]

            if self.state_estimate[i + 1] < min_n:
                min_n = self.state_estimate[i + 1]
            if self.state_estimate[i + 1] > max_n:
                max_n = self.state_estimate[i + 1]

            i += 3  # 3 components per state estimate vector entry for epuck pose and landmarks

        # take the number of rows and columns either side of the origin as the minimum values to over-shoot the bounding
        # coordinates, so that no landmark cannot be encoded within a grid cell
        #
        # half a cell side length is taken off to account for the cell that's explicitly added for the origin coord to lie in
        #
        # (ie. half of the origin cell lies in each of the Cartesian quadrants)
        left_of_origin_cols = int(
            np.ceil((np.abs(min_m) - (0.5 * self.CELL_SIZE)) / self.CELL_SIZE))  # abs so this quantity is positive
        right_of_origin_cols = int(np.ceil((max_m - (0.5 * self.CELL_SIZE)) / self.CELL_SIZE))

        below_origin_rows = int(
            np.ceil((np.abs(min_n) - (0.5 * self.CELL_SIZE)) / self.CELL_SIZE))  # abs so this quantity is positive
        above_origin_rows = int(np.ceil((max_n - (0.5 * self.CELL_SIZE)) / self.CELL_SIZE))

        # take the total row and column counts as one more than the respective counts on either side of the origin, so
        # that there's explicitly a cell for the origin coord to lie in
        row_count = below_origin_rows + above_origin_rows + 1
        col_count = left_of_origin_cols + right_of_origin_cols + 1

        # initialise the matrix with zeros - ie. all cells unoccupied unless explicitly set to be occupied
        occupancy_grid = np.full((row_count, col_count), 0)

        # populate cell that epuck centre is in as free - epuck has been able to drive into it, so probably doesn't
        # contain a landmark
        epuck_x = self.state_estimate[0]
        epuck_y = self.state_estimate[1]

        # calculate what proportion of the way it falls between the distance representable in the number of columns and
        # rows determined required
        #
        # in the epuck_x_dist quantity, 0.5 is added to the number of columns left of the origin to account for the half
        # a cell side length held in the negative x quadrant by the origin cell
        #
        # in the col_pos quantity, the floor is taken as - for example - if it's 4.3/10 cols for example, want to
        # declare it as in a cell in column 4 - this also takes care of converting this from a count to an index
        x_dist = col_count * self.CELL_SIZE
        epuck_x_dist = epuck_x - (-1 * (left_of_origin_cols + 0.5) * self.CELL_SIZE)
        x_proportion = epuck_x_dist / x_dist
        col_pos = int(np.floor(
            x_proportion * col_count))

        y_dist = row_count * self.CELL_SIZE
        epuck_y_dist = epuck_y - (-1 * (below_origin_rows + 0.5) * self.CELL_SIZE)
        y_proportion = epuck_y_dist / y_dist
        row_pos = int(np.floor(y_proportion * row_count))

        # need to augment the row pos because the occupancy matrix will be later flipped vertically for intuition wrt.
        # the Cartesian plane when printed/displayed
        epuck_cell = (row_count - row_pos - 1, col_pos)

        occupancy_grid[row_pos, col_pos] = 0  # 0 for free


        # iterate through the landmarks and apply the same logic as above for the epuck, but filling in 1 for each
        # landmark instead of 0, to indicate cell not free

        # first 3 state estimate components correspond to epuck pose - 4th element is where landmarks (if any) start
        i = 3
        while i < len(self.state_estimate):
            landmark_x = self.state_estimate[i]
            landmark_y = self.state_estimate[i + 1]

            x_dist = col_count * self.CELL_SIZE
            landmark_x_dist = landmark_x - (-1 * (left_of_origin_cols + 0.5) * self.CELL_SIZE)
            x_proportion = landmark_x_dist / x_dist
            col_pos = int(np.floor(x_proportion * col_count))

            y_dist = row_count * self.CELL_SIZE
            landmark_y_dist = landmark_y - (-1 * (below_origin_rows + 0.5) * self.CELL_SIZE)
            y_proportion = landmark_y_dist / y_dist
            row_pos = int(np.floor(y_proportion * row_count))

            occupancy_grid[row_pos, col_pos] = 1

            i += 3

        # flip in the up/down direction to get negative columns at the bottom rather than at the top for when printing
        # out - less disturbing to look at
        occupancy_grid = np.flipud(occupancy_grid)

        # the origin cell row index is now given by the number of above origin rows - because it's one more than this
        # count, hence taking it without +1 is fine, and it's above origin row count rather than below because of the
        # flip done to the matrix to make it more intuitive for when printed out
        origin_cell = (above_origin_rows, left_of_origin_cols)

        return origin_cell, epuck_cell, occupancy_grid