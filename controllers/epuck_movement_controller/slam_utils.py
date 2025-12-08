import config


def calculate_control_vector(motion_controller):

    left_vel, right_vel = motion_controller.get_current_velocities()

    # Convert wheel velocities (rad/s) to robot velocities
    # Linear velocity: average of wheel velocities * wheel radius
    linear_velocity = (left_vel + right_vel) * config.WHEEL_RADIUS / 2.0

    # Angular velocity: difference in wheel velocities * wheel radius / axle length
    angular_velocity = (right_vel - left_vel) * config.WHEEL_RADIUS / config.AXLE_LENGTH

    return [linear_velocity, angular_velocity]


def update_pose_from_odometry(odometry, x_t):
    current_x, current_y, current_theta = odometry.get_pose()
    x_t[0] = current_x
    x_t[1] = current_y
    x_t[2] = current_theta