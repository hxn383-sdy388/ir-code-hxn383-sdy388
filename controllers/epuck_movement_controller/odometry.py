"""Odometry system for tracking robot pose with calibration support."""

import math


class Odometry:
    """Tracks robot position and orientation using wheel encoders."""

    def __init__(self, wheel_radius, axle_length,
                 linear_scale=1.0,
                 angular_scale=1.0,
                 wheel_left_scale=1.0,
                 wheel_right_scale=1.0):

        self.wheel_radius = wheel_radius
        self.axle_length = axle_length

        # Calibration parameters
        self.linear_scale = linear_scale
        self.angular_scale = angular_scale
        self.wheel_left_scale = wheel_left_scale
        self.wheel_right_scale = wheel_right_scale

        # Pose state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Previous encoder values
        self.prev_left = 0.0
        self.prev_right = 0.0

    def update(self, left_encoder_val, right_encoder_val):
        # Calculate encoder changes
        d_left = left_encoder_val - self.prev_left
        d_right = right_encoder_val - self.prev_right

        # Apply wheel-specific calibration and convert to linear distances
        l_distance = d_left * self.wheel_radius * self.wheel_left_scale * self.linear_scale
        r_distance = d_right * self.wheel_radius * self.wheel_right_scale * self.linear_scale

        # Calculate center displacement and orientation change
        center_distance = (l_distance + r_distance) / 2.0
        d_theta = (r_distance - l_distance) / self.axle_length * self.angular_scale

        # Update orientation using midpoint angle (more accurate for arcs)
        mid_theta = self.theta + d_theta / 2.0
        self.theta += d_theta
        self.theta = self._normalize_angle(self.theta)

        # Update position using midpoint angle
        self.x += center_distance * math.cos(mid_theta)
        self.y += center_distance * math.sin(mid_theta)

        # Store encoder values
        self.prev_left = left_encoder_val
        self.prev_right = right_encoder_val

    def get_pose(self):
        """Return current pose as (x, y, theta)."""
        return self.x, self.y, self.theta

    def reset(self, x=0.0, y=0.0, theta=0.0):
        """Reset odometry to specified pose."""
        self.x = x
        self.y = y
        self.theta = theta
        self.prev_left = 0.0
        self.prev_right = 0.0

    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))