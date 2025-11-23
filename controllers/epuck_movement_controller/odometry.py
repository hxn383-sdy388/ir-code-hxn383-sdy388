"""Odometry system for tracking robot pose."""

import math

class Odometry:
    """Tracks robot position and orientation using wheel encoders."""
    
    def __init__(self, wheel_radius, axle_length):
        self.wheel_radius = wheel_radius
        self.axle_length = axle_length
        
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
        
        # Convert to linear distances
        l_distance = d_left * self.wheel_radius
        r_distance = d_right * self.wheel_radius
        
        # Calculate center displacement and orientation change
        center_distance = (l_distance + r_distance) / 2.0
        d_theta = (r_distance - l_distance) / self.axle_length
        
        # Update orientation
        self.theta += d_theta
        self.theta = self._normalize_angle(self.theta)
        
        # Update position
        self.x += center_distance * math.cos(self.theta)
        self.y += center_distance * math.sin(self.theta)
        
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