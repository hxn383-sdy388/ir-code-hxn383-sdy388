"""Motion control for differential drive robot with smooth acceleration."""

class EPuckMotionController:
    
    def __init__(self, left_motor, right_motor, timestep,
                 max_acceleration=2.0,
                 max_linear_deceleration=3.0,
                 max_angular_deceleration=4.0,
                 turning_factor=0.1):

        self.left_motor = left_motor
        self.right_motor = right_motor
        self.timestep_seconds = timestep / 1000.0  # Convert to seconds
        self.max_accel = max_acceleration
        self.max_linear_decel = max_linear_deceleration
        self.max_angular_decel = max_angular_deceleration
        self.turning_factor = turning_factor
        
        # Current velocities
        self.current_left_vel = 0.0
        self.current_right_vel = 0.0
        
        # Target velocities (what we want to reach)
        self.target_left_vel = 0.0
        self.target_right_vel = 0.0
        
        # Set motors to velocity control mode
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.stop()
    
    def update(self):
        """
        Update motor velocities with acceleration limiting.
        """
        # Determine if movement is angular (rotation) or linear
        is_angular = self._is_angular_movement()

        # Smoothly approach target velocities with appropriate deceleration
        self.current_left_vel = self._ramp_velocity(
            self.current_left_vel,
            self.target_left_vel,
            is_angular
        )
        self.current_right_vel = self._ramp_velocity(
            self.current_right_vel,
            self.target_right_vel,
            is_angular
        )

        # Apply to motors
        self.left_motor.setVelocity(self.current_left_vel)
        self.right_motor.setVelocity(self.current_right_vel)
    
    def _is_angular_movement(self):
        """
        Determine if the current movement is primarily angular (rotation).
        Angular movement is when wheels move in opposite directions.
        """
        # Check if target velocities have opposite signs (indicating rotation)
        # or if one is significantly smaller than the other (sharp turning)
        if (self.target_left_vel * self.target_right_vel < 0):
            # Opposite signs - pure rotation
            return True
        elif (abs(self.target_left_vel - self.target_right_vel) >
              0.5 * max(abs(self.target_left_vel), abs(self.target_right_vel))):
            # Significant difference - sharp turning (treat as angular)
            return True
        else:
            # Similar velocities - linear movement
            return False

    def _ramp_velocity(self, current, target, is_angular=False):
        """
        Smoothly ramp velocity from current to target.
        """
        difference = target - current

        # Already at target (within threshold)
        if abs(difference) < 0.01:
            return target

        # Calculate max velocity change this timestep
        if abs(target) < abs(current):
            # Decelerating - use appropriate deceleration constant
            if is_angular:
                max_change = self.max_angular_decel * self.timestep_seconds
            else:
                max_change = self.max_linear_decel * self.timestep_seconds
        else:
            # Accelerating
            max_change = self.max_accel * self.timestep_seconds

        # Apply change (limited by max_change)
        if difference > 0:
            # Need to increase velocity
            return min(current + max_change, target)
        else:
            # Need to decrease velocity
            return max(current - max_change, target)
    
    def stop(self):
        """Gradually stop all motor movement."""
        self.target_left_vel = 0.0
        self.target_right_vel = 0.0
    
    def emergency_stop(self):
        """Immediately stop (no acceleration limiting)."""
        self.current_left_vel = 0.0
        self.current_right_vel = 0.0
        self.target_left_vel = 0.0
        self.target_right_vel = 0.0
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
    
    def forward(self, speed):
        """Set target to move forward at specified speed."""
        self.target_left_vel = speed
        self.target_right_vel = speed
    
    def backward(self, speed):
        """Set target to move backward at specified speed."""
        self.target_left_vel = -speed
        self.target_right_vel = -speed
    
    def rotate_left(self, speed):
        """Set target to rotate in place counterclockwise."""
        self.target_left_vel = -speed
        self.target_right_vel = speed
    
    def rotate_right(self, speed):
        """Set target to rotate in place clockwise."""
        self.target_left_vel = speed
        self.target_right_vel = -speed
    
    def turn_left(self, speed):
        """Set target to turn left while moving forward."""
        self.target_left_vel = speed * self.turning_factor
        self.target_right_vel = speed
    
    def turn_right(self, speed):
        """Set target to turn right while moving forward."""
        self.target_left_vel = speed
        self.target_right_vel = speed * self.turning_factor
    
    def set_target_velocities(self, left_vel, right_vel):
        """Set target velocities directly."""
        self.target_left_vel = left_vel
        self.target_right_vel = right_vel
    
    def get_current_velocities(self):
        """Return current motor velocities."""
        return self.current_left_vel, self.current_right_vel
    
    def is_stopped(self, threshold=0.01):
        """Check if robot is effectively stopped."""
        return (abs(self.current_left_vel) < threshold and 
                abs(self.current_right_vel) < threshold)