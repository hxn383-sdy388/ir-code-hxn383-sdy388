"""Motion control for differential drive robot."""

class EPuckMotionController:
    
    def __init__(self, left_motor, right_motor, turning_factor=0.1):
        self.left_motor = left_motor
        self.right_motor = right_motor
        self.turning_factor = turning_factor
        
        # Set motors to velocity control mode
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        # Initialize stopped
        self.stop()
    
    def stop(self):
        """Stop all motor movement."""
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
    
    def forward(self, speed):
        """Move forward at specified speed."""
        self.left_motor.setVelocity(speed)
        self.right_motor.setVelocity(speed)
    
    def backward(self, speed):
        """Move backward at specified speed."""
        self.left_motor.setVelocity(-speed)
        self.right_motor.setVelocity(-speed)
    
    def rotate_left(self, speed):
        """Rotate in place counterclockwise."""
        self.left_motor.setVelocity(-speed)
        self.right_motor.setVelocity(speed)
    
    def rotate_right(self, speed):
        """Rotate in place clockwise."""
        self.left_motor.setVelocity(speed)
        self.right_motor.setVelocity(-speed)
    
    def turn_left(self, speed):
        """Turn left while moving forward."""
        self.left_motor.setVelocity(speed * self.turning_factor)
        self.right_motor.setVelocity(speed)
    
    def turn_right(self, speed):
        """Turn right while moving forward."""
        self.left_motor.setVelocity(speed)
        self.right_motor.setVelocity(speed * self.turning_factor)
    
    def set_velocities(self, left_vel, right_vel):
        """Set motor velocities directly."""
        self.left_motor.setVelocity(left_vel)
        self.right_motor.setVelocity(right_vel)