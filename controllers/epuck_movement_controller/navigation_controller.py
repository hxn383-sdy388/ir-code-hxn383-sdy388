"""High-level navigation controller for waypoint following."""

import math

class NavigationController:
    """Handles turn-then-move navigation strategy with state management."""
    
    def __init__(self, motion_controller, heading_tolerance, 
                 waypoint_tolerance, turn_speed, forward_speed,
                 stabilize_iterations=5, final_approach_distance=0.03):

        self.motion = motion_controller
        self.heading_tolerance = heading_tolerance
        self.waypoint_tolerance = waypoint_tolerance
        self.turn_speed = turn_speed
        self.forward_speed = forward_speed
        self.stabilize_iterations = stabilize_iterations
        self.final_approach_distance = final_approach_distance
        
        self.state = 'idle'
        self.stabilize_counter = 0
    
    def reset(self):
        """Reset navigation state to idle."""
        self.state = 'idle'
        self.stabilize_counter = 0
        self.motion.stop()
    
    def get_state(self):
        """Get current navigation state."""
        return self.state
    
    def get_stabilize_counter(self):
        """Get stabilization counter for status display."""
        return self.stabilize_counter
    
    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))
    
    def calculate_heading_error(self, current_theta, target_theta):
        """Calculate heading error normalized to [-pi, pi]."""
        error = target_theta - current_theta
        return self.normalize_angle(error)
    
    def navigate_to_waypoint(self, current_x, current_y, current_theta, 
                            target_x, target_y, is_final_waypoint=False):
        """Navigate to waypoint using turn-then-move strategy."""
        dx = target_x - current_x
        dy = target_y - current_y
        distance = math.sqrt(dx*dx + dy*dy)
        required_heading = math.atan2(dy, dx)
        heading_error = self.calculate_heading_error(current_theta, required_heading)
        
        if distance < self.waypoint_tolerance:
            self.reset()
            return False
        
        if is_final_waypoint and distance < self.final_approach_distance:
            self.state = 'moving'
            self.motion.forward(self.forward_speed * 0.5)
            return True
        
        if self.state == 'idle':
            self._handle_idle_state(heading_error)
        
        elif self.state == 'turning':
            self._handle_turning_state(heading_error)
        
        elif self.state == 'stabilizing':
            self._handle_stabilizing_state(heading_error)
        
        elif self.state == 'moving':
            self._handle_moving_state(heading_error)
        
        return True
    
    def _handle_idle_state(self, heading_error):
        """Handle idle state - determine initial action."""
        if abs(heading_error) > self.heading_tolerance:
            self.state = 'turning'
            if heading_error > 0:
                self.motion.turn_left(self.turn_speed)
            else:
                self.motion.turn_right(self.turn_speed)
        else:
            self.state = 'moving'
            self.motion.forward(self.forward_speed)
    
    def _handle_turning_state(self, heading_error):
        """Handle turning state - continue or complete turn."""
        if abs(heading_error) > self.heading_tolerance:
            if heading_error > 0:
                self.motion.turn_left(self.turn_speed)
            else:
                self.motion.turn_right(self.turn_speed)
        else:
            self.state = 'stabilizing'
            self.stabilize_counter = self.stabilize_iterations
            self.motion.stop()
    
    def _handle_stabilizing_state(self, heading_error):
        """Handle stabilizing state - brief pause after turning."""
        self.motion.stop()
        self.stabilize_counter -= 1
        
        if self.stabilize_counter <= 0:
            if abs(heading_error) > self.heading_tolerance * 2:
                self.state = 'turning'
            else:
                self.state = 'moving'
                self.motion.forward(self.forward_speed)
    
    def _handle_moving_state(self, heading_error):
        """Handle moving state - continue or correct heading."""
        moving_tolerance = self.heading_tolerance * 10
        
        if abs(heading_error) > moving_tolerance:
            self.state = 'turning'
            self.motion.stop()
        else:
            self.motion.forward(self.forward_speed)