"""High-level navigation controller for waypoint following."""

import math
import config

class NavigationController:
    """Handles turn-then-move navigation strategy with state management."""

    def __init__(self, motion_controller, heading_tolerance,
                 waypoint_tolerance, turn_speed, forward_speed,
                 stabilize_iterations=None, final_approach_distance=None):

        self.motion = motion_controller
        self.heading_tolerance = heading_tolerance
        self.waypoint_tolerance = waypoint_tolerance
        self.turn_speed = turn_speed
        self.forward_speed = forward_speed
        self.stabilize_iterations = stabilize_iterations if stabilize_iterations is not None else config.NAV_STABILIZE_ITERATIONS
        self.final_approach_distance = final_approach_distance if final_approach_distance is not None else config.NAV_FINAL_APPROACH_DISTANCE

        self.state = 'idle'
        self.stabilize_counter = 0

        # PID control parameters for heading
        self.previous_heading_error = 0
        self.heading_error_integral = 0
        self.last_update_time = None
    
    def reset(self):
        """Reset navigation state to idle."""
        self.state = 'idle'
        self.stabilize_counter = 0
        self.motion.stop()
        # Reset PID parameters
        self.previous_heading_error = 0
        self.heading_error_integral = 0
    
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

    def calculate_turn_rate_pid(self, heading_error):
        """
        PID-style heading control to prevent overshooting with damping.
        Especially important for negative turning factors.
        """
        # Get turning factor from motion controller
        turning_factor = self.motion.turning_factor

        # PID gains - adjust based on turning factor using config values
        if turning_factor < 0:
            # For negative factors, reduce P gain and increase D gain
            kp = config.NAV_PID_KP_NEGATIVE * (1 + turning_factor * config.NAV_TURNING_FACTOR_SCALE)
            kd = config.NAV_PID_KD_NEGATIVE  # Increased derivative for damping
            ki = config.NAV_PID_KI_NEGATIVE  # Small integral to eliminate steady-state error
        else:
            # Normal positive turning factor
            kp = config.NAV_PID_KP_NORMAL * (1 + turning_factor * config.NAV_TURNING_FACTOR_NORMAL_SCALE)
            kd = config.NAV_PID_KD_NORMAL
            ki = config.NAV_PID_KI_NORMAL

        # Proportional term
        p_term = heading_error * kp

        # Derivative term (damping to prevent oscillation)
        d_term = (heading_error - self.previous_heading_error) * kd

        # Integral term (with windup protection using config limit)
        self.heading_error_integral += heading_error * ki
        self.heading_error_integral = max(-config.NAV_PID_INTEGRAL_LIMIT, min(config.NAV_PID_INTEGRAL_LIMIT, self.heading_error_integral))
        i_term = self.heading_error_integral

        # Combine PID terms
        total_correction = p_term - d_term + i_term  # Note: negative D for damping

        # Store current error for next derivative calculation
        self.previous_heading_error = heading_error

        # Limit maximum correction to prevent aggressive movements
        max_correction = self.turn_speed
        if turning_factor < 0:
            # Further limit correction for negative turning factors using config
            max_correction *= config.NAV_PID_CORRECTION_LIMIT_FACTOR

        total_correction = max(-max_correction, min(max_correction, total_correction))

        return total_correction
    
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
            self.motion.forward(self.forward_speed * config.NAV_FINAL_APPROACH_SPEED_FACTOR)
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
            # Use PID control from the start
            turn_rate = self.calculate_turn_rate_pid(heading_error)

            # Ensure minimum turn rate to prevent getting stuck
            if abs(turn_rate) < config.NAV_PID_TURN_RATE_MIN:
                speed = config.NAV_PID_TURN_RATE_MIN
            else:
                speed = min(abs(turn_rate), self.turn_speed)

            if heading_error > 0:
                self.motion.rotate_left(speed)
            else:
                self.motion.rotate_right(speed)
        else:
            self.state = 'moving'
            self.motion.forward(self.forward_speed)
    
    def _handle_turning_state(self, heading_error):
        """Handle turning state - continue or complete turn with PID control."""
        if abs(heading_error) > self.heading_tolerance:
            # Use PID-controlled turning to prevent overshooting
            turn_rate = self.calculate_turn_rate_pid(heading_error)

            # Ensure minimum turn rate to prevent getting stuck
            # If PID outputs too small a value, use minimum speed in correct direction
            if abs(turn_rate) < config.NAV_PID_TURN_RATE_MIN:
                # Apply minimum turn rate in the correct direction based on heading error
                if heading_error > 0:
                    speed = config.NAV_PID_TURN_RATE_MIN
                    self.motion.rotate_left(speed)
                else:
                    speed = config.NAV_PID_TURN_RATE_MIN
                    self.motion.rotate_right(speed)
            elif turn_rate > 0:
                # Turn left with PID-controlled speed
                speed = min(abs(turn_rate), self.turn_speed)
                self.motion.rotate_left(speed)
            else:
                # Turn right with PID-controlled speed
                speed = min(abs(turn_rate), self.turn_speed)
                self.motion.rotate_right(speed)
        else:
            self.state = 'stabilizing'
            self.stabilize_counter = self.stabilize_iterations
            self.motion.stop()
            # Reset integral when we reach target
            self.heading_error_integral = 0
    
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
        """Handle moving state - continue or correct heading with smooth control."""
        moving_tolerance = self.heading_tolerance * config.NAV_MOVING_TOLERANCE_FACTOR

        if abs(heading_error) > moving_tolerance:
            # Significant heading error - switch to turning state
            self.state = 'turning'
            self.motion.stop()
        elif abs(heading_error) > self.heading_tolerance * config.NAV_MOVING_CORRECTION_FACTOR:
            # Small heading error - apply differential drive correction while moving
            # This prevents oscillation by making smooth corrections
            turning_factor = self.motion.turning_factor

            # Calculate differential speed based on error and turning factor
            base_speed = self.forward_speed
            correction = heading_error * config.NAV_MOVING_DIFFERENTIAL_FACTOR  # Gentle correction from config

            # Apply correction with consideration for negative turning factors
            if turning_factor < 0:
                # With negative turning factor, apply gentler corrections
                correction *= config.NAV_MOVING_NEGATIVE_DAMPING

            # Apply differential speeds for smooth path following
            if heading_error > 0:
                # Need to turn left - slow down left wheel
                left_speed = base_speed * (1 - abs(correction))
                right_speed = base_speed
            else:
                # Need to turn right - slow down right wheel
                left_speed = base_speed
                right_speed = base_speed * (1 - abs(correction))

            # Ensure speeds don't go negative or too low using config factor
            left_speed = max(base_speed * config.NAV_MOVING_MIN_SPEED_FACTOR, left_speed)
            right_speed = max(base_speed * config.NAV_MOVING_MIN_SPEED_FACTOR, right_speed)

            self.motion.set_target_velocities(left_speed, right_speed)
        else:
            # Heading is good - move straight
            self.motion.forward(self.forward_speed)