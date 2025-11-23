"""Collision avoidance system using IR proximity sensors."""

import math

class CollisionAvoidance:
    """
    Monitors proximity sensors and provides safe velocity commands.
    Takes absolute precedence over all other movement commands.
    """
    
    def __init__(self, robot, timestep, 
                 sensor_positions,
                 sensor_groups,
                 obstacle_threshold=80.0,
                 danger_threshold=150.0,
                 critical_threshold=300.0,
                 danger_speed_factor=0.3,
                 obstacle_speed_reduction=0.7,
                 approach_angle_threshold=math.pi/2,
                 min_velocity_threshold=0.01):

        self.robot = robot
        self.timestep = timestep
        
        # Store sensor configuration
        self.sensor_positions = sensor_positions
        self.sensor_groups = sensor_groups
        
        # Detection thresholds
        self.obstacle_threshold = obstacle_threshold
        self.danger_threshold = danger_threshold
        self.critical_threshold = critical_threshold
        
        # Response parameters
        self.danger_speed_factor = danger_speed_factor
        self.obstacle_speed_reduction = obstacle_speed_reduction
        self.approach_angle_threshold = approach_angle_threshold
        self.min_velocity_threshold = min_velocity_threshold
        
        # Initialize proximity sensors
        self.sensors = []
        for sensor_info in self.sensor_positions:
            sensor = robot.getDevice(sensor_info['name'])
            sensor.enable(timestep)
            self.sensors.append({
                'device': sensor,
                'angle': sensor_info['angle'],
                'name': sensor_info['name']
            })
        
        # State tracking
        self.sensor_values = [0.0] * len(self.sensors)
        self.is_obstacle_detected = False
        self.is_danger_zone = False
        self.is_critical = False
        
    def update(self):
        """
        Update sensor readings and obstacle detection state.
        """
        self.sensor_values = [s['device'].getValue() for s in self.sensors]
        
        # Determine detection states
        max_reading = max(self.sensor_values)
        self.is_critical = max_reading > self.critical_threshold
        self.is_danger_zone = max_reading > self.danger_threshold
        self.is_obstacle_detected = max_reading > self.obstacle_threshold
    
    def get_safe_velocities(self, desired_left_vel, desired_right_vel):
        """
        Filter desired velocities to ensure collision avoidance.
        """
        
        # Emergency stop if critical obstacle
        if self.is_critical:
            return 0.0, 0.0
        
        # No obstacles - allow full speed
        if not self.is_obstacle_detected:
            return desired_left_vel, desired_right_vel
        
        # Analyze obstacle location
        obstacle_direction = self._get_obstacle_direction()
        movement_direction = self._get_movement_direction(desired_left_vel, desired_right_vel)
        
        # Check if moving towards obstacle
        if self._is_moving_towards_obstacle(movement_direction, obstacle_direction):
            # In danger zone - stop or reverse only
            if self.is_danger_zone:
                return self._handle_danger_zone(desired_left_vel, desired_right_vel, obstacle_direction)
            else:
                # Obstacle detected but not critical - reduce speed
                return self._reduce_speed(desired_left_vel, desired_right_vel)
        
        # Moving away from obstacle or turning - allow movement
        return desired_left_vel, desired_right_vel
    
    def _get_obstacle_direction(self):
        """
        Calculate weighted average direction of detected obstacles.
        """
        x_sum = 0.0
        y_sum = 0.0
        total_weight = 0.0
        
        for i, value in enumerate(self.sensor_values):
            if value > self.obstacle_threshold:
                weight = value - self.obstacle_threshold
                angle = self.sensors[i]['angle']
                x_sum += weight * math.cos(angle)
                y_sum += weight * math.sin(angle)
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return math.atan2(y_sum, x_sum)
    
    def _get_movement_direction(self, left_vel, right_vel):
        """
        Determine direction of intended movement.
        """
        avg_vel = (left_vel + right_vel) / 2.0
        
        if abs(avg_vel) < self.min_velocity_threshold:
            # Rotating in place - no linear movement
            return None
        
        # Forward is 0 radians, backward is pi radians
        return 0.0 if avg_vel > 0 else math.pi
    
    def _is_moving_towards_obstacle(self, movement_dir, obstacle_dir):
        """
        Check if movement direction is towards the obstacle.
        """
        if movement_dir is None:
            # Rotating in place - generally safe
            return False
        
        # Calculate angle difference
        angle_diff = abs(self._normalize_angle(movement_dir - obstacle_dir))
        
        # If angle difference < threshold, moving towards obstacle
        return angle_diff < self.approach_angle_threshold
    
    def _handle_danger_zone(self, desired_left, desired_right, obstacle_dir):
        """
        Handle movement when in danger zone - only allow backwards or lateral movement.
        """
        avg_vel = (desired_left + desired_right) / 2.0
        
        # Only allow backward movement or stopping
        if avg_vel <= 0:
            # Moving backward or stopping - allow with reduced speed
            return (desired_left * self.danger_speed_factor, 
                    desired_right * self.danger_speed_factor)
        else:
            # Trying to move forward in danger zone - stop
            return 0.0, 0.0
    
    def _reduce_speed(self, left_vel, right_vel):
        """
        Reduce speed proportionally based on closest obstacle distance.
        """
        max_reading = max(self.sensor_values)
        
        # Calculate reduction factor (closer = more reduction)
        if max_reading > self.danger_threshold:
            factor = self.danger_speed_factor
        elif max_reading > self.obstacle_threshold:
            # Linear interpolation between thresholds
            range_size = self.danger_threshold - self.obstacle_threshold
            distance_into_range = max_reading - self.obstacle_threshold
            factor = 1.0 - (self.obstacle_speed_reduction * distance_into_range / range_size)
        else:
            factor = 1.0
        
        return left_vel * factor, right_vel * factor
    
    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))
    
    def get_sensor_readings(self):
        """Return current sensor values for debugging/visualization."""
        return self.sensor_values.copy()
    
    def get_status(self):
        """
        Get current collision avoidance status.
        """
        return {
            'obstacle_detected': self.is_obstacle_detected,
            'danger_zone': self.is_danger_zone,
            'critical': self.is_critical,
            'max_sensor_value': max(self.sensor_values),
            'obstacle_direction': self._get_obstacle_direction() if self.is_obstacle_detected else None
        }
    
    def is_path_clear(self, direction='forward'):
        """
        Check if path is clear in specified direction.
        """
        if direction not in self.sensor_groups:
            return False
        
        indices = self.sensor_groups[direction]
        return all(self.sensor_values[i] < self.obstacle_threshold for i in indices)