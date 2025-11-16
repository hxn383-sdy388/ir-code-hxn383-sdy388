"""Configuration constants for the e-puck robot."""
import math

# Robot Physical Parameters
WHEEL_RADIUS = 0.02        # meters
AXLE_LENGTH = 0.05         # meters
ENCODER_RESOLUTION = 160   # ticks per radian

# Motor Control
MAXIMUM_SPEED = 5.0        # rad/s
NORMAL_SPEED = MAXIMUM_SPEED / 2
TURNING_FACTOR = 0.5

# Acceleration Parameters
MAX_ACCELERATION = 2.0      # rad/s² - maximum rate of velocity change
MAX_DECELERATION = 3.0      # rad/s² - can brake faster than accelerate
VELOCITY_THRESHOLD = 0.01   # rad/s - consider stopped below this

# Collision Avoidance - Detection Thresholds
OBSTACLE_THRESHOLD = 80.0    # IR sensor value indicating obstacle presence
DANGER_THRESHOLD = 150.0     # IR sensor value for danger zone (reduced speed)
CRITICAL_THRESHOLD = 300.0   # IR sensor value requiring emergency stop

# Collision Avoidance - Response Parameters
DANGER_ZONE_SPEED_FACTOR = 0.3      # Speed multiplier when in danger zone
OBSTACLE_SPEED_REDUCTION = 0.7      # Maximum speed reduction for obstacles
APPROACH_ANGLE_THRESHOLD = math.pi / 2  # Radians - consider "moving towards" obstacle
MIN_VELOCITY_THRESHOLD = 0.01           # rad/s - threshold for rotation detection

# E-puck Sensor Configuration
SENSOR_POSITIONS = [
    {'angle': math.radians(-15), 'name': 'ps0'},   # Front-right
    {'angle': math.radians(-45), 'name': 'ps1'},   # Right-front
    {'angle': math.radians(-90), 'name': 'ps2'},   # Right
    {'angle': math.radians(-150), 'name': 'ps3'},  # Right-rear
    {'angle': math.radians(150), 'name': 'ps4'},   # Left-rear
    {'angle': math.radians(90), 'name': 'ps5'},    # Left
    {'angle': math.radians(45), 'name': 'ps6'},    # Left-front
    {'angle': math.radians(15), 'name': 'ps7'},    # Front-left
]

# Sensor groupings for directional path checking
SENSOR_GROUPS = {
    'forward': [0, 7],      # ps0, ps7
    'right': [1, 2],        # ps1, ps2
    'backward': [3, 4],     # ps3, ps4
    'left': [5, 6]          # ps5, ps6
}

# Control Modes
MODE_MANUAL = 'manual'
MODE_AUTONOMOUS = 'autonomous'