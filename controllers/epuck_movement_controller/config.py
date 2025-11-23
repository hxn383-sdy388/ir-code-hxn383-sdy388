"""Configuration constants for the e-puck robot."""
import math

# Robot Physical Parameters
WHEEL_RADIUS = 0.02        # meters
AXLE_LENGTH = 0.058        # meters - CALIBRATED
ENCODER_RESOLUTION = 160   # ticks per radian

# Motor Control
MAXIMUM_SPEED = 5.0        # rad/s
NORMAL_SPEED = MAXIMUM_SPEED / 2
TURNING_FACTOR = -1

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

# Path Planning - Occupancy Grid Parameters
GRID_CELL_SIZE = 0.05       # meters - size of each grid cell (5cm resolution)
GRID_ORIGIN_MARKER = '/'    # Marker in occupancy grid to indicate origin cell
GRID_SAFETY_BUFFER = 1      # cells - safety buffer distance from obstacles (10cm at 5cm/cell)
GRID_DIAGONAL_RESTRICTION_BUFFER = 1  # cells - restrict diagonals near obstacles
PLANNING_GRID_SIZE = 60     # cells - 60x60 grid for path planning (3m x 3m at 5cm/cell)
DISPLAY_GRID_SIZE = 20      # cells - 20x20 grid for display visualization

# Path Planning - A* Algorithm Parameters
ASTAR_DIAGONAL_COST = 1.414  # Cost for diagonal movement (sqrt(2))
ASTAR_STRAIGHT_COST = 1.0    # Cost for straight movement
ASTAR_ALLOW_DIAGONAL = True  # Allow diagonal movement in path planning
ASTAR_GOAL_TOLERANCE = 0.15  # meters - distance to goal considered "reached"
ASTAR_AGGRESSIVE_SMOOTHING = True  # Use aggressive path smoothing

# Navigation Parameters
HEADING_TOLERANCE = 0.08     # radians (~4.6 degrees) - heading alignment tolerance
WAYPOINT_DISTANCE_TOLERANCE = 0.02  # meters - distance to waypoint considered "reached" (2cm)
NAVIGATION_TURN_SPEED = NORMAL_SPEED  # Speed for turning during navigation
NAVIGATION_FORWARD_SPEED = NORMAL_SPEED  # Speed for forward movement during navigation

# Control Modes
MODE_MANUAL = 'manual'
MODE_AUTONOMOUS = 'autonomous'

# Smart Display Grid - Zoom Levels
DISPLAY_ZOOM_DETAIL = 0.025    # meters/cell - 2.5cm - High detail for nearby objects
DISPLAY_ZOOM_NORMAL = 0.05     # meters/cell - 5cm - Standard view
DISPLAY_ZOOM_WIDE = 0.10       # meters/cell - 10cm - Medium range view
DISPLAY_ZOOM_OVERVIEW = 0.15   # meters/cell - 15cm - Maximum overview

# Smart Display Grid - Distance Thresholds
DISPLAY_LANDMARK_NEAR_DIST = 0.3    # meters - landmarks closer than this are "near"
DISPLAY_LANDMARK_MID_DIST = 1.0     # meters - landmarks between near and this are obstacles
DISPLAY_MIN_RANGE = 0.5             # meters - minimum display range
DISPLAY_ZOOM_NEAR_THRESHOLD = 0.3   # meters - distance for detail zoom
DISPLAY_ZOOM_NORMAL_THRESHOLD = 0.5 # meters - distance for normal zoom
DISPLAY_ZOOM_WIDE_THRESHOLD = 1.5   # meters - distance for wide zoom
DISPLAY_CENTER_WEIGHT = 0.7         # weight for robot vs origin centering (0.7 = 70% robot)

# Navigation PID Control - Normal Turning Factors
NAV_PID_KP_NORMAL = 1.0       # Proportional gain for positive turning factors
NAV_PID_KD_NORMAL = 0.2       # Derivative gain for positive turning factors
NAV_PID_KI_NORMAL = 0.01      # Integral gain for positive turning factors

# Navigation PID Control - Negative Turning Factors
NAV_PID_KP_NEGATIVE = 0.6     # Proportional gain for negative turning factors (reduced)
NAV_PID_KD_NEGATIVE = 0.4     # Derivative gain for negative turning factors (increased damping)
NAV_PID_KI_NEGATIVE = 0.005   # Integral gain for negative turning factors (reduced)

# Navigation PID Control - Limits
NAV_PID_INTEGRAL_LIMIT = 0.5  # Maximum integral accumulation
NAV_PID_TURN_RATE_MIN = 0.1   # Minimum turn rate to apply
NAV_PID_CORRECTION_LIMIT_FACTOR = 0.7  # Factor to limit corrections for negative turning
NAV_TURNING_FACTOR_SCALE = 0.3  # Scale factor for turning factor influence on Kp
NAV_TURNING_FACTOR_NORMAL_SCALE = 0.7  # Scale for normal turning factor influence

# Navigation Movement Control
NAV_MOVING_TOLERANCE_FACTOR = 10    # Factor for heading tolerance while moving
NAV_MOVING_CORRECTION_FACTOR = 3    # Factor for applying corrections while moving
NAV_MOVING_DIFFERENTIAL_FACTOR = 0.3  # Gentle correction factor for differential drive
NAV_MOVING_NEGATIVE_DAMPING = 0.5   # Extra damping for negative turning factors
NAV_MOVING_MIN_SPEED_FACTOR = 0.3   # Minimum speed as factor of base speed
NAV_FINAL_APPROACH_SPEED_FACTOR = 0.5  # Speed factor for final approach to waypoint

# Navigation State Machine
NAV_STABILIZE_ITERATIONS = 5        # Number of iterations to stabilize after turning
NAV_FINAL_APPROACH_DISTANCE = 0.03  # meters - distance for final approach mode

# Wall Detection Parameters
WALL_MAX_DISTANCE = 0.5      # meters - maximum distance to consider as wall
WALL_MIN_ANGLE_DIFF = 5.0     # degrees - minimum angle difference between wall measurements
WALL_ANGLE_TOLERANCE = 15.0   # degrees - tolerance for consecutive wall measurements

# Path Planning Limits
PATH_PLANNING_MAX_RANGE = 1.5  # meters - maximum planning range from robot (half of 3m grid)