"""Configuration constants for the e-puck robot."""

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

# Control Modes
MODE_MANUAL = 'manual'
MODE_AUTONOMOUS = 'autonomous'