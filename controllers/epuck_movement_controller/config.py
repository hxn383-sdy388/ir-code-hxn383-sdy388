"""Configuration constants for the e-puck robot."""

# Robot Physical Parameters
WHEEL_RADIUS = 0.02        # meters
AXLE_LENGTH = 0.05         # meters
ENCODER_RESOLUTION = 160   # ticks per radian

# Motor Control
MAXIMUM_SPEED = 5.0        # rad/s
NORMAL_SPEED = MAXIMUM_SPEED / 2
TURNING_FACTOR = 0.1

# Control Modes
MODE_MANUAL = 'manual'
MODE_AUTONOMOUS = 'autonomous'