# ---------- IMPORTS ----------
import numpy as np


# ---------- CLASSES ----------
class MeasurementController:
    # ---------- CONSTANTS ----------
    TIMESTEP = 0


    # ---------- VARIABLES & DATA STRUCTURES ---------
    robot = None
    camera = None
    lidar = None
    fov = None
    num_of_beams = None


    # ---------- CONSTRUCTOR ----------
    def __init__(self, robot, timestep):
        self.robot = robot
        self.TIMESTEP = timestep

        # get the camera and enable the recognition node
        self.camera = self.robot.getDevice('recognitioncamera')
        self.camera.enable(timestep)
        self.camera.recognitionEnable(timestep)

        # get the lidar
        self.lidar = self.robot.getLidar('lidar')
        self.lidar.enable(timestep)

        # get fov and number of beams
        self.fov = self.lidar.getFov()
        self.num_of_beams = self.lidar.getHorizontalResolution()


    # ---------- METHODS ----------
    def cam_recog_measure_landmarks(self):
        # using the camera's recognition feature, measure the landmark positions
        z = []  # measurements

        recognised_objects = self.camera.getRecognitionObjects()
        for object in recognised_objects:
            rel_x, rel_y, rel_z = object.getPosition()  # position is relative to the epuck

            distance = np.hypot(float(rel_y),
                                float(rel_x))  # calculate straight line distance between epuck and landmark

            alpha = np.arctan2(rel_y, rel_x)  # calculate the relative bearing between the epuck and landmark
            alpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # bound the bearing to be between -pi and +pi

            z.append((float(distance), float(alpha), 0))

        return z

    def measure_hardcoded_landmarks(self, x_t):
        z = []  # measurements
        landmarks = [(0.25, 0.25, 0), (-0.25, 0.25, 0), (-0.25, -0.25, 0), (0.25, -0.25, 0)]

        # consider each landmark
        for landmark in landmarks:
            # get distance between epuck (ie. current pose) and landmark
            distance = np.sqrt(np.square((landmark[0] - x_t[0])) + np.square((landmark[1] - x_t[1])))

            # in this case, opposite is delta y, and adjacent is delta x
            alpha = np.arctan2(landmark[1] - x_t[1], landmark[0] - x_t[0]) - x_t[2]
            alpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # bound alpha to be between -pi and +pi

            z.append((distance, alpha, 0))

        return z

    def lidar_measure_landmarks(self):
        z = []  # measurements

        # array of distance measurements
        range_image = self.lidar.getRangeImage()

        # Calculate the angle increment between consecutive beams
        # The FOV spans from the first beam to the last beam
        if self.num_of_beams > 1:
            angle_increment = self.fov / (self.num_of_beams - 1)
        else:
            angle_increment = 0

        # Starting angle (leftmost beam is at -fov/2)
        start_angle = -self.fov / 2

        for i, measurement in enumerate(range_image):
            # Calculate angle for this beam
            alpha = start_angle + (i * angle_increment)
            alpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # bound the angle

            # skip beams that aren't measuring anything
            if measurement != np.inf:
                z.append((float(measurement), float(alpha), 0))

        return z