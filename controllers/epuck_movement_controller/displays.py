# ---------- IMPORTS ----------
import numpy as np


# ---------- CLASSES ----------
class DisplayController:
    # ---------- CONSTANTS ----------
    PIXELS_PER_METRE = 100  # ratio of pixels per metre - i.e. 1 pixels corresponds to 1cm


    # ---------- VARIABLES & DATA STRUCTURES ---------
    robot = None
    display = None
    display_width = None
    display_height = None
    display_origin_x = None
    display_origin_y = None
    occ_grid_disp = None
    occ_grid_disp_width = None
    occ_grid_disp_height = None


    # ---------- CONSTRUCTOR ----------
    def __init__(self, robot):
        self.robot = robot

        # get display (the standard one that shows the pose and landmarks - not the occupancy grid one)
        self.display = robot.getDevice('display')
        self.display_width = self.display.getWidth()
        self.display_height = self.display.getHeight()

        # visualise the epuck's x-y origin (0,0) in the centre of the display
        self.display_origin_x = self.display_width / 2
        self.display_origin_y = self.display_height / 2

        # get occupancy grid display
        self.occ_grid_disp = robot.getDevice('occupancy-grid')
        self.occ_grid_disp_width = self.occ_grid_disp.getWidth()
        self.occ_grid_disp_height = self.occ_grid_disp.getHeight()


    # ---------- METHODS ----------
    def world_coords_to_display_cords(self, x, y):
        # convert world coordinates to display coordinates
        # in the environment, x is right, y is up, whereas on the display, x is right, y is down
        display_x = int(self.display_origin_x + x * self.PIXELS_PER_METRE)
        display_y = int(self.display_origin_y - y * self.PIXELS_PER_METRE)
        # (above) subtract because of display y direction being opposite to world y direction

        return display_x, display_y

    def draw_true_pose(self, x_t):
        # x_t = [x_coord, y_coord, theta]

        # draw epuck body
        self.display.setColor(0x1026cc)
        epuck_radius = 0.037 * self.PIXELS_PER_METRE  # epuck has 7.4cm diameter => 0.037m radius

        # subtract because of display y direction being opposite to world y direction
        display_x, display_y = self.world_coords_to_display_cords(x_t[0], x_t[1])

        self.display.fillOval(display_x, display_y, epuck_radius, epuck_radius)

        # draw heading of epuck
        heading_x = display_x + (int(epuck_radius * np.cos(x_t[2])))
        heading_y = display_y - (int(epuck_radius * np.sin(x_t[2])))
        self.display.setColor(0xe01fb3)
        self.display.drawLine(int(display_x), int(display_y), int(heading_x), int(heading_y))
        return

    def draw_state_estimate(self, state_estimate):
        # state_estimate = [x_coord, y_coord, theta, landmarks...]

        # draw epuck body
        self.display.setColor(0x24fc03)
        epuck_radius = 0.037 * self.PIXELS_PER_METRE  # epuck has 7.4cm diameter => 0.037m radius

        # convert world coordinates to display coordinates
        display_x, display_y = self.world_coords_to_display_cords(state_estimate[0], state_estimate[1])

        self.display.fillOval(display_x, display_y, epuck_radius, epuck_radius)

        # draw heading of epuck
        heading_x = display_x + (int(epuck_radius * np.cos(state_estimate[2])))
        heading_y = display_y - (int(epuck_radius * np.sin(state_estimate[2])))
        self.display.setColor(0x2a4226)
        self.display.drawLine(int(display_x), int(display_y), int(heading_x), int(heading_y))

        # draw landmarks
        i = 3
        while i < len(state_estimate):
            landmark_x = state_estimate[i]
            landmark_y = state_estimate[i + 1]

            if not (np.isnan(landmark_x)) and not (np.isnan(landmark_y)):
                self.display.setColor(0xbf22bd)

                # convert world coordinates to display coordinates
                disp_x, disp_y = self.world_coords_to_display_cords(landmark_x, landmark_y)

                self.display.fillOval(disp_x, disp_y, 1, 1)

            i += 3

        return

    def temp_draw_landmarks(self):
        # temporary function - draws on display the true location of landmarks so that it can be seen
        # how well the EKF-SLAM algorithm is mapping landmarks
        # ie. coords of the landmarks actually in the environment
        landmarks = [(-0.25, 0.25, 0), (0.25, 0.25, 0), (-0.25, -0.25, 0),(0.25, -0.25, 0)]

        for x, y, signature in landmarks:
            self.display.setColor(0x3d3527)

            # convert world coordinates to display coordinates
            display_x, display_y = self.world_coords_to_display_cords(x, y)

            radius = 0.03 * self.PIXELS_PER_METRE # environment cylinder objects currently have 0.03 radius

            self.display.fillOval(display_x, display_y, radius, radius)
        return

    def draw_occupancy_grid(self, origin_cell, epuck_cell, occupancy_grid):
        # Visualises the occupancy grid on a display
        #
        # Unoccupied cells are drawn as alternating shades of grey (for visual differentiation)
        #
        # Occupied cells are drawn in red
        #
        # The cell containing the origin is drawn in pink (to make it easier to visually landmark cells in the
        # occupancy grid)
        #
        # The cell currently occupied by the centre of the epuck is drawn in green
        #
        # Throughout epuck movement, the drawn grid may appear to become distorted
        #
        # This is a consequence of the limited size of the display, and the occupancy grid growing unbalanced in either
        # its number of rows or columns
        #
        # Despite this visual behaviour, each cell in the occupancy actually represents a square area at all times
        #
        # Another visual anomaly may be white lines appearing between grid cells on the display
        #
        # This attributable to precision issues mapping points from within the occupancy grid space to the display
        # space, where the former's size is dynamic and unlimited, and the latter is static and (hence) limited

        # ------------------------------

        # take the number of rows/cols, divide display width by num of rows/cols, that's cell width/height
        row_count = np.shape(occupancy_grid)[0]
        col_count = np.shape(occupancy_grid)[1]

        cell_width = self.occ_grid_disp_width / col_count
        cell_height = self.occ_grid_disp_height / row_count


        # iterate through rows, and then through columns, to draw cells one at a time

        # first of the two alternators - two so that grey used for empty cells alternates in both row and col directions
        row_alternator = False
        for row in range(np.shape(occupancy_grid)[0]):
            col_alternator = row_alternator
            for col in range(np.shape(occupancy_grid)[1]):
                # Check cell value for smart encoding system
                cell_value = occupancy_grid[row, col]

                # Complete smart display grid color mapping:
                # 0: Empty (dark grey background)
                # 1: Obstacle (bright red)
                # 2: Origin (bright blue)
                # 3: Robot (bright green)
                # 4: Path (yellow)
                # 5: Near landmark (light red - danger zone)
                # 6: Far landmark (dark red)
                # 7: Wall (medium grey)
                # 8: Buffer zone (dark yellow - safety margin)
                # 9: Waypoint (orange)
                # 10: Current waypoint (cyan)
                # 11: Diagonal buffer (dark orange)
                # 12: Robot heading (medium green)

                if cell_value == 3:  # Robot position
                    self.occ_grid_disp.setColor(0x00ff00)  # Bright green
                    self.occ_grid_disp.fillRectangle(col * cell_width, row * cell_height, cell_width, cell_height)
                elif cell_value == 2:  # Origin
                    self.occ_grid_disp.setColor(0x0080ff)  # Bright blue
                    self.occ_grid_disp.fillRectangle(col * cell_width, row * cell_height, cell_width, cell_height)
                elif cell_value == 1:  # Obstacle
                    self.occ_grid_disp.setColor(0xff4444)  # Bright red
                    self.occ_grid_disp.fillRectangle(col * cell_width, row * cell_height, cell_width, cell_height)
                elif cell_value == 4:  # Path
                    self.occ_grid_disp.setColor(0xffff00)  # Yellow
                    self.occ_grid_disp.fillRectangle(col * cell_width, row * cell_height, cell_width, cell_height)
                elif cell_value == 5:  # Near landmark
                    self.occ_grid_disp.setColor(0xff8888)  # Light red
                    self.occ_grid_disp.fillRectangle(col * cell_width, row * cell_height, cell_width, cell_height)
                elif cell_value == 6:  # Far landmark
                    self.occ_grid_disp.setColor(0x884444)  # Dark red
                    self.occ_grid_disp.fillRectangle(col * cell_width, row * cell_height, cell_width, cell_height)
                elif cell_value == 7:  # Wall
                    self.occ_grid_disp.setColor(0x666666)  # Medium grey
                    self.occ_grid_disp.fillRectangle(col * cell_width, row * cell_height, cell_width, cell_height)
                elif cell_value == 8:  # Buffer zone
                    self.occ_grid_disp.setColor(0x4a4a00)  # Dark yellow
                    self.occ_grid_disp.fillRectangle(col * cell_width, row * cell_height, cell_width, cell_height)
                elif cell_value == 9:  # Waypoint
                    self.occ_grid_disp.setColor(0xffaa00)  # Orange
                    self.occ_grid_disp.fillRectangle(col * cell_width, row * cell_height, cell_width, cell_height)
                elif cell_value == 10:  # Current waypoint
                    self.occ_grid_disp.setColor(0x00ffff)  # Cyan
                    self.occ_grid_disp.fillRectangle(col * cell_width, row * cell_height, cell_width, cell_height)
                elif cell_value == 11:  # Diagonal buffer
                    self.occ_grid_disp.setColor(0x6a3a00)  # Dark orange
                    self.occ_grid_disp.fillRectangle(col * cell_width, row * cell_height, cell_width, cell_height)
                elif cell_value == 12:  # Robot heading
                    self.occ_grid_disp.setColor(0x00aa00)  # Medium green
                    self.occ_grid_disp.fillRectangle(col * cell_width, row * cell_height, cell_width, cell_height)
                else:  # Empty space (0 or unknown)
                    self.occ_grid_disp.setColor(0x202020)  # Dark grey background
                    self.occ_grid_disp.fillRectangle(col * cell_width, row * cell_height, cell_width, cell_height)

                col_alternator = not col_alternator  # flip the alternator before drawing the cell in the next column
            row_alternator = not row_alternator  # flip the alternator before drawing the cell in the next row

        return

    def clean_displays(self):
        # removes anything not explicitly drawn this time-step on the display
        self.display.setColor(0xffffff)  # make display background white
        self.display.fillRectangle(0, 0, self.display_width, self.display_height)

        # do the same for the occupancy grid display
        self.occ_grid_disp.setColor(0xffffff)
        self.occ_grid_disp.fillRectangle(0, 0, self.occ_grid_disp_width, self.occ_grid_disp_height)
        return