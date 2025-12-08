# Intelligent Robotics Source Code

*Harvey Nicholson, Salmaan Yehia*

---

## About the Project

This project implements the "explore and return" problem using an E-Puck in Webots. 

The main controller is in `epuck_movement_controller/epuck_movement_controller.py`.

## Implementation Details

Salmaan implemented the autonomous navigation system comprising the motion controller (with smooth velocity ramping), waypoint navigation module (using turn-then-move strategy), collision avoidance system (IR sensor-based safety layer), and A* path planner (featuring corridor centering and adaptive resolution). Additionally, he co-developed the smart display grid for real-time visualization of the robot's state, environment mapping, and path planning.

Harvey implemented the EKF-SLAM algorithm, using Webots' built-in LiDAR device for landmark measurement readings. Throughout the development process, a turret-mounted camera with automatic landmark detection, and the Supervisor API for globally aware positional data were used. However, neither of these are used in the final algorithm implementation, being substituted for the afforementioned LiDAR measurements, and Salmaan's odometry data respectively. NumPy has been used for mathematical operations.