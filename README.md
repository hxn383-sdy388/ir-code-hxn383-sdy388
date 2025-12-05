# Intelligent Robotics Source Code

*Harvey Nicholson, Salmaan Yehia*

---

## About the Project

This project implements the "explore and return" problem using an E-Puck in Webots. 

## Implementation Details

Salmaan implemented ... using ...

Harvey implemented the EKF-SLAM algorithm, using Webots' built-in LiDAR device for landmark measurement readings. Throughout the development process, a turret-mounted camera with automatic landmark detection, and the Supervisor API for globally aware positional data were used. However, neither of these are used in the final algorithm implementation, being substituted for the afforementioned LiDAR measurements, and Salmaan's odometry data respectively. NumPy has been used for mathematical operations.