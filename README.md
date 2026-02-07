# Drone Project 

A computer vision-based drone control system that translates hand gestures into flight commands. This project features a built-in physics simulator, a geometric shape correction engine for path planning, and an AI-powered "Follow Me" mode using face tracking.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20%26%20Face%20Tracking-orange)
![Status](https://img.shields.io/badge/Status-Prototype-yellow)

## Overview

This application uses your webcam to detect hand landmarks and facial features in real-time. It processes these inputs to control a virtual drone displayed on a radar-style interface. The system includes robust logic to differentiate between similar gestures (e.g., a "Fist" vs. "Thumbs Up") and includes safety timers to prevent accidental commands.

## Key Features

### 1. Gesture Control Engine
* **Takeoff/Landing:** Safety-locked thumbs-up/down recognition.
* **Acrobatics:** Detects a "Fist" gesture to trigger a backflip animation.
* **Anti-Jitter:** Uses a history buffer (deque) to smooth out detection noise.

### 2. Geometric Shape Correction (Smart Pathing)
* **Draw-to-Fly:** Use your index finger to draw a path in the air.
* **Shape Perfecting:** The engine analyzes your drawn path using Convex Hulls and Circularity formulas.
* **Auto-Correction:** It automatically recognizes if you drew a rough Circle, Square, or Triangle and converts it into a mathematically perfect flight path for the drone to execute.

### 3. "Follow Me" Mode
* **Face Tracking:** Uses MediaPipe Face Detection to lock onto the user.
* **PID-like Logic:** Calculates the error between the face center and the frame center to issue Yaw (Rotate) and Pitch (Forward/Backward) commands, simulating a drone keeping the user in frame.

### 4. Radar Simulator
* **Visual Feedback:** A real-time 2D radar display showing the drone's relative position ($X, Y$), altitude ($Z$), and flight trail.
* **Simulated Physics:** Includes inertia, altitude clamps, and auto-landing sequences.

---
## 🕹️ Controls & Gestures

The system uses a state machine to prevent conflicting commands. Hold gestures for **1 second** to trigger safety locks.

| Action | Gesture | Description |
| :--- | :--- | :--- |
| **TAKEOFF** | 👍 **Thumbs Up** | Hold for 1s to arm motors and takeoff. |
| **LAND** | 👎 **Thumbs Down** | Initiates auto-landing sequence. |
| **BACKFLIP** | ✊ **Fist** | Hold for 1s to perform a stunt. |
| **FOLLOW ME** | ✌️ **Peace Sign** | Hold for 1s to toggle Face Tracking mode. |
| **HOVER/STOP** | 🖐 **Open Palm** | Stops movement; also used to "Finish" a drawing. |
| **DRAW PATH** | 🤟 **Spider-Man** | (Index + Pinky extended) Enters "Path Mode". |
| **PAINT** | ☝️ **Index Finger** | While in "Path Mode", moves the cursor to draw. |
| **UP** | ☝️ **Index Finger Up** | Moves the drone up in the air (Altitude +). |
| **DOWN** | 👇 **Index Finger Down** | Moves the drone downwards (Altitude -). |
---
## Installation

### Prerequisites

1. Download all the files from the google drive link below
   https://drive.google.com/drive/folders/143lcgNMuB1ER2t8CXExgNqHvA_XPZiz1?usp=sharing
   
2. Ensure you have Python installed. This project relies on the following libraries:

```bash
pip install opencv-python mediapipe numpy tensorflow
```


