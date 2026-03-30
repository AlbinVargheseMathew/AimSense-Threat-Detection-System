# AimSense – AI Assisted Gun Turret with Human-in-the-Loop Fire Control

## Introduction

AimSense is an AI-based surveillance and targeting system designed to detect, track, and assist in handling potential threats in real time.

The system uses deep learning and computer vision to identify weapons and humans from a live camera feed. However, it does not operate autonomously — a human operator is always required to approve any action.

This ensures both **high speed (AI)** and **ethical control (Human)** in the system.

---

## Problem Statement

Modern defense systems require fast decision-making, but fully autonomous systems can be unsafe.

The challenge is to:
- Use AI for fast detection
- Keep humans in control of critical actions

AimSense solves this by using a **Human-in-the-Loop (HIL)** approach. :contentReference[oaicite:0]{index=0}

---

## Features

- Real-time object detection using YOLOv8  
- Detection of weapons like guns, grenades, explosives  
- Automatic target tracking using servo motors  
- Human approval required before firing  
- Works in both real-world setup and Unreal Engine simulation  

---

## System Workflow

1. Camera captures live video  
2. YOLO model detects objects (person + weapon)  
3. System checks interaction using pose estimation  
4. If threat is confirmed → system generates a shoot request  
5. Human operator approves or rejects  
6. Action is executed based on decision  

This multi-stage detection reduces false positives and improves accuracy. :contentReference[oaicite:1]{index=1}

---

## Technologies Used

- Python  
- OpenCV  
- YOLOv8 (Ultralytics)  
- PyQt5  
- Arduino  
- PySerial  
- Unreal Engine  

---

## Model Files

This project uses two YOLOv8 models:

- `best.pt` → Main object detection model  
  - Detects: person, gun, grenade, explosive  
  - Used for primary threat detection  

- `yolov8n-pose.pt` → Pose estimation model  
  - Detects human keypoints (wrist, shoulder, etc.)  
  - Used to verify if a weapon is being held  

 
## Results

- Real-time detection and tracking achieved  
- Accurate weapon detection  
- Reduced false positives using dual-model approach  
- Safe operation ensured through human control  

---

## Applications

- Defense systems  
- Surveillance  
- Border security  
- High-security areas  

---

## Future Work

- Improve detection accuracy  
- Add advanced tracking methods  
- Deploy on embedded systems  


 ## References

- YOLOv8 (Ultralytics)
- OpenCV Documentation
