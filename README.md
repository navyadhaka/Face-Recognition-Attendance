# Face-Recognition-Attendance
This project is an AI-powered attendance system that automates the process of marking attendance using facial recognition. It integrates YOLO for real-time face detection and DeepFace (with ArcFace) for identity verification, ensuring fast and accurate recognition.

## Features

- **Automated Attendance**: No manual marking required.
- **Real Time Face Detection** using YOLO.
- **Identity Verification** using DeepFace(ArcFace).
- **Image Pre-processing** for improved accuracy (resizing, resolution enhancement).

## Requirements
Install the required Python dependencies:
```sh
pip install deepface opencv-python pillow numpy torch torchvision
```

## Technologies Used
- **YOLO**: Object detection.
- **DeepFace (ArcFace)**: Face recognition.
- **OpenCV**: Image processing.
- **NumPy**: Array handling.
