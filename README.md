# Fitness AI

AI-powered fitness application that uses real-time human pose estimation and machine learning to recognize exercises, count repetitions, and analyze exercise form.

The project is designed as a complete AI + mobile + backend + IoT system. The AI pipeline is developed in Python and will later be integrated into a React Native mobile application.

---

## Project Objective

The objective is to develop a fitness application capable of understanding physical exercises through a camera.

The planned system will:

* Detect the user's body position in real time.
* Extract human body landmarks.
* Recognize exercises such as push-ups and squats.
* Count repetitions automatically.
* Evaluate exercise execution and form.
* Track physical activity such as steps.
* Connect the AI system to a mobile application.
* Use MQTT for future IoT communication.
* Provide a progression/level system inspired by fitness games.

Example:

```text
User performs 20 push-ups
        ↓
Camera
        ↓
YOLO26-Pose
        ↓
Body landmarks
        ↓
Our exercise recognition model
        ↓
Push-up detected
        ↓
Repetition counter
        ↓
20 repetitions completed
        ↓
Level progression
```

---

# Current AI Architecture

The AI system is divided into several stages:

```text
Camera / Video
       ↓
YOLO26-Pose
       ↓
Human Body Keypoints
       ↓
Data Preprocessing
       ↓
Temporal Sequence
       ↓
Our Exercise Classification Model
       ↓
Exercise Recognition
       ↓
Repetition Counting
       ↓
Form Analysis
```

The pose estimation model and the exercise recognition model have different responsibilities.

### Pose estimation

A pretrained YOLO26-Pose model detects human body keypoints.

### Exercise recognition

Our own machine learning model will learn to recognize exercises from sequences of these keypoints.

This separation allows the project to use a strong pretrained pose detector while training a specialized model specifically for fitness exercises.

---

# YOLO26-Pose

The project currently uses **YOLO26-Pose** from Ultralytics.

YOLO is a family of real-time computer vision models.

The pose version is designed to detect human body keypoints instead of only detecting objects.

For a person, the model provides approximately 17 body keypoints, including:

* Nose
* Eyes
* Ears
* Shoulders
* Elbows
* Wrists
* Hips
* Knees
* Ankles

These points provide the information required by the next stage of the AI pipeline.

We are **not training YOLO26-Pose from scratch**.

Instead, we integrate the pretrained pose model and use its output as input for our own exercise-recognition model.

The YOLO model therefore acts as the visual perception component of the system.

---

# Why Pose Landmarks?

Using raw camera images directly for exercise recognition would require a much larger and more complex model.

Instead, the system first converts the image into body coordinates.

For example:

```text
Camera image
     ↓
Person detected
     ↓
17 body keypoints
     ↓
(x, y) coordinates
     ↓
34 numerical values
```

For every frame:

```text
17 keypoints × 2 coordinates = 34 values
```

These values can then be used to train a machine learning model.

---

# Technology Stack

## AI

* Python 3.10
* PyTorch
* Ultralytics YOLO26-Pose
* NumPy
* OpenCV
* scikit-learn
* TensorFlow/Keras for future experiments if required

## Backend

Planned:

* Node.js
* Express.js
* MySQL
* REST APIs

## Mobile

Planned:

* React Native
* Expo

## IoT

Planned:

* MQTT
* Sensors / IoT devices
* MQTT broker

---

# Development Environment

Current development environment:

```text
Operating System:
Windows 10

Python:
3.10.11

GPU:
NVIDIA GeForce RTX 4050 Laptop GPU

GPU VRAM:
6 GB

CUDA:
12.8

PyTorch:
2.11.0+cu128

Ultralytics:
8.4.171
```

The AI development is currently performed locally using the NVIDIA GPU.

---

# Project Structure

Current structure:

```text
fitness-ai/
│
├── README.md
├── .gitignore
├── requirements-pose.txt
│
├── scripts/
│   ├── pose_test.py
│   ├── yolo_pose_test.py
│   └── collect_data.py
│
├── dataset/
│   ├── pushup/
│   └── squat/
│
└── models/
```

The `dataset` and `models` directories will contain generated training data and trained models during later development.

Large generated files are excluded from GitHub using `.gitignore`.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/Seddik-Ben-Hamouda/fitness-ai.git
cd fitness-ai
```

---

## 2. Create the Python environment

The pose/YOLO environment uses Python 3.10.

On Windows:

```powershell
py -3.10 -m venv venv-pose
```

Activate it:

```powershell
venv-pose\Scripts\activate
```

If using the Thonny Python installation:

```powershell
"C:\Program Files (x86)\Thonny\python.exe" -m venv venv-pose
```

Then:

```powershell
venv-pose\Scripts\activate
```

---

# Install AI Dependencies

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the required packages:

```powershell
pip install -r requirements-pose.txt
```

---

# PyTorch CUDA Installation

The project uses the CUDA-enabled PyTorch build so that inference can run on the NVIDIA GPU.

If the standard requirements installation installs a CPU version of PyTorch, install the CUDA build with:

```powershell
pip uninstall torch torchvision -y
```

Then:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected result:

```text
PyTorch: 2.11.0+cu128
CUDA available: True
CUDA: 12.8
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

---

# Test YOLO Pose Detection

Run:

```powershell
python scripts\yolo_pose_test.py
```

The first execution may automatically download the YOLO26-Pose model weights.

The webcam should open and display the detected human body keypoints.

Press:

```text
Q
```

to close the application.

---

# Dataset Collection

The next stage is to create our own exercise dataset.

The project currently supports:

```text
pushup
squat
```

Run:

```powershell
python scripts\collect_data.py
```

The program asks:

```text
Enter exercise (pushup/squat):
```

Enter:

```text
pushup
```

or:

```text
squat
```

The webcam then captures body landmarks.

Each frame contains:

```text
17 keypoints × 2 coordinates = 34 values
```

The coordinates are normalized relative to the image dimensions.

A sequence contains:

```text
30 frames
```

Therefore each training sample has approximately:

```text
30 frames × 34 values
```

When the sequence is complete, press:

```text
S
```

to save it.

Press:

```text
Q
```

to quit.

---

# Dataset Structure

Training samples will be organized as:

```text
dataset/
│
├── pushup/
│   ├── pushup_0.npy
│   ├── pushup_1.npy
│   ├── pushup_2.npy
│   └── ...
│
└── squat/
    ├── squat_0.npy
    ├── squat_1.npy
    ├── squat_2.npy
    └── ...
```

Each `.npy` file represents one temporal sequence.

---

# Exercise Recognition Model

After collecting enough data, the next AI stage is to train our own model.

The planne
