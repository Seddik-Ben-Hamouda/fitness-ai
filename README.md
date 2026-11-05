# Fitness AI

A local Python prototype that uses YOLO pose estimation to count squat and
push-up repetitions from a webcam or saved video. The longer-term app flow is
for the user to choose an exercise; the current prototypes are for testing the
computer-vision counters.

## Current status

- The squat counter tracks knee angle; the push-up counter tracks elbow angle.
- A repetition is counted after the movement reaches its lower-angle threshold
  and returns to its standing/top threshold.
- Both counters display the detected pose, angle, count, and basic feedback.
- These are range-of-motion experiments, not complete form or safety checks.
- An experimental LSTM classifier was also trained to distinguish squat from
  push-up. It is separate from the counters and is not required to run them.

## Setup

Use Python 3.10 and the `venv-pose` environment on Windows:

```powershell
py -3.10 -m venv venv-pose
venv-pose\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-pose.txt
```

The project runs on CPU or CUDA. If PyTorch was installed without CUDA support,
install the CUDA-enabled PyTorch build appropriate for your system using the
official PyTorch instructions. Check the installation with:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The first YOLO run needs the `yolo26m-pose.pt` model weights in the project
folder. Keep these weights local; they are excluded from Git.

## Run the counters

Start the squat counter with your webcam:

```powershell
venv-pose\Scripts\python.exe scripts\squat_counter.py --source camera
```

Start the push-up counter with your webcam:

```powershell
venv-pose\Scripts\python.exe scripts\pushup_counter.py --source camera
```

For either counter, replace `camera` with a video path to process a saved video:

```powershell
venv-pose\Scripts\python.exe scripts\squat_counter.py --source "C:\path\to\squat.mp4"
venv-pose\Scripts\python.exe scripts\pushup_counter.py --source "C:\path\to\pushup.mp4"
```

The video is displayed with the pose overlay, movement angle, status, and
repetition count. The final count remains visible for five seconds after the
video ends. Press **Q** in the video window or terminal, **Esc** in the window,
or **Ctrl+C** in the terminal to stop. For best results, use a side view and
keep the relevant body joints visible.

### Publish repetitions to EMQX Cloud

Both counters can optionally publish each completed repetition to an EMQX
Serverless broker over TLS. MQTT is disabled unless `--mqtt` is supplied.

1. Install the added MQTT dependencies with `pip install -r requirements-pose.txt`.
2. Copy `.env.example` to `.env` in the project root and fill in the broker
   address, MQTT username, and MQTT password from EMQX. Do not use the EMQX API
   key here.
3. Download the deployment CA certificate and save it as `emqxsl-ca.crt` in
   the project root, or set `MQTT_CA_CERT` in `.env` to its path.
4. Run either counter with `--mqtt`:

```powershell
venv-pose\Scripts\python.exe scripts\squat_counter.py --source camera --mqtt
venv-pose\Scripts\python.exe scripts\pushup_counter.py --source camera --mqtt
```

Each completed repetition publishes a JSON message to `fitness/squat/reps` or
`fitness/push-up/reps`. Use an MQTT client such as MQTTX to subscribe to that
topic and capture the received messages. The `.env` file and downloaded CA
certificate are excluded from Git.

### Adjusting thresholds

The default squat thresholds are 110 degrees at the bottom and 155 degrees
standing. The default push-up thresholds are 100 degrees at the bottom and
155 degrees at the top. For example, to allow a squat to count when the detected
knee angle returns to 140 degrees:

```powershell
venv-pose\Scripts\python.exe scripts\squat_counter.py --source "C:\path\to\squat.mp4" --standing-angle 140
```

Push-up thresholds can be changed with `--bottom-angle` and `--top-angle`;
squat thresholds use `--bottom-angle` and `--standing-angle`. Adjustments can
change which movements count, so compare the displayed count with the actual
repetitions.

## Dataset and classifier experiments

The downloaded videos are expected under `dataset/push-up/` and `dataset/squat/`.
Extract fixed-length YOLO pose sequences with:

```powershell
venv-pose\Scripts\python.exe scripts\extract_video_poses.py
```

This saves 30-frame, 17-keypoint sequences under `dataset/pose_sequences/`.
Train the experimental exercise classifier with:

```powershell
venv-pose\Scripts\python.exe scripts\train_exercise_classifier.py
```

Test that classifier visually on a video or webcam with:

```powershell
venv-pose\Scripts\python.exe scripts\test_exercise_classifier.py --source "C:\path\to\exercise.mp4"
venv-pose\Scripts\python.exe scripts\test_exercise_classifier.py --source camera
```

The classifier predicts the exercise type; it does not decide whether form is
correct and is not used by the squat or push-up counters. Its dataset and test
set are small, so treat its scores as experimental.

## Project files

```text
scripts/
  extract_video_poses.py          Convert labelled videos to pose sequences
  pushup_counter.py               Count push-up repetitions
  squat_counter.py                Count squat repetitions
  train_exercise_classifier.py    Train the experimental LSTM/GRU classifier
  test_exercise_classifier.py     Visually test the classifier
  yolo_pose_test.py               Preview YOLO pose detection from a webcam
  collect_data.py                 Collect pose sequences from a webcam
  pose_test.py                    Older MediaPipe pose demo
requirements-pose.txt             Python package requirements
```

Downloaded datasets, generated sequences, model reports, environments, and
model weights are local files and are excluded from Git by `.gitignore`. Check
the dataset's license on its source page before redistributing it.
