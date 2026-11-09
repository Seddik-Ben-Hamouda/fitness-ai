# Fitness AI

## Count your reps with a camera

Fitness AI is a student project that uses a webcam or a saved video to count
**squats** and **push-ups**. It shows a live picture, a repetition count, and
simple prompts such as when to lower or stand back up.

> **In one picture:**<br>
>  Your movement →  The app follows body joints →  Your rep count

It is a learning prototype, not a personal trainer. It does not confirm that an
exercise is safe or that your full form is correct.

## What you can try

| Choose | What you see |
|---|---|
| Squat | Your pose, knee movement, count, and a short prompt |
| Push-up | Your pose, elbow movement, count, and a short prompt |

Use your webcam or a video file. A side-on view with the needed joints visible
usually works best. Press **Q** to close the video.

## How to run it

The project is currently set up for **Windows**. To try it, you need Python
3.10, the project packages, and the YOLO pose-model file
(`yolo26m-pose.pt`). Run these commands in the project folder:

```powershell
py -3.10 -m venv venv-pose
venv-pose\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-pose.txt
```

Then start either counter:

```powershell
python scripts\squat_counter.py --source camera
python scripts\pushup_counter.py --source camera
```

To count from a saved video, replace `camera` with its file path, for example:

```powershell
python scripts\squat_counter.py --source "C:\Videos\my-squats.mp4"
```

## What is in the project?

| File | In plain language |
|---|---|
| `scripts/squat_counter.py` | Watches knee movement and counts squats |
| `scripts/pushup_counter.py` | Watches elbow movement and counts push-ups |
| `scripts/mqtt_publisher.py` | Optional helper that sends counts to an online message service |
| `scripts/extract_video_poses.py` | Prepares exercise videos for a separate experiment |
| `scripts/train_exercise_classifier.py` | Trains an experiment to tell squats from push-ups |
| `scripts/test_exercise_classifier.py` | Tries that experiment on a video or webcam |
| `scripts/collect_data.py` | Records examples from a webcam |
| `scripts/yolo_pose_test.py` | Shows the detected body pose |
| `scripts/pose_test.py` | Earlier pose-detection demo |

The Python files include comments explaining their main sections and important
steps. The classifier experiment is separate from the live repetition counters.

## Dataset

The `dataset/push-up/` and `dataset/squat/` video folders are the exercise
examples used by this project. Git is set up to include only those two folders;
the original large archive, other exercise videos, and generated files stay
out of the upload.

Before sharing the videos publicly, check that their original source permits
redistribution and give credit where required. The counters can also be tried
with your own videos.
