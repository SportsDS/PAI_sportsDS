# Color Courier — AI-assisted robotic sorting

A self-contained 2-D simulation in which a differential-drive robot classifies three colored parcels, plans obstacle-avoiding paths, and delivers each parcel to the matching bin.

## Run

Requirements: Python 3.10+, NumPy, Pillow, and ffmpeg for MP4 generation.

```bash
python -m pip install -r requirements.txt
python simulation.py --video color_courier_demo.mp4 --seconds 120 --fps 12
```

The included demo is 120 seconds at 12 fps. It is not sped up.

## Components

- **RGB perception:** nearest Euclidean distance from a simulated camera RGB observation to one of three color prototypes. Gaussian observation noise has SD 7.
- **Path planning:** A* search on an 8-connected occupancy grid. Obstacles are inflated by a safety margin.
- **Motion control:** simplified planar differential-drive kinematics with waypoint following.
- **Task logic:** state machine for parcel approach, pickup, delivery, and release.

## Reproducibility and limitations

The per-class camera noise is seeded deterministically, so RGB readings and classifications are reproducible. This is a custom 2-D visualization simulation, not ROS/Gazebo/Genesis or a physics engine. Grasping is represented by task-state transitions; contact dynamics, occlusion, friction, wheel slip, and hardware safety are not modeled. The classifier is limited to three known colors.

## Files

- `simulation.py` — perception, A* planning, motion, task logic, and video renderer
- `color_courier_demo.mp4` — completed task video
- `説明資料_PAI最終課題_アカウントID.pdf` — short project report; replace the placeholder in the filename before submission
