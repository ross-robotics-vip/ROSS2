# ROSS2 (Remote Observation & Swarm System v2)

A small, cheap search-and-rescue robot car that can be sent into collapsed or
burning buildings to find people and animals, so firefighters don't have to
go in blind.

## What we're building

After a fire or a structural collapse, the inside of a building is full of
rubble, smoke and unstable ground. It's often too risky to send a person in
just to look around. ROSS2 is a low-cost robot car that firefighters can send
in first. It streams video back, spots humans or animals, and marks where it
found them on a map, so the rescue team knows where to go and can plan the
right procedure before anyone steps inside.

Right now we're focused on getting **one robot** working well end to end.
Once that's solid, the longer-term plan is autonomy and a small swarm of
robots working together (see [Roadmap](#roadmap)).

ROSS2 grew out of the original [ROSS](https://github.com/ssophia-k/ROSS)
project, which used an ESP32-CAM. ROSS2 is built around a Raspberry Pi 5.

## The team

We're a five-person team, and every part below is needed to make this work.

| Who | Focus |
|---|---|
| Austin, Morsalin | Computer vision: detection, visual SLAM, mapping |
| Sanjay | Hardware and networking |
| Jiang | Camera stability |
| Tiger | Mechanical design and CAD |

## Where things stand

### Hardware

- Raspberry Pi 5 with the Raspberry Pi AI HAT
- One camera (monocular), read through Picamera2
- A small car chassis with front wheel motors, driven through SunFounder's
  PiCar-X library

### Software on the Pi

The Pi runs a small web server (FastAPI) that streams the camera as MJPEG.
You open the stream in a browser on a laptop on the same network and pick
which detector to run:

- **Raw video**, for driving
- **Face detection** with a Haar cascade
- **People detection** with OpenCV's HOG detector
- **People detection** with YOLO (`yolo26n`)
- **Video plus remote driving**, using the keyboard in the Pi's terminal

Detection currently runs on the Pi's CPU. Moving it onto the AI HAT is next
(see [Roadmap](#roadmap)).

### Computer vision drafts

The vision work happens in `computer_vision_draft/`. These are standalone
prototypes we run on a laptop with a webcam or an iPhone camera before
porting them to the Pi:

- **Visual SLAM from one camera.** It tracks feature points from frame to
  frame to estimate how the camera turns and moves, and draws a top-down map
  of the path. The first version (`rotate-orbslam.py`) matched ORB features;
  the newer one (`rotate-klt-slam.py`) uses KLT optical-flow tracking, which
  is much lighter and better suited to a Pi.
- **People on the map.** While mapping, YOLO looks for people. Each person
  gets a rough distance (from the size of their bounding box) and a dot on
  the map, and repeat sightings are merged so the count stays honest.
- **360° room scan.** MiDaS depth estimation turns a slow 360° turn into a
  rough 3D model of the room.
- **Detector experiments** with HOG and YOLO.

A single camera can't measure real distances, so the map's scale is still a
guess. That's what the IMU and motor data on the roadmap are for.

## Roadmap

**Know where the robot is.** Combine IMU data (gyro and accelerometer) and
motor rotation data with vision, so the robot knows its position in metres
from where it started, can retrace its path, and can tell when it's stuck.

**Detection on the AI HAT.** Run the person and animal detectors on the
Hailo accelerator to free up the Pi's CPU for mapping.

**Camera stability.** Rough terrain shakes the camera. A stable view is what
makes reliable detection possible on uneven ground.

**Mechanics.** Wheels that can adjust and have claws for climbing stairs and
rubble, and a frame that protects the electronics while staying flexible
enough to move over obstacles.

**Networking.** Choosing the right link (Wi-Fi, Bluetooth or something else)
and sending video and map data back to the laptop efficiently and reliably,
all on cheap hardware that still does the job.

**Autonomy.** Obstacle detection and driving on its own.

**Swarm.** Several robots that spread out, talk to each other, and send
their data back through a "mother" server robot that acts as the base
station. This comes after we have one robot working.

## Getting started

Everything Python runs through [uv](https://docs.astral.sh/uv/). The `ross`
command is the project's CLI.

```bash
git clone https://github.com/ross-robotics-vip/ROSS2.git
cd ROSS2
uv sync
uv run ross --help
```

### On a laptop (webcam)

A quick way to test streaming without the robot. Run it from the repo root,
because the face detector loads `haarcascade_frontalface_default.xml` from
there.

```bash
uv run ross stream --port 8080
```

Then open:

- `http://<laptop-ip>:8080/live`: raw webcam
- `http://<laptop-ip>:8080/human_detection`: face detection

### On the Raspberry Pi (robot)

The Pi version also needs packages that aren't installed by `uv sync`:

- `picamera2`, which ships with Raspberry Pi OS. Create the environment with
  `uv venv --system-site-packages` so it can see it.
- SunFounder's `robot-hat` and `picar-x` libraries, installed with SunFounder's
  PiCar-X instructions.
- `readchar`, for keyboard driving.

```bash
uv venv --system-site-packages
uv sync
uv run --with readchar ross stream_pi --port 8080
```

From a laptop on the same network, open one of:

| URL | What it shows |
|---|---|
| `http://<pi-ip>:8080/live` | Raw camera |
| `http://<pi-ip>:8080/cascade` | Face detection (Haar cascade) |
| `http://<pi-ip>:8080/hog` | People detection (HOG) |
| `http://<pi-ip>:8080/yolo` | People detection (YOLO) |
| `http://<pi-ip>:8080/move_vision` | Camera, and turns on keyboard driving |

Driving works in the terminal where the server is running (usually an SSH
session into the Pi): `W` `A` `S` `D` to move, `O`/`P` for faster/slower,
`F` to stop, `Ctrl+C` to quit driving.

### Computer vision drafts

The drafts open their own windows, so they need regular OpenCV instead of
the headless version the project installs. Run them in a throwaway
environment:

```bash
# Visual SLAM + people on the map (set SOURCE at the top: 0/1 = camera)
uv run --no-project --with opencv-python --with ultralytics \
    python computer_vision_draft/rotate-klt-slam.py

# 360° MiDaS room scan
uv run --no-project --python 3.12 --with opencv-python --with torch \
    --with timm --with open3d --with scipy python computer_vision_draft/midasroom.py
```

YOLO weights (`yolo26n.pt`) download automatically on first run. Saved maps
go to `outputs/`.

## Repository layout

```text
ROSS2/
├── ross2/                     Python package behind the `ross` CLI
│   ├── cli.py                 Registers the CLI commands
│   ├── config.py              Shared paths (outputs/, models/) and .env loading
│   ├── commands/
│   │   ├── stream.py          `ross stream`: laptop webcam server
│   │   ├── stream_pi.py       `ross stream_pi`: Pi camera server + driving
│   │   └── flash, fuel, serial, slam, teleop .py   Carried over from ROSS v1
│   ├── slam/
│   │   ├── vision.py          Laptop camera capture + face detection
│   │   ├── vision_pi.py       Pi camera, detectors (cascade/HOG/YOLO), PiCar-X driving
│   │   └── pipeline, reconstruct, io .py           ROSS v1 MiDaS room scanner
│   ├── net/                   ROSS v1 robot discovery + HTTP client
│   └── drivers/               ROSS v1 GPIO and battery fuel-gauge drivers
├── computer_vision_draft/     Standalone vision prototypes (laptop first)
│   ├── rotate-klt-slam.py     Visual SLAM (KLT) + people marked on the map
│   ├── rotate-orbslam.py      Earlier visual SLAM using ORB features
│   ├── midasroom.py           360° room scan with MiDaS depth → 3D mesh
│   ├── yolo_human_detection.py   YOLO people detection on a webcam
│   └── hog.py                 HOG people detection on a webcam
├── docs/notes/                Team meeting notes
├── yolo_test.py               Quick YOLO test on the Pi camera
├── haarcascade_frontalface_default.xml   Face detector model used by `ross stream`
├── pyproject.toml             Dependencies and lint settings
├── uv.lock                    Pinned dependency versions
└── .github/workflows/         CI: ruff lint and format checks
```

The ROSS v1 files are kept for reference. They aren't hooked up to the CLI,
and they still import from the old `ROSS1` package, so they won't run as is.

## License

MIT. See [LICENSE](LICENSE).
