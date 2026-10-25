"""
KLT + IMU SLAM — rotate-klt-slam with the iPhone's camera AND motion sensors
=============================================================================

Same idea as rotate-klt-slam.py (2-D map of where the camera went, people
marked on it), plus the iPhone's IMU (gyroscope, accelerometer, compass)
streamed through phone_stream.py, copied with permission from
github.com/aublaylock/SLAMTest.

VIDEO SOURCE (setting VIDEO)
  "camera": video through OpenCV like rotate-klt-slam.py (SOURCE = 0/1 for
            Continuity Camera, or a stream URL); the phone page sends IMU only.
            Caveat: Continuity Camera pauses if Safari is open on the phone,
            and Safari stops motion data when it's in the background, so the
            IMU may drop out. The HUD then shows "IMU: none" and "(vision)".
  "phone":  video and IMU both from the Safari page (phone's clock for both).

WHAT THE IMU ADDS
  1. Turning comes from the gyroscope, measured over exactly the time
     between two frames (both use the phone's clock). It is far more
     accurate than guessing the turn from the image, and it keeps working
     when the camera sees nothing useful (dark, smoke, blank wall).
  2. No long-term heading drift. phone_stream.py keeps an orientation
     estimate that the compass slowly corrects; our heading is gently
     pulled toward it whenever the phone isn't turning fast.
  3. De-rotation. The gyro says how the phone rotated (turn, tilt and
     roll), so we can predict where each KLT point would land from rotation
     alone. Whatever movement is left over is real forward/backward motion.
     That means walking forward is detected even while turning, which
     rotate-klt-slam.py had to ignore.
  4. Without IMU data (motion permission denied, phone just connected) it
     falls back to the vision-only estimate from rotate-klt-slam.py.

STILL LIMITED
  - Distance: one camera can't measure it, and integrating the
    accelerometer drifts within seconds. WALK_SPEED is still a guess until
    the robot's motor data replaces it.
  - Person distance comes from box size (~1.7 m body): rough (±25 %).

SETUP (one-time)
  Phone and laptop on the same Wi-Fi. Either:
  - Tailscale on both, same account, with MagicDNS + HTTPS certificates
    enabled in the admin page. The script prints an https://...ts.net URL.
  - Or no Tailscale: the script makes a self-signed certificate and prints
    an http:// link to install it on the iPhone (Settings > General > VPN &
    Device Management, then About > Certificate Trust Settings > enable).
  Then open the printed https:// URL in Safari and tap Start Streaming.
  With VIDEO = "camera", add ?imu to the end of the URL so the page sends
  motion data only (e.g. https://hamids-macbook-air.tail95f2c9.ts.net/?imu).
  With Tailscale from the Mac App Store, run this in a second terminal and
  use the ts.net URL without a port:
      /Applications/Tailscale.app/Contents/MacOS/Tailscale serve https+insecure://localhost:8443

RUN (from the repo root)
  uv run --no-project --with opencv-python --with ultralytics --with websockets \\
      python computer_vision_draft/klt-imu-slam.py

CONTROLS: Q = quit (the map is saved to outputs/)
"""

import math
import time
from pathlib import Path

import cv2
import numpy as np

from phone_stream import PhoneStream

# ═══════════════════════════════════════════════════════════════
#  SETTINGS
# ═══════════════════════════════════════════════════════════════

# Video: "camera" = OpenCV camera/stream (Continuity Camera), IMU from the
# phone page with ?imu.  "phone" = video and IMU both from the phone page.
VIDEO = "camera"
SOURCE = 0                # (VIDEO = "camera") 0/1 = local camera, or a stream URL
WIDTH = 640               # (VIDEO = "camera") process frames at this width
HFOV_DEG = 70.0           # (VIDEO = "camera") camera's horizontal field of view
CAM_IMU_OFFSET_S = 0.0    # (VIDEO = "camera") shift frame times to line up with
                          # the gyro; tune if turns look smeared or lagged

# (VIDEO = "phone") Safari sends portrait 640-px-wide frames. A diagonal FOV
# gives the right focal length whichever way round the frame is (81° ≈ the
# iPhone main camera's 69° horizontal on a 4:3 sensor).
DIAGONAL_FOV_DEG = 81.0

# Rear camera (OpenCV axes: x right, y down, z forward) → phone axes
# (x right, y up the screen, z out of the screen). Safari already rotates
# frames to portrait, so this is just two sign flips. (With VIDEO = "camera"
# the picture may be landscape, so the mapping is worked out from gravity.)
CAM_TO_DEV = np.diag([1.0, -1.0, -1.0])

# Motion
WALK_SPEED = 0.8          # m/s assumed when moving forward (placeholder)
EXPANSION_RATE = 0.03     # radial growth per SECOND that counts as "moving"
TURN_THRESH_DEG_S = 12.0  # faster than this = "turning"
STILL_PX = 0.35           # median flow below this = not moving at all
FLOW_MAX_RATE = 1.5       # rad/s; turning faster than this, skip forward motion
COMPASS_PULL = 0.02       # fraction of heading error corrected per frame
COMPASS_PULL_MAX_RATE = 0.2   # rad/s; only correct heading when turning slowly

# KLT tracking
MAX_CORNERS = 400
MIN_TRACKS = 150          # find new corners when fewer than this survive
MIN_GOOD = 20             # fewer tracked points than this = tracking lost
FB_MAX_PX = 1.0           # forward-backward check tolerance
EDGE_MARGIN = 0.06        # ignore this fraction of the image at each edge

# People
DETECTOR = "yolo"         # "yolo" (laptop), "hailo" (Pi AI HAT), "none"
YOLO_WEIGHTS = "yolo26n.pt"
HAILO_HEF = "/usr/share/hailo-models/yolov8s_h8l.hef"  # check this dir on the Pi
DETECT_EVERY = 2          # run the detector every N frames (the phone sends ~15 fps)
PERSON_CONF = 0.5
PERSON_LENGTH_M = 1.7     # standing height ≈ lying length
MERGE_RADIUS_M = 1.5      # detections closer than this are the same person
MIN_SIGHTINGS = 3         # seen this often before they count

MAP_PX = 600
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

LK_PARAMS = dict(
    winSize=(21, 21),     # search window around each point
    maxLevel=3,           # image pyramid levels: lets KLT follow bigger moves
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)


# ═══════════════════════════════════════════════════════════════
#  CAMERA MODEL
# ═══════════════════════════════════════════════════════════════

def focal_from_fov(width, hfov_deg):
    """Pinhole camera: half the image width sits at half the horizontal FOV."""
    return (width / 2) / math.tan(math.radians(hfov_deg) / 2)


def focal_from_diagonal(w, h, diag_fov_deg):
    """Pinhole camera: half the image diagonal sits at half the diagonal FOV."""
    return (math.hypot(w, h) / 2) / math.tan(math.radians(diag_fov_deg) / 2)


# ═══════════════════════════════════════════════════════════════
#  KLT TRACKER  (same as rotate-klt-slam.py)
# ═══════════════════════════════════════════════════════════════

class KLTTracker:
    def __init__(self):
        self.prev_gray = None
        self.prev_pts = None

    def find_corners(self, gray):
        h, w = gray.shape
        mx, my = int(w * EDGE_MARGIN), int(h * EDGE_MARGIN)
        mask = np.zeros_like(gray)
        mask[my:h - my, mx:w - mx] = 255
        return cv2.goodFeaturesToTrack(gray, maxCorners=MAX_CORNERS,
                                       qualityLevel=0.01, minDistance=8, mask=mask)

    def track(self, gray):
        """Return (old_points, new_points) as N×2 arrays, or None."""
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray, self.prev_pts = gray, self.find_corners(gray)
            return None

        # Points die as they leave the frame; top up when too few remain.
        if self.prev_pts is None or len(self.prev_pts) < MIN_TRACKS:
            self.prev_pts = self.find_corners(self.prev_gray)
        if self.prev_pts is None or len(self.prev_pts) < MIN_GOOD:
            self.prev_gray = gray  # blank wall, darkness, smoke...
            return None

        # Track forward (old → new), then backward (new → old). A point that
        # doesn't come back to where it started was tracked wrongly.
        p1 = self.prev_pts
        p2, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p1, None, **LK_PARAMS)
        p1_back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p2, None, **LK_PARAMS)
        fb_error = np.linalg.norm((p1 - p1_back).reshape(-1, 2), axis=1)
        good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb_error < FB_MAX_PX)

        q1, q2 = p1.reshape(-1, 2)[good], p2.reshape(-1, 2)[good]
        h, w = gray.shape
        inside = (q2[:, 0] >= 0) & (q2[:, 0] < w) & (q2[:, 1] >= 0) & (q2[:, 1] < h)
        q1, q2 = q1[inside], q2[inside]

        # The new points become next frame's old points (that's "tracking").
        self.prev_gray, self.prev_pts = gray, q2.reshape(-1, 1, 2)
        if len(q1) < MIN_GOOD:
            return None
        return q1, q2


# ═══════════════════════════════════════════════════════════════
#  MOTION: GYRO + FLOW
# ═══════════════════════════════════════════════════════════════

def cam_to_dev_from_gravity(up_dev):
    """Camera axes → phone axes, for a picture kept upright (Continuity Camera).

    The rear camera always looks along the phone's −z. Which phone axis is
    "down in the picture" depends on how the phone is held (portrait or
    landscape), so take gravity's direction across the screen and snap it to
    the nearest phone axis (the picture turns in 90° steps).
    """
    down = -up_dev
    axis = int(np.argmax(np.abs(down[:2])))     # 0 = phone x, 1 = phone y
    y_cam = np.zeros(3)
    y_cam[axis] = np.sign(down[axis])            # image down, in phone axes
    z_cam = np.array([0.0, 0.0, -1.0])           # out of the rear lens
    x_cam = np.cross(y_cam, z_cam)               # image right (x = y × z)
    return np.column_stack([x_cam, y_cam, z_cam])


def yaw_from_rotation(R_dev, up_dev):
    """How much the phone turned left/right (radians, + = right).

    R_dev is the phone's 3-D rotation between two frames. As a rotation
    vector (axis × angle) its component along "up" is the turn about the
    vertical axis. Right-hand rule makes that positive for a LEFT turn,
    hence the minus. This works however the phone is held.
    """
    rotvec = cv2.Rodrigues(R_dev)[0].ravel()
    return -float(rotvec @ up_dev)


def derotate(points, R_cam, f, centre):
    """Where each point would land if the camera had only rotated.

    Pixel → 3-D viewing ray, turn the ray by R_cam, ray → pixel. Also returns
    a mask that drops rays the rotation swung behind the camera.
    """
    rays = np.column_stack([(points - centre) / f, np.ones(len(points))])
    turned = rays @ R_cam.T
    ahead = turned[:, 2] > 1e-6
    return centre + f * turned[ahead, :2] / turned[ahead, 2:3], ahead


def radial_expansion(p_from, p_to, centre):
    """Median growth of distance-from-centre: + = moved forward, − = back."""
    r1 = np.linalg.norm(p_from - centre, axis=1)
    r2 = np.linalg.norm(p_to - centre, axis=1)
    far = r1 > 0.2 * centre[0]   # points near the centre barely move; skip them
    return float(np.median((r2[far] - r1[far]) / r1[far])) if far.sum() >= 10 else 0.0


def estimate_motion_vision(q1, q2, f, cx, cy):
    """Vision-only fallback (from rotate-klt-slam.py): (dyaw, expansion).

    Turning shifts every point's viewing angle equally; forward motion
    pushes the left and right halves apart. Averaging the two halves'
    median shifts cancels the forward part and leaves the turn.
    """
    a1 = np.arctan((q1[:, 0] - cx) / f)
    a2 = np.arctan((q2[:, 0] - cx) / f)
    shift = a2 - a1
    left, right = a1 < 0, a1 >= 0
    if left.sum() >= 5 and right.sum() >= 5:
        turn = (np.median(shift[left]) + np.median(shift[right])) / 2
    else:
        turn = np.median(shift)
    x2 = cx + f * np.tan(a2 - turn)
    y2 = q2[:, 1] - np.median(q2[:, 1] - q1[:, 1])
    expansion = radial_expansion(q1, np.column_stack([x2, y2]), np.array([cx, cy]))
    return -float(turn), expansion


def wrap(angle):
    """Wrap radians into −π..π, so 359° and 1° count as 2° apart."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Pose:
    """Position in metres: x = right, z = forward at start. Yaw + = right."""

    def __init__(self):
        self.x = self.z = self.yaw = 0.0

    def move_forward(self, dist):
        self.x += dist * math.sin(self.yaw)
        self.z += dist * math.cos(self.yaw)

    def to_world(self, ahead, right):
        """A point `ahead` m in front and `right` m to the right → map (x, z)."""
        s, c = math.sin(self.yaw), math.cos(self.yaw)
        return self.x + ahead * s + right * c, self.z + ahead * c - right * s


# ═══════════════════════════════════════════════════════════════
#  PEOPLE  (same as rotate-klt-slam.py)
# ═══════════════════════════════════════════════════════════════

def load_detector():
    """Return a function frame → list of person boxes (x1, y1, x2, y2)."""
    if DETECTOR == "yolo":
        from ultralytics import YOLO
        model = YOLO(YOLO_WEIGHTS)

        def detect(frame):
            r = model(frame, classes=[0], conf=PERSON_CONF, verbose=False)[0]
            return [tuple(int(v) for v in box) for box in r.boxes.xyxy.cpu().numpy()]
        return detect

    if DETECTOR == "hailo":
        # Pi only (`sudo apt install hailo-all`). Not yet run on hardware;
        # follows picamera2's examples/hailo/detect.py. The NMS output is one
        # array per class of rows [y0, x0, y1, x1, score], normalised 0–1.
        from picamera2.devices import Hailo
        hailo = Hailo(HAILO_HEF)
        in_h, in_w, _ = hailo.get_input_shape()

        def detect(frame):
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(cv2.resize(frame, (in_w, in_h)), cv2.COLOR_BGR2RGB)
            persons = hailo.run(rgb)[0]   # class 0 = person
            return [(int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
                    for y0, x0, y1, x1, score in (d[:5] for d in persons)
                    if score >= PERSON_CONF]
        return detect

    return lambda frame: []


def locate_person(box, f, cx):
    """Where is this person relative to the camera? Returns (ahead, right) m.

    Similar triangles: size_px / f = PERSON_LENGTH_M / distance. Using the
    larger box side works for standing (tall box) and lying (wide box).
    """
    x1, y1, x2, y2 = box
    size_px = max(x2 - x1, y2 - y1, 1)
    distance = min(max(f * PERSON_LENGTH_M / size_px, 0.5), 15.0)
    bearing = math.atan(((x1 + x2) / 2 - cx) / f)
    return distance * math.cos(bearing), distance * math.sin(bearing)


class PeopleMap:
    """Every sighting lands on the map; nearby sightings merge into one person."""

    def __init__(self):
        self.people = []   # dicts: id, x, z, sightings

    def add(self, x, z):
        nearest, best = None, MERGE_RADIUS_M
        for p in self.people:
            d = math.hypot(p["x"] - x, p["z"] - z)
            if d < best:
                nearest, best = p, d
        if nearest is None:
            nearest = {"id": len(self.people) + 1, "x": x, "z": z, "sightings": 0}
            self.people.append(nearest)
        n = nearest["sightings"]   # running average smooths the noisy range
        nearest["x"] = (nearest["x"] * n + x) / (n + 1)
        nearest["z"] = (nearest["z"] * n + z) / (n + 1)
        nearest["sightings"] = n + 1
        return nearest

    def confirmed(self):
        return [p for p in self.people if p["sightings"] >= MIN_SIGHTINGS]


# ═══════════════════════════════════════════════════════════════
#  DRAWING  (same as rotate-klt-slam.py)
# ═══════════════════════════════════════════════════════════════

def draw_map(trajectory, pose, people_map):
    img = np.full((MAP_PX, MAP_PX, 3), 18, np.uint8)

    # Zoom so everything fits (at least 4 m across), centred on the content.
    pts = trajectory + [(p["x"], p["z"]) for p in people_map.people]
    xs, zs = zip(*pts)
    mid_x, mid_z = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
    span = max(max(xs) - min(xs), max(zs) - min(zs), 4.0) * 1.2
    scale = MAP_PX / span   # pixels per metre

    def to_px(x, z):   # map z points up on screen
        return int(MAP_PX / 2 + (x - mid_x) * scale), int(MAP_PX / 2 - (z - mid_z) * scale)

    for m in range(math.floor(mid_x - span), math.ceil(mid_x + span)):   # 1 m grid
        cv2.line(img, to_px(m, mid_z - span), to_px(m, mid_z + span), (40, 40, 40), 1)
    for m in range(math.floor(mid_z - span), math.ceil(mid_z + span)):
        cv2.line(img, to_px(mid_x - span, m), to_px(mid_x + span, m), (40, 40, 40), 1)

    if len(trajectory) > 1:
        cv2.polylines(img, [np.array([to_px(x, z) for x, z in trajectory])], False,
                      (0, 255, 255), 2)

    for p in people_map.people:
        ok = p["sightings"] >= MIN_SIGHTINGS
        color = (60, 60, 255) if ok else (110, 110, 110)   # red = confirmed
        c = to_px(p["x"], p["z"])
        cv2.circle(img, c, 7, color, -1)
        cv2.putText(img, f"P{p['id']}", (c[0] + 9, c[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Arrow for the camera: tip points along the heading.
    cx, cz = to_px(pose.x, pose.z)
    tri = [(cx + r * math.sin(pose.yaw + a), cz - r * math.cos(pose.yaw + a))
           for r, a in ((18, 0.0), (9, 2.5), (9, -2.5))]
    cv2.drawContours(img, [np.array(tri, np.int32)], 0, (0, 0, 255), -1)

    cv2.putText(img, f"PEOPLE: {len(people_map.confirmed())}", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 255), 2)
    cv2.putText(img, "grid 1 m", (12, MAP_PX - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    return img


# ═══════════════════════════════════════════════════════════════
#  ONE FRAME AT A TIME
# ═══════════════════════════════════════════════════════════════

class Slam:
    def __init__(self, stream, detect):
        self.stream, self.detect = stream, detect
        self.tracker, self.pose, self.people_map = KLTTracker(), Pose(), PeopleMap()
        self.trajectory = [(0.0, 0.0)]
        self.boxes, self.frame_i = [], 0
        self.t_prev = None
        self.heading_offset = None   # phone heading when our map's yaw was 0

    def step(self, frame, t):
        """Process one frame captured at time t (seconds). Returns the view."""
        h, w = frame.shape[:2]
        f = (focal_from_fov(w, HFOV_DEG) if VIDEO == "camera"
             else focal_from_diagonal(w, h, DIAGONAL_FOV_DEG))
        centre = np.array([w / 2, h / 2])
        dt = t - self.t_prev if self.t_prev is not None else 0.0
        valid_dt = 0.0 < dt < 0.5
        tracks = self.tracker.track(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # 1. How did the phone rotate since the last frame? (gyro)
        R_dev, rate, rot_ok = (self.stream.get_rotation(self.t_prev, t) if valid_dt
                               else (np.eye(3), 0.0, False))
        attitude = self.stream.get_attitude_matrix()   # phone → world, world z = up
        use_imu = rot_ok and attitude is not None

        dyaw, expansion, motion_ok, flow_px = 0.0, 0.0, False, 0.0
        if use_imu:
            up_dev = attitude.T @ np.array([0.0, 0.0, 1.0])   # "up" in phone axes
            dyaw = yaw_from_rotation(R_dev, up_dev)
            yaw_source = "gyro"
        else:
            yaw_source = "vision"

        # 2. What's left after removing that rotation is forward motion.
        if tracks is not None:
            q1, q2 = tracks
            flow_px = float(np.median(np.linalg.norm(q2 - q1, axis=1)))
            if use_imu:
                if rate < FLOW_MAX_RATE:
                    C = cam_to_dev_from_gravity(up_dev) if VIDEO == "camera" else CAM_TO_DEV
                    R_cam = C.T @ R_dev.T @ C   # the same rotation, in camera axes
                    q1r, ahead = derotate(q1, R_cam, f, centre)
                    if ahead.sum() >= MIN_GOOD:
                        expansion = radial_expansion(q1r, q2[ahead], centre)
                        motion_ok = True
            elif valid_dt:
                dyaw, expansion = estimate_motion_vision(q1, q2, f, *centre)
                # without the gyro, expansion can't be trusted during a turn
                motion_ok = abs(dyaw) / dt < math.radians(TURN_THRESH_DEG_S)

        # 3. Update the heading, and slowly pull it toward the phone's
        #    compass-corrected heading so it doesn't drift.
        self.pose.yaw += dyaw
        heading, _, _, ready = self.stream.get_orientation()
        if use_imu and ready:
            if self.heading_offset is None:
                self.heading_offset = math.radians(heading) - self.pose.yaw
            elif rate < COMPASS_PULL_MAX_RATE:
                error = wrap(math.radians(heading) - self.heading_offset - self.pose.yaw)
                self.pose.yaw += COMPASS_PULL * error

        # 4. Move forward/back, and describe what happened.
        turning = valid_dt and abs(dyaw) / dt > math.radians(TURN_THRESH_DEG_S)
        status = []
        if tracks is None:
            status.append("LOST" + (" (gyro still tracking turns)" if use_imu else ""))
        elif motion_ok and abs(expansion) / dt > EXPANSION_RATE:
            forward = expansion > 0
            self.pose.move_forward((1 if forward else -1) * WALK_SPEED * dt)
            status.append("FORWARD" if forward else "BACKWARD")
        if turning:
            status.append("TURNING " + ("RIGHT" if dyaw > 0 else "LEFT"))
        if not status:
            status.append("STILL" if flow_px < STILL_PX else "IDLE")

        last_x, last_z = self.trajectory[-1]
        if math.hypot(self.pose.x - last_x, self.pose.z - last_z) > 0.02:
            self.trajectory.append((self.pose.x, self.pose.z))
        self.t_prev = t

        # 5. Detect people every few frames and put them on the map.
        if self.frame_i % DETECT_EVERY == 0:
            self.boxes = []
            for box in self.detect(frame):
                ahead, right = locate_person(box, f, centre[0])
                person = self.people_map.add(*self.pose.to_world(ahead, right))
                self.boxes.append((box, person, math.hypot(ahead, right)))
        self.frame_i += 1

        return self.draw(frame, tracks, turning, " + ".join(status), yaw_source)

    def draw(self, frame, tracks, turning, status, yaw_source):
        vis = frame.copy()
        if tracks is not None:
            color = (0, 0, 255) if turning else (0, 255, 0)
            for (u1, v1), (u2, v2) in zip(tracks[0].astype(int), tracks[1].astype(int)):
                cv2.line(vis, (u1, v1), (u2, v2), color, 1)
        for (x1, y1, x2, y2), person, dist in self.boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis, f"P{person['id']} ~{dist:.1f}m", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        imu = self.stream.get_latest_imu()
        compass = self.stream.get_compass()
        imu_txt = "IMU: none (motion permission?)" if imu is None else f"IMU: {imu[2]:.0f} Hz"
        if compass is not None:
            imu_txt += "  compass: " + ("correcting" if compass[2] else "idle")
        lines = [(f"{status}", 0.7, (0, 255, 255)),
                 (f"YAW {math.degrees(self.pose.yaw):+.0f} deg ({yaw_source})", 0.55,
                  (255, 255, 255)),
                 (f"POS ({self.pose.x:+.1f}, {self.pose.z:+.1f}) m", 0.55, (255, 255, 255)),
                 (imu_txt, 0.5, (200, 200, 200))]
        for i, (text, size, color) in enumerate(lines):
            cv2.putText(vis, text, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, size,
                        (0, 0, 0), 4)
            cv2.putText(vis, text, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1)

        h = vis.shape[0]
        map_img = draw_map(self.trajectory, self.pose, self.people_map)
        return np.hstack((vis, cv2.resize(map_img, (h, h))))


# ═══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════

WINDOW = "KLT + IMU SLAM"


def run_with_camera(slam):
    """Video from OpenCV (Continuity Camera / webcam / stream), IMU from the phone."""
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source {SOURCE!r}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # phone_stream stamps IMU samples on this laptop's time.time() clock,
        # so frames are stamped on the same clock when they arrive.
        t = time.time() + CAM_IMU_OFFSET_S
        h0, w0 = frame.shape[:2]
        frame = cv2.resize(frame, (WIDTH, round(h0 * WIDTH / w0)))
        cv2.imshow(WINDOW, slam.step(frame, t))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()


def run_with_phone_video(stream, slam):
    """Video and IMU both from the phone's Safari page."""
    last_arrival = None
    while True:
        frame, arrival, capture = stream.get_latest_frame()
        if frame is None:
            waiting = np.zeros((200, 560, 3), np.uint8)
            cv2.putText(waiting, "Waiting for the phone... (Q quits)", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.imshow(WINDOW, waiting)
        elif arrival != last_arrival:   # only process frames we haven't seen
            last_arrival = arrival
            # capture time (phone clock mapped to ours) lines frames up with the gyro
            cv2.imshow(WINDOW, slam.step(frame, capture))
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break


def run():
    stream = PhoneStream()
    stream.start()          # prints the URL to open on the phone
    if VIDEO == "camera":
        print("VIDEO = 'camera': video comes from this laptop (SOURCE). On the phone,")
        print("add ?imu to the end of the https:// URL so it sends motion data only.")
    slam = Slam(stream, load_detector())
    if VIDEO == "camera":
        run_with_camera(slam)
    else:
        run_with_phone_video(stream, slam)

    cv2.destroyAllWindows()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / f"klt_imu_map_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(str(out), draw_map(slam.trajectory, slam.pose, slam.people_map))
    print(f"{len(slam.people_map.confirmed())} people confirmed. Map saved to {out}")
    for p in slam.people_map.confirmed():
        print(f"  P{p['id']}: x={p['x']:+.1f} m  z={p['z']:+.1f} m  ({p['sightings']} sightings)")


if __name__ == "__main__":
    run()
