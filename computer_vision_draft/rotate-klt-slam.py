"""
Rotate-KLT SLAM — 2-D map of where the camera went, with people marked on it
=============================================================================

Fixed successor to rotate-orbslam.py. One file; runs on a laptop (YOLO) or a
Raspberry Pi 5 with the AI HAT (Hailo).

WHAT CHANGED FROM rotate-orbslam.py
  1. Real units. Yaw is measured from how far points moved (pixels → angle
     through the focal length) instead of adding a fixed ±2° per frame, and
     forward motion is speed × elapsed time instead of a fixed 2 units per
     frame. The map no longer depends on the frame rate.
  2. A real camera model. FOCAL_LENGTH / PP were KITTI-dataset numbers that
     nothing used. The focal length now comes from your camera's field of view.
  3. No crash on short match lists (`for m, n in matches`). KLT has no knn
     matching step at all.
  4. Medians instead of means, so a person walking through the frame does not
     drag the estimate.
  5. People: detected with YOLO (laptop) or the Hailo AI HAT (Pi), placed on
     the map in metres, de-duplicated, and counted.

WHY KLT INSTEAD OF ORB
  ORB finds corners in every frame from scratch, computes a binary descriptor
  for each, then compares every descriptor in frame A with every descriptor
  in frame B (3000 × 3000 comparisons in the old file). That machinery exists
  to recognise a place you saw long ago (loop closure, relocalisation).
  Consecutive video frames are nearly identical, so we don't need it. KLT
  (Lucas–Kanade optical flow) takes last frame's points and searches a small
  window around each for where it moved, using image gradients: sub-pixel
  accurate, no descriptors, no all-pairs matching. Measured on one M1 core:
  ~4 ms/frame (640×480, 400 points) vs ~76 ms/frame for the old ORB + brute-
  force path (1280×720, 3000 points). Roughly 2–3× slower on a Pi 5 core.
  The trade-off: KLT loses points in fast motion or blur (image pyramids help)
  and cannot recognise old places. ORB comes back when we add loop closure.

LIMITS (next steps)
  - One camera cannot measure distance. WALK_SPEED is an assumption until we
    add motor commands / IMU ("movement data").
  - Heading drifts slowly; the gyro will fix most of that.
  - Person distance comes from box size (~1.7 m body), so it is rough (±25 %)
    and too far for people who are partly hidden or cut off by the frame.

CONTROLS: Q = quit (the map is saved to outputs/)
"""

import math
import time
from pathlib import Path

import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════════
#  SETTINGS
# ═══════════════════════════════════════════════════════════════

# Camera: 0/1 = local camera (the iPhone via Continuity Camera is often 1),
# a stream URL (e.g. the robot's "http://<ip>:81/stream"), or a video file.
SOURCE = 0
WIDTH = 640               # process frames at this width (smaller = faster)
HFOV_DEG = 70.0           # camera's horizontal field of view

# Motion
WALK_SPEED = 0.8          # m/s assumed when moving forward (placeholder)
EXPANSION_THRESH = 0.002  # radial growth per frame that counts as "moving"
TURN_THRESH_DEG_S = 12.0  # faster than this = "turning"
STILL_PX = 0.35           # median flow below this = not moving at all

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
DETECT_EVERY = 3          # run the detector every N frames
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
    """Pinhole camera: half the image width sits at half the FOV angle.

    tan(hfov / 2) = (width / 2) / f   →   f = (width / 2) / tan(hfov / 2)
    """
    return (width / 2) / math.tan(math.radians(hfov_deg) / 2)


# ═══════════════════════════════════════════════════════════════
#  KLT TRACKER  (replaces FeatureExtractor + BFMatcher)
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
        if self.prev_gray is None:
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
#  MOTION FROM FLOW
# ═══════════════════════════════════════════════════════════════

def estimate_motion(q1, q2, f, cx, cy):
    """Split the flow into a turn (radians) and a forward expansion (ratio).

    Turning: every point's viewing angle shifts by the same amount.
    Forward: points move away from the centre (left half goes left, right
    half goes right). Taking the median shift of each half and averaging
    them cancels the forward part and leaves the turn.
    """
    a1 = np.arctan((q1[:, 0] - cx) / f)   # viewing angle of each point
    a2 = np.arctan((q2[:, 0] - cx) / f)
    shift = a2 - a1
    left, right = a1 < 0, a1 >= 0
    if left.sum() >= 5 and right.sum() >= 5:
        turn = (np.median(shift[left]) + np.median(shift[right])) / 2
    else:
        turn = np.median(shift)
    dyaw = -float(turn)   # scene slid right → camera turned left (yaw + = right)

    # Undo the turn (and any up/down bob), then measure how much points moved
    # away from the centre. Near-centre points barely move, so skip them.
    x2 = cx + f * np.tan(a2 - turn)
    y2 = q2[:, 1] - np.median(q2[:, 1] - q1[:, 1])
    r1 = np.hypot(q1[:, 0] - cx, q1[:, 1] - cy)
    r2 = np.hypot(x2 - cx, y2 - cy)
    far = r1 > 0.2 * cx
    expansion = float(np.median((r2[far] - r1[far]) / r1[far])) if far.sum() >= 10 else 0.0

    flow_px = float(np.median(np.hypot(*(q2 - q1).T)))
    return dyaw, expansion, flow_px


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
#  PEOPLE
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
#  DRAWING
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
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════

def run():
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source {SOURCE!r}")
    is_file = isinstance(SOURCE, str) and Path(SOURCE).is_file()

    detect = load_detector()
    tracker, pose, people_map = KLTTracker(), Pose(), PeopleMap()
    trajectory = [(0.0, 0.0)]
    boxes, frame_i, t_prev = [], 0, None
    print("Rotate-KLT SLAM running. Q = quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h0, w0 = frame.shape[:2]
        frame = cv2.resize(frame, (WIDTH, round(h0 * WIDTH / w0)))
        h, w = frame.shape[:2]
        f, cx, cy = focal_from_fov(w, HFOV_DEG), w / 2, h / 2

        # Elapsed time: a recorded video uses its own clock, a live camera the wall clock.
        now = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if is_file else time.monotonic()
        dt = min(max(now - t_prev, 0.0), 0.2) if t_prev is not None else 0.0
        t_prev = now

        # 1. Track points and turn their motion into yaw + forward motion.
        tracks = tracker.track(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        status, turning = "LOST", False
        if tracks is not None:
            q1, q2 = tracks
            dyaw, expansion, flow_px = estimate_motion(q1, q2, f, cx, cy)
            pose.yaw += dyaw
            turning = dt > 0 and abs(dyaw) / dt > math.radians(TURN_THRESH_DEG_S)

            # Expansion is only trusted when not turning: perspective during a
            # turn also stretches the image a little.
            if turning:
                status = "TURNING " + ("RIGHT" if dyaw > 0 else "LEFT")
            elif abs(expansion) > EXPANSION_THRESH:
                forward = expansion > 0
                pose.move_forward((1 if forward else -1) * WALK_SPEED * dt)
                status = "FORWARD" if forward else "BACKWARD"
            else:
                status = "STILL" if flow_px < STILL_PX else "IDLE"

            last_x, last_z = trajectory[-1]
            if math.hypot(pose.x - last_x, pose.z - last_z) > 0.02:
                trajectory.append((pose.x, pose.z))

        # 2. Detect people every few frames and put them on the map.
        if frame_i % DETECT_EVERY == 0:
            boxes = []
            for box in detect(frame):
                ahead, right = locate_person(box, f, cx)
                person = people_map.add(*pose.to_world(ahead, right))
                boxes.append((box, person, math.hypot(ahead, right)))
        frame_i += 1

        # 3. Draw.
        vis = frame.copy()
        if tracks is not None:
            color = (0, 0, 255) if turning else (0, 255, 0)
            for (u1, v1), (u2, v2) in zip(q1.astype(int), q2.astype(int)):
                cv2.line(vis, (u1, v1), (u2, v2), color, 1)
        for (x1, y1, x2, y2), person, dist in boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis, f"P{person['id']} ~{dist:.1f}m", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(vis, f"STATUS: {status}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(vis, f"YAW: {math.degrees(pose.yaw):+.0f} deg   "
                         f"POS: ({pose.x:+.1f}, {pose.z:+.1f}) m", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        map_img = draw_map(trajectory, pose, people_map)
        cv2.imshow("Rotate-KLT SLAM", np.hstack((vis, cv2.resize(map_img, (h, h)))))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / f"klt_map_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(str(out), draw_map(trajectory, pose, people_map))
    print(f"{len(people_map.confirmed())} people confirmed. Map saved to {out}")
    for p in people_map.confirmed():
        print(f"  P{p['id']}: x={p['x']:+.1f} m  z={p['z']:+.1f} m  ({p['sightings']} sightings)")


if __name__ == "__main__":
    run()
