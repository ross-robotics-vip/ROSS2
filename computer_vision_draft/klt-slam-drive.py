"""
KLT SLAM + DRIVE — drive the robot from a web page while it maps
=================================================================

Runs on the Raspberry Pi. Combines rotate-klt-slam.py (camera → KLT tracking
→ map + people) with the Yahboom Quad MD motor board (motor_draft/quad_md.py)
and serves ONE web page where you watch the live video and map and drive.

CONTROLS (click the page once so it gets the keyboard)
  W / S        all four wheels forward / backward
  A / D        spin left / right on the spot (left and right sides opposite)
  W+A, W+D …   curve: the inside wheels slow down
  H J K L      test one motor at a time: M1 M2 M3 M4 forward (hold Shift = backward)
  - / =        slower / faster
  Space        stop everything
  The motors stop when you let go of the keys, when the page loses focus, and
  by themselves if no command arrives for half a second (closed tab, lost
  connection).

FIRST TIME (wheels off the ground!)
  Hold H, J, K, L one at a time and note which wheel turns and which way.
  Then set LEFT_MOTORS / RIGHT_MOTORS / MOTOR_DIRECTION below so that W rolls
  every wheel forward.

RUN (on the Pi, from ~/ROSS2), then open http://ross-test.local:8090
  python3 computer_vision_draft/klt-slam-drive.py                   # Hailo people detection
  python3 computer_vision_draft/klt-slam-drive.py --detector none   # SLAM + driving only
  Ctrl+C on the Pi stops the motors and saves the map to outputs/.
  For testing: --no-motors (no motor board), --laptop (webcam instead of Pi camera).
"""

import argparse
import importlib.util
import json
import math
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "motor_draft"))
import quad_md  # noqa: E402  (the motor board, from motor_draft/quad_md.py)

# rotate-klt-slam.py has a dash in its name, so it's loaded by path.
_spec = importlib.util.spec_from_file_location("rotate_klt_slam", HERE / "rotate-klt-slam.py")
rk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rk)

# ═══════════════════════════════════════════════════════════════
#  SETTINGS
# ═══════════════════════════════════════════════════════════════

VIEW_PORT = 8090

# Which motor port is which wheel. Yahboom labels M1 as the front-left wheel;
# check the rest with H J K L and change these if your wiring differs.
LEFT_MOTORS = (3, 4)    # swapped with RIGHT so A/D stay the same after the W/S flip
RIGHT_MOTORS = (1, 2)
# +1 or -1 per motor: flip any wheel that spins BACKWARDS when you press W
# (motors on the right side are often mounted mirror-image).
MOTOR_DIRECTION = {1: 1, 2: 1, 3: 1, 4: 1}   # tested on the car: these make W go forward

DEFAULT_SPEED = 1500      # motor power, out of 3600
SPEED_STEP = 300          # how much - and = change it
CURVE_INNER = 0.4         # inside wheels' share of speed while curving (W+A, W+D)
TURN_SCALE = 1.0          # spinning on the spot (A, D) relative to the speed
WATCHDOG_S = 0.5          # stop if no command for this long
MOTOR_RATE_HZ = 20        # how often motor commands are (re)sent

# Map distance: "commands" = from the motor power you apply (needs the setting
# below measured once); "vision" = from image expansion, like rotate-klt-slam.py.
MOTION = "commands"
SPEED_AT_FULL_PWM = 0.6   # m/s at power 3600 - PLACEHOLDER: drive 3 s at full
                          # power, measure the distance, divide by 3, put it here


# ═══════════════════════════════════════════════════════════════
#  KEYS → MOTOR POWER
# ═══════════════════════════════════════════════════════════════

def plan_motors(keys, speed, shift=False):
    """Turn the held keys into power for M1-M4.

    Returns (pwm, forward): pwm = [M1, M2, M3, M4] ready for the board, and
    forward = how much the robot as a whole moves forward (-1..1), used to
    measure distance on the map.
    """
    drive = ("w" in keys) - ("s" in keys)       # +1 forward, -1 backward
    turn = ("d" in keys) - ("a" in keys)        # +1 right, -1 left

    if drive and turn:                  # curve: inside wheels slower
        left = drive * speed * (1.0 if turn > 0 else CURVE_INNER)
        right = drive * speed * (CURVE_INNER if turn > 0 else 1.0)
    elif drive:                         # straight: all four wheels the same
        left = right = drive * speed
    elif turn:                          # spin on the spot: sides opposite
        left, right = turn * speed * TURN_SCALE, -turn * speed * TURN_SCALE
    else:
        left = right = 0

    power = {m: 0.0 for m in (1, 2, 3, 4)}
    for m in LEFT_MOTORS:
        power[m] = left
    for m in RIGHT_MOTORS:
        power[m] = right
    for key, m in (("h", 1), ("j", 2), ("k", 3), ("l", 4)):   # single-motor tests
        if key in keys:
            power[m] = -speed if shift else speed

    pwm = [int(power[m] * MOTOR_DIRECTION[m]) for m in (1, 2, 3, 4)]
    forward = (left + right) / 2 / quad_md.PWM_MAX
    return pwm, forward


class Driver:
    """Owns the motor board. A background thread re-sends the latest command
    20 times a second and switches the motors off if commands stop coming."""

    def __init__(self, board):
        self.board = board            # None = no board (--no-motors): just pretend
        self.lock = threading.Lock()
        self.pwm, self.forward, self.last_command = [0, 0, 0, 0], 0.0, 0.0
        self.encoders, self.battery = [0, 0, 0, 0], None
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def command(self, pwm, forward):
        with self.lock:
            self.pwm, self.forward, self.last_command = pwm, forward, time.monotonic()

    def current(self):
        """(pwm, forward) being applied now - zero if the watchdog has fired."""
        with self.lock:
            if time.monotonic() - self.last_command > WATCHDOG_S:
                return [0, 0, 0, 0], 0.0
            return list(self.pwm), self.forward

    def _loop(self):
        sent, last_sent, last_battery = None, 0.0, -1e9
        while self.running:
            pwm, _ = self.current()
            now = time.monotonic()
            if self.board is not None:
                try:
                    if pwm != sent or now - last_sent > 0.2:
                        self.board.set_pwm(*pwm)
                        sent, last_sent = pwm, now
                    self.encoders = self.board.encoder_totals()
                    if now - last_battery > 10:
                        self.battery = self.board.battery_volts()
                        last_battery = now
                except OSError as e:
                    print(f"[motors] board error: {e}")
                    time.sleep(0.5)
            time.sleep(1 / MOTOR_RATE_HZ)

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)
        if self.board is not None:
            self.board.stop()         # always leave the motors off
            self.board.close()


def connect_board():
    """The motor board over USB if plugged in, otherwise I2C."""
    port = quad_md.find_usb_port()
    board = quad_md.QuadMDSerial(port) if port else quad_md.QuadMD()
    volts = board.battery_volts()
    print(f"Motor board found over {port or 'I2C'}. Motor battery: {volts:.1f} V")
    if volts < 5.0:
        print("Warning: motor battery looks off or flat - motors may not move.")
    board.stop()
    return board


# ═══════════════════════════════════════════════════════════════
#  WEB PAGE: video + map + keyboard driving
# ═══════════════════════════════════════════════════════════════

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>ROSS drive</title>
<style>
 body { margin: 0; background: #111; color: #ddd; font: 14px system-ui, sans-serif; }
 #wrap { display: flex; flex-wrap: wrap; gap: 12px; padding: 12px; }
 #view { flex: 1 1 640px; max-width: 100%; border: 1px solid #333; }
 #panel { flex: 0 0 260px; }
 .keys { display: grid; grid-template-columns: repeat(4, 44px); gap: 6px; margin: 8px 0 14px; }
 .k { border: 1px solid #555; border-radius: 6px; text-align: center; padding: 8px 0; }
 .k.on { background: #2d7dff; border-color: #2d7dff; color: #fff; }
 .blank { visibility: hidden; }
 pre { background: #000; padding: 8px; border-radius: 6px; white-space: pre-wrap; }
 #focus { color: #f90; }
</style></head><body>
<div id="wrap">
 <img id="view" src="/stream">
 <div id="panel">
  <b>Drive</b> <span id="focus">(click this page first)</span>
  <div class="keys">
   <div class="k blank"></div><div class="k" id="k-w">W</div><div class="k blank"></div><div class="k blank"></div>
   <div class="k" id="k-a">A</div><div class="k" id="k-s">S</div><div class="k" id="k-d">D</div><div class="k blank"></div>
  </div>
  <b>Single motor</b> (Shift = backward)
  <div class="keys">
   <div class="k" id="k-h">H M1</div><div class="k" id="k-j">J M2</div>
   <div class="k" id="k-k">K M3</div><div class="k" id="k-l">L M4</div>
  </div>
  Speed <b id="speed"></b> / 3600 &nbsp; (- / =) &nbsp; Space = stop
  <pre id="state">waiting...</pre>
 </div>
</div>
<script>
const KEYS = "wasdhjkl";
const held = new Set();
let shift = false, speed = __SPEED__;
const show = () => {
  for (const k of KEYS) document.getElementById("k-" + k).classList.toggle("on", held.has(k));
  document.getElementById("speed").textContent = speed;
};
const send = () => fetch("/drive", {method: "POST",
  body: JSON.stringify({keys: [...held], shift: shift, speed: speed})}).catch(() => {});
document.addEventListener("keydown", e => {
  const k = e.key.toLowerCase(); shift = e.shiftKey;
  if (KEYS.includes(k) && k.length === 1) { held.add(k); e.preventDefault(); }
  else if (k === " ") { held.clear(); e.preventDefault(); }
  else if (k === "-" || k === "_") speed = Math.max(300, speed - __STEP__);
  else if (k === "=" || k === "+") speed = Math.min(3600, speed + __STEP__);
  else return;
  show(); send();
});
document.addEventListener("keyup", e => {
  held.delete(e.key.toLowerCase()); shift = e.shiftKey; show(); send();
});
window.addEventListener("blur", () => { held.clear(); show(); send(); });  // stop if focus is lost
window.addEventListener("focus", () => document.getElementById("focus").textContent = "");
// While keys are held, keep repeating the command; the robot stops if these stop.
setInterval(() => { if (held.size) send(); }, 150);
setInterval(async () => {
  try {
    const s = await (await fetch("/state")).json();
    document.getElementById("state").textContent =
      `${s.status}\\nyaw ${s.yaw} deg   pos (${s.x}, ${s.z}) m\\npeople ${s.people}\\n` +
      `motor power [${s.pwm}]\\nencoders [${s.encoders}]\\nbattery ${s.battery}`;
  } catch (e) {}
}, 300);
show();
</script></body></html>"""


class ControlServer:
    """Serves the page, the MJPEG video stream, /state, and takes /drive commands."""

    def __init__(self, port, on_drive, get_state):
        self.jpeg = None
        self.new_frame = threading.Condition()
        server_self = self
        page = (PAGE.replace("__SPEED__", str(DEFAULT_SPEED))
                .replace("__STEP__", str(SPEED_STEP)).encode())

        class Handler(BaseHTTPRequestHandler):
            def _send(self, body, content_type):
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/state":
                    self._send(json.dumps(get_state()).encode(), "application/json")
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while True:
                            with server_self.new_frame:
                                server_self.new_frame.wait(timeout=2)
                                jpeg = server_self.jpeg
                            if jpeg is not None:
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                                 + jpeg + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self._send(page, "text/html; charset=utf-8")

            def do_POST(self):
                if self.path != "/drive":
                    self.send_error(404)
                    return
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    keys = {k for k in body.get("keys", []) if k in "wasdhjkl"}
                    speed = max(0, min(quad_md.PWM_MAX, int(body.get("speed", DEFAULT_SPEED))))
                    on_drive(keys, speed, bool(body.get("shift")))
                except (ValueError, TypeError):
                    pass
                self._send(b"ok", "text/plain")

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()

    def show(self, image):
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self.new_frame:
                self.jpeg = buf.tobytes()
                self.new_frame.notify_all()


# ═══════════════════════════════════════════════════════════════
#  MAIN LOOP  (rotate-klt-slam.py's loop + driving)
# ═══════════════════════════════════════════════════════════════

def get_local_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        socket.connect(("8.8.8.8", 80))
        ip = socket.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        socket.close()
    return ip


def run(args):
    rk.DETECTOR = args.detector
    camera = cv2.VideoCapture(rk.SOURCE) if args.laptop else rk.PiCamera()
    if not camera.isOpened():
        raise RuntimeError("Cannot open the camera")
    hfov = rk.HFOV_DEG if args.laptop else rk.PI_HFOV_DEG

    driver = Driver(None if args.no_motors else connect_board())
    state = {"status": "starting", "yaw": 0, "x": 0, "z": 0, "people": 0,
             "pwm": [0, 0, 0, 0], "encoders": [0, 0, 0, 0], "battery": "?"}

    def on_drive(keys, speed, shift):
        driver.command(*plan_motors(keys, speed, shift))

    view = ControlServer(args.port, on_drive, lambda: state)
    detect = rk.load_detector()
    tracker, pose, people_map = rk.KLTTracker(), rk.Pose(), rk.PeopleMap()
    trajectory = [(0.0, 0.0)]
    boxes, frame_i, t_prev = [], 0, None
    ip = get_local_ip()
    print(f"Open http://{ip}.local:{args.port}/ to watch and drive. Ctrl+C = stop + save.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            h0, w0 = frame.shape[:2]
            frame = cv2.resize(frame, (rk.WIDTH, round(h0 * rk.WIDTH / w0)))
            h, w = frame.shape[:2]
            f, cx, cy = rk.focal_from_fov(w, hfov), w / 2, h / 2
            now = time.monotonic()
            dt = min(now - t_prev, 0.2) if t_prev is not None else 0.0
            t_prev = now
            pwm, forward = driver.current()

            # 1. Turning from the camera (KLT), distance from the motors (or vision).
            tracks = tracker.track(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            status, turning = "LOST", False
            if tracks is not None:
                q1, q2 = tracks
                dyaw, expansion, flow_px = rk.estimate_motion(q1, q2, f, cx, cy)
                pose.yaw += dyaw
                turning = dt > 0 and abs(dyaw) / dt > math.radians(rk.TURN_THRESH_DEG_S)
                still = flow_px < rk.STILL_PX
                if MOTION == "commands":
                    if abs(forward) > 0.01 and still:
                        status = "STUCK? (motors on, view not moving)"
                    elif abs(forward) > 0.01:
                        pose.move_forward(forward * SPEED_AT_FULL_PWM * dt)
                        status = "FORWARD" if forward > 0 else "BACKWARD"
                    else:
                        status = "STILL" if still else "IDLE"
                elif not turning and abs(expansion) > rk.EXPANSION_THRESH:
                    pose.move_forward((1 if expansion > 0 else -1) * rk.WALK_SPEED * dt)
                    status = "FORWARD" if expansion > 0 else "BACKWARD"
                else:
                    status = "STILL" if still else "IDLE"
                if turning:
                    status = "TURNING " + ("RIGHT" if dyaw > 0 else "LEFT") + (
                        "" if status in ("STILL", "IDLE") else " + " + status)

                last_x, last_z = trajectory[-1]
                if math.hypot(pose.x - last_x, pose.z - last_z) > 0.02:
                    trajectory.append((pose.x, pose.z))

            # 2. People.
            if frame_i % rk.DETECT_EVERY == 0:
                boxes = []
                for box in detect(frame):
                    ahead, right = rk.locate_person(box, f, cx)
                    person = people_map.add(*pose.to_world(ahead, right))
                    boxes.append((box, person, math.hypot(ahead, right)))
            frame_i += 1

            # 3. Draw and publish.
            vis = frame.copy()
            if tracks is not None:
                color = (0, 0, 255) if turning else (0, 255, 0)
                for (u1, v1), (u2, v2) in zip(q1.astype(int), q2.astype(int)):
                    cv2.line(vis, (u1, v1), (u2, v2), color, 1)
            for (x1, y1, x2, y2), person, dist in boxes:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(vis, f"P{person['id']} ~{dist:.1f}m", (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(vis, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(vis, f"power {pwm}", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2)
            map_img = rk.draw_map(trajectory, pose, people_map)
            view.show(np.hstack((vis, cv2.resize(map_img, (h, h)))))

            state.update(status=status, yaw=round(math.degrees(pose.yaw)),
                         x=round(pose.x, 2), z=round(pose.z, 2),
                         people=len(people_map.confirmed()), pwm=pwm,
                         encoders=driver.encoders,
                         battery="?" if driver.battery is None else f"{driver.battery:.1f} V")
    except KeyboardInterrupt:
        pass
    finally:
        driver.close()                # motors off first
        camera.release()

    rk.OUTPUT_DIR.mkdir(exist_ok=True)
    out = rk.OUTPUT_DIR / f"klt_drive_map_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(str(out), rk.draw_map(trajectory, pose, people_map))
    print(f"\nMotors off. {len(people_map.confirmed())} people confirmed. Map saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KLT SLAM + drive from a web page")
    parser.add_argument("--detector", choices=["hailo", "yolo", "none"], default="hailo")
    parser.add_argument("--port", type=int, default=VIEW_PORT)
    parser.add_argument("--no-motors", action="store_true", help="run without the motor board")
    parser.add_argument("--laptop", action="store_true", help="webcam instead of the Pi camera")
    run(parser.parse_args())
