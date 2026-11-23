"""
Yahboom Quad MD (4-channel encoder motor driver) — Raspberry Pi motor test
==========================================================================

The board has its own STM32 chip that drives the motors and counts encoder
pulses. The Pi sends it short commands, either over a USB cable or over I2C,
using the protocol from https://www.yahboom.net/study/Quad-MD-Module
("Control command" page). It uses one or the other, not both at once.

CONNECTING (Pi off while wiring)
  USB (easiest): board USB-C -> a Pi USB port. The board shows up as
    /dev/ttyUSB0 and takes text commands like "$pwm:1000,0,0,0#".
  I2C: board SDA -> Pi pin 3, SCL -> pin 5, GND -> pin 6 (address 0x26).
    Leave the board's 5V pin unconnected while the Pi has its own USB-C
    supply, so two 5 V sources aren't tied together.
  Either way: motor battery -> the board's power input (5-12 V: about 12 V
  for 520 motors, 7.4 V for 310 / TT motors). Motor -> M1.
  The script uses USB if the board is plugged in, otherwise I2C
  (or force one with --usb / --i2c).

RUN (on the Pi)
  python3 motor_draft/quad_md.py                   # M1: forward, stop, backward, stop
  python3 motor_draft/quad_md.py --pwm 1800        # more power if it doesn't move
  python3 motor_draft/quad_md.py --type tt         # also configure motor type (encoders)
  python3 motor_draft/quad_md.py --motors 1 2 3 4  # later, all four
  Ctrl+C stops the motors at any time.
"""

import argparse
import glob
import struct
import time

ADDRESS = 0x26               # the board's I2C address
I2C_BUS = 1                  # Pi pins 3/5 are I2C bus 1 (/dev/i2c-1)

# Registers (from Yahboom's "Control command" page)
REG_MOTOR_TYPE = 0x01        # uint8: 1=520, 2=310, 3=TT with encoder, 4=TT without
REG_DEADZONE = 0x02          # uint16, 0-3600
REG_PULSE_LINES = 0x03       # uint16: encoder magnetic-ring lines
REG_GEAR_RATIO = 0x04        # uint16: gearbox reduction ratio
REG_WHEEL_DIAMETER = 0x05    # float, mm (little-endian, unlike the others)
REG_SPEED = 0x06             # 4 x int16, mm/s, -1000..1000 (encoder motors only)
REG_PWM = 0x07               # 4 x int16, -3600..3600 (any motor)
REG_BATTERY = 0x08           # uint16, volts x 10
REG_TOTAL_ENCODER = 0x20     # M1 = 0x20 (high word) + 0x21 (low word), M2 = 0x22 ...

PWM_MAX = 3600
SPEED_MAX = 1000

# Setup values Yahboom's own Pi example uses for each motor type
MOTOR_PRESETS = {
    "520": dict(type=1, ratio=30, lines=11, wheel=67.0, deadzone=1900),
    "310": dict(type=2, ratio=20, lines=13, wheel=48.0, deadzone=1600),
    "tt": dict(type=3, ratio=45, lines=13, wheel=68.0, deadzone=1250),
    "tt-noencoder": dict(type=4, ratio=48, deadzone=1000),
}


class QuadMD:
    """The board over I2C (register map)."""

    def __init__(self, bus=I2C_BUS, address=ADDRESS):
        try:
            from smbus2 import SMBus
        except ImportError:      # python3-smbus has the same functions
            from smbus import SMBus
        self.bus = SMBus(bus)
        self.address = address

    # ── low level ───────────────────────────────────────────────
    def _write(self, register, data):
        self.bus.write_i2c_block_data(self.address, register, list(data))
        time.sleep(0.01)     # give the board's chip a moment between commands

    def _read(self, register, length):
        return self.bus.read_i2c_block_data(self.address, register, length)

    # ── setup ───────────────────────────────────────────────────
    def configure(self, preset):
        """Tell the board which motor is attached (needed for encoders/speed)."""
        p = MOTOR_PRESETS[preset]
        self._write(REG_MOTOR_TYPE, [p["type"]])
        time.sleep(0.1)
        self._write(REG_DEADZONE, struct.pack(">H", p["deadzone"]))
        self._write(REG_GEAR_RATIO, struct.pack(">H", p["ratio"]))
        if "lines" in p:
            self._write(REG_PULSE_LINES, struct.pack(">H", p["lines"]))
            self._write(REG_WHEEL_DIAMETER, struct.pack("<f", p["wheel"]))
        time.sleep(0.1)

    # ── driving ─────────────────────────────────────────────────
    def set_pwm(self, m1=0, m2=0, m3=0, m4=0):
        """Raw power per motor, -3600..3600 (negative = backwards)."""
        values = [max(-PWM_MAX, min(PWM_MAX, int(v))) for v in (m1, m2, m3, m4)]
        self._write(REG_PWM, struct.pack(">4h", *values))   # 4 big-endian int16

    def set_speed(self, m1=0, m2=0, m3=0, m4=0):
        """Target speed in mm/s, -1000..1000. Needs encoder motors + configure()."""
        values = [max(-SPEED_MAX, min(SPEED_MAX, int(v))) for v in (m1, m2, m3, m4)]
        self._write(REG_SPEED, struct.pack(">4h", *values))

    def stop(self):
        self.set_pwm(0, 0, 0, 0)

    # ── reading ─────────────────────────────────────────────────
    def battery_volts(self):
        b = self._read(REG_BATTERY, 2)
        return ((b[0] << 8) | b[1]) / 10.0

    def encoder_total(self, motor):
        """Total encoder count since power-on for motor 1-4 (0 if no encoder)."""
        reg = REG_TOTAL_ENCODER + 2 * (motor - 1)
        high, low = self._read(reg, 2), self._read(reg + 1, 2)
        value = (((high[0] << 8) | high[1]) << 16) | ((low[0] << 8) | low[1])
        return value - (1 << 32) if value & 0x80000000 else value   # signed 32-bit

    def close(self):
        self.bus.close()


class QuadMDSerial:
    """The same board over its USB cable: text commands like "$pwm:a,b,c,d#".

    Same methods as QuadMD, so the rest of the script doesn't care which one
    it has.
    """

    BAUD = 115200

    def __init__(self, port):
        import serial            # pyserial
        self.port = serial.Serial(port, self.BAUD, timeout=0.1)
        time.sleep(0.2)
        self.port.reset_input_buffer()
        self.encoders = [0, 0, 0, 0]
        self._send("$upload:1,0,0#")   # ask for total encoder counts ($MAll:...#)

    def _send(self, command):
        self.port.write(command.encode("ascii"))
        time.sleep(0.01)

    def _read_lines(self, until=None, timeout=0.05):
        """Read the replies waiting right now; keep the latest encoder counts.

        Returns the first line starting with `until`, if asked for one. Never
        runs longer than `timeout` seconds: the board streams encoder counts
        non-stop, so "read until nothing is left" would never finish.
        """
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if not self.port.in_waiting:
                time.sleep(0.005)
                continue
            line = self.port.readline().decode("ascii", "ignore").strip()
            if line.startswith("$MAll:") and line.endswith("#"):
                try:
                    self.encoders = [int(v) for v in line[6:-1].split(",")][:4]
                except ValueError:
                    pass
            if until and line.startswith(until):
                return line
        return None

    def configure(self, preset):
        p = MOTOR_PRESETS[preset]
        commands = [f"$mtype:{p['type']}#", f"$deadzone:{p['deadzone']}#",
                    f"$mphase:{p['ratio']}#"]
        if "lines" in p:
            commands += [f"$mline:{p['lines']}#", f"$wdiameter:{p['wheel']:.2f}#"]
        for c in commands:
            self._send(c)
            time.sleep(0.1)
        self._read_lines()       # swallow the "...OK" replies

    def set_pwm(self, m1=0, m2=0, m3=0, m4=0):
        v = [max(-PWM_MAX, min(PWM_MAX, int(x))) for x in (m1, m2, m3, m4)]
        self._send("$pwm:{},{},{},{}#".format(*v))

    def set_speed(self, m1=0, m2=0, m3=0, m4=0):
        v = [max(-SPEED_MAX, min(SPEED_MAX, int(x))) for x in (m1, m2, m3, m4)]
        self._send("$spd:{},{},{},{}#".format(*v))

    def stop(self):
        self.set_pwm(0, 0, 0, 0)

    def battery_volts(self):
        self._send("$read_vol#")
        line = self._read_lines(until="$Battery:", timeout=1.0)   # "$Battery:7.40V#"
        if line is None:
            raise OSError("no reply to $read_vol# - is the board powered?")
        return float(line[len("$Battery:"):].rstrip("#").rstrip("Vv"))

    def encoder_total(self, motor):
        self._read_lines()
        return self.encoders[motor - 1]

    def close(self):
        self._send("$upload:0,0,0#")    # stop the encoder stream
        self.port.close()


def find_usb_port():
    """The board's USB serial device, e.g. /dev/ttyUSB0, or None."""
    ports = sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
    return ports[0] if ports else None


def run_motors(board, motors, pwm, seconds):
    """Spin the chosen motors forward, pause, backward, pause, printing encoders."""
    for label, direction in (("FORWARD", 1), ("stop", 0), ("BACKWARD", -1), ("stop", 0)):
        power = [pwm * direction if m in motors else 0 for m in (1, 2, 3, 4)]
        board.set_pwm(*power)
        print(f"{label:9s} pwm {power}")
        end = time.monotonic() + (seconds if direction else 1.0)
        while time.monotonic() < end:
            counts = "  ".join(f"M{m}: {board.encoder_total(m):+7d}" for m in motors)
            print(f"   encoders  {counts}", end="\r")
            time.sleep(0.25)
        print()


def main():
    parser = argparse.ArgumentParser(description="Yahboom Quad MD motor test")
    parser.add_argument("--motors", type=int, nargs="+", default=[1], choices=[1, 2, 3, 4],
                        help="which motor ports to spin (default: 1)")
    parser.add_argument("--pwm", type=int, default=1000,
                        help=f"power 0-{PWM_MAX} (default 1000, about 28%%)")
    parser.add_argument("--seconds", type=float, default=2.0, help="how long each direction")
    parser.add_argument("--type", choices=sorted(MOTOR_PRESETS),
                        help="configure the motor type first (needed for encoder readings)")
    link = parser.add_mutually_exclusive_group()
    link.add_argument("--usb", nargs="?", const="auto", metavar="PORT",
                      help="talk over the USB cable (default if plugged in)")
    link.add_argument("--i2c", action="store_true", help="talk over I2C pins 3/5")
    args = parser.parse_args()

    port = None if args.i2c else (find_usb_port() if args.usb in (None, "auto") else args.usb)
    if args.usb and port is None:
        raise SystemExit("No USB serial device found (ls /dev/ttyUSB*). Is the USB-C cable in?")
    try:
        board = QuadMDSerial(port) if port else QuadMD()
        volts = board.battery_volts()
    except OSError as e:
        if port:
            raise SystemExit(f"Can't talk to the board on {port} ({e}).\n"
                             "Check the cable and that the motor battery is on. If it says "
                             "'Permission denied': sudo usermod -aG dialout $USER, then log in again.")
        raise SystemExit(f"Can't reach the board at I2C address 0x{ADDRESS:02x} ({e}).\n"
                         "Check SDA/SCL/GND wiring, that I2C is enabled, and run: i2cdetect -y 1")

    print(f"Board found over {port or 'I2C'}. Motor battery: {volts:.1f} V")
    if volts < 5.0:
        print("Warning: the motor battery looks off or flat - motors may not move.")
    if args.type:
        board.configure(args.type)
        print(f"Configured motor type: {args.type}")

    try:
        run_motors(board, args.motors, abs(args.pwm), args.seconds)
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C")
    finally:
        board.stop()          # always leave the motors off
        board.close()
        print("Motors off.")


if __name__ == "__main__":
    main()
