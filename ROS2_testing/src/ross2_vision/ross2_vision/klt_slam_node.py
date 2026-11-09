import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import cv2
import numpy as np


class KLTSlamNode(Node):
    def __init__(self):
        super().__init__('klt_slam_node')
        self.get_logger().info("Initializing ROS 2 KLT-SLAM Node...")

        # Queue size 1 drops stale frames to keep optical flow real-time
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 1
        )
        self.image_pub = self.create_publisher(Image, '/camera/image_slam', 1)
        self.pose_pub = self.create_publisher(PoseStamped, '/slam/pose', 10)

        # Shi-Tomasi corner detection parameters
        self.feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )

        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # SLAM state variables
        self.prev_gray = None
        self.prev_pts = None
        self.K = None  # Estimated on first frame based on resolution

        # Global pose state (Rotation matrix R and translation vector t)
        self.R_total = np.eye(3, dtype=np.float64)
        self.t_total = np.zeros((3, 1), dtype=np.float64)
        self.yaw_deg = 0.0

        # Mini-map canvas for 2D top-down trajectory visualization
        self.map_size = 90
        self.traj_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)

    def estimate_calibration_matrix(self, width, height):
        """Estimates pinhole camera intrinsics K for the ESP32-S3 OV2640 lens."""
        focal_length = 0.8 * width  # ~65-70 deg FOV approximation
        cx = width / 2.0
        cy = height / 2.0
        return np.array([
            [focal_length, 0.0, cx],
            [0.0, focal_length, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def image_callback(self, msg):
        # 1. Unpack ROS 2 Image message without cv_bridge
        frame = np.ndarray(
            shape=(msg.height, msg.width, 3),
            dtype=np.uint8,
            buffer=msg.data
        ).copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize calibration matrix K on the first incoming frame
        if self.K is None:
            self.K = self.estimate_calibration_matrix(msg.width, msg.height)
            self.get_logger().info(
                f"Estimated Intrinsics K for {msg.width}x{msg.height}: f={self.K[0,0]:.1f}"
            )

        # Initialize features on the first frame or replenish if features drop below 60
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 60:
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            self.prev_gray = gray
            self.publish_outputs(frame, msg.header, 0)
            return

        # 2. Calculate Kanade-Lucas-Tomasi (KLT) Optical Flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )

        annotated = frame.copy()
        num_tracked = 0

        if curr_pts is not None and status is not None:
            good_old = self.prev_pts[status.flatten() == 1]
            good_new = curr_pts[status.flatten() == 1]
            num_tracked = len(good_new)

            # Draw optical flow tracks (motion vectors)
            for new, old in zip(good_new, good_old):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                cv2.line(annotated, (a, b), (c, d), (0, 255, 0), 2)
                cv2.circle(annotated, (a, b), 3, (0, 0, 255), -1)

            # 3. Estimate Rotation & Pose via Essential Matrix
            if num_tracked >= 8:
                try:
                    pts_new_reshaped = good_new.reshape(-1, 1, 2)
                    pts_old_reshaped = good_old.reshape(-1, 1, 2)

                    E, mask = cv2.findEssentialMat(
                        pts_new_reshaped,
                        pts_old_reshaped,
                        self.K,
                        method=cv2.RANSAC,
                        prob=0.999,
                        threshold=1.0
                    )

                    if E is not None and E.shape == (3, 3):
                        _, R, t, _ = cv2.recoverPose(
                            E, pts_new_reshaped, pts_old_reshaped, self.K, mask=mask
                        )

                        # Filter out noise when camera is nearly stationary
                        pixel_motion = np.mean(np.linalg.norm(good_new - good_old, axis=2))
                        if pixel_motion > 1.5:
                            scale = min(pixel_motion * 0.05, 0.5)
                            self.t_total = self.t_total + scale * (self.R_total @ t)
                            self.R_total = R @ self.R_total

                            # Extract Yaw (rotation around Y-axis in camera optical frame)
                            self.yaw_deg = math.degrees(
                                math.atan2(self.R_total[0, 2], self.R_total[2, 2])
                            )
                except cv2.error:
                    pass

            # Update previous state and replenish features if running low
            if num_tracked < 80:
                new_detected = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
                if new_detected is not None:
                    self.prev_pts = np.vstack((good_new.reshape(-1, 1, 2), new_detected))[:200]
                else:
                    self.prev_pts = good_new.reshape(-1, 1, 2)
            else:
                self.prev_pts = good_new.reshape(-1, 1, 2)

        self.prev_gray = gray
        self.publish_outputs(annotated, msg.header, num_tracked)

    def draw_hud_and_minimap(self, img, num_tracked):
        # Text telemetry overlay
        cv2.putText(
            img, f"KLT Features: {num_tracked}", (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
        )
        cv2.putText(
            img, f"Est. Yaw: {self.yaw_deg:.1f} deg", (5, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1
        )

        # Update 2D Top-Down Trajectory Minimap in top-right corner
        cx, cy = self.map_size // 2, self.map_size // 2
        draw_x = int(cx + self.t_total[0, 0] * 4.0) % self.map_size
        draw_y = int(cy - self.t_total[2, 0] * 4.0) % self.map_size

        # Slowly fade old trajectory and plot current position + heading vector
        self.traj_map = cv2.addWeighted(self.traj_map, 0.99, np.zeros_like(self.traj_map), 0.01, 0)
        cv2.circle(self.traj_map, (draw_x, draw_y), 2, (0, 255, 255), -1)

        map_overlay = self.traj_map.copy()
        yaw_rad = math.radians(self.yaw_deg)
        hx = int(draw_x + 15 * math.sin(yaw_rad))
        hy = int(draw_y - 15 * math.cos(yaw_rad))
        cv2.arrowedLine(map_overlay, (draw_x, draw_y), (hx, hy), (0, 0, 255), 2)
        cv2.rectangle(map_overlay, (0, 0), (self.map_size - 1, self.map_size - 1), (255, 255, 255), 1)
        cv2.putText(map_overlay, "2D Pose / Heading", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Overlay minimap onto the frame if frame is large enough
        h, w, _ = img.shape
        if h > self.map_size + 20 and w > self.map_size + 20:
            img[10:10 + self.map_size, w - self.map_size - 10:w - 10] = map_overlay

        return img

    def rotmat_to_quaternion(self, R):
        """Converts a 3x3 rotation matrix to (x, y, z, w) quaternion using pure NumPy."""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        return x, y, z, w

    def publish_outputs(self, annotated_img, header, num_tracked):
        # 1. Publish annotated image to /camera/image_slam
        final_img = self.draw_hud_and_minimap(annotated_img, num_tracked)

        out_msg = Image()
        out_msg.header = header
        out_msg.height = final_img.shape[0]
        out_msg.width = final_img.shape[1]
        out_msg.encoding = 'bgr8'
        out_msg.is_bigendian = 0
        out_msg.step = final_img.shape[1] * 3
        out_msg.data = final_img.tobytes()
        self.image_pub.publish(out_msg)

        # 2. Publish 3D PoseStamped to /slam/pose
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = 'odom'
        pose_msg.pose.position.x = float(self.t_total[0, 0])
        pose_msg.pose.position.y = float(self.t_total[1, 0])
        pose_msg.pose.position.z = float(self.t_total[2, 0])

        qx, qy, qz, qw = self.rotmat_to_quaternion(self.R_total)
        pose_msg.pose.orientation.x = float(qx)
        pose_msg.pose.orientation.y = float(qy)
        pose_msg.pose.orientation.z = float(qz)
        pose_msg.pose.orientation.w = float(qw)
        self.pose_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = KLTSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
