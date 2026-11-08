import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np

class ESP32CamNode(Node):
    def __init__(self):
        super().__init__('esp32_cam_node')
        self.stream_url = 'http://172.20.10.2:81/stream' 
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)
        self.get_logger().info(f"Connecting to ESP32-S3 stream at: {self.stream_url}")
        
        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open stream. Check IP and Wi-Fi.")
            
        self.timer = self.create_timer(0.05, self.timer_callback)

    def timer_callback(self):
        if not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if ret and frame is not None:
            # Manually package the ROS 2 Image message
            img_msg = Image()
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = 'camera_link'
            img_msg.height = frame.shape[0]
            img_msg.width = frame.shape[1]
            img_msg.encoding = 'bgr8'
            img_msg.is_bigendian = 0
            img_msg.step = frame.shape[1] * 3
            img_msg.data = frame.tobytes()
            
            self.publisher_.publish(img_msg)
        else:
            self.get_logger().warn("Dropped frame or stream interrupted")

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ESP32CamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
