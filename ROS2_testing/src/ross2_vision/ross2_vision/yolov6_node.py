import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import torch
import numpy as np
import sys
import os

# Force Python to find the YOLOv6 classes we just cloned
sys.path.append(os.path.join(os.path.expanduser('~'), 'ROSS2_ws', 'YOLOv6'))

class YoloV6Node(Node):
    def __init__(self):
        super().__init__('yolov6_node')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Loading YOLOv6 on device: {self.device}")
        
        self.weights_path = 'yolov6s.pt'
        try:
            # Security bypass added here
            self.model = torch.load(self.weights_path, map_location=self.device, weights_only=False)['model'].float().eval()
            self.get_logger().info("Model loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.publisher_ = self.create_publisher(Image, '/camera/image_yolo', 10)

    def image_callback(self, msg):
        cv_image = np.ndarray(shape=(msg.height, msg.width, 3), dtype=np.uint8, buffer=msg.data)
        
        img_tensor = self.preprocess_image(cv_image)
        with torch.no_grad():
            predictions = getattr(self, 'model', lambda x: None)(img_tensor)
            
        annotated_image = self.draw_boxes(cv_image.copy(), predictions)
        
        out_msg = Image()
        out_msg.header = msg.header
        out_msg.height = annotated_image.shape[0]
        out_msg.width = annotated_image.shape[1]
        out_msg.encoding = 'bgr8'
        out_msg.is_bigendian = 0
        out_msg.step = annotated_image.shape[1] * 3
        out_msg.data = annotated_image.tobytes()
        
        self.publisher_.publish(out_msg)

    def preprocess_image(self, img):
        img_resized = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().to(self.device)
        img_tensor /= 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        return img_tensor

    def draw_boxes(self, img, predictions):
        if predictions is None:
            return img

        # The model returns a tuple in eval mode; extract the main tensor
        if isinstance(predictions, (tuple, list)):
            pred_tensor = predictions[0]
        else:
            pred_tensor = predictions
            
        # Move data from GPU/CPU to a standard NumPy array [num_boxes, 85]
        pred = pred_tensor[0].cpu().numpy()
        
        # We scaled the image down to 640x640 for the AI. 
        # We must scale the resulting coordinates back up to fit your camera's actual resolution.
        h_orig, w_orig, _ = img.shape
        x_scale = w_orig / 640.0
        y_scale = h_orig / 640.0
        
        boxes = []
        confidences = []
        class_ids = []
        
        # Parse the YOLO tensor
        for row in pred:
            obj_conf = row[4]
            if obj_conf > 0.4: # Only parse confident detections
                classes_scores = row[5:]
                class_id = np.argmax(classes_scores)
                confidence = obj_conf * classes_scores[class_id]
                
                if confidence > 0.4:
                    cx, cy, w, h = row[0], row[1], row[2], row[3]
                    
                    # Convert YOLO center x,y to OpenCV top-left x,y
                    x = int((cx - w/2) * x_scale)
                    y = int((cy - h/2) * y_scale)
                    width = int(w * x_scale)
                    height = int(h * y_scale)
                    
                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))
        
        # Apply NMS to remove overlapping boxes on the same object
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.4, nms_threshold=0.45)
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                conf = confidences[i]
                
                # Draw the bounding box and label
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, f"Class ID {class_ids[i]} | {conf:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        cv2.putText(img, "YOLOv6 Inference Active", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img
def main(args=None):
    rclpy.init(args=args)
    node = YoloV6Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
