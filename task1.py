import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import time
import matplotlib.pyplot as plt

class LaneDetectionSystem:
    def __init__(self, model_weights=None, video_source=0, output_size=(720, 1280)):
        """
        Initialize the lane detection system
        
        Args:
            model_weights: Path to pre-trained model weights
            video_source: Camera index or video file path
            output_size: Output display resolution (height, width)
        """
        # Hardware configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model initialization
        self.net = parsingNet(pretrained=False, backbone='50',
                            cls_dim=(101, 56, 4), use_aux=True).to(self.device)
        if model_weights:
            self.load_model(model_weights)
        self.net.eval()
        
        # Video configuration
        self.cap = cv2.VideoCapture(video_source)
        self.set_camera_resolution(1280, 720)  # HD resolution recommended
        self.output_size = output_size
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
        
        # Lane visualization settings
        self.lane_colors = [
            (0, 0, 255),   # Red - left lane
            (0, 255, 0),    # Green - right lane
            (255, 0, 0),    # Blue - additional lanes
            (0, 255, 255)   # Yellow - additional lanes
        ]
        
        # Performance metrics
        self.frame_count = 0
        self.fps = 0
        self.processing_times = []
        
        # Kalman filter for smooth lane tracking
        self.kalman_filters = [self.init_kalman_filter() for _ in range(4)]

    def load_model(self, model_path):
        """Load trained model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model'])
        print(f"Model loaded from {model_path}")

    def set_camera_resolution(self, width, height):
        """Set camera input resolution"""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.input_size = (width, height)

    def init_kalman_filter(self):
        """Initialize Kalman filter for lane tracking"""
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        return kf

    def preprocess_frame(self, frame):
        """Prepare frame for model input"""
        # Convert color space and resize
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        return self.transform(img_pil).unsqueeze(0).to(self.device)

    def postprocess_output(self, output, frame):
        """Convert model output to lane coordinates"""
        output = output.squeeze().cpu().numpy()
        lanes = []
        
        for lane_idx in range(output.shape[2]):
            lane_points = []
            for y in range(output.shape[1]):
                prob = output[:, y, lane_idx]
                if np.max(prob) > 0.5:  # Confidence threshold
                    x = np.argmax(prob)
                    # Convert to image coordinates
                    img_x = int(x * self.input_size[0] / 100)
                    img_y = int(y * self.input_size[1] / 56)
                    lane_points.append((img_x, img_y))
            
            if len(lane_points) > 5:  # Minimum points threshold
                lanes.append(lane_points)
        
        return self.filter_and_smooth_lanes(lanes, frame)

    def filter_and_smooth_lanes(self, lanes, frame):
        """Apply Kalman filtering to smooth lane detection"""
        smoothed_lanes = []
        for i, lane in enumerate(lanes[:4]):  # Process max 4 lanes
            if len(lane) < 2:
                continue
                
            # Convert to numpy array
            points = np.array(lane, dtype=np.float32)
            
            # Predict and correct with Kalman filter
            self.kalman_filters[i].predict()
            measurement = np.mean(points, axis=0)
            estimated = self.kalman_filters[i].correct(measurement)
            
            # Create smoothed lane
            x_start = int(estimated[0])
            x_end = int(estimated[0] + estimated[2] * 10)
            y_start = frame.shape[0]
            y_end = int(frame.shape[0] * 0.6)
            
            smoothed_lanes.append([(x_start, y_start), (x_end, y_end)])
        
        return smoothed_lanes

    def draw_lanes(self, frame, lanes):
        """Visualize detected lanes on the frame"""
        for i, lane in enumerate(lanes):
            if len(lane) == 2:
                cv2.line(frame, lane[0], lane[1], self.lane_colors[i], 5, cv2.LINE_AA)
        
        # Add info overlay
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Lanes: {len(lanes)}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame

    def calculate_curvature(self, lanes):
        """Calculate lane curvature for driver feedback"""
        if len(lanes) < 2:
            return 0
        
        left_lane, right_lane = lanes[0], lanes[1]
        # Calculate curvature using polynomial fitting
        # (Implementation depends on your coordinate system)
        return curvature

    def process_frame(self, frame):
        """Complete processing pipeline for one frame"""
        start_time = time.time()
        
        # Preprocess
        input_tensor = self.preprocess_frame(frame)
        
        # Model inference
        with torch.no_grad():
            output = self.net(input_tensor)
        
        # Postprocess
        lanes = self.postprocess_output(output, frame)
        
        # Visualize
        output_frame = self.draw_lanes(frame.copy(), lanes)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            self.fps = 10 / sum(self.processing_times[-10:])
        
        return output_frame, lanes

    def run(self):
        """Main processing loop"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                output_frame, lanes = self.process_frame(frame)
                
                # Display
                cv2.imshow('Lane Detection', cv2.resize(output_frame, self.output_size[::-1]))
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.save_performance_report()

    def save_performance_report(self):
        """Save system performance metrics"""
        avg_fps = self.frame_count / sum(self.processing_times)
        with open('performance_report.txt', 'w') as f:
            f.write(f"Total frames processed: {self.frame_count}\n")
            f.write(f"Average FPS: {avg_fps:.2f}\n")
            f.write(f"Average processing time: {1000*np.mean(self.processing_times):.2f}ms\n")
            f.write(f"Min/Max processing time: {1000*np.min(self.processing_times):.2f}ms/{1000*np.max(self.processing_times):.2f}ms\n")

if __name__ == "__main__":
    # Initialize with optional model weights
    detector = LaneDetectionSystem(
        model_weights="lane_detection_model.pth",
        video_source="highway.mp4",  # Or use 0 for webcam
        output_size=(720, 1280)
    )
    
    # Start processing
    detector.run()
