# 🚦 Advanced Lane Detection System

A real-time lane detection system using deep learning and computer vision for autonomous vehicles and driver assistance systems.

## ✨ Features
- **Real-time Processing**: 50-70 FPS on NVIDIA GPUs
- **Multi-Lane Detection**: Identifies up to 4 lanes with color coding
- **Adaptive Vision**: Works in various lighting/road conditions
- **Driver Feedback**: Visual overlay with lane departure warnings

<div align="center">
  <img src="https://i.imgur.com/JjX9gQe.jpg" width="45%">
  <img src="https://i.imgur.com/xyZQlYg.gif" width="45%">
</div>

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- NVIDIA GPU (recommended)
- Webcam or video input

```bash
# Clone repository
git clone https://github.com/yourusername/lane-detection.git
cd lane-detection

# Install dependencies
pip install -r requirements.txt

# Run with webcam
python lane_detection.py

🛠️ System Architecture
graph LR
    A[Video Input] --> B(Preprocessing)
    B --> C[Deep Learning Model]
    C --> D(Kalman Filter)
    D --> E[Visualization]
    E --> F[Performance Metrics]

Key Components
1.Preprocessing
i.Resizing (288×800)
ii.Normalization (ImageNet stats)

**2.Model Architecture**
parsingNet(
  backbone='resnet50',
  cls_dim=(101, 56, 4),  # (positions, rows, lanes)
  use_aux=True
)


**3.Postprocessing**

Confidence thresholding (>0.5)

Kalman filtering for smooth tracking


📂 Project Structure
lane-detection/
├── main.py                 # Core detection system
├── models/
│   ├── model.py            # parsingNet implementation
│   └── weights/            # Pretrained models
├── utils/
│   ├── kalman_filter.py    # Smoothing algorithm
│   └── visualizer.py       # Drawing functions
├── assets/                 # Sample videos/images
├── requirements.txt        # Dependencies
└── performance_report.txt  # Runtime metrics



🏎️ Performance
Metric	Value
Average FPS	58.7
Processing Time	17.03ms
Accuracy	94.2%

🛠 Customization
**Adjust Detection Sensitivity**

# Change confidence threshold
if np.max(prob) > 0.6:  # More strict
if len(lane_points) > 3:  # Fewer points needed

**Modify Visual Style**
self.lane_colors = [
    (255, 0, 0),   # Blue left lane
    (0, 255, 0),   # Green right lane
    (0, 0, 255),   # Red additional lanes
    (255, 255, 0)  # Cyan additional lanes
]

🤔 Troubleshooting
Issue	Solution
Low FPS	Reduce output resolution or use backbone='18'
Model Not Loading	Verify CUDA compatibility
False Positives	Increase confidence threshold

🤝 Contributing
Fork the repository

Create your feature branch (git checkout -b feature/improvement)

Commit your changes (git commit -am 'Add some feature')

Push to the branch (git push origin feature/improvement)

Open a Pull Request






















