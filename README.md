# Interactive AR Hand Tracker
<img width="1392" alt="Screenshot 2025-06-09 at 1 25 42 PM" src="https://github.com/user-attachments/assets/769410cc-f918-4d20-a045-0ea5acc9431b" />

---

**University project for Augmented Reality (AR) with hand gesture control**

An interactive augmented reality system that allows real-time control of 3D objects using hand gestures through a laptop camera. The project is developed using computer vision and machine learning technologies.

## Features

### Gesture Control
- **Object Movement** - using index finger
- **Size Adjustment** - with "zoom" gesture (thumb + index finger)
- **3D Rotation** - with open palm and control across all axes
- **Object Switching** - with peace sign ✌️ (thumb + index + middle finger)


### 3D Model Support
- **Custom models**: automatic loading of .obj files from `models/` folder
- **OBJ format**: full support for vertices, faces, and normals
- **Automatic normalization**: scaling models to standard size
- **Triangulation**: automatic breakdown of complex polygons into triangles

### Activation Delay System
- **1-second delay** to prevent accidental gesture activation
- **Visual progress bar** shows activation process
- **Status indicators**: [WAITING] / [ACTIVE]
- **Smooth transitions** between control modes

## Technologies

- **Python 3.7+**
- **OpenCV** - computer vision and image processing
- **MediaPipe** - real-time hand detection and tracking
- **NumPy** - mathematical calculations for 3D transformations
- **Custom OBJ loader** - 3D model parsing

## Installation

### Requirements
- Python 3.7 or higher
- Webcam
- macOS / Windows / Linux

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/om1ji/hand-tracker.git
cd hand-tracker
```

2. **Create virtual environment:**
```bash
python -m venv ar_env
source ar_env/bin/activate  # On Windows: ar_env\Scripts\activate
```

3. **Install dependencies:**
```bash
pip3 install -r requirements.txt
```

4. **Run the program:**
```bash
python3 interactive_ar_hand.py
```

## Hand Gestures (hold for 1 second to activate):

| Gesture | Description | Function |
|---------|-------------|----------|
| 👆 **1 finger** | Only index finger raised | Object movement |
| 🤏 **Zoom** | Thumb + index finger | Size adjustment |
| ✋ **Open palm** | All fingers raised | 3D object rotation |
| ✌️ **Peace sign** | Thumb + index + middle | Object type switching |

### 3D Rotation Control:
- **ROLL** (rotation): rotate the thumb ↔ pinky line
- **PITCH** (tilt): move index finger closer/farther from wrist  
- **YAW** (turn): change distance between thumb and pinky

### Hotkeys:
- **ESC** - exit program

## 📁 Project Structure

```
interactive-ar-hand-tracker/
├── interactive_ar_hand.py    # Main program
├── obj_loader.py            # 3D model loader
├── requirements.txt         # Python dependencies
├── models/                  # Folder for .obj files
│   ├── monkey.obj          # Example: Suzanne monkey from Blender
│   └── star.obj            # Example: star
└── README.md               # This file
```

## Adding Custom 3D Models

1. **Export model from Blender to .obj format:**
   - File → Export → Wavefront (.obj)
   - Enable options: Write Normals, Triangulate Faces
   - Recommended size: 1-2 Blender units

2. **Place .obj file in `models/` folder**

3. **Restart the program** - model will load automatically


## Configuration

### Code parameters:
- `gesture_activation_delay` - gesture activation delay time (default: 15 frames ≈ 1 sec)
- `focal_length` - focal length for perspective projection (default: 600)
- `camera_distance` - camera distance from object (default: 400)
- Rotation sensitivity for each axis


## Troubleshooting

### Gesture recognition issues:
- Ensure good lighting (!!!)
- Keep hand 30-100 cm from camera
- Avoid complex background behind hand


## Technical Specifications

- **Frame rate**: up to 15 FPS (hardware dependent)
- **Detection accuracy**: 21 hand keypoints (MediaPipe)
- **Model support**: unlimited .obj files
- **Maximum model complexity**: ~10000 polygons for smooth operation
- **Control latency**: < 50ms after gesture activation
