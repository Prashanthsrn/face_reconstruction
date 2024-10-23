# 3D Face Reconstruction

A real-time facial landmark detection and 3D model generation tool using MediaPipe and OpenCV. This application captures facial landmarks through your webcam and allows you to export them as 3D models in OBJ format.

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python face_reconstruction_main.py
```

### Controls
- Press 's' to capture and save the current face mesh as a 3D OBJ file
- Press 'q' to quit the application

The 3D models will be saved in an `output` directory with timestamps (e.g., `face_mesh_20240323-143021.obj`).

## Features

- Real-time face mesh detection
- Live FPS counter
- One-click 3D model export
- Automatic mesh topology generation
- Vertex normal calculation
- Properly scaled and positioned face mesh

## Requirements

See `requirements.txt` for detailed dependencies. Main requirements:
- OpenCV
- MediaPipe
- NumPy

## Output

Generated OBJ files include:
- 3D vertices
- Face topology
- Vertex normals

## Limitations

- Single face detection only - cannot process multiple faces simultaneously
- Performance depends heavily on lighting conditions
- Limited depth accuracy due to single camera input
- Face must be directly facing the camera for best results
- May struggle with extreme facial expressions or head rotations
- No texture mapping support in the OBJ export
- Real-time performance depends on system hardware capabilities
