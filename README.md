# VisionCanvas 🎨🖐️

VisionCanvas is a real-time, gesture-controlled air-canvas application powered by **OpenCV** and **MediaPipe**. Draw, erase, select colors, and switch brushes seamlessly in the air using natural hand gestures through your webcam.

---

## ✨ Features

- **Real-Time Hand Tracking:** Powered by MediaPipe's Hand Landmarker model for fast, accurate tracking.
- **Multiple Brush Presets:**
  - Hard Brush
  - Soft Brush
  - Neon Brush
  - Pencil Brush
  - Watercolor Brush
- **Smooth Stroke Interpolation:** Integrated **One Euro Filter** reduces jitter while maintaining low-latency responsiveness.
- **Intuitive Gestures & Shortcuts:**
  - Index finger pointing to draw / interact.
  - Quick keyboard hotkeys for canvas controls, brush switching, and canvas snapshots.
- **Save & Export:** Save drawings directly to the `output/` directory in PNG format.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+ (or compatible virtual environment)
- A working webcam

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shashwat2506/VisionCanvas.git
   cd VisionCanvas
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # Windows (cmd):
   venv\Scripts\activate.bat
   # Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 Running the Application

Run the main application script:

```bash
python main.py
```
*(Or directly with your venv on Windows: `.\venv\Scripts\python.exe main.py`)*

---

## ⌨️ Controls & Keyboard Shortcuts

| Key | Action |
| :--- | :--- |
| `b` | Switch to Brush mode |
| `e` | Switch to Eraser mode |
| `1` - `5` | Switch brush preset (Hard, Soft, Neon, Pencil, Watercolor) |
| `[` / `]` | Decrease / Increase brush thickness |
| `c` | Clear canvas |
| `z` | Undo stroke |
| `y` | Redo stroke |
| `s` | Save canvas snapshot to `output/` |
| `q` / `Esc` | Quit application |

---

## 📁 Project Structure

```
VisionCanvas/
├── assets/                  # Project assets
├── docs/                    # Documentation
├── output/                  # Saved drawings (PNG)
├── src/
│   ├── brushes/             # Brush implementations (Hard, Soft, Neon, etc.)
│   ├── camera/              # Webcam capture abstraction
│   ├── canvas/              # Drawing canvas logic and stroke rendering
│   ├── gestures/            # Gesture recognition and virtual cursor
│   ├── hand_tracking/       # MediaPipe HandLandmarker wrapper & task file
│   ├── ui/                  # UI components and layout
│   ├── utils/               # Smoothing filter (One Euro Filter)
│   └── config.py            # Global project configuration
├── main.py                  # Application entry point
├── requirements.txt         # Project dependencies
└── README.md
```

---

## 📄 License

This project is licensed under the MIT License - feel free to use and modify it for your projects!
