# VisionCanvas 🎨🖐️

VisionCanvas is a real-time, gesture-controlled air-canvas application powered by **OpenCV** and **MediaPipe**. Draw, erase, select colors, and switch brushes seamlessly in the air using natural hand gestures through your webcam.

---

## ✨ Features

- **Real-Time Hand Tracking:** Powered by MediaPipe's Hand Landmarker model for fast, accurate tracking.
- **AI Handwriting-to-Text Conversion (Text AI):**
  - Air-draw letters (A–Z) and numbers (0–9) or words in the air with index finger.
  - Automatically recognizes handwriting and presents an interactive review card with Accept, Retry, and Dismiss options.
  - Converts rough air-drawn strokes into clean, TrueType typeset glyphs (Arial, Calibri, Times New Roman, Segoe UI) preserving color, scaling, and position.
  - Full collapsible undo/redo support.
- **Smart Shape Beautification:** Real-time geometric circle, rectangle, line, and triangle recognition and auto-snapping.
- **Multiple Brush Presets:**
  - Hard Brush, Soft Brush, Neon Glow Brush, Pencil Brush, Watercolor Brush.
- **Smooth Stroke Interpolation:** Integrated **One Euro Filter** eliminates jitter while maintaining low latency.
- **Interactive Studio UI & Focus Mode:** Glassmorphism UI panel, color wheel swatches, brush size sliders, and borderless canvas Focus Mode.
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

```powershell
.\venv\Scripts\python.exe main.py
```
*(Or `python main.py` with activated venv)*

---

## ⌨️ Controls & Keyboard Shortcuts

| Key | Action |
| :--- | :--- |
| `t` | Switch to **Text AI Mode** (Handwriting-to-Typeset conversion) |
| `b` | Switch to **Brush Mode** (or toggle Bold in Text mode) |
| `i` | Toggle Italic in Text mode |
| `e` | Switch to **Eraser Mode** |
| `s` | Toggle **Smart Shapes** (Auto-beautification) |
| `f` | Toggle **Focus Mode** (Canvas maximized, collapsed sidebars) |
| `Enter` | Accept recognized text / trigger text recognition |
| `Space` | Manually trigger recognition on drawn strokes |
| `r` | Retry recognized text (or Redo in normal mode) |
| `1` - `5` | Switch brush preset (Hard, Soft, Neon, Pencil, Watercolor) |
| `[` / `]` | Decrease / Increase brush thickness |
| `c` | Clear canvas |
| `z` | Undo stroke |
| `y` | Redo stroke |
| `x` | Save/export snapshot to `output/` |
| `q` / `Esc` | Dismiss text review / Quit application |

---

## 🧪 Automated Testing

VisionCanvas includes a full automated test suite with 45+ unit tests and 16 black-box GUI automation scenarios:

```powershell
# Run unit tests
.\venv\Scripts\pytest -v

# Run black-box integration tests and generate HTML report
.\venv\Scripts\python run_black_box_tests.py
```
Test report output: [`reports/black_box_test_report.html`](reports/black_box_test_report.html)

---

## 📁 Project Structure

```
VisionCanvas/
├── assets/                  # UI assets and snapshots
├── output/                  # Exported canvas drawings (PNG)
├── reports/                 # Black-box test reports (HTML)
├── src/
│   ├── brushes/             # Brush implementations (Hard, Soft, Neon, etc.)
│   ├── camera/              # Webcam capture abstraction with mock fallback
│   ├── canvas/              # Drawing canvas, FontManager, Shape & Text recognizers
│   ├── gestures/            # Gesture recognition and virtual cursor
│   ├── hand_tracking/       # MediaPipe HandLandmarker wrapper & task file
│   ├── ui/                  # Studio UI components and layout
│   ├── utils/               # One Euro Filter smoothing
│   └── config.py            # Global project configuration
├── tests/                   # Pytest suite (Unit & Black-Box GUI tests)
├── main.py                  # Application entry point
├── run_black_box_tests.py   # Black-box test runner & report generator
├── requirements.txt         # Project dependencies
└── README.md
```

---

## 📄 License

This project is licensed under the MIT License - feel free to use and modify it for your projects!
