# ============================================================
# VisionCanvas — Centralized Configuration
# ============================================================

# --- Camera ---
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_BUFFER_SIZE = 1  # Minimize frame buffering for low latency

# --- Hand Tracking ---
MAX_HANDS = 1
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.7
MODEL_PATH = "src/hand_tracking/hand_landmarker.task"

# --- One Euro Filter (Smoothing) ---
# mincutoff: lower = smoother but laggier. 1.0-10.0 good for hand tracking
# beta: higher = more responsive to fast movement. 0.01-0.5 typical
# dcutoff: derivative cutoff, usually left at 1.0
SMOOTHING_ENABLED = True
ONE_EURO_MIN_CUTOFF = 10.0
ONE_EURO_BETA = 0.05
ONE_EURO_DCUTOFF = 1.0

# --- Drawing ---
MAX_INTERPOLATION_DISTANCE = 5.0
MAX_CURSOR_JUMP = 150  # Ignore jumps larger than this (pixel distance)
BRUSH_DEFAULT_SIZE = 12
BRUSH_DEFAULT_COLOR = (180, 80, 255)  # BGR purple

# --- Canvas ---
CANVAS_WIDTH = 640   # Matched to camera for 1:1 mapping
CANVAS_HEIGHT = 480

# --- UI Window ---
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# --- Gesture Debounce ---
GESTURE_DEBOUNCE_TIME = 0.5  # seconds for undo/redo/save

# --- Performance / Debug ---
DEBUG_PERFORMANCE = False
TARGET_FPS = 30

# --- Hand Landmark Overlay ---
DRAW_HAND_LANDMARKS = True
LANDMARK_COLOR = (255, 120, 255)      # BGR neon pink
LANDMARK_CONNECTION_COLOR = (180, 80, 255)  # BGR purple
LANDMARK_DOT_RADIUS = 4
LANDMARK_LINE_THICKNESS = 2
