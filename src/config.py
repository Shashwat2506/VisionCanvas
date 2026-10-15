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
# mincutoff: lower = smoother at low speed. 0.8-1.5 eliminates jitter
# beta: higher = more responsive to fast movement. 0.03-0.08 is optimal
SMOOTHING_ENABLED = True
ONE_EURO_MIN_CUTOFF = 1.0
ONE_EURO_BETA = 0.06
ONE_EURO_DCUTOFF = 1.0

# --- Drawing ---
MAX_INTERPOLATION_DISTANCE = 4.0
MAX_CURSOR_JUMP = 200  # Pixel distance for glitch rejection
BRUSH_DEFAULT_SIZE = 4  # Fine, natural starting brush size (4px)
BRUSH_SIZES = [2, 4, 8, 14, 22]  # Fine, Normal, Medium, Thick, Marker
BRUSH_DEFAULT_COLOR = (180, 80, 255)  # BGR purple

# --- AI & Recognition ---
SMART_SHAPES_DEFAULT = True
SHAPE_CONFIDENCE_THRESHOLD = 0.75
TEXT_RECOGNITION_ENABLED = True

# --- Canvas ---
CANVAS_WIDTH = 640   # Base camera resolution
CANVAS_HEIGHT = 480

# --- UI Window ---
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# --- Gesture Debounce ---
GESTURE_DEBOUNCE_TIME = 0.4  # seconds for undo/redo/save

# --- Performance / Debug ---
DEBUG_PERFORMANCE = False
TARGET_FPS = 30

# --- Hand Landmark Overlay ---
DRAW_HAND_LANDMARKS = True
LANDMARK_COLOR = (255, 120, 255)      # BGR neon pink
LANDMARK_CONNECTION_COLOR = (180, 80, 255)  # BGR purple
LANDMARK_DOT_RADIUS = 3
LANDMARK_LINE_THICKNESS = 1

