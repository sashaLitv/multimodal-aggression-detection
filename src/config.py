import os
import sys
from PIL import ImageFont

# =====================================================================
# 1. PROJECT ROOT SETUP
# =====================================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# PROJECT_ROOT = "/content/drive/MyDrive/Cursova"
# SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =====================================================================
# 2. MAIN DIRECTORIES
# =====================================================================
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
MODELS_DIR = os.path.join(ASSETS_DIR, "models")

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
UI_PATH = os.path.join(SRC_DIR, "ui", "mainwindow.ui")
# =====================================================================
# 3. WEIGHTS AND MODELS PATHS
# =====================================================================
VIDEO_MODEL_PATH = os.path.join(MODELS_DIR, "model_video.weights.h5")
AUDIO_MODEL_PATH = os.path.join(MODELS_DIR, "model_audio.keras") 
FUSION_MODEL_PATH = os.path.join(MODELS_DIR, "fusion_mlp.weights.h5") 
VIDEO_MODEL_NEW_PATH = os.path.join(MODELS_DIR, "model_video_new.keras")
AUDIO_MODEL_NEW_PATH = os.path.join(MODELS_DIR, "model_audio_new.keras")
MODEL_TASK_PATH = os.path.join(MODELS_DIR, "pose_landmarker_heavy.task")
YOLO_PATH = os.path.join(MODELS_DIR, "yolo11m-pose.pt")
THRESHOLD_FILE = os.path.join(MODELS_DIR, "video_threshold.txt")

# =====================================================================
# 4. DATASET PATHS
# =====================================================================
VIDEO_DATA_DIR = os.path.join(DATASET_DIR, "video")
AUDIO_DATA_DIR = os.path.join(DATASET_DIR, "audio")
FUSION_DATA_DIR = os.path.join(DATASET_DIR, "fusion")

# =====================================================================
# 5. OUTPUTS AND PLOTS PATHS
# =====================================================================
PROCESSED_VIDEO_PARTS_DIR = os.path.join(OUTPUTS_DIR, "processed_video_parts")
PROCESSED_AUDIO_PATH = os.path.join(OUTPUTS_DIR, "processed_audio_parts/audio_data.npz")
PROCESSED_MANUALLY_CLEANED_VIDEO_PARTS_DIR = os.path.join(OUTPUTS_DIR, "processed_manually_cleaned_video_parts")

OUTPUT_PATH_DASHBOARD = os.path.join(OUTPUTS_DIR, "dashboard")
OUTPUT_PATH_FIGURES = os.path.join(OUTPUTS_DIR, "figures")
DIST_PLOT_FILE = os.path.join(OUTPUTS_DIR, "video_anomaly_distribution.png")
CM_PLOT_FILE = os.path.join(OUTPUTS_DIR, "video_confusion_matrix.png")

# =====================================================================
# 6. UTILS PATHS
# =====================================================================
COUNTS_FILE = os.path.join(PROJECT_ROOT, "src/utils/dataset_counts.json")

# =====================================================================
# 7. GLOBAL ANALYSIS CONSTANTS
# =====================================================================
SEQUENCE_LENGTH = 45
STEP = 3
HORIZON = 15
HISTORY_LEN = SEQUENCE_LENGTH - HORIZON
JOINT_COUNT = 13
FEATURE_DIM = 6
CONSECUTIVE = 2
GLOBAL_VEL_STD = 0.05668
GLOBAL_ACC_STD = 0.04303
KINETIC_CLIP_BOUND = 0.5

W_POSS = 1
W_VELL = 2
W_ACC = 3

SAMPLE_RATE = 22050
DURATION = 3            
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
AUDIO_THRESHOLD = 0.6 

# =====================================================================
# 8. UI CONFIG
# =====================================================================
FONT_SIZE_MAIN = 24
FONT_SIZE_SMALL = 14

try:
    font_main = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 54)
    font_small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 36)
except IOError:
    try:
        font_main = ImageFont.truetype("arial.ttf", FONT_SIZE_MAIN)
        font_small = ImageFont.truetype("arial.ttf", FONT_SIZE_SMALL)
    except IOError:
        print("Системний шрифт не знайдено, кирилиця може не відображатись!")
        font_main = ImageFont.load_default()
        font_small = ImageFont.load_default()

# =====================================================================
# 9. SECURITY CONFIG
# =====================================================================
SECRET_PIN = "2026" 

WEBHOOK_URL = "https://chat.googleapis.com/v1/spaces/AAQACokBzDw/messages?example_token=YOUR_WEBHOOK_TOKEN"  