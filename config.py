import os

# Dataset Configuration
RAW_DATASET_DIR = "Dataset/raw"  # Put your dataset here
PROCESSED_DATASET_DIR = "Dataset/processed"

# Model Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
INITIAL_EPOCHS = 25
FINE_TUNE_EPOCHS = 25
LEARNING_RATE = 0.001
FINE_TUNE_LR = 0.0001

# Data Split Configuration
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Model Architecture
BASE_MODEL = 'MobileNetV2'
DENSE_UNITS = 256
DROPOUT_RATE = 0.5
FINE_TUNE_LAYERS = 50

# Training Configuration
EARLY_STOPPING_PATIENCE = 10
LR_REDUCTION_PATIENCE = 5
LR_REDUCTION_FACTOR = 0.2

# File Paths
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/cattle_breed_model.h5"
BEST_MODEL_PATH = f"{MODEL_DIR}/best_cattle_breed_model.h5"
TFLITE_MODEL_PATH = f"{MODEL_DIR}/cattle_breed_model.tflite"
CLASS_INDICES_PATH = f"{MODEL_DIR}/class_indices.json"
LABELS_PATH = f"{MODEL_DIR}/labels.txt"

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATASET_DIR, exist_ok=True)
