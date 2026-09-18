import os
import torch

DATA_DIR = os.path.join("data", "rice_leaf_diseases")

SAVE_DIR = "weights"
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_mobilenetv4.pth")
ONNX_EXPORT_PATH = os.path.join(SAVE_DIR, "mobilenetv4_quantized.onnx")

CLASS_NAMES = [
    "Bacterial_Leaf_Blight",
    "Brown_Spot",
    "Rice_Blast",
    "Healthy"
]

NUM_CLASSES = len(CLASS_NAMES)

MODEL_NAME = "mobilenetv4_conv_medium"

IMAGE_SIZE = 512

BATCH_SIZE = 16

NUM_EPOCHS = 15

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2

LABEL_SMOOTHING = 0.1

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")