import torch

FINE_TUNED_MODEL_PATH = "./models/fine_tuned_model"
DISTILBERT_BASE = "distilbert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64
POSITIVE_SENTIMENT_THRESHOLD = 7.5
NEUTRAL_SENTIMENT_THRESHOLD = 4.0
NUM_EPOCHS = 5
DATASET_PATH = "datasets/AWARE_Comprehensive.csv"