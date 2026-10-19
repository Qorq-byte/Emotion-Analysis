from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

TOKENIZER_MODE = "jieba"  # "jieba" or "char"; changing it requires reprocessing
SEQ_LEN = 128
MIN_TOKEN_FREQ = 2
MAX_VOCAB_SIZE = 30000
SAMPLE_FRACTION = 0.1
TEST_FRACTION = 0.2
VALIDATION_FRACTION = 0.2  # split from train.jsonl
SEED = 42

BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.3
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
EPOCHS = 20
PATIENCE = 4
