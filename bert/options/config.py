import torch

# Configuration
class ConfigBertBase:
    MODEL_NAME = 'bert-base-uncased'
    NUM_LABELS = 5  # SST-5 classes
    MAX_LENGTH = 128
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LORA = False
    CHECKPOINT = f"best_{MODEL_NAME}{'_lora' if LORA else ''}.pth"
    
# Configuration
class ConfigBertLarge:
    MODEL_NAME = 'bert-large-uncased'
    NUM_LABELS = 5  # SST-5 classes
    MAX_LENGTH = 128
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5
    EPOCHS = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LORA = False
    CHECKPOINT = f"best_{MODEL_NAME}{'_lora' if LORA else ''}.pth"
    
# Configuration
class ConfigBertBaseLora:
    MODEL_NAME = 'bert-base-uncased'
    NUM_LABELS = 5  # SST-5 classes
    MAX_LENGTH = 128
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LORA = True
    CHECKPOINT = f"best_{MODEL_NAME}{'_lora' if LORA else ''}.pth"
    
# Configuration
class ConfigBertLargeLora:
    MODEL_NAME = 'bert-large-uncased'
    NUM_LABELS = 5  # SST-5 classes
    MAX_LENGTH = 128
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5
    EPOCHS = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LORA = True
    CHECKPOINT = f"best_{MODEL_NAME}{'_lora' if LORA else ''}.pth"
