import torch

# Configuration
class ConfigBertBase:
    MODEL_NAME = 'bert-base-uncased'
    NUM_LABELS = 5  # SST-5 classes
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-5
    EPOCHS = 5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT = './best_bert_sst5_model.pth'
    LORA = False
    
# Configuration
class ConfigBertLarge:
    MODEL_NAME = 'bert-large-uncased'
    NUM_LABELS = 5  # SST-5 classes
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-5
    EPOCHS = 5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT = './best_bert_sst5_model.pth'
    LORA = False
    
# Configuration
class ConfigBertBaseLora:
    MODEL_NAME = 'bert-base-uncased'
    NUM_LABELS = 5  # SST-5 classes
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-5
    EPOCHS = 5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT = './best_bert_sst5_model.pth'
    LORA = True
    
# Configuration
class ConfigBertLargeLora:
    MODEL_NAME = 'bert-large-uncased'
    NUM_LABELS = 5  # SST-5 classes
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-5
    EPOCHS = 5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT = './best_bert_sst5_model.pth'
    LORA = True