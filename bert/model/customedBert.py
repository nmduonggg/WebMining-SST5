import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

class CustomedBert(nn.Module):
    def __init__(self, config, num_labels, lora=False):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        
        self.encoder = BertModel.from_pretrained(self.config)
        # self.classifier = nn.Sequential(
        #     nn.Linear()
        # )
        
    def lora_setup(self):
        pass
        
    def encode(self, **kwargs):
        return self.encoder(**kwargs)
    
    def forward(self, encoding, label):
        features = self.encode(**encoding)
        print(features.shape)
        
        