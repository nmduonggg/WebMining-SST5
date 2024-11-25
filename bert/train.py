import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm

import loss
from options import parsing
from data import SSTDataset
from eval import evaluate_model
from model import CustomedBert

# Training Function
def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, device, epochs, Config=None, outdir='./', scheduler=None):
    best_val_accuracy = 0
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            loss = criterion(outputs, labels)
            
            train_losses.append(loss.item())
            
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, total=len(val_dataloader)):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids, 
                    attention_mask=attention_mask
                )
                
                predictions = torch.argmax(outputs, dim=1)
                val_predictions.extend(predictions.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
        
        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        if scheduler is not None: 
            scheduler.step(val_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training Loss: {np.mean(train_losses):.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        os.makedirs(outdir, exist_ok=True)
        if val_accuracy > best_val_accuracy:
            print("Update best acc: ", val_accuracy)
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(outdir, Config.CHECKPOINT))
    
    return model


def main():

    # Argument
    args, Config = parsing()
    
    # Initialize tokenizer and model
    model = CustomedBert(
        config=Config.MODEL_NAME, 
        num_labels=Config.NUM_LABELS,
        lora=args.lora, 
        full_finetune=args.full_finetune
    ).to(Config.DEVICE)
    
    for n, p in model.named_parameters():
        if p.requires_grad: 
            print(n, end='\t')
    print()
    
    # Create datasets and dataloaders

    train_dataset = SSTDataset(split="train", binary=False, model_name=Config.MODEL_NAME, max_length=Config.MAX_LENGTH)
    val_dataset = SSTDataset(split="dev", binary=False, model_name=Config.MODEL_NAME, max_length=Config.MAX_LENGTH)
    test_dataset = SSTDataset(split="test", binary=False, model_name=Config.MODEL_NAME, max_length=Config.MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Optimizer and Loss
    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': Config.LEARNING_RATE},
        {'params': model.classifier.parameters()}
    ], lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    # criterion = nn.CrossEntropyLoss()
    criterion = loss.FocalLoss(gamma=5)
    
    # Train model
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        criterion, 
        Config.DEVICE, 
        Config.EPOCHS,
        Config,
        args.outdir,
        scheduler
    )
    
    # Evaluate model
    predictions, true_labels = evaluate_model(
        trained_model, 
        test_loader, 
        Config.DEVICE
    )

if __name__ == '__main__':
    main()
