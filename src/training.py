import os
import sys
from src.logger_config import logger
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from src.config import BASE_PATH

def train_linear_probe(df, target_col, model_name):
    X = np.array(df['embedding'].tolist(), dtype=np.float32)
    
    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(df[target_col].values)
    num_classes = len(le.classes_)
    
    logger.info(f'training {model_name} ({num_classes} classes, {len(X)} samples)...')
    
    # normalize
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    X = (X - mean) / std
    
    # tensors
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader  = DataLoader(dataset, batch_size=64, shuffle=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # simple linear layer on top of ViT features
    model = nn.Sequential(
        nn.Linear(768, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # training loop
    for epoch in range(50):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch+1) % 10 == 0:
            logger.info(f'  epoch {epoch+1}/50 — loss: {total_loss/len(loader):.4f}')
    
    # save everything needed for prediction
    from src.config import MODEL_PATH
    os.makedirs(MODEL_PATH, exist_ok=True)
    joblib.dump({'mean': mean, 'std': std, 'le': le, 'num_classes': num_classes},
                f'{MODEL_PATH}/{model_name}_meta.pkl')
    torch.save(model.state_dict(), f'{MODEL_PATH}/{model_name}_model.pt')
    
    logger.info(f'{model_name} done!')
    return model, le, mean, std

def train_all_models():
    logger.info('loading embeddings...')
    df = pd.read_pickle(BASE_PATH + '/embeddings_vit.pkl')

    df_train = df[df['split'].isin(['train', 'val'])].reset_index(drop=True)
    df_test  = df[df['split'] == 'test'].reset_index(drop=True)

    logger.info(f'train: {len(df_train)} | test: {len(df_test)}')

    train_linear_probe(df_train, 'manufacturer', 'manufacturer')
    train_linear_probe(df_train, 'family',       'family')
    train_linear_probe(df_train, 'variant',      'variant')

if __name__ == '__main__':
    train_all_models()
    logger.info('all models trained and saved!')