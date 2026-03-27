import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import logging
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from src.config import BASE_PATH, OUTPUT_PATH

logger = logging.getLogger(__name__)

def load_model(model_name):
    meta = joblib.load(f'{BASE_PATH}/sklearn_models/{model_name}_meta.pkl')
    num_classes = meta['num_classes']

    model = nn.Sequential(
        nn.Linear(768, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    model.load_state_dict(torch.load(f'{BASE_PATH}/sklearn_models/{model_name}_model.pt',
                                      map_location='cpu'))
    model.eval()
    return model, meta

def evaluate_model(model_name, df_test, target_col):
    model, meta = load_model(model_name)
    le   = meta['le']
    mean = meta['mean']
    std  = meta['std']

    X = np.array(df_test['embedding'].tolist(), dtype=np.float32)
    X = (X - mean) / std
    y_true = le.transform(df_test[target_col].values)

    with torch.no_grad():
        logits = model(torch.tensor(X))
        y_pred = logits.argmax(dim=1).numpy()

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='weighted')

    logger.info(f'\n{model_name}')
    logger.info(f'  accuracy : {acc*100:.2f}%')
    logger.info(f'  f1-score : {f1*100:.2f}%')
    return acc, f1

def save_scores_json(results):
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    scores = {
        'manufacturer': {
            'accuracy': round(results['manufacturer'][0] * 100, 2),
            'f1':       round(results['manufacturer'][1] * 100, 2),
            'classes':  30
        },
        'family': {
            'accuracy': round(results['family'][0] * 100, 2),
            'f1':       round(results['family'][1] * 100, 2),
            'classes':  70
        },
        'variant': {
            'accuracy': round(results['variant'][0] * 100, 2),
            'f1':       round(results['variant'][1] * 100, 2),
            'classes':  100
        }
    }
    path = OUTPUT_PATH + '/scores.json'
    with open(path, 'w') as f:
        json.dump(scores, f, indent=2)
    logger.info(f'scores saved → {path}')

def plot_results(results):
    logger.info('Creating plot results')
    labels = ['Manufacturer\n(30 classes)', 'Family\n(70 classes)', 'Variant\n(100 classes)']
    accs = [results['manufacturer'][0]*100, results['family'][0]*100, results['variant'][0]*100]
    f1s  = [results['manufacturer'][1]*100, results['family'][1]*100, results['variant'][1]*100]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    bars1 = axes[0].bar(labels, accs, color=['#4CAF50', '#2196F3', '#FF9800'], width=0.5)
    axes[0].set_title('Accuracy per classification level')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_ylim(0, 100)
    for bar, v in zip(bars1, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, v+1, f'{v:.1f}%', ha='center', fontweight='bold')

    bars2 = axes[1].bar(labels, f1s, color=['#4CAF50', '#2196F3', '#FF9800'], width=0.5)
    axes[1].set_title('F1-Score per classification level')
    axes[1].set_ylabel('F1-Score (%)')
    axes[1].set_ylim(0, 100)
    for bar, v in zip(bars2, f1s):
        axes[1].text(bar.get_x() + bar.get_width()/2, v+1, f'{v:.1f}%', ha='center', fontweight='bold')

    plt.suptitle('Aircraft Classification — ViT + Linear Probe (PyTorch)', fontsize=14)
    plt.tight_layout()
    plt.show()

def score_all_models():
    df = pd.read_pickle(BASE_PATH + '/embeddings_vit.pkl')
    df_test = df[df['split'] == 'test'].reset_index(drop=True)
    logger.info(f'test set : {len(df_test)} images')

    acc_m, f1_m = evaluate_model('manufacturer', df_test, 'manufacturer')
    acc_f, f1_f = evaluate_model('family',       df_test, 'family')
    acc_v, f1_v = evaluate_model('variant',      df_test, 'variant')

    results = {
        'manufacturer': (acc_m, f1_m),
        'family':       (acc_f, f1_f),
        'variant':      (acc_v, f1_v),
    }

    plot_results(results)
    save_scores_json(results)  
    return results

if __name__ == '__main__':
    score_all_models()