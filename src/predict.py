import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import joblib
import torch
import torch.nn as nn
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel

from src.config import BASE_PATH, MODEL_PATH

def load_vit():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vit_model')
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)
    vit = ViTModel.from_pretrained(model_path)
    vit.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vit = vit.to(device)
    return feature_extractor, vit, device

def load_classifier(model_name):
    meta = joblib.load(f'{MODEL_PATH}/{model_name}_meta.pkl')
    num_classes = meta['num_classes']

    model = nn.Sequential(
        nn.Linear(768, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    model.load_state_dict(torch.load(
        f'{MODEL_PATH}/{model_name}_model.pt',
        map_location='cpu'
    ))
    model.eval()
    return model, meta

def predict_image(image_path, feature_extractor, vit, device):
    # load and extract ViT embedding
    img = Image.open(image_path).convert('RGB')
    inputs = feature_extractor(images=[img], return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = vit(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

    results = {}
    for target in ['manufacturer', 'family', 'variant']:
        model, meta = load_classifier(target)
        le   = meta['le']
        mean = meta['mean']
        std  = meta['std']

        # normalize
        X = (embedding - mean) / std
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = model(X_tensor)
            probs  = torch.softmax(logits, dim=1).numpy()[0]
            pred   = probs.argmax()

        label      = le.classes_[pred]
        confidence = probs[pred] * 100

        results[target] = {
            'label':      label,
            'confidence': confidence,
            'top3': [
                {'label': le.classes_[i], 'confidence': probs[i]*100}
                for i in probs.argsort()[-3:][::-1]
            ]
        }

    return results, img

if __name__ == '__main__':
    from src.config import DATA_PATH

    print('loading ViT...')
    feature_extractor, vit, device = load_vit()

    # test on a random image
    images_dir = DATA_PATH + 'images/'
    sample = os.listdir(images_dir)[0]
    image_path = images_dir + sample

    print(f'predicting : {sample}')
    results, img = predict_image(image_path, feature_extractor, vit, device)

    print(f'\nmanufacturer : {results["manufacturer"]["label"]} ({results["manufacturer"]["confidence"]:.1f}%)')
    print(f'family       : {results["family"]["label"]} ({results["family"]["confidence"]:.1f}%)')
    print(f'variant      : {results["variant"]["label"]} ({results["variant"]["confidence"]:.1f}%)')