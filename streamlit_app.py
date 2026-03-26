import os
import sys
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from PIL import Image
import random

from src.config import BASE_PATH, DATA_PATH, MODEL_PATH

st.set_page_config(
    page_title='Aircraft Classifier',
    page_icon='✈️',
    layout='wide'
)

# sidebar navigation
page = st.sidebar.selectbox('Navigation', [
    '📊 Dataset Overview',
    '📈 Model Scoring',
    '🖼️ Prediction Examples',
    '🔍 Live Demo'
])

# load embeddings once
@st.cache_data
def load_data():
    return pd.read_pickle(BASE_PATH + '/embeddings_vit.pkl')

@st.cache_resource
def load_classifier(model_name):
    meta = joblib.load(f'{MODEL_PATH}/{model_name}_meta.pkl')
    import torch
    import torch.nn as nn
    num_classes = meta['num_classes']
    model = nn.Sequential(
        nn.Linear(768, 512), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    model.load_state_dict(torch.load(
        f'{MODEL_PATH}/{model_name}_model.pt', map_location='cpu'
    ))
    model.eval()
    return model, meta

# ── PAGE 1 : Dataset Overview ──
if page == '📊 Dataset Overview':
    st.title('📊 Dataset Overview')
    st.markdown('**FGVC-Aircraft** — Fine-Grained Visual Classification of Aircraft')

    df = load_data()

    # stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Images', '10,000')
    col2.metric('Manufacturers', df['manufacturer'].nunique())
    col3.metric('Families', df['family'].nunique())
    col4.metric('Variants', df['variant'].nunique())

    st.markdown('---')

    # split distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Split Distribution')
        split_counts = df['split'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(split_counts, labels=split_counts.index, autopct='%1.1f%%',
               colors=['#4CAF50', '#2196F3', '#FF9800'])
        ax.set_title('Train / Val / Test')
        st.pyplot(fig)

    with col2:
        st.subheader('Top 10 Manufacturers')
        top_manuf = df['manufacturer'].value_counts().head(10)
        fig, ax = plt.subplots()
        top_manuf.plot(kind='barh', ax=ax, color='#2196F3')
        ax.set_xlabel('Number of images')
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown('---')
    st.subheader('Top 15 Families')
    top_family = df['family'].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(12, 4))
    top_family.plot(kind='bar', ax=ax, color='#4CAF50')
    ax.set_ylabel('Number of images')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

# ── PAGE 2 : Model Scoring ──
elif page == '📈 Model Scoring':
    st.title('📈 Model Scoring')
    st.markdown('Evaluation of the 3 classification models on the test set.')

    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    import seaborn as sns

    df = load_data()
    df_test = df[df['split'] == 'test'].reset_index(drop=True)

    results = {}
    for target in ['manufacturer', 'family', 'variant']:
        model, meta = load_classifier(target)
        X = np.array(df_test['embedding'].tolist(), dtype=np.float32)
        X = (X - meta['mean']) / meta['std']
        y_true = meta['le'].transform(df_test[target].values)
        with torch.no_grad():
            logits = model(torch.tensor(X))
            y_pred = logits.argmax(dim=1).numpy()
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average='weighted')
        results[target] = {'acc': acc, 'f1': f1, 'y_true': y_true, 'y_pred': y_pred, 'meta': meta}

    # metrics
    col1, col2, col3 = st.columns(3)
    col1.metric('Manufacturer Accuracy', f"{results['manufacturer']['acc']*100:.1f}%")
    col2.metric('Family Accuracy',       f"{results['family']['acc']*100:.1f}%")
    col3.metric('Variant Accuracy',      f"{results['variant']['acc']*100:.1f}%")

    st.markdown('---')

    # bar charts
    labels = ['Manufacturer\n(30 classes)', 'Family\n(70 classes)', 'Variant\n(100 classes)']
    accs = [results['manufacturer']['acc']*100, results['family']['acc']*100, results['variant']['acc']*100]
    f1s  = [results['manufacturer']['f1']*100,  results['family']['f1']*100,  results['variant']['f1']*100]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bars1 = axes[0].bar(labels, accs, color=['#4CAF50', '#2196F3', '#FF9800'], width=0.5)
    axes[0].set_title('Accuracy per level')
    axes[0].set_ylim(0, 100)
    for bar, v in zip(bars1, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, v+1, f'{v:.1f}%', ha='center', fontweight='bold')

    bars2 = axes[1].bar(labels, f1s, color=['#4CAF50', '#2196F3', '#FF9800'], width=0.5)
    axes[1].set_title('F1-Score per level')
    axes[1].set_ylim(0, 100)
    for bar, v in zip(bars2, f1s):
        axes[1].text(bar.get_x() + bar.get_width()/2, v+1, f'{v:.1f}%', ha='center', fontweight='bold')

    plt.suptitle('Aircraft Classification — ViT + Linear Probe (PyTorch)')
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown('---')

    # confusion matrix
    st.subheader('Confusion Matrix')
    selected = st.selectbox('Select model', ['manufacturer', 'family', 'variant'])
    r = results[selected]
    cm = confusion_matrix(r['y_true'], r['y_pred'])
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, ax=ax, cmap='Blues', fmt='d',
                xticklabels=r['meta']['le'].classes_,
                yticklabels=r['meta']['le'].classes_)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    st.pyplot(fig)

# ── PAGE 3 : Prediction Examples ──
elif page == '🖼️ Prediction Examples':
    st.title('🖼️ Prediction Examples')
    st.markdown('Random examples from the test set with model predictions.')

    import torch

    df = load_data()
    df_test = df[df['split'] == 'test'].reset_index(drop=True)

    filter_type = st.radio('Show', ['All', 'Correct only', 'Wrong only'], horizontal=True)
    n_examples = st.slider('Number of examples', 4, 20, 8)

    if st.button('🔀 Generate examples'):
        predictions = {}
        for target in ['manufacturer', 'family', 'variant']:
            model, meta = load_classifier(target)
            X = np.array(df_test['embedding'].tolist(), dtype=np.float32)
            X = (X - meta['mean']) / meta['std']
            with torch.no_grad():
                logits = model(torch.tensor(X))
                preds = logits.argmax(dim=1).numpy()
            predictions[target] = meta['le'].inverse_transform(preds)

        df_test['pred_manufacturer'] = predictions['manufacturer']
        df_test['pred_family']       = predictions['family']
        df_test['pred_variant']      = predictions['variant']
        df_test['correct'] = (
            (df_test['manufacturer'] == df_test['pred_manufacturer']) &
            (df_test['family']       == df_test['pred_family'])
        )

        if filter_type == 'Correct only':
            df_show = df_test[df_test['correct']].sample(n_examples)
        elif filter_type == 'Wrong only':
            df_show = df_test[~df_test['correct']].sample(n_examples)
        else:
            df_show = df_test.sample(n_examples)

        images_dir = DATA_PATH + 'images/'
        cols = st.columns(4)
        for idx, (_, row) in enumerate(df_show.iterrows()):
            with cols[idx % 4]:
                try:
                    img = Image.open(images_dir + row['image_id'] + '.jpg')
                    st.image(img, use_column_width=True)
                except:
                    st.write('image not found')
                correct = '✅' if row['correct'] else '❌'
                st.markdown(f"**{correct} {row['manufacturer']}**")
                st.caption(f"True: {row['variant']}")
                st.caption(f"Pred: {row['pred_variant']}")

# ── PAGE 4 : Live Demo ──
elif page == '🔍 Live Demo':
    st.title('🔍 Live Demo')
    st.markdown('Upload an aircraft photo to get a live prediction.')

    from src.predict import load_vit, predict_image

    @st.cache_resource
    def load_vit_model():
        return load_vit()

    with st.spinner('Loading ViT model...'):
        feature_extractor, vit, device = load_vit_model()

    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, width=400)

        if st.button('🔍 Predict', use_container_width=True):
            with st.spinner('Analyzing...'):
                temp_path = BASE_PATH + '/temp_predict.jpg'
                img.save(temp_path)
                results, _ = predict_image(temp_path, feature_extractor, vit, device)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Manufacturer', results['manufacturer']['label'],
                          f"{results['manufacturer']['confidence']:.1f}%")
                for t in results['manufacturer']['top3']:
                    st.progress(int(t['confidence']),
                                text=f"{t['label']} {t['confidence']:.1f}%")
            with col2:
                st.metric('Family', results['family']['label'],
                          f"{results['family']['confidence']:.1f}%")
                for t in results['family']['top3']:
                    st.progress(int(t['confidence']),
                                text=f"{t['label']} {t['confidence']:.1f}%")
            with col3:
                st.metric('Variant', results['variant']['label'],
                          f"{results['variant']['confidence']:.1f}%")
                for t in results['variant']['top3']:
                    st.progress(int(t['confidence']),
                                text=f"{t['label']} {t['confidence']:.1f}%")