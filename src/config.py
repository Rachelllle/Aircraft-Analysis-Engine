import os

# Main paths
BASE_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'aircraft_data')
DATA_PATH   = os.path.join(BASE_PATH, 'fgvc-aircraft-2013b', 'fgvc-aircraft-2013b', 'data') + '/'
CSV_PATH    = BASE_PATH + '/'
MODEL_PATH  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')

# Model parameters
IMG_SIZE    = 224
BATCH_SIZE  = 32
PCA_K       = 256
MAX_ITER    = 150
STEP_SIZE   = 0.001
SEED        = 42

# Quick check 
if __name__ == '__main__':
    paths = [BASE_PATH, DATA_PATH, CSV_PATH,
             CSV_PATH + 'train.csv',
             DATA_PATH + 'images/']
    for p in paths:
        status = 'ok' if os.path.exists(p) else 'MISSING'
        print(f'{status} - {p}')