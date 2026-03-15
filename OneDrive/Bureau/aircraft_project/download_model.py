import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'

from huggingface_hub import hf_hub_download

model_id = 'google/vit-base-patch16-224'
save_dir = './vit_model'
os.makedirs(save_dir, exist_ok=True)

files = [
    'config.json',
    'preprocessor_config.json',
    'model.safetensors',   
]

for filename in files:
    print(f'downloading {filename}...')
    try:
        path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=save_dir,
            force_download=False
        )
        print(f'ok - {filename}')
    except Exception as e:
        print(f'error - {filename} : {e}')

print('done!')