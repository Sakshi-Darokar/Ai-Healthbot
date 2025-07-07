import os
import requests

# Set Hugging Face base path
HF_BASE = "https://huggingface.co/datasets/DarokarSakshi/desc2025-dataset/resolve/main/"

def download_from_hf(filename):
    url = HF_BASE + filename
    if not os.path.exists(filename):
        print(f"📥 Downloading: {filename} from Hugging Face...")
        r = requests.get(url)
        if r.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(r.content)
        else:
            raise Exception(f"Failed to download {filename}. HTTP {r.status_code}")
    else:
        print(f"✅ {filename} already exists, skipping download.")

def download_all_files():
    download_from_hf("desc2025.xml")
    download_from_hf("mesh_faiss.index")
    download_from_hf("mesh_labels.pkl")
