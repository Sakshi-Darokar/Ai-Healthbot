import os
import requests

# ✅ Hugging Face base path (your dataset repo)
HF_BASE = "https://huggingface.co/datasets/DarokarSakshi/desc2025-dataset/resolve/main/"

# 🔽 Helper to download individual file
def download_from_hf(filename):
    url = HF_BASE + filename
    if not os.path.exists(filename):
        print(f"📥 Downloading: {filename} from Hugging Face...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"✅ {filename} downloaded successfully.")
        else:
            raise Exception(f"❌ Failed to download {filename}. Status code: {response.status_code}")
    else:
        print(f"✔️ {filename} already exists locally. Skipping.")

# 📦 Call this from app.py to download all files
def download_all_files():
    download_from_hf("desc2025.xml")
    download_from_hf("mesh_faiss.index")
    download_from_hf("mesh_labels.pkl")
