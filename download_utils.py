import os
import time
from huggingface_hub import hf_hub_download

REPO_ID = "DarokarSakshi/desc2025-dataset"

def download_from_hf(filename, retries=5):
    if os.path.exists(filename):
        print(f"✔️ {filename} already exists locally.")
        return

    for attempt in range(1, retries + 1):
        try:
            print(f"📥 Attempting download ({attempt}) of {filename}...")
            file_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=".",  # This is enough — no need for symlink flags
            )
            print(f"✅ {filename} downloaded successfully to {file_path}")
            return
        except Exception as e:
            wait_time = 2 ** attempt
            print(f"⚠️ Attempt {attempt} failed: {e}")
            if attempt < retries:
                print(f"⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"❌ Failed to download {filename} after {retries} attempts. Error: {e}")

def download_all_files():
    download_from_hf("desc2025.xml")
    download_from_hf("mesh_faiss.index")
    download_from_hf("mesh_labels.pkl")
