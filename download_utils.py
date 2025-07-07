import os
import time
from huggingface_hub import hf_hub_download, LocalEntryNotFoundError

REPO_ID = "DarokarSakshi/desc2025-dataset"

def download_from_hf(filename, retries=3):
    for attempt in range(1, retries + 1):
        if os.path.exists(filename):
            print(f"✔️ {filename} already exists locally. Skipping.")
            return
        try:
            print(f"📥 Attempting download ({attempt}) → {filename}")
            file_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=".",  # current dir
            )
            print(f"✅ {filename} downloaded and saved at: {file_path}")
            return
        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed for {filename}: {e}")
            if attempt < retries:
                wait_time = 2 ** attempt  # exponential backoff
                print(f"⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise Exception(f"❌ Failed to download {filename} after {retries} attempts.\n{e}")

def download_all_files():
    download_from_hf("desc2025.xml")         # usually works
    download_from_hf("mesh_faiss.index")     # large → may need retry
    download_from_hf("mesh_labels.pkl")      # medium → retry may help
