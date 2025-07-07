import os
import time
from huggingface_hub import hf_hub_download

REPO_ID = "DarokarSakshi/desc2025-dataset"

def download_from_hf(filename, retries=5):
    """
    Downloads file from Hugging Face Hub with retry mechanism
    """
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
                local_dir=".",
            )
            print(f"✅ {filename} downloaded successfully.")
            return
        except Exception as e:
            wait_time = 2 ** attempt
            print(f"⚠️ Attempt {attempt} failed: {str(e)}")
            if attempt < retries:
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"❌ Final attempt failed. Giving up on {filename}.")
                raise e

def download_all_files():
    download_from_hf("desc2025.xml")
    download_from_hf("mesh_faiss.index")
    download_from_hf("mesh_labels.pkl")
