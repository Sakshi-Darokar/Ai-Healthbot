import os
from huggingface_hub import hf_hub_download

# Set Hugging Face dataset repo
REPO_ID = "DarokarSakshi/desc2025-dataset"

def download_from_hf(filename):
    if not os.path.exists(filename):
        print(f"📥 Downloading {filename} from Hugging Face Hub...")
        try:
            file_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=".",  # Current directory
                local_dir_use_symlinks=False
            )
            print(f"✅ {filename} downloaded and saved at: {file_path}")
        except Exception as e:
            raise Exception(f"❌ Failed to download {filename}: {str(e)}")
    else:
        print(f"✔️ {filename} already exists, skipping.")

def download_all_files():
    download_from_hf("desc2025.xml")
    download_from_hf("mesh_faiss.index")
    download_from_hf("mesh_labels.pkl")
