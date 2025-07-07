# build_faiss_index.py
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from mesh_parser import load_mesh_synonyms
from tqdm import tqdm
import pickle
import numpy as np

# Load PubMedBERT
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load MeSH terms
mesh_dict = load_mesh_synonyms("desc2025.xml")
unique_mesh_terms = list(set(mesh_dict.values()))  # only main MeSH terms

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token

# Build index
embeddings = []
labels = []

print(f"Embedding {len(unique_mesh_terms)} MeSH terms...")
for term in tqdm(unique_mesh_terms):
    emb = get_embedding(term)
    embeddings.append(emb[0])
    labels.append(term)

# Convert to FAISS index
embedding_dim = embeddings[0].shape[0]
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings))

# Save index + labels
faiss.write_index(index, "mesh_faiss.index")
with open("mesh_labels.pkl", "wb") as f:
    pickle.dump(labels, f)

print("✅ FAISS index built and saved.")
np.save("embeddings.npy", embeddings)
with open("labels.pkl", "wb") as f:
    pickle.dump(labels, f)
print("✅ Embeddings and labels saved.")
