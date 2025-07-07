# query_faiss.py
import faiss
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Load PubMedBERT
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load FAISS index and MeSH labels
index = faiss.read_index("mesh_faiss.index")
with open("mesh_labels.pkl", "rb") as f:
    mesh_labels = pickle.load(f)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def search_mesh_terms(symptom_text, top_k=5):
    query_vector = get_embedding(symptom_text)
    distances, indices = index.search(query_vector, top_k)
    matched_terms = [(mesh_labels[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
    return matched_terms

# ----------- TEST -------------
if __name__ == "__main__":
    user_input = input("Enter symptoms (English): ")
    results = search_mesh_terms(user_input)

    print("\n🔍 Top MeSH Matches:")
    for i, (term, dist) in enumerate(results, 1):
        print(f"{i}. {term}  (Distance: {dist:.2f})")
