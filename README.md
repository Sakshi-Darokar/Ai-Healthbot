# 🩺 AI HealthBot - Intelligent Medical Symptom Assistant

**AI HealthBot** is an intelligent conversational agent designed to help users understand their symptoms by analyzing natural language inputs, asking smart follow-up questions, and providing possible health insights backed by PubMed medical literature.

> ⚠️ This bot is built for **educational and informational purposes only**. It is **not a substitute for professional medical advice, diagnosis, or treatment**.

---

## 🚀 Features

- 🌐 **Multilingual Input Handling**: Understands English, Hindi, and Roman Hindi symptom descriptions.
- ✨ **Preprocessing Pipeline**: Translation, spell correction, slang normalization, and medical keyword matching using MeSH.
- 🧠 **Smart Follow-Up Logic**: Asks non-repetitive, relevant follow-up questions based on conversation history and missing information.
- 🏥 **Condition Prediction**: Predicts potential conditions based on symptoms and returns medical reasoning + care tips.
- 📚 **PubMed Integration**: Retrieves real-time, evidence-backed medical articles using MeSH embeddings + FAISS + PubMed API.
- ⚠️ **Emergency & Ethical Filters**: Detects urgent cases (e.g., chest pain, breathing issues) and advises emergency care with disclaimers.

---

## 🛠️ Tech Stack

| Component | Tool |
|----------|------|
| LLM Backend | [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) via [Together.ai](https://www.together.ai/) API |
| Framework | [LangChain](https://www.langchain.com/) |
| Medical NLP | [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract) |
| Vector Search | FAISS |
| UI (Optional) | Streamlit |
| Translation | Deep Translator |
| Document Search | PubMed E-Utils API |
| Memory & Prompting | LangChain ChatMemory + Prompt Templates |

---

## 📁 Project Modules

- `main.py`: Core chatbot logic (can be integrated with Streamlit or CLI)
- `phase1_preprocess.py`: Input translation, cleaning, MeSH mapping
- `symptom_profiler.py`: Extracts structured symptom data using LLM
- `smart_dialog_manager.py`: Generates intelligent follow-up questions
- `disease_predictor.py`: Predicts conditions based on symptom profile
- `query_faiss.py`: MeSH term search using PubMedBERT and FAISS
- `pubmed_utils.py`: Filters and ranks relevant PubMed articles
- `mesh_parser.py`: Loads MeSH synonyms from XML

---

## 🔍 Example Use Case

**User Input**:  
`"Bukhar aur gala dard ho raha hai kal se"`

**Bot Response**:
- Translates & corrects → "Fever and throat pain since yesterday"
- Extracts profile → `{ symptom: "fever", location: "throat", duration: "1 day", severity: "moderate" }`
- Asks: `"Is the throat pain constant or does it come and go?"`
- Predicts → `Possible Condition: Viral Pharyngitis`
- Fetches PubMed links like:
  - [Management of acute sore throat](https://pubmed.ncbi.nlm.nih.gov/XXXXX/)
  - [Streptococcal throat infections in children](https://pubmed.ncbi.nlm.nih.gov/YYYYY/)

---

## ⚠️ Limitations

- Mistral-7B performs well for logic and question flow but:
  - It is **not medically fine-tuned** (unlike GPT-4Med or MedPaLM)
  - May **hallucinate** or rephrase emergency messages unnecessarily
  - JSON extraction can sometimes break with longer replies
- PubMed API may occasionally return articles with low relevance — filtered using a scoring function.

---

## 🧠 Future Improvements

- Add structured ICD code matching  
- Build a front-end UI with Streamlit  
- Improve PubMed article ranking using advanced embeddings  
- Fine-tune an LLM for medical-specific dialog  

---

## 🙏 Feedback Welcome!

This project is actively evolving and feedback is greatly appreciated.  
Feel free to open issues, contribute, or suggest improvements!

---


AI-HealthBot/
│
├── main.py
├── README.md
├── requirements.txt
│
├── phase1_preprocessing/
│   └── phase1_preprocess.py
│
├── pubmed/
│   ├── pubmed_utils.py
│   ├── pubmed_search.py
│   └── query_faiss.py
│
├── symptom_analysis/
│   ├── smart_dialog_manager.py
│   ├── symptom_profiler.py
│   ├── symptom_classifier.py
│   └── disease_predictor.py
│
├── utils/
│   └── mesh_parser.py
│
└── data/
    ├── desc2025.xml
    ├── mesh_faiss.index
    └── mesh_labels.pkl

⚠️ Note on Dataset and FAISS File Upload
The following files could not be uploaded to this GitHub repository due to their large size:

pgsql
Copy
Edit
data/
├── desc2025.xml
├── mesh_faiss.index
└── mesh_labels.pkl
These files are essential for the vector search and MeSH-based medical mapping in the AI HealthBot system. Due to GitHub's file size limits and LFS restrictions, we are storing and loading these files externally during deployment.

If you're trying to run this project locally, please make sure to manually add the above files inside the data/ folder.





## 📜 Disclaimer

This project is **not intended to provide medical advice**. Always consult a licensed healthcare provider for any medical concerns.

---
