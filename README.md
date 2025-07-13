# AI-Assistant-for-Summarization-

# 📘 Study Buddy — Document-Aware GenAI Assistant

A powerful Streamlit web app that leverages LLMs (like **LLaMA 3 via Ollama**) and **LangChain** to read documents (PDF/TXT), understand their content, and:

✅ Answer comprehension-based questions  
🧩 Generate logic-based questions with justification  
📝 Summarize uploaded academic content  
🧠 Evaluate user responses

---

## 🚀 Features

- **📄 File Upload**  
  Upload `.pdf` or `.txt` documents and extract readable content automatically.

- **📝 Document Summarization**  
  Get a clean, concise summary using LLMs — great for revision or analysis.

- **💬 Comprehension Q&A**  
  Ask questions directly from the document. Uses keyword-aware chunk selection and LangChain QA.

- **🧠 Logic-Based Question Generation**  
  Automatically generates challenging questions, correct answers, and justifications based on content.

- **✍️ Answer Evaluation**  
  Submit your response to the generated question and receive an AI-powered evaluation.

---

## 🛠️ Tech Stack

| Component       | Tech Used               |
|----------------|-------------------------|
| Frontend        | [Streamlit](https://streamlit.io) |
| LLM Backend     | [Ollama](https://ollama.com/) with `llama3` model |
| Document Parsing| PyMuPDF, LangChain Loaders |
| Embeddings      | FAISS Vector Store + Ollama Embeddings |
| QA & Generation | [LangChain](https://www.langchain.com) |

---

## 📦 Installation

### 1. Clone the repository

### 2. Install Python dependencies
pip install -r requirements.txt

### 3. Start Ollama and pull LLaMA 3 model
ollama serve
ollama pull llama3

### 4. Run the app
streamlit run app.py


## 🧪 Example Use Cases
📚 University exam preparation

🧑‍🏫 Teachers generating logic-based quiz questions

🧠 Interactive comprehension tool

🤖 AI-assisted tutoring


## ⚠️ Known Limitations
Currently supports only .pdf and .txt files.

Requires llama3 to be installed locally via Ollama.

FAISS and chain generation are not session-persistent yet.

## 📄 License
MIT License — feel free to use, modify, and share.

## 🙌 Acknowledgements
Ollama for local LLM hosting

LangChain for LLM pipelines

Streamlit for interactive UI

## ✨ Future Enhancements
⏳ Add chat history sidebar

📥 Support more file formats (.docx, .pptx)

🔐 User sessions and saving answers

📊 Evaluation scoring and analytics

    
```bash
git clone https://github.com/your-username/study-buddy-genai.git
cd study-buddy-genai


