import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import fitz  # PyMuPDF

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain

# ---------- Functions ----------

@st.cache_resource
def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyMuPDFLoader(file_path).load()
    elif ext == ".txt":
        return TextLoader(file_path).load()
    else:
        raise ValueError("Unsupported file type")


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=300)
    return splitter.split_documents(docs)


@st.cache_resource
def create_vectorstore(_chunks):
    embeddings = OllamaEmbeddings(model="llama3")
    return FAISS.from_documents(chunks, embeddings)


@st.cache_resource
def create_qa_chain(_vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = Ollama(model="llama3")
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        return_source_documents=True
    )


def ask_question_direct(chunks, query):
    matched_chunks = [
        chunk for chunk in chunks 
        if any(word.lower() in chunk.page_content.lower() for word in query.split())
    ]

    if not matched_chunks:
        return "❌ Could not find relevant content."

    llm = Ollama(model="llama3")
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.invoke({
        "input_documents": matched_chunks,
        "question": query
    })

    return response['output_text']


def generate_question(llm, docs):
    doc = docs[0]
    prompt_text = f"""
You are a professor creating a logic-based question from syllabus content.
Read the content and return:

Q: <logic question>
A: <correct answer>
Justification: <why this is correct, using info from the content>

Content:
\"\"\"
{doc.page_content}
\"\"\"
"""
    return llm.invoke(prompt_text)


def evaluate_response(llm, question, correct_answer, user_answer, justification):
    prompt = f"""
Evaluate this student's answer against the correct one.

Q: {question}
✅ Correct Answer: {correct_answer}
🧑‍🎓 Student's Answer: {user_answer}
📚 Justification: {justification}

Respond whether the answer is correct and explain why.
"""
    return llm.invoke(prompt)


def summarize_document(llm, chunks):
    combined_text = "\n\n".join([doc.page_content for doc in chunks[:5]])
    prompt = f"""
You are an assistant tasked with summarizing academic syllabus content clearly and concisely.

Please summarize the following document text:
\"\"\" 
{combined_text}
\"\"\"

Summary:"""
    return llm.invoke(prompt)


# ---------- Streamlit UI ----------

st.set_page_config(page_title="📘 Study Buddy AI", layout="wide")
st.title("📘 Study Buddy: Document-Aware GenAI Assistant")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    file_path = f"./uploaded_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ File uploaded successfully!")

    with st.spinner("🔄 Loading and processing document..."):
        docs = load_document(file_path)
        chunks = split_documents(docs)
        vectordb = create_vectorstore(chunks)
        qa_chain = create_qa_chain(vectordb)

    if st.button("📚 Summarize Document"):
        with st.spinner("Summarizing..."):
            summary = summarize_document(Ollama(model="llama3"), chunks)
            st.subheader("📝 Document Summary:")
            st.write(summary)

    st.markdown("---")
    st.subheader("💬 Ask a Comprehension-Based Question")
    user_query = st.text_input("Enter your question:")
    if st.button("🔍 Get Answer"):
        with st.spinner("Thinking..."):
            answer = ask_question_direct(chunks, user_query)
            st.subheader("Answer:")
            st.write(answer)

    st.markdown("---")
    st.subheader("🧠 Logic-Based Question Generator")
    if st.button("Generate Logic-Based Q&A"):
        with st.spinner("Generating..."):
            llm = Ollama(model="llama3")
            logic_qna = generate_question(llm, chunks)
            st.code(logic_qna, language="markdown")

            # Extract parts
            lines = logic_qna.strip().split("\n")
            question, correct_answer, justification = "", "", ""
            for line in lines:
                if line.startswith("Q:"):
                    question = line[2:].strip()
                elif line.startswith("A:"):
                    correct_answer = line[2:].strip()
                elif line.startswith("Justification:"):
                    justification = line[len("Justification:"):].strip()

            if question and correct_answer:
                user_answer = st.text_input("✍️ Your Answer to the Logic-Based Question")
                if st.button("🧪 Evaluate My Answer"):
                    with st.spinner("Evaluating..."):
                        evaluation = evaluate_response(llm, question, correct_answer, user_answer, justification)
                        st.subheader("Evaluation:")
                        st.write(evaluation)
