import streamlit as st
import os
import sqlite3
import google.generativeai as genai
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# Hardcoded Gemini API Key
GEMINI_API_KEY = "AIzaSyCC430r2ShgtyH_V0Dqop9eW9DWdtVtH3Y"  # Replace with your actual API key

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

# Initialize database
conn = sqlite3.connect('study_assistant.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS files
             (id INTEGER PRIMARY KEY, name TEXT, path TEXT, size INTEGER)''')
c.execute('''CREATE TABLE IF NOT EXISTS notes
             (id INTEGER PRIMARY KEY, content TEXT, path TEXT)''')
conn.commit()

# Initialize Sentence Transformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def process_document(file_path):
    """Extract text from PDF and create embeddings"""
    text = ""
    if file_path.endswith('.pdf'):
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings
    embeddings = embedding_model.encode(chunks)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return text, index, chunks

def generate_with_gemini(prompt):
    """Generate text using Gemini"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini error: {e}")
        return None

def save_note_as_pdf(note_text, filename):
    """Save note as PDF"""
    try:
        pdf_path = os.path.join("notes", filename)
        os.makedirs("notes", exist_ok=True)
        
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        
        text = c.beginText(40, height - 40)
        text.setFont("Helvetica", 12)
        for line in note_text.split('\n'):
            text.textLine(line)
        c.drawText(text)
        c.save()
        
        # Save to database
        c = conn.cursor()
        c.execute("INSERT INTO notes (content, path) VALUES (?, ?)",
                  (note_text, pdf_path))
        conn.commit()
        
        return pdf_path
    except Exception as e:
        st.error(f"PDF creation error: {e}")
        return None

def data_ingestion():
    """File upload and note taking functionality"""
    st.subheader("Upload Study Materials")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
    if uploaded_file:
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Process document
        text, _, _ = process_document(file_path)
        
        # Store in database
        c = conn.cursor()
        c.execute("INSERT INTO files (name, path, size) VALUES (?, ?, ?)",
                  (uploaded_file.name, file_path, os.path.getsize(file_path)))
        conn.commit()
        st.success("File uploaded and processed successfully!")
    
    # Manual notes
    st.subheader("Create Manual Notes")
    note_text = st.text_area("Enter your notes here:")
    note_filename = st.text_input("Note filename:")
    if st.button("Save Note"):
        if note_text and note_filename:
            pdf_path = save_note_as_pdf(note_text, f"{note_filename}.pdf")
            if pdf_path:
                st.success(f"Note saved as {pdf_path}")

def summarizer():
    """Document summarization"""
    st.subheader("Document Summarization")
    
    # Get list of documents
    c = conn.cursor()
    c.execute("SELECT name, path FROM files")
    documents = c.fetchall()
    
    if not documents:
        st.warning("No documents found!")
        return
    
    selected_doc = st.selectbox("Select document", [doc[0] for doc in documents])
    word_limit = st.slider("Summary length (words)", 50, 500, 200)
    
    if st.button("Generate Summary"):
        doc_path = [doc[1] for doc in documents if doc[0] == selected_doc][0]
        text, _, _ = process_document(doc_path)
        
        prompt = f"""
        Create a concise summary of the following document in about {word_limit} words.
        Focus on key points and main ideas. Use clear, simple language.
        
        Document content:
        {text[:5000]}  # Limiting input size for demo
        """
        
        summary = generate_with_gemini(prompt)
        if summary:
            st.subheader("Summary")
            st.write(summary)

def quiz_generator():
    """Quiz generation"""
    st.subheader("Quiz Generator")
    
    c = conn.cursor()
    c.execute("SELECT name, path FROM files")
    documents = c.fetchall()
    
    if not documents:
        st.warning("No documents found!")
        return
    
    selected_doc = st.selectbox("Select document", [doc[0] for doc in documents])
    difficulty = st.selectbox("Difficulty level", ["Easy", "Medium", "Hard"])
    
    if st.button("Generate Quiz"):
        doc_path = [doc[1] for doc in documents if doc[0] == selected_doc][0]
        text, _, _ = process_document(doc_path)
        
        prompt = f"""
        Generate a 5-question {difficulty.lower()} level quiz based on the following content.
        Format as a valid JSON object with the following structure:
        {{
            "quiz": [
                {{
                    "question": "",
                    "options": ["", "", "", ""],
                    "answer": ""
                }}
            ]
        }}
        
        Content:
        {text[:5000]}
        """
        
        quiz_json = generate_with_gemini(prompt)
        if quiz_json:
            try:
                # Clean the JSON response
                quiz_json = quiz_json.strip().strip('```json').strip('```')
                st.write("Cleaned Quiz JSON Response:", quiz_json)  # Debugging
                quiz_data = json.loads(quiz_json)
                st.session_state.quiz = quiz_data['quiz']
                st.session_state.user_answers = [None] * len(quiz_data['quiz'])
            except json.JSONDecodeError as e:
                st.error(f"Failed to parse quiz: {e}. Response: {quiz_json}")

def document_query():
    """Document Q&A"""
    st.subheader("Document Query")
    
    c = conn.cursor()
    c.execute("SELECT name, path FROM files")
    documents = c.fetchall()
    
    if not documents:
        st.warning("No documents found!")
        return
    
    selected_doc = st.selectbox("Select document", [doc[0] for doc in documents])
    question = st.text_input("Enter your question")
    
    if question and st.button("Get Answer"):
        doc_path = [doc[1] for doc in documents if doc[0] == selected_doc][0]
        text, index, chunks = process_document(doc_path)
        
        # Find relevant chunks
        query_embedding = embedding_model.encode([question])
        _, indices = index.search(np.array(query_embedding).astype('float32'), k=3)
        
        context = " ".join([chunks[i] for i in indices[0]])
        
        prompt = f"""
        Answer this question based on the provided context.
        Question: {question}
        Context: {context}
        If the answer isn't in the context, say you don't know.
        Provide a concise answer in 2-3 sentences.
        """
        
        answer = generate_with_gemini(prompt)
        if answer:
            st.write("### Answer")
            st.write(answer)

def main():
    st.title("Study Assistant - Gemini Edition")
    
    menu = ["Data Ingestion", "Summarizer", "Quiz Generator", "Document Query"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Data Ingestion":
        data_ingestion()
    elif choice == "Summarizer":
        summarizer()
    elif choice == "Quiz Generator":
        quiz_generator()
    elif choice == "Document Query":
        document_query()

if __name__ == "__main__":
    main()
