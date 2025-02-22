# import streamlit as st
# import os
# import sqlite3
# import google.generativeai as genai
# from reportlab.pdfgen import canvas
# from reportlab.lib.pagesizes import letter
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import json

# # Hardcoded Gemini API Key
# GEMINI_API_KEY = "AIzaSyCC430r2ShgtyH_V0Dqop9eW9DWdtVtH3Y"  # Replace with your actual API key

# # Configure Gemini
# genai.configure(api_key=GEMINI_API_KEY)
# generation_config = {
#     "temperature": 0.7,
#     "top_p": 0.95,
#     "top_k": 40,
#     "max_output_tokens": 8192,
# }

# model = genai.GenerativeModel(
#     model_name="gemini-2.0-flash-exp",
#     generation_config=generation_config,
# )

# # Initialize database
# conn = sqlite3.connect('study_assistant.db')
# c = conn.cursor()
# c.execute('''CREATE TABLE IF NOT EXISTS files
#              (id INTEGER PRIMARY KEY, name TEXT, path TEXT, size INTEGER)''')
# c.execute('''CREATE TABLE IF NOT EXISTS notes
#              (id INTEGER PRIMARY KEY, content TEXT, path TEXT)''')
# conn.commit()

# # Initialize Sentence Transformer
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# def process_document(file_path):
#     """Extract text from PDF and create embeddings"""
#     text = ""
#     if file_path.endswith('.pdf'):
#         reader = PdfReader(file_path)
#         for page in reader.pages:
#             text += page.extract_text()
    
#     # Split text into chunks
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )
#     chunks = text_splitter.split_text(text)
    
#     # Create embeddings
#     embeddings = embedding_model.encode(chunks)
    
#     # Create FAISS index
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(np.array(embeddings).astype('float32'))
    
#     return text, index, chunks

# def generate_with_gemini(prompt):
#     """Generate text using Gemini"""
#     try:
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         st.error(f"Gemini error: {e}")
#         return None

# def save_note_as_pdf(note_text, filename):
#     """Save note as PDF"""
#     try:
#         pdf_path = os.path.join("notes", filename)
#         os.makedirs("notes", exist_ok=True)
        
#         c = canvas.Canvas(pdf_path, pagesize=letter)
#         width, height = letter
        
#         text = c.beginText(40, height - 40)
#         text.setFont("Helvetica", 12)
#         for line in note_text.split('\n'):
#             text.textLine(line)
#         c.drawText(text)
#         c.save()
        
#         # Save to database
#         c = conn.cursor()
#         c.execute("INSERT INTO notes (content, path) VALUES (?, ?)",
#                   (note_text, pdf_path))
#         conn.commit()
        
#         return pdf_path
#     except Exception as e:
#         st.error(f"PDF creation error: {e}")
#         return None

# def data_ingestion():
#     """File upload and note taking functionality"""
#     st.subheader("Upload Study Materials")
    
#     # File upload
#     uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
#     if uploaded_file:
#         upload_dir = "uploads"
#         os.makedirs(upload_dir, exist_ok=True)
#         file_path = os.path.join(upload_dir, uploaded_file.name)
        
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
            
#         # Process document
#         text, _, _ = process_document(file_path)
        
#         # Store in database
#         c = conn.cursor()
#         c.execute("INSERT INTO files (name, path, size) VALUES (?, ?, ?)",
#                   (uploaded_file.name, file_path, os.path.getsize(file_path)))
#         conn.commit()
#         st.success("File uploaded and processed successfully!")

#     # Manual notes
#     st.subheader("Create Manual Notes")
#     note_text = st.text_area("Enter your notes here:")
#     note_filename = st.text_input("Note filename:")
#     if st.button("Save Note"):
#         if note_text and note_filename:
#             pdf_path = save_note_as_pdf(note_text, f"{note_filename}.pdf")
#             if pdf_path:
#                 st.success(f"Note saved as {pdf_path}")
# def generate_filename(note_text):
#     """Generate a filename using Gemini"""
#     prompt = f"Given the content of the note between <file> and </file> describe it in at max 4 words without any special characters. DO NOT give anything extra <file> '{note_text}' </file>"
    
#     try:
#         response = generate_with_gemini(prompt)
#         if response:
#             # Clean up the filename (remove extra quotes or spaces)
#             filename = re.sub(r'["\s]+', '', response)  # Removes quotes and spaces
#             return filename
#         else:
#             return "note.pdf"  # Default filename if Gemini fails
#     except Exception as e:
#         st.error(f"Error generating filename: {e}")
#         return "note.pdf"  # Default filename on error
# def get_uploaded_files():
#     """Fetch and return the list of files from the SQLite database."""
#     try:
#         c = conn.cursor()
#         c.execute("SELECT name, size FROM files")
#         files = c.fetchall()
        
#         if not files:
#             return []
        
#         return [
#             {
#                 "File Name": file[0],
#                 "Size (KB)": round(file[1] / 1024, 2),  # Convert bytes to KB
#             }
#             for file in files
#         ]
#     except Exception as e:
#         st.error(f"Error fetching document list: {e}")
#         return []
# def delete_file(file_name):
#     """Delete a file from the SQLite database and local storage."""
#     try:
#         # Get the file path from the database
#         c = conn.cursor()
#         c.execute("SELECT path FROM files WHERE name = ?", (file_name,))
#         file_path = c.fetchone()
        
#         if file_path:
#             # Delete the file from local storage
#             os.remove(file_path[0])
            
#             # Delete the file record from the database
#             c.execute("DELETE FROM files WHERE name = ?", (file_name,))
#             conn.commit()
            
#             st.success(f"File '{file_name}' has been deleted successfully!")
#         else:
#             st.error(f"File '{file_name}' not found.")
#     except Exception as e:
#         st.error(f"Error deleting file '{file_name}': {e}")
# def uploaded_files():
#     """Display the uploaded files with an option to delete them."""
#     file_data = get_uploaded_files()

#     if not file_data:
#         st.info("No files have been uploaded yet.")
#         return

#     st.subheader("Uploaded Files")
    
#     for file in file_data:
#         col1, col2, col3 = st.columns([3, 1, 1])
        
#         col1.text(f"📄 {file['File Name']}")
#         col2.text(f"{file['Size (KB)']} KB")
        
#         if col3.button("Delete", key=file['File Name']):
#             delete_file(file['File Name'])
# def summarizer():
#     """Document summarization"""
#     st.subheader("Document Summarization")
    
#     # Get list of documents
#     c = conn.cursor()
#     c.execute("SELECT name, path FROM files")
#     documents = c.fetchall()
    
#     if not documents:
#         st.warning("No documents found!")
#         return
    
#     selected_doc = st.selectbox("Select document", [doc[0] for doc in documents])
#     word_limit = st.slider("Summary length (words)", 50, 500, 200)
    
#     if st.button("Generate Summary"):
#         doc_path = [doc[1] for doc in documents if doc[0] == selected_doc][0]
#         text, _, _ = process_document(doc_path)
        
#         prompt = f"""
#         Create a concise summary of the following document in about {word_limit} words.
#         Focus on key points and main ideas. Use clear, simple language.
        
#         Document content:
#         {text[:5000]}  # Limiting input size for demo
#         """
        
#         summary = generate_with_gemini(prompt)
#         if summary:
#             st.subheader("Summary")
#             st.write(summary)

# def quiz_generator():
#     """Quiz generation"""
#     st.subheader("Quiz Generator")
    
#     c = conn.cursor()
#     c.execute("SELECT name, path FROM files")
#     documents = c.fetchall()
    
#     if not documents:
#         st.warning("No documents found!")
#         return
    
#     selected_doc = st.selectbox("Select document", [doc[0] for doc in documents])
#     difficulty = st.selectbox("Difficulty level", ["Easy", "Medium", "Hard"])
    
#     if st.button("Generate Quiz"):
#         doc_path = [doc[1] for doc in documents if doc[0] == selected_doc][0]
#         text, _, _ = process_document(doc_path)
        
#         prompt = f"""
#         Generate a 5-question {difficulty.lower()} level quiz based on the following content.
#         Format as a valid JSON object with the following structure:
#         {{
#             "quiz": [
#                 {{
#                     "question": "",
#                     "options": ["", "", "", ""],
#                     "answer": ""
#                 }}
#             ]
#         }}
        
#         Content:
#         {text[:5000]}
#         """
        
#         quiz_json = generate_with_gemini(prompt)
#         if quiz_json:
#             try:
#                 # Clean the JSON response
#                 quiz_json = quiz_json.strip().strip('```json').strip('```')
#                 # st.write("Cleaned Quiz JSON Response:", quiz_json)  # Debugging
                
#                 # Parse the JSON
#                 quiz_data = json.loads(quiz_json)
                
#                 # Ensure the JSON has the expected structure
#                 if "quiz" in quiz_data and isinstance(quiz_data["quiz"], list):
#                     st.session_state.quiz = quiz_data["quiz"]
#                     st.session_state.user_answers = [None] * len(quiz_data["quiz"])
#                     st.session_state.correct_answers = [False] * len(quiz_data["quiz"])
#                     st.session_state.is_quiz_generated = True
#                     st.session_state.quiz_submitted = False
#                 else:
#                     st.error("Invalid quiz format. Expected a list of questions under the 'quiz' key.")
#             except json.JSONDecodeError as e:
#                 st.error(f"Failed to parse quiz: {e}. Response: {quiz_json}")
#             except Exception as e:
#                 st.error(f"An error occurred: {e}")

#     # Display the quiz if generated
#     if st.session_state.get("is_quiz_generated", False):
#         quiz = st.session_state.quiz
        
#         for i, question in enumerate(quiz):
#             st.write(f"**Question {i + 1}:** {question['question']}")
            
#             # Use a unique key for each radio button
#             user_answer = st.radio(
#                 f"Select an option for Question {i + 1}:",
#                 question["options"],
#                 key=f"question_{i}_radio"  # Unique key for each question
#             )
            
#             # Update session state with the user's answer
#             st.session_state.user_answers[i] = user_answer
        
#         if st.button("Submit Quiz") and not st.session_state.quiz_submitted:
#             st.session_state.quiz_submitted = True
#             for i, question in enumerate(quiz):
#                 st.session_state.correct_answers[i] = st.session_state.user_answers[i] == question["answer"]

#         if st.session_state.quiz_submitted:
#             total_correct = 0
#             for i, question in enumerate(quiz):
#                 if st.session_state.correct_answers[i]:
#                     st.success(f"Correct! ✅ Question {i + 1}: {question['question']}")
#                     total_correct += 1
#                 else:
#                     st.error(f"Wrong ❌ Question {i + 1}: {question['question']}")
#                     st.info(f"The correct answer is: {question['answer']}")

#             st.write(f"### Final Score: {total_correct} out of {len(quiz)}")
# def take_notes():
#     """Generate notes from a selected document"""
#     st.subheader("Generate Notes from Documents")
    
#     try:
#         # Fetch the list of documents from the database
#         c = conn.cursor()
#         c.execute("SELECT name, path FROM files")
#         documents = c.fetchall()
        
#         if not documents:
#             st.warning("No documents found in the database.")
#             return
        
#         # Display a dropdown to select a document
#         doc_list = [doc[0] for doc in documents]
#         selected_doc = st.selectbox("Choose a document to summarize", doc_list)
        
#         # Button to generate notes
#         if st.button("Generate Notes"):
#             st.info(f"Generating notes for: {selected_doc}")
            
#             # Get the path of the selected document
#             doc_path = [doc[1] for doc in documents if doc[0] == selected_doc][0]
            
#             # Extract text from the document
#             text, _, _ = process_document(doc_path)
            
#             # Generate notes using Gemini
#             prompt = (
#                 f"Make short notes from the following document content. "
#                 f"Ensure the notes are concise and presented as bullet points. "
#                 f"Focus on key concepts, definitions, and important details.\n\n"
#                 f"Document Content:\n{text[:5000]}"  # Limit input size for Gemini
#             )
            
#             notes = generate_with_gemini(prompt)
            
#             if notes:
#                 st.subheader("Generated Notes")
#                 st.text_area("Notes Preview", notes, height=300)
                
#                 # Allow users to download the notes as a text file
#                 download_filename = f"{os.path.splitext(selected_doc)[0]}_notes.txt"
#                 st.download_button(
#                     label="Download Notes",
#                     data=notes,
#                     file_name=download_filename,
#                     mime="text/plain",
#                 )
#             else:
#                 st.warning("No notes generated. Please check the document or try again.")
    
#     except Exception as e:
#         st.error(f"An error occurred: {e}")

# def init_chat_session():
#     """Initialize chat session state."""
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []
#     if "selected_doc" not in st.session_state:
#         st.session_state.selected_doc = None
#     if "use_chat_history" not in st.session_state:
#         st.session_state.use_chat_history = True

# def clear_chat_history():
#     """Clear the chat history."""
#     st.session_state.chat_history = []
#     st.rerun()

# def get_chat_history():
#     """Retrieve recent messages from the chat history."""
#     chat_history = []
#     for message in st.session_state.chat_history:
#         chat_history.append(f"{message['role']}: {message['content']}")
#     return "\n".join(chat_history)

# def document_query():
#     """Document Q&A with chat history and document selection."""
#     st.subheader("Document Query")
    
#     # Initialize chat session
#     init_chat_session()
    
#     # Fetch the list of documents from the database
#     c = conn.cursor()
#     c.execute("SELECT name, path FROM files")
#     documents = c.fetchall()
    
#     if not documents:
#         st.warning("No documents found!")
#         return
    
#     # Document selection
#     if st.session_state.selected_doc is None:
#         st.session_state.selected_doc = st.selectbox("Select a file to chat with", [doc[0] for doc in documents])
#         st.info(f"Selected document: {st.session_state.selected_doc}")
    
#     # Sidebar options
#     with st.sidebar:
#         st.subheader("Chat Settings")
#         st.session_state.use_chat_history = st.checkbox("Enable Chat History", value=st.session_state.use_chat_history)
#         if st.button("Clear Chat History"):
#             clear_chat_history()
    
#     # Display chat history
#     st.subheader("Chat")
#     for message in st.session_state.chat_history:
#         with st.chat_message(message["role"]):
#             st.write(message["content"])
    
#     # User input
#     user_input = st.chat_input("Enter your question")
    
#     if user_input:
#         # Add user message to chat history
#         st.session_state.chat_history.append({"role": "user", "content": user_input})
        
#         # Get the path of the selected document
#         doc_path = [doc[1] for doc in documents if doc[0] == st.session_state.selected_doc][0]
        
#         # Extract text from the document
#         text, index, chunks = process_document(doc_path)
        
#         # Find relevant chunks using FAISS
#         query_embedding = embedding_model.encode([user_input])
#         _, indices = index.search(np.array(query_embedding).astype('float32'), k=3)
        
#         # Combine relevant chunks into context
#         context = " ".join([chunks[i] for i in indices[0]])
        
#         # Generate prompt with chat history (if enabled)
#         if st.session_state.use_chat_history:
#             chat_history = get_chat_history()
#             prompt = f"""
#             Answer this question based on the provided context and chat history.
#             Chat History:
#             {chat_history}
            
#             Question: {user_input}
#             Context: {context}
#             If the answer isn't in the context, say you don't know.
#             Provide a concise answer in 2-3 sentences.
#             """
#         else:
#             prompt = f"""
#             Answer this question based on the provided context.
#             Question: {user_input}
#             Context: {context}
#             If the answer isn't in the context, say you don't know.
#             Provide a concise answer in 2-3 sentences.
#             """
        
#         # Get AI response using Gemini
#         answer = generate_with_gemini(prompt)
        
#         if answer:
#             # Add AI response to chat history
#             st.session_state.chat_history.append({"role": "assistant", "content": answer})
            
#             # Display AI response in a streaming fashion
#             with st.chat_message("assistant"):
#                 response_container = st.empty()
#                 full_response = ""
#                 for chunk in answer.split():
#                     full_response += chunk + " "
#                     response_container.markdown(full_response + "▌")
#                     time.sleep(0.1)  # Simulate streaming
#                 response_container.markdown(full_response)
#     # Clear chat history button
#     # if st.button("Clear Chat History"):
#     #     st.session_state.chat_history = []
#     #     st.rerun()  # Refresh the app to reflect the cleared chat history
                
# def main():
#     st.title("Study Assistant - Gemini Edition")
    
#     menu = ["Data Ingestion", "Summarizer", "Quiz Generator", "Document Query", "Notes", "Uploaded Files"]
#     choice = st.sidebar.selectbox("Menu", menu)
    
#     if choice == "Data Ingestion":
#         data_ingestion()
#     elif choice == "Summarizer":
#         summarizer()
#     elif choice == "Quiz Generator":
#         quiz_generator()
#     elif choice == "Document Query":
#         document_query()  # Includes chat history
#     elif choice == "Notes":
#         take_notes()
#     elif choice == "Uploaded Files":
#         uploaded_files()

# if __name__ == "__main__":
#     main()
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
import time  # Import the time module

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
def generate_filename(note_text):
    """Generate a filename using Gemini"""
    prompt = f"Given the content of the note between <file> and </file> describe it in at max 4 words without any special characters. DO NOT give anything extra <file> '{note_text}' </file>"
    
    try:
        response = generate_with_gemini(prompt)
        if response:
            # Clean up the filename (remove extra quotes or spaces)
            filename = re.sub(r'["\s]+', '', response)  # Removes quotes and spaces
            return filename
        else:
            return "note.pdf"  # Default filename if Gemini fails
    except Exception as e:
        st.error(f"Error generating filename: {e}")
        return "note.pdf"  # Default filename on error
def get_uploaded_files():
    """Fetch and return the list of files from the SQLite database."""
    try:
        c = conn.cursor()
        c.execute("SELECT name, size FROM files")
        files = c.fetchall()
        
        if not files:
            return []
        
        return [
            {
                "File Name": file[0],
                "Size (KB)": round(file[1] / 1024, 2),  # Convert bytes to KB
            }
            for file in files
        ]
    except Exception as e:
        st.error(f"Error fetching document list: {e}")
        return []
def delete_file(file_name):
    """Delete a file from the SQLite database and local storage."""
    try:
        # Get the file path from the database
        c = conn.cursor()
        c.execute("SELECT path FROM files WHERE name = ?", (file_name,))
        file_path = c.fetchone()
        
        if file_path:
            # Delete the file from local storage
            os.remove(file_path[0])
            
            # Delete the file record from the database
            c.execute("DELETE FROM files WHERE name = ?", (file_name,))
            conn.commit()
            
            st.success(f"File '{file_name}' has been deleted successfully!")
        else:
            st.error(f"File '{file_name}' not found.")
    except Exception as e:
        st.error(f"Error deleting file '{file_name}': {e}")
def uploaded_files():
    """Display the uploaded files with an option to delete them."""
    file_data = get_uploaded_files()

    if not file_data:
        st.info("No files have been uploaded yet.")
        return

    st.subheader("Uploaded Files")
    
    for file in file_data:
        col1, col2, col3 = st.columns([3, 1, 1])
        
        col1.text(f"📄 {file['File Name']}")
        col2.text(f"{file['Size (KB)']} KB")
        
        if col3.button("Delete", key=file['File Name']):
            delete_file(file['File Name'])
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
                # st.write("Cleaned Quiz JSON Response:", quiz_json)  # Debugging
                
                # Parse the JSON
                quiz_data = json.loads(quiz_json)
                
                # Ensure the JSON has the expected structure
                if "quiz" in quiz_data and isinstance(quiz_data["quiz"], list):
                    st.session_state.quiz = quiz_data["quiz"]
                    st.session_state.user_answers = [None] * len(quiz_data["quiz"])
                    st.session_state.correct_answers = [False] * len(quiz_data["quiz"])
                    st.session_state.is_quiz_generated = True
                    st.session_state.quiz_submitted = False
                else:
                    st.error("Invalid quiz format. Expected a list of questions under the 'quiz' key.")
            except json.JSONDecodeError as e:
                st.error(f"Failed to parse quiz: {e}. Response: {quiz_json}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Display the quiz if generated
    if st.session_state.get("is_quiz_generated", False):
        quiz = st.session_state.quiz
        
        for i, question in enumerate(quiz):
            st.write(f"**Question {i + 1}:** {question['question']}")
            
            # Use a unique key for each radio button
            user_answer = st.radio(
                f"Select an option for Question {i + 1}:",
                question["options"],
                key=f"question_{i}_radio"  # Unique key for each question
            )
            
            # Update session state with the user's answer
            st.session_state.user_answers[i] = user_answer
        
        if st.button("Submit Quiz") and not st.session_state.quiz_submitted:
            st.session_state.quiz_submitted = True
            for i, question in enumerate(quiz):
                st.session_state.correct_answers[i] = st.session_state.user_answers[i] == question["answer"]

        if st.session_state.quiz_submitted:
            total_correct = 0
            for i, question in enumerate(quiz):
                if st.session_state.correct_answers[i]:
                    st.success(f"Correct! ✅ Question {i + 1}: {question['question']}")
                    total_correct += 1
                else:
                    st.error(f"Wrong ❌ Question {i + 1}: {question['question']}")
                    st.info(f"The correct answer is: {question['answer']}")

            st.write(f"### Final Score: {total_correct} out of {len(quiz)}")
def take_notes():
    """Generate notes from a selected document"""
    st.subheader("Generate Notes from Documents")
    
    try:
        # Fetch the list of documents from the database
        c = conn.cursor()
        c.execute("SELECT name, path FROM files")
        documents = c.fetchall()
        
        if not documents:
            st.warning("No documents found in the database.")
            return
        
        # Display a dropdown to select a document
        doc_list = [doc[0] for doc in documents]
        selected_doc = st.selectbox("Choose a document to summarize", doc_list)
        
        # Button to generate notes
        if st.button("Generate Notes"):
            st.info(f"Generating notes for: {selected_doc}")
            
            # Get the path of the selected document
            doc_path = [doc[1] for doc in documents if doc[0] == selected_doc][0]
            
            # Extract text from the document
            text, _, _ = process_document(doc_path)
            
            # Generate notes using Gemini
            prompt = (
                f"Make short notes from the following document content. "
                f"Ensure the notes are concise and presented as bullet points. "
                f"Focus on key concepts, definitions, and important details.\n\n"
                f"Document Content:\n{text[:5000]}"  # Limit input size for Gemini
            )
            
            notes = generate_with_gemini(prompt)
            
            if notes:
                st.subheader("Generated Notes")
                st.text_area("Notes Preview", notes, height=300)
                
                # Allow users to download the notes as a text file
                download_filename = f"{os.path.splitext(selected_doc)[0]}_notes.txt"
                st.download_button(
                    label="Download Notes",
                    data=notes,
                    file_name=download_filename,
                    mime="text/plain",
                )
            else:
                st.warning("No notes generated. Please check the document or try again.")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")

def init_chat_session():
    """Initialize chat session state."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "selected_doc" not in st.session_state:
        st.session_state.selected_doc = None
    if "use_chat_history" not in st.session_state:
        st.session_state.use_chat_history = True

def clear_chat_history():
    """Clear the chat history."""
    st.session_state.chat_history = []
    st.rerun()

def get_chat_history():
    """Retrieve recent messages from the chat history."""
    chat_history = []
    for message in st.session_state.chat_history:
        chat_history.append(f"{message['role']}: {message['content']}")
    return "\n".join(chat_history)

from langchain_community.tools.tavily_search import TavilySearchResults

# Set the Tavily API key
os.environ["TAVILY_API_KEY"] = "your_tavily_api_key"  # Replace with your actual Tavily API key
web_search_tool = TavilySearchResults(k=3)  # Retrieve top 3 results

def document_query():
    """Document Q&A with chat history, document selection, and web search fallback."""
    st.subheader("Document Query")
    
    # Initialize chat session
    init_chat_session()
    
    # Fetch the list of documents from the database
    c = conn.cursor()
    c.execute("SELECT name, path FROM files")
    documents = c.fetchall()
    
    if not documents:
        st.warning("No documents found!")
        return
    
    # Document selection
    if st.session_state.selected_doc is None:
        st.session_state.selected_doc = st.selectbox("Select a file to chat with", [doc[0] for doc in documents])
        st.info(f"Selected document: {st.session_state.selected_doc}")
    
    # Sidebar options
    with st.sidebar:
        st.subheader("Chat Settings")
        st.session_state.use_chat_history = st.checkbox("Enable Chat History", value=st.session_state.use_chat_history)
        if st.button("Clear Chat History"):
            clear_chat_history()
    
    # Display chat history
    st.subheader("Chat")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # User input
    user_input = st.chat_input("Enter your question")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display the user's question
        with st.chat_message("user"):
            st.write(f"**Question:** {user_input}")
        
        # Get the path of the selected document
        doc_path = [doc[1] for doc in documents if doc[0] == st.session_state.selected_doc][0]
        
        # Extract text from the document
        text, index, chunks = process_document(doc_path)
        
        # Find relevant chunks using FAISS
        query_embedding = embedding_model.encode([user_input])
        _, indices = index.search(np.array(query_embedding).astype('float32'), k=3)
        
        # Combine relevant chunks into context
        context = " ".join([chunks[i] for i in indices[0]])
        
        # Generate prompt with chat history (if enabled)
        if st.session_state.use_chat_history:
            chat_history = "\n".join(
                [f"{message['role']}: {message['content']}" for message in st.session_state.chat_history]
            )
            prompt = f"""
            You are a helpful assistant. Answer the user's question based on the provided context and chat history.
            Chat History:
            {chat_history}
            
            Context:
            {context}
            
            Question: {user_input}
            If the answer isn't in the context, say you don't know.
            Provide a concise and accurate answer in 2-3 sentences.
            """
        else:
            prompt = f"""
            You are a helpful assistant. Answer the user's question based on the provided context.
            Context:
            {context}
            
            Question: {user_input}
            If the answer isn't in the context, say you don't know.
            Provide a concise and accurate answer in 2-3 sentences.
            """
        
        # Get AI response using Gemini
        answer = generate_with_gemini(prompt)
        
        if answer:
            # Check if the assistant couldn't find the answer in the context
            if "I don't know" in answer or "not in the context" in answer:
                # Perform a web search using Tavily
                web_search_results = web_search_tool.invoke({"query": user_input})
                
                # Generate a new prompt with the web search results
                web_search_prompt = f"""
                The original query was: {user_input}

                Here are some web search results that might be relevant:
                {web_search_results}

                Please provide a structured answer based on the above information.
                """
                
                # Generate the final answer using Gemini
                web_search_answer = generate_with_gemini(web_search_prompt)
                
                # Update the answer with web search results
                answer = f"{answer}\n\n**Web Search Results:**\n{web_search_answer}"
            
            # Add AI response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            
            # Display AI response in a streaming fashion
            with st.chat_message("assistant"):
                response_container = st.empty()
                full_response = ""
                for chunk in answer.split():
                    full_response += chunk + " "
                    response_container.markdown(f"**Answer:** {full_response}▌")
                    time.sleep(0.1)  # Simulate streaming
                response_container.markdown(f"**Answer:** {full_response}")
def main():
    st.title("Study Assistant - Gemini Edition")
    
    menu = ["Data Ingestion", "Summarizer", "Quiz Generator", "Document Query", "Notes", "Uploaded Files"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Data Ingestion":
        data_ingestion()
    elif choice == "Summarizer":
        summarizer()
    elif choice == "Quiz Generator":
        quiz_generator()
    elif choice == "Document Query":
        document_query()  # Includes chat history
    elif choice == "Notes":
        take_notes()
    elif choice == "Uploaded Files":
        uploaded_files()

if __name__ == "__main__":
    main()
