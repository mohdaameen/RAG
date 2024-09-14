# Install necessary packages

# Import required libraries
import streamlit as st
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import tempfile
from groq import Groq  # Placeholder for Groq or any other LLM API

load_dotenv()

# Function to process the uploaded file, create embeddings, and initialize vector store
def process_file(uploaded_file):
    if uploaded_file is not None:
        # Create a temporary file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            file_path = temp_file.name

        # Load the file
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()

        # Split the documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        # Create embeddings
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cpu"}
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

        # Create a vector store
        db = Chroma.from_documents(docs, embeddings, persist_directory=None)

        return db
    return None

# Function to generate chatbot response using RAG
def generate_chatbot_response(query, db):
    """Generates a response based on relevant document chunks."""
    
    # Retrieve relevant documents based on the query
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.2},  # Adjust k and threshold as needed
    )
    relevant_docs = retriever.invoke(query)

    # If no relevant documents found, return a fallback response
    if not relevant_docs:
        return "I'm not sure about that. Could you ask something else?"

    # Join the retrieved chunks as context for generation
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Assuming Groq API client is correctly configured (replace with actual Groq client setup)
    client = Groq()  # Replace with actual API key

    # Generate response using the retrieved context
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",  # Replace with the appropriate model name
        messages=[
            {
                "role": "system",
                "content": f"Use the following context to assist the user:\n\n{context} if no context, reply 'not sure'."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0.2,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""
    
    return response

# Streamlit UI
st.title("RAG-based Chatbot with File Upload")

# File upload interface
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

# Process the uploaded file and create embeddings
if uploaded_file:
    db = process_file(uploaded_file)
    st.success("File processed and embeddings created!")

    # Chatbot interface
    st.subheader("Ask the Chatbot")
    query = st.text_input("Enter your question")

    if query:
        response = generate_chatbot_response(query, db)
        st.write(f"Chatbot Response: {response}")

