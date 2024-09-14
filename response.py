import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")


model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

query = "What does hotpenguin offer? "


retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.3},
)
relevant_docs = retriever.invoke(query)



def generate_chatbot_response(query, db):
    """Generates a response from the Groq API based on relevant document chunks."""
    
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.2},
    )
    relevant_docs = retriever.invoke(query)

    print(relevant_docs)

   
    context = "\n".join([doc.page_content for doc in relevant_docs])

    client = Groq()

    
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"Use the following context to assist the user:\n\n{context} if no context reply not sure"
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

print(generate_chatbot_response('Tell me about botpenguin?',db))
