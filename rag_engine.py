import google.generativeai as genai
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")



def split_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )

    chunks = splitter.split_text(text)

    return chunks


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding= embeddings,
        persist_directory="chroma_db"
    )

    return vectordb

def retrieve_context(vectordb, question):

    docs = vectordb.similarity_search(
        question,
        k=3
    )

    return docs

genai.configure(
    api_key = GEMINI_API_KEY
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)



def ask_question(
        vectordb,
        question
):
    
    docs = retrieve_context(
        vectordb,
        question
    )
    context = "\n\n".join(
    [doc.page_content for doc in docs]
)

    prompt = f"""

You are an AI document assistant.
Answer ONLY from the supplied context.
If the answer is not present in the context,
respond exactly:
"I could not find that information in the uploaded document."
Never make up information.
Provide concise and accurate answers.

Context:
{context}

Question:
{question}

"""
    
    response = model.generate_content(
        prompt
    )

    if response.candidates:
        candidate = response.candidates[0]

        if candidate.content.parts:
            return candidate.content.parts[0].text, docs

    return "No answer generated.", docs