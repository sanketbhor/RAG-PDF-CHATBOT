# RAG PDF Chatbot

Upload one or more PDFs and ask questions about them — answers are generated only from the document content, with source chunks shown for verification, so you're not stuck ctrl+F-ing through long PDFs or trusting an LLM's memory instead of your actual document.

**🔴 Live demo: _[coming soon — deploying to Streamlit Community Cloud]_**

## Architecture

```
User → Streamlit UI → PDF Loader (pypdf) → Text Splitter (LangChain)
                                                  │
                                                  ▼
                                    HuggingFace Embeddings (MiniLM-L6-v2)
                                                  │
                                                  ▼
                                          Chroma Vector Store
                                                  │
                              question ──► similarity search (top-3 chunks)
                                                  │
                                                  ▼
                                         Gemini 2.5 Flash (context-grounded)
                                                  │
                                                  ▼
                                        Answer + cited source chunks
```

## Tech stack

- **Streamlit** — UI and chat/session state
- **Google Gemini 2.5 Flash** — answer generation
- **LangChain** — text splitting and Chroma integration
- **ChromaDB** — vector store for retrieval
- **HuggingFace `sentence-transformers/all-MiniLM-L6-v2`** — embeddings
- **pypdf** — PDF text extraction

## How it works

1. Upload one or more PDFs — text is extracted and chunked (1000 chars, 200 overlap).
2. Chunks are embedded and stored in a Chroma vector store.
3. Each question retrieves the top-3 most relevant chunks and passes them to Gemini as grounding context.
4. Gemini answers strictly from that context — if the answer isn't in the document, it says so instead of guessing.
5. Source chunks are shown alongside every answer so you can verify it.

## Run locally

```bash
git clone https://github.com/sanketbhor/RAG-PDF-CHATBOT.git
cd RAG-PDF-CHATBOT
pip install -r requirements.txt
```

Add your Gemini API key (get one from [Google AI Studio](https://aistudio.google.com/apikey)):

```bash
cp .env.example .env
# then edit .env and set GEMINI_API_KEY
```

Run the app:

```bash
streamlit run app.py
```

Open `http://localhost:8501`, upload a PDF, and start asking questions.
