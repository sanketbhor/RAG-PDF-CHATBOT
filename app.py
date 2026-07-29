import streamlit as st

from pdf_loader import extract_text
from rag_engine import (
    split_text,
    create_vector_store,
    ask_question
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("📄 RAG PDF Chatbot")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    with st.spinner("Reading PDFs..."):

        all_text = ""

        for pdf in uploaded_files:
            all_text += extract_text(pdf)
            all_text += "\n\n"

        chunks = split_text(all_text)

        vectordb = create_vector_store(chunks)

    st.success("✅ PDFs Processed!")

    question = st.text_input("Ask a question")

    if question:

        answer, docs = ask_question(
            vectordb,
            question
        )

        st.write(answer)

        # Save chat history
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })

        # Show source chunks
        st.subheader("📄 Source Chunks")

        for i, doc in enumerate(docs):

            with st.expander(f"Chunk {i+1}"):

                st.write(doc.page_content)

# Conversation History
if st.session_state.chat_history:

    st.subheader("💬 Conversation")

    for chat in st.session_state.chat_history:

        st.markdown(f"**🧑 You:** {chat['question']}")
        st.markdown(f"**🤖 AI:** {chat['answer']}")
        st.divider()

# Clear Chat Button
if st.button("🗑 Clear Chat"):

    st.session_state.chat_history = []

    st.rerun()