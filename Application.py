import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import boto3

st.set_page_config(page_title="Financial RAG Assistant")

st.title("📊 Financial RAG Assistant")

# Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader(
    "Upload Financial PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    bucket_name = "financial-rag-pdf-bucket"

    s3 = boto3.client("s3")

    # Upload PDF to S3
    s3.upload_fileobj(
        uploaded_file,
        bucket_name,
        uploaded_file.name
    )

    st.success("PDF uploaded to S3 successfully")

    # Download PDF from S3
    s3.download_file(
        bucket_name,
        uploaded_file.name,
        "uploaded.pdf"
    )

    # Read PDF
    reader = PdfReader("uploaded.pdf")

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    # Create chunks
    chunk_size = 50

    chunks = [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]

    # Embedding Model
    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings = model.encode(chunks)

    # FAISS Index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    st.success("Knowledge Base Created")

    question = st.text_input(
        "Ask a question about the PDF"
    )

    if st.button("Get Answer"):

        question_embedding = model.encode(
            [question]
        )

        D, I = index.search(
            np.array(question_embedding),
            k=3
        )

        answer = ""

        for idx in I[0]:
            if idx < len(chunks):
                answer += chunks[idx] + "\n"

        final_answer = answer

        patterns = [
            "Revenue",
            "Net Profit",
            "Operating Margin",
            "Employees",
            "Total Assets",
            "Total Liabilities"
        ]

        found = False

        for item in patterns:

            if item.lower() in question.lower():

                match = re.search(
                    rf"{item}[:\s]*([^\n]+)",
                    answer,
                    re.IGNORECASE
                )

                if match:
                    final_answer = (
                        f"{item}: {match.group(1).strip()}"
                    )
                    found = True
                    break

        # Employee variations
        if not found and (
            "employee" in question.lower()
            or "employees" in question.lower()
        ):

            match = re.search(
                r"Employees[:\s]*([^\n]+)",
                answer,
                re.IGNORECASE
            )

            if match:
                final_answer = (
                    f"Employees: {match.group(1).strip()}"
                )

        st.subheader("Answer")
        st.success(final_answer)

        # Save Chat History
        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": final_answer
            }
        )

# Chat History
if len(st.session_state.chat_history) > 0:

    st.markdown("---")
    st.subheader("💬 Chat History")

    for chat in reversed(
        st.session_state.chat_history
    ):

        st.markdown(
            f"**👤 Question:** {chat['question']}"
        )

        st.markdown(
            f"**🤖 Answer:** {chat['answer']}"
        )

        st.markdown("---")
