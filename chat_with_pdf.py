import streamlit as st
from openai import OpenAI
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")


client = OpenAI(
	api_key=os.environ["OPENAI_API_KEY"],
	base_url=os.environ["OPENAI_BASE_URL"],
)

st.title("📝 File Q&A with OpenAI")
uploaded_files = st.file_uploader("Upload a document", type=("txt", "pdf"), accept_multiple_files=True)

question = st.chat_input(
    "Ask something about the document",
    disabled=not uploaded_files,
)

# Keep track of conversation state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask something about the article"}]

# Maintain a list of already processed documents to avoid duplication
if "existing_docs" not in st.session_state:
    st.session_state["existing_docs"] = []

# Maintain the vectorstore in session state to avoid re-computing for each user query
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question and uploaded_files:
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        # Check if the file has already been processed (don't want to duplicate it in the vectorstore)
        if uploaded_file.file_id in st.session_state.existing_docs:
            print(f"File {uploaded_file.name} already processed.")
            continue
        st.session_state.existing_docs.append(uploaded_file.file_id)
        print(f"Processing file: {uploaded_file.name}")

        # Read the content of the uploaded file
        # Extract the file type
        file_extension = uploaded_file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            # Write the upload into a temp file to be compatible with LangChain
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Handle different extensions differently with LangChain
        if file_extension == "txt":
            loader = TextLoader(tmp_path)
        elif file_extension == "pdf":
            loader = PyPDFLoader(tmp_path)

        # Load the documents
        documents = loader.load()

        # Chunk the documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 200,
            chunk_overlap = 0
        )
        chunks = text_splitter.split_documents(documents)

        # Calculate total token usage for pricing estimation
        total_tokens = 0
        for chunk in chunks:
            tokens = len(enc.encode(chunk.page_content))
            total_tokens += tokens
        print(f"Total tokens in all chunks: {total_tokens}")

        # We only want to create a vectorstore once, then we can add to it for subsequent files
        if st.session_state.vectorstore is None:
            print("Creating new vectorstore")
            st.session_state.vectorstore = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings(model="openai.text-embedding-3-large"))
        else:
            print("Adding to existing vectorstore")
            st.session_state.vectorstore.add_documents(chunks)

    def format_docs(docs):
        return "\n\n---\n\n".join(d.page_content for d in docs)

    docs = st.session_state.vectorstore.similarity_search(question, k=5)

    context = format_docs(docs)
    system_instructions = (
        "You are a helpful assistant for question answering.\n"
        "Use ONLY the provided context to answer concisely (<=3 sentences).\n"
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}"
    )

    # Append the user's question to the messages
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.chat_message("assistant"):
        # Stream the response for better user experience
        stream = client.chat.completions.create(
            model="openai.gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instructions},
                *st.session_state.messages
            ],
            stream=True
        )
        response = st.write_stream(stream)

        # Calculate total token usage for pricing estimation
        total_tokens = 0
        for message in st.session_state.messages:
            tokens = len(enc.encode(message["content"]))
            total_tokens += tokens
        total_tokens += len(enc.encode(system_instructions))
        print(f"Total tokens in query: {total_tokens}")

    # Append the assistant's response to the messages
    st.session_state.messages.append({"role": "assistant", "content": response})