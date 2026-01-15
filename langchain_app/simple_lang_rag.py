import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from chromadb import Client
from chromadb.config import Settings
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_classic.chains import RetrievalQA

# simple_lang_rag.py (adapted to the imports at top)
# Requirements (examples):
#   pip install streamlit langchain chromadb ollama langchain_text_splitters
# Run:
#   streamlit run simple_lang_rag.py

st.set_page_config(page_title="Ollama + LangChain RAG (chromadb)", layout="wide")

# --- Helpers ----
def get_llm():
    return Ollama(model="phi3", base_url="http://localhost:11434", verbose=False)

def create_retriever_from_text(
    text,
    embeddings_model=None,
    chunk_size=500,
    chunk_overlap=50,
    k=3,
    persist_directory="./chromadb_store",
    collection_name="langchain_demo",
):
    # default to OllamaEmbeddings (imported at top)
    if embeddings_model is None:
        embeddings_model = OllamaEmbeddings(model="phi3")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    if not chunks:
        raise ValueError("No text chunks to index.")

    docs = [Document(page_content=c) for c in chunks]
    
    # prefer to pass client settings to Chroma; fall back to alternate constructors for compatibility
    try:
        client_settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory,
            anonymized_telemetry=False,
        )
        chroma = Chroma.from_documents(
            documents=docs,
            embedding=embeddings_model,
            collection_name=collection_name,
            client_settings=client_settings,
        )
    except Exception:
        # fallback: try from_documents with persist_directory param
        try:
            chroma = Chroma.from_documents(
                documents=docs,
                embedding=embeddings_model,
                collection_name=collection_name,
                persist_directory=persist_directory,
            )
        except Exception:
            # final fallback: create chromadb client manually and then Chroma
            try:
                client = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))
            except Exception:
                client = Client()
            # Chroma may accept a chromadb client directly depending on version
            chroma = Chroma.from_documents(
                documents=docs,
                embedding=embeddings_model,
                collection_name=collection_name,
                client=client,
            )

    # create retriever with desired k
    retriever = chroma.as_retriever(search_kwargs={"k": k})
    # return retriever and the chroma vectorstore (and optionally the client if available)
    client_obj = None
    try:
        client_obj = getattr(chroma, "_client", None) or getattr(chroma, "client", None)
    except Exception:
        client_obj = None
    return retriever, chroma, client_obj

# --- UI ----
st.title("Simple RAG with Ollama + LangChain (chromadb)")
st.markdown("Ingest a short paragraph, then ask questions about it. Ollama is used as the LLM; embeddings use OllamaEmbeddings and vector store uses chromadb via Chroma.")

default_paragraph = """LangChain is a framework for developing applications powered by language models.
It provides components for prompt management, LLM wrappers, chains, and vector-based retrieval (RAG).
This simple demo ingests text into a chromadb-backed vector store and answers user questions using a retrieval-augmented chain."""

with st.expander("Example paragraph", expanded=False):
    st.write(default_paragraph)

paragraph = st.text_area("Paragraph to ingest (or paste your text)", value=default_paragraph, height=200)
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Ingest paragraph"):
        if not paragraph.strip():
            st.warning("Paste a paragraph to ingest first.")
        else:
            with st.spinner("Creating embeddings and vector store (chromadb)..."):
                try:
                    retriever, chroma_store, client = create_retriever_from_text(paragraph)
                except Exception as e:
                    st.error(f"Failed to create vector store: {e}")
                    raise
                st.session_state["retriever"] = retriever
                st.session_state["chroma_store"] = chroma_store
                st.session_state["chromadb_client"] = client
                st.success("Ingested paragraph into chromadb-backed Chroma vector store.")

with col2:
    if "retriever" in st.session_state:
        st.info("Vector store ready. Ask questions in the box below.")
    else:
        st.info("No vector store yet. Click 'Ingest paragraph' to create one.")

query = st.text_input("Ask a question about the ingested paragraph")

if st.button("Ask") and query.strip():
    if "retriever" not in st.session_state:
        st.warning("Please ingest a paragraph first.")
    else:
        llm = None
        try:
            llm = get_llm()
        except Exception as e:
            st.error(f"Could not initialize Ollama LLM: {e}")
        if llm:
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    qa = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state["retriever"],
                        return_source_documents=True,
                    )
                    result = qa({"query": query})
                    answer = result.get("result") or result.get("output_text") or ""
                    sources = result.get("source_documents") or []
                    st.markdown("### Answer")
                    st.write(answer)
                    if sources:
                        st.markdown("### Retrieved chunks")
                        for i, doc in enumerate(sources, 1):
                            content = getattr(doc, "page_content", str(doc))
                            st.write(f"Chunk {i}: {content}")
                except Exception as e:
                    st.error(f"Error during QA: {e}")

# Optional: small UI to clear session (for re-ingest)
if st.button("Clear ingested data"):
    for key in ["retriever", "chroma_store", "chromadb_client"]:
        if key in st.session_state:
            del st.session_state[key]
    st.stop()
