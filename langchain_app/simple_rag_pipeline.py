from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from chromadb import Client
from chromadb.config import Settings

# -------- 1. Sample Knowledge (Very Small) --------
texts = [
    "SIP stands for Systematic Investment Plan. It helps people invest small amounts regularly.",
    "Mutual funds pool money from many investors and invest in stocks or bonds.",
    "Long-term investing helps reduce risk and improve returns.",
    "Romil is working as a software engineer in LTI Mindtree organization."
]

documents = [Document(page_content=text) for text in texts]

# -------- 2. Split Text --------
splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)
chunks = splitter.split_documents(documents)

# -------- 3. Create Chroma Client --------
client = Client(Settings(persist_directory="./chroma_db", anonymized_telemetry=False))

# -------- 4. Embeddings --------
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# -------- 5. Create Vector Store --------
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="simple_rag_collection",
    client=client,
)

# -------- 6. Load LLM (phi3) --------
llm = Ollama(model="phi3", num_predict=100, temperature=0)

# -------- 7. Simple Chatbot Loop --------
print("\n🤖 RAG Chatbot is ready! Type 'exit' to quit.\n")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        print("Goodbye 👋")
        break
    # retrieve relevant documents from the vector store
    relevant_docs = vector_store.similarity_search(query, k=2)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    # Prompt
    prompt = f"""
        Use the context below to answer the question.

        Context:
        {context}

        Question:
        {query}

        Answer in simple language:
    """
    # Ollama LLM instance is not callable; use generate() and extract the text
    result = llm.generate([prompt])
    response = result.generations[0][0].text
    print(f"🤖 Bot: {response}")
