from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings

from langchain_community.vectorstores import Chroma
from chromadb import Client
from chromadb.config import Settings

from langchain_community.llms.ollama import Ollama
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from langchain_core.runnables import RunnablePassthrough

app = FastAPI()

class AnswerResponse(BaseModel):
    question: str
    answer: str

class AIResponse(BaseModel):
    answer: str

# Globals for single reusable instances
_global_embedding_model = None
_global_vector_store = None
_global_retriever = None
_global_ollama_llm = None

class LLM:
    def __init__(self, model: str, num_predict: int, temperature: float):
        self.model = model
        self.num_predict = num_predict
        self.temperature = temperature

    def create_embeddings_model(self):
        global _global_embedding_model, _global_ollama_llm
        if _global_embedding_model is None:
            _global_ollama_llm = Ollama(model=self.model, temperature=self.temperature, num_predict=self.num_predict)
            start_time = time.time()
            _global_ollama_llm.invoke("warm up")
            print("time taken 11: ", time.time() - start_time)
            _global_embedding_model = OllamaEmbeddings(model=self.model)
        return _global_embedding_model

    def create_vector_store(self, chunks: list[Document]) -> Chroma:
        """
        Create the global vector store once. Subsequent calls reuse it.
        We persist the Chroma DB to ./chroma_db so it can survive restarts.
        """
        global _global_vector_store
        if _global_vector_store is None:
            client = Client(Settings(persist_directory="./chroma_db", anonymized_telemetry=False))
            _global_vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.create_embeddings_model(),
                collection_name="fastapi_rag_collection",
                client=client,
            )
        # If you need to add new documents on the fly, implement add_documents here.
        return _global_vector_store

    def create_retriever_from_text(self, docs: list[str]):
        """
        Split docs into chunks, ensure vector store exists, and return the global retriever.
        """
        global _global_retriever
        # Build Document objects and split into chunks
        documents = [Document(page_content=text) for text in docs]
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        # Ensure the global vector store is created (first call will create it)
        vs = self.create_vector_store(chunks)
        if _global_retriever is None:
            _global_retriever = vs.as_retriever(search_kwargs={"k": 3})
        return _global_retriever

    def retrieve_context(self, inputs):
        global _global_retriever
        retriever = _global_retriever
        # some retrievers expose `invoke`, others use `get_relevant_documents`
        if hasattr(retriever, "invoke"):
            start_time = time.time()
            docs = retriever.invoke(inputs["question"])
            print("time taken 8: ", time.time() - start_time)
        else:
            start_time = time.time()
            docs = retriever.get_relevant_documents(inputs["question"])
            print("time taken 9: ", time.time() - start_time)
        start_time = time.time()
        result = "\n".join([d.page_content for d in docs])
        print("time taken 10: ", time.time() - start_time)
        print("Retrieved context:", result)
        return result

    async def generate(self, question: str, docs: list[str]) -> dict:
        start_time = time.time()
        global _global_ollama_llm
        ollama_llm = _global_ollama_llm
        print("time taken 1: ", time.time() - start_time)
        start_time = time.time()
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a financial assistant. "
             "Answer using only the provided context.\n\n"
             "Context:\n{context}\n\n"
            #  "{format_instructions}"
            ),
            ("human", "{question}")
        ])
        print("time taken 2: ", time.time() - start_time)
        start_time = time.time()
        # parser = JsonOutputParser(pydantic_object=AIResponse)
        print("time taken 3: ", time.time() - start_time)
        start_time = time.time()
        reg_chain = (
            {
                "context": self.retrieve_context,
                "question": RunnablePassthrough(),
                # "format_instructions": lambda _: parser.get_format_instructions(),
            } | prompt | ollama_llm
            # | parser
        )
        print("time taken 4: ", time.time() - start_time)
        start_time = time.time()
        response = reg_chain.invoke({"question": question})
        print("time taken 5: ", time.time() - start_time)
        return response

# Create one LLM instance and initialize global retriever/vector store at startup
llm = LLM(model="phi3:mini", num_predict=150, temperature=0)

@app.on_event("startup")
async def startup_event():
    # initial docs to populate the persistent global store once
    initial_docs = [
        "SIP is a disciplined way to invest in mutual funds.",
        "SIP helps in rupee cost averaging.",
        "Long-term SIPs benefit from compounding.",
        "SIP stands for Systematic Investment Plan. It helps people invest small amounts regularly.",
        "Mutual funds pool money from many investors and invest in stocks or bonds.",
        "Long-term investing helps reduce risk and improve returns.",
        "Romil is working as a software engineer in LTI Mindtree organization and living in Noida."
    ]
    try:
        # This will create the embedding model, vector store and retriever if not present
        llm.create_retriever_from_text(initial_docs)
        print("Global embedding model, vector store and retriever initialized.")
    except Exception as e:
        print("Failed to initialize global store; will create on-demand. Error:", e)

@app.get("/knowledge-chat", response_model=AnswerResponse)
async def knowledge_chat(question: str):
    start_time = time.time()
    response = await llm.generate(question=question, docs=[])
    print("time taken 6: ", time.time() - start_time)
    # If you want to use the startup docs by default, pass them in; here we reuse retriever's docs
    # If response is a pydantic object, access .answer; if a dict, use ["answer"]
    start_time = time.time()
    if hasattr(response, "answer"):
        answer_text = response.answer
    elif isinstance(response, dict):
        answer_text = response.get("answer", "")
    else:
        answer_text = str(response)

    print("time taken 7: ", time.time() - start_time)
    return AnswerResponse(question=question, answer=answer_text)

if __name__ == "__main__":
    # run module app; change module path if you rename file
    uvicorn.run("rag:app", host="127.0.0.1", port=8000, reload=True)