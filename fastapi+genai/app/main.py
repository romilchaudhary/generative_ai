from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

app = FastAPI(title="QA Service")

llm = ChatOllama(model="phi3", temperature=0)

class AnswerResponse(BaseModel):
    question: str
    answer: str

memory_store = {}

def get_chat_history(session_id: str):
    if session_id not in memory_store:
        memory_store[session_id] = InMemoryChatMessageHistory()
    return memory_store[session_id]

def generate_answer(question: str) -> str:
    """
    Simple placeholder answer generator.
    Replace this with a call to an LLM or other service as needed.
    """
    prompt = ChatPromptTemplate(
        messages=[
            {"role": "system", "content": "You are a helpful assistant, give answer in one line."},
            {"role": "ai", "content": "Conversation so far:\n{history}"},
            {"role": "user", "content": "{question}"}
        ]
    )
    chain = prompt | llm
    chatbot = RunnableWithMessageHistory(
        chain,
        get_chat_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    result = chatbot.invoke({"question": question}, config={"configurable": {"session_id": "default_session"}})
    history = get_chat_history("default_session")
    print(history)
    return result.content, history


@app.get("/ask", response_model=AnswerResponse)
async def ask(question: Optional[str] = Query(None, min_length=1, description="Question text")):
    """
    GET /ask?question=...
    Returns a JSON object with the original question and a generated answer.
    """
    if not question:
        raise HTTPException(status_code=400, detail="Query parameter 'question' is required")

    answer, history = generate_answer(question)
    return AnswerResponse(question=question, answer=answer)


if __name__ == "__main__":

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)