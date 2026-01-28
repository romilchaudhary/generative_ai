from fastapi import FastAPI
from pydantic import BaseModel
from agent import agent_executor
from database import init_db

app = FastAPI()

init_db() # initialize DB on startup

class AgentRequest(BaseModel):
    query: str

@app.post("/agent/invoke")
async def invoke_agent(request: AgentRequest):
    # Here you would integrate the agent logic
    response = agent_executor.invoke({"input": request.query})
    print(response)
    return {"response": response["output"]}