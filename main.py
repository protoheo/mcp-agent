from fastapi import FastAPI
import uvicorn
from core.main_body import agent_body

model_manager, agent_model = agent_body()

app = FastAPI(
    title="Agentic AI API",
    version="1.0.0",
    docs_url="/",
    redoc_url="/redoc"
)


@app.post("/llm")
async def chat_transfer(input_msg: str):
    output = agent_model.run_chat(input_msg)
    return output


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

