from fastapi import FastAPI

app = FastAPI(
    title="Irish & EU Regulation RAG Assistant API",
    description="API for the open-source RAG assistant for Irish & EU regulation.",
    version="0.1.0",
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Irish & EU Regulation RAG Assistant API"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# To run this app (from the project root directory, after activating venv):
# uvicorn app.main:app --reload
