from fastapi import FastAPI, HTTPException
from prometheus_client import make_asgi_app, Counter
from .core.rag import RAGOrchestrator
from .core.config import settings

app = FastAPI(
    title="CogniFlow AI Orchestrator",
    version="1.0.0",
    description="Enterprise-Grade RAG Orchestrator for Cloud-Native Environments."
)

# Prometheus Metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
query_counter = Counter("cogniflow_queries_total", "Total number of queries processed")

rag = RAGOrchestrator()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/query")
async def process_query(user_query: str):
    try:
        query_counter.inc()
        response = rag.generate_response(user_query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
