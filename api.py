from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sys
import os

# Add the current directory to the path so we can import our agents
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.orchestrator import Orchestrator

app = FastAPI(title="ROCm Forge API", description="Autonomous CUDA to AMD ROCm Migration API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
os.makedirs(os.path.join(os.path.dirname(__file__), "static"), exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

class MigrationRequest(BaseModel):
    code: str
    code_type: str = "auto"
    groq_api_key: str = ""

@app.post("/api/migrate")
async def migrate_code(request: MigrationRequest):
    try:
        # Initialize orchestrator
        orchestrator = Orchestrator(groq_api_key=request.groq_api_key)
        
        # Run migration pipeline
        result = orchestrator.run_migration(request.code, request.code_type)
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
            
        # Serialize the result
        return {
            "success": result.success,
            "original_code": request.code,
            "refactored_code": result.refactored_code,
            "total_duration_ms": result.total_duration_ms,
            "analysis": {
                "code_type": result.analysis.code_type,
                "migration_score": result.analysis.migration_score,
                "migration_level": result.analysis.migration_level,
                "summary": result.analysis.summary,
                "known_issues": result.analysis.known_issues,
                "detected_patterns": [
                    {
                        "pattern": p.pattern,
                        "rocm_equivalent": p.rocm_equivalent,
                        "category": p.category,
                        "severity": p.severity,
                        "line_number": p.line_number,
                        "note": p.note
                    } for p in result.analysis.detected_patterns
                ]
            },
            "refactoring_changes": result.refactoring_changes,
            "agent_steps": [
                {
                    "agent_name": step.agent_name,
                    "status": step.status,
                    "message": step.message,
                    "details": step.details,
                    "duration_ms": step.duration_ms,
                    "icon": step.icon
                } for step in result.agent_steps
            ],
            "deployment": result.deployment,
            "llm_analysis": result.llm_analysis,
            "llm_review": result.llm_review
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
