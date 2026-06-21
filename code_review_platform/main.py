import sys
import os
from pathlib import Path

# Add the current directory to sys.path to ensure modules can be imported correctly
sys.path.append(str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from agents.supervisor import Supervisor
from db import init_db, save_review, get_all_reviews, get_review_by_id, delete_review

app = FastAPI(title="Multi-Agent Code Review Platform", version="1.0.0")

# Request Models
class ReviewRequest(BaseModel):
    code: str
    language: str = "python"
    filename: str = "main.py"

# Initialize supervisor (creates agent instances)
supervisor = Supervisor()

# Initialize DB on startup
@app.on_event("startup")
def on_startup():
    init_db()

@app.post("/review")
async def review(request: ReviewRequest):
    try:
        # 1. Run all agents (Bug, Quality, Security, and Scoring)
        reports = await supervisor.run_all(
            code=request.code,
            language=request.language,
            filename=request.filename
        )
        
        bug_report = reports["bug_report"]
        quality_report = reports["quality_report"]
        security_report = reports["security_report"]
        scoring_report = reports["scoring_report"]
        
        # 2. Save the review to the SQLite Database
        review_id = save_review(
            filename=request.filename,
            code=request.code,
            overall_score=int(scoring_report["score"]),
            bug_report=bug_report,
            quality_report=quality_report,
            security_report=security_report,
            overall_summary=scoring_report["summary"]
        )
        
        # 3. Retrieve the final saved model to return to frontend
        saved = get_review_by_id(review_id)
        if not saved:
            raise HTTPException(status_code=500, detail="Failed to save review to database.")
            
        saved["scoring_report"] = scoring_report
        return saved
        
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/reviews")
async def list_reviews():
    try:
        return get_all_reviews()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/reviews/{review_id}")
async def get_review(review_id: int):
    try:
        review_data = get_review_by_id(review_id)
        if not review_data:
            raise HTTPException(status_code=404, detail="Review not found.")
        
        # Parse or reconstruct the scoring report based on DB scores & summary
        review_data["scoring_report"] = {
            "score": review_data["overall_score"],
            "grade": get_grade_for_score(review_data["overall_score"]),
            "summary": review_data["overall_summary"],
            "strengths": ["Loaded from database archives."],
            "recommendations": []
        }
        return review_data
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.delete("/reviews/{review_id}")
async def remove_review(review_id: int):
    try:
        success = delete_review(review_id)
        if not success:
            raise HTTPException(status_code=404, detail="Review not found.")
        return {"status": "success", "message": f"Review {review_id} deleted."}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# Serving UI directly at '/'
@app.get("/", response_class=HTMLResponse)
async def read_index():
    static_file = Path(__file__).parent / "static" / "index.html"
    if static_file.exists():
        return HTMLResponse(content=static_file.read_text(encoding="utf-8"))
    
    # Fallback message if static file is missing
    return HTMLResponse(
        content="""
        <html>
            <body style="font-family:sans-serif; text-align:center; padding-top:100px; background:#0b0f19; color:#fff;">
                <h1>Multi-Agent Code Review Platform</h1>
                <p style="color:#a0aec0;">Static frontend template (index.html) is currently missing or being built.</p>
            </body>
        </html>
        """,
        status_code=404
    )

def get_grade_for_score(score: int) -> str:
    if score >= 95: return "A+"
    elif score >= 90: return "A"
    elif score >= 80: return "B"
    elif score >= 70: return "C"
    elif score >= 60: return "D"
    return "F"

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
