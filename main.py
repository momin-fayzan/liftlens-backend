from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tempfile, os, shutil, anthropic
from pipeline.extract_frames import extract_frames
from pipeline.pose_estimation import run_pose_estimation
from pipeline.analysis import analyze_lift

app = FastAPI(title="LiftLens API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this once frontend URL is known
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_KEY", "")

@app.get("/")
def health():
    return {"status": "ok", "service": "LiftLens API"}


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    exercise: str
    analysis: dict
    messages: List[Message]


@app.post("/chat")
async def chat(req: ChatRequest):
    if not ANTHROPIC_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_KEY not set")

    system = f"""You are LiftLens, an expert powerlifting coach. The athlete just uploaded a {req.exercise} video.

Here are the biomechanics findings from the video analysis:
{req.analysis}

Your job:
- Explain what happened in plain, direct language
- Identify the most important issue to fix first
- Give 1-2 actionable cues they can use on their next set
- Keep responses concise and practical — no fluff
- Use "you" not "the athlete"

Start with a brief summary of the rep (2-3 sentences), then the main issue, then what to fix."""

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=system,
        messages=[{"role": m.role, "content": m.content} for m in req.messages],
    )
    return {"reply": response.content[0].text}

@app.post("/analyze")
async def analyze(
    exercise: str,
    file: UploadFile = File(...)
):
    """
    Accepts a lift video and returns pose landmarks + biomechanics analysis.
    exercise: one of 'squat', 'bench', 'deadlift'
    """
    if exercise not in ("squat", "bench", "deadlift"):
        raise HTTPException(status_code=400, detail="exercise must be squat, bench, or deadlift")

    allowed = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    # Save upload to a temp file
    suffix = os.path.splitext(file.filename)[-1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        frames      = extract_frames(tmp_path)
        landmarks   = run_pose_estimation(frames)
        analysis    = analyze_lift(exercise, landmarks)
    finally:
        os.unlink(tmp_path)

    return {
        "exercise": exercise,
        "frame_count": len(frames),
        "analysis": analysis,
    }
