# LiftLens Backend — Phase 2

FastAPI backend with video frame extraction and MediaPipe pose estimation.

## Project structure

```
liftlens-backend/
├── main.py                  # FastAPI app + /analyze endpoint
├── pipeline/
│   ├── extract_frames.py    # OpenCV frame extraction
│   ├── pose_estimation.py   # MediaPipe landmark detection
│   └── analysis.py          # Biomechanics engine (squat, bench, deadlift)
├── requirements.txt
├── render.yaml              # Render.com deploy config
└── README.md
```

## Run locally

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn main:app --reload
```

API is now live at http://localhost:8000
Interactive docs at http://localhost:8000/docs

## Test with curl

```bash
curl -X POST "http://localhost:8000/analyze?exercise=squat" \
  -F "file=@your_squat.mp4"
```

## Deploy to Render

1. Push this folder to a GitHub repo
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Render will auto-detect `render.yaml` and configure everything
5. Click Deploy — your API will be live at `https://liftlens-api.onrender.com`

## API response shape

```json
{
  "exercise": "squat",
  "frame_count": 42,
  "analysis": {
    "min_knee_angle_deg": 82.4,
    "min_hip_angle_deg": 67.1,
    "depth_reached": true,
    "knee_cave_detected": false,
    "flags": [],
    "frames_analyzed": 42
  }
}
```

Flags can include: `insufficient_depth`, `knee_cave`, `excessive_forward_lean`,
`excessive_elbow_flare`, `excessive_wrist_extension`, `hips_rising_early`, `excessive_back_angle`

## Next: Phase 3 — LLM coaching layer

Pass `analysis` JSON to Claude API with a system prompt like:
> "You are an expert powerlifting coach. Given these biomechanics findings, explain what the lifter did wrong, why it matters, and what to fix first."
