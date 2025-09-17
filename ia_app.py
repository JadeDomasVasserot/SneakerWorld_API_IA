from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

class RecommendRequest(BaseModel):
    user_id: int

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/recommend")
async def recommend(req: RecommendRequest):
    # Dummy recommendation logic
    recommendations = ["item1", "item2", "item3"]
    return {"user_id": req.user_id, "recommendations": recommendations}

@app.post("/generate")
async def generate(req: GenerateRequest):
    # Dummy generation logic
    generated_text = f"Generated response for: {req.prompt}"
    return {"prompt": req.prompt, "generated": generated_text}