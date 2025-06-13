import os
import tensorflow as tf
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate_music")
async def generate_music_endpoint(max_val: float = 1.0, min_val: float = 0.0, noise_dim: int = 100):
    from api import generate_music
    generator = tf.keras.models.load_model("models/generator.keras")
    audio = generate_music(generator, noise_dim=noise_dim, max_val=max_val, min_val=min_val)
    return {"audio_file": "generated/generated_music.wav"}

@app.get("/audio")
async def get_audio():
    return FileResponse("generated/generated_music.wav", media_type="audio/wav")

if __name__ == "__main__":
    os.makedirs("generated", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=8080)