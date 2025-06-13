from fastapi import FastAPI

try:
    from tensorflow.keras import models
except ImportError:
    from tensorflow import keras

    models = keras.models
from generate import generate_music

app = FastAPI()


@app.post("/generate_music")
async def generate_music_endpoint(max_val: float = 1.0, min_val: float = 0.0):
    """
    Эндпоинт для генерации музыки.

    Args:
        max_val (float): Максимальное значение для денормализации.
        min_val (float): Минимальное значение для денормализации.

    Returns:
        dict: Путь к сгенерированному аудиофайлу.
    """
    generator = models.load_model("models/generator.keras")
    audio = generate_music(generator, noise_dim=100, max_val=max_val, min_val=min_val)
    return {"audio_file": "generated/generated_music.wav"}