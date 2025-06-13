import numpy as np
import soundfile as sf
import librosa


def generate_music(generator, noise_dim=100, max_val=1.0, min_val=0.0, sr=22050):
    """
    Генерирует аудиофайл из шума с использованием обученного генератора.

    Args:
        generator: Обученная модель генератора.
        noise_dim (int): Размер входного шума.
        max_val (float): Максимальное значение для денормализации.
        min_val (float): Минимальное значение для денормализации.
        sr (int): Частота дискретизации.

    Returns:
        np.ndarray: Сгенерированный аудиосигнал.
    """
    noise = np.random.normal(0, 1, (1, noise_dim))
    gen_spec = generator.predict(noise, verbose=0)
    gen_spec = gen_spec * (max_val - min_val) + min_val  # Денормализация
    audio = librosa.feature.inverse.mel_to_audio(gen_spec[0, ..., 0], sr=sr)  # Убираем лишние измерения
    sf.write("generated/generated_music.wav", audio, sr)
    return audio