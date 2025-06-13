import tensorflow as tf
from preprocess import load_dataset
from model import build_generator, build_discriminator, build_gan
from train import train_gan

if __name__ == "__main__":
    mel_specs, max_val, min_val = load_dataset(duration=10)
    noise_dim = 100
    generator = build_generator(noise_dim, target_shape=(128, 346, 1))
    discriminator = build_discriminator(spec_shape=(128, 346, 1))
    discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    gan = build_gan(generator, discriminator, noise_dim=noise_dim)
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    train_gan(generator, discriminator, gan, mel_specs, epochs=50)
    generator.save("models/generator.keras")