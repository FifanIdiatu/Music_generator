import tensorflow as tf

try:
    from tensorflow.keras import layers, models
except ImportError:
    from tensorflow import keras

    layers = keras.layers
    models = keras.models

def build_generator(noise_dim=100, target_shape=(128, 346, 1)):
    model = models.Sequential([
        layers.Input(shape=(noise_dim,)),
        layers.Dense(8 * 22 * 256),  # Начальная форма (8, 22, 256)
        layers.Reshape((8, 22, 256)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu'),  # (16, 44, 128)
        layers.BatchNormalization(),
        layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', activation='relu'),   # (32, 88, 64)
        layers.BatchNormalization(),
        layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', activation='relu'),   # (64, 176, 32)
        layers.BatchNormalization(),
        layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same', activation='relu'),   # (128, 352, 16)
        layers.BatchNormalization(),
        layers.Conv2DTranspose(1, (4, 4), strides=(1, 1), padding='same', activation='tanh'),    # (128, 352, 1)
        layers.Cropping2D(cropping=((0, 0), (6, 0)))  # Обрезка до (128, 346, 1)
    ])
    return model

def build_discriminator(spec_shape=(128, 346, 1)):
    model = models.Sequential([
        layers.Input(shape=spec_shape),
        layers.Conv2D(16, (4, 4), strides=(2, 2), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_gan(generator, discriminator, noise_dim=100):
    discriminator.trainable = False
    gan_input = layers.Input(shape=(noise_dim,))
    gan_output = discriminator(generator(gan_input))
    model = models.Model(gan_input, gan_output)
    return model