import tensorflow as tf
import numpy as np
from model import build_generator, build_discriminator, build_gan
try:
    from tensorflow.keras import layers, models
except ImportError:
    from tensorflow import keras

    layers = keras.layers
    models = keras.models

def train_gan(generator, discriminator, gan, dataset, epochs=50, batch_size=64, noise_dim=100):
    """
    Обучает GAN на датасете.

    Args:
        generator: Модель генератора.
        discriminator: Модель дискриминатора.
        gan: Комбинированная модель GAN.
        dataset: tf.data.Dataset с мел-спектрограммами.
        epochs (int): Количество эпох.
        batch_size (int): Размер батча.
        noise_dim (int): Размер входного шума.
    """
    generator.compile(optimizer='adam', loss='binary_crossentropy')
    discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    total_batches = sum(1 for _ in dataset)  # Подсчёт общего числа батчей
    print(f"Total batches per epoch: {total_batches}")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        d_loss_total = 0.0
        g_loss_total = 0.0
        num_batches = 0

        for i, batch in enumerate(dataset):
            num_batches += 1
            batch_size_actual = tf.shape(batch)[0]
            real_specs = tf.reshape(batch, [batch_size_actual, 128, 346, 1])  # Приведение формы

            print(f"Epoch {epoch + 1}, Batch {i + 1}/{total_batches}, Processing...", end='\r')

            # Генерация шума
            noise = tf.random.normal([batch_size_actual, noise_dim])
            generated_specs = generator(noise, training=True)

            # Обучение дискриминатора
            real_labels = tf.ones((batch_size_actual, 1))
            fake_labels = tf.zeros((batch_size_actual, 1))
            with tf.GradientTape() as d_tape:
                real_output = discriminator(real_specs, training=True)
                fake_output = discriminator(generated_specs, training=True)
                d_loss_real = tf.reduce_mean(tf.keras.losses.binary_crossentropy(real_labels, real_output))
                d_loss_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(fake_labels, fake_output))
                d_loss = d_loss_real + d_loss_fake
            d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
            discriminator.optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

            # Обучение генератора
            with tf.GradientTape() as g_tape:
                noise = tf.random.normal([batch_size_actual, noise_dim])
                generated_specs = generator(noise, training=True)
                fake_output = discriminator(generated_specs, training=True)
                g_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(real_labels, fake_output))
            g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
            generator.optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

            d_loss_total += d_loss
            g_loss_total += g_loss

        avg_d_loss = d_loss_total / num_batches
        avg_g_loss = g_loss_total / num_batches
        print(f"\nEpoch {epoch + 1} completed, D Loss: {avg_d_loss.numpy()}, G Loss: {avg_g_loss.numpy()}")