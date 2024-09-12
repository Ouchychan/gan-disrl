import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import os
import wandb

# Initialize WandB
wandb.init(project="wgan-experiment", config={
    "epochs": 10000,
    "batch_size": 32,
    "latent_dim": 100,
    "save_interval": 500,
    "checkpoint_interval": 500,  # Save checkpoint every 1000 iterations
    "gradient_penalty_lambda": 10  # Lambda for gradient penalty
})
config = wandb.config

# Load and preprocess data
(x_train, _), (_, _) = mnist.load_data()
x_train = (x_train - 127.5) / 127.5  # Normalize images
x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension
batch_size = config.batch_size

# Build WGAN Generator
def build_generator():
    model = Sequential([
        Dense(128, input_dim=config.latent_dim),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dense(784, activation='tanh'),
        Reshape((28, 28, 1))
    ])
    return model

# Build WGAN Discriminator (Critic)
def build_discriminator():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128),
        LeakyReLU(alpha=0.2),
        Dropout(0.4),
        Dense(1)
    ])
    return model

# Build WGAN
def build_wgan(generator, discriminator):
    model = Sequential([generator, discriminator])
    return model

generator = build_generator()
discriminator = build_discriminator()
wgan = build_wgan(generator, discriminator)

# Wasserstein loss function
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

# Compile models
discriminator.compile(loss=wasserstein_loss, optimizer=Adam(0.0002, 0.5))
wgan.compile(loss=wasserstein_loss, optimizer=Adam(0.0002, 0.5))

# Gradient penalty
def gradient_penalty(batch_size, real_images, fake_images):
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated_images = alpha * real_images + (1 - alpha) * fake_images
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        validity = discriminator(interpolated_images)
    gradients = tape.gradient(validity, interpolated_images)
    norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    penalty = tf.reduce_mean((norm - 1.0) ** 2)
    return penalty

def train_wgan(epochs=config.epochs, save_interval=config.save_interval, checkpoint_interval=config.checkpoint_interval):
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_path = os.path.join(checkpoint_dir, 'wgan_generator_weights_epoch_{epoch:04d}.weights.h5')
    
    # Determine the starting epoch based on existing checkpoints
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.weights.h5')]
    checkpoint_files.sort()  # Sort by file name to get the latest one
    
    start_epoch = 0
    if checkpoint_files:
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
        print(f"Loading weights from checkpoint: {latest_checkpoint}")
        generator.load_weights(latest_checkpoint)
        start_epoch = int(checkpoint_files[-1].split('_')[-1].split('.')[0])
    
    for epoch in range(start_epoch, epochs):
        try:
            for _ in range(5):  # Training the critic more frequently than the generator
                # Train discriminator (critic)
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                real_images = x_train[idx]
                fake_images = generator.predict(np.random.normal(0, 1, (batch_size, config.latent_dim)))
                
                # Compute the gradient penalty
                d_loss_real = discriminator.train_on_batch(real_images, -np.ones((batch_size, 1)))
                d_loss_fake = discriminator.train_on_batch(fake_images, np.ones((batch_size, 1)))
                gp = gradient_penalty(batch_size, real_images, fake_images)
                d_loss = d_loss_real + d_loss_fake + config.gradient_penalty_lambda * gp  # Lambda for gradient penalty

            # Train generator
            g_loss = wgan.train_on_batch(np.random.normal(0, 1, (batch_size, config.latent_dim)), -np.ones((batch_size, 1)))
            print(f"Epoch {epoch}/{epochs} - D Loss: {d_loss} - G Loss: {g_loss}")
            
            # Log training losses to WandB
            wandb.log({"D Loss Real": d_loss_real, "D Loss Fake": d_loss_fake, "G Loss": g_loss})
            
            # Save generated images
            if (epoch + 1) % save_interval == 0:
                save_generated_images(epoch + 1, generator)
            
            # Save checkpoints at intervals
            if (epoch + 1) % checkpoint_interval == 0:
                generator.save_weights(checkpoint_path.format(epoch=epoch + 1))
                print(f"Checkpoint saved at epoch {epoch + 1}.")
        
        except Exception as e:
            print(f"An error occurred: {e}")
            break

def save_generated_images(epoch, generator, latent_dim=config.latent_dim):
    noise = np.random.normal(0, 1, (25, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = (generated_images + 1) / 2.0  # Rescale to [0, 1]
    
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    cnt = 0
    for i in range(5):
        for j in range(5):
            axs[i,j].imshow(generated_images[cnt, :, :, 0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    img_path = f'wgan_generated_image_epoch_{epoch}.png'
    plt.savefig(img_path)
    plt.close()
    
    # Log images to WandB
    wandb.log({"Generated Images": wandb.Image(img_path)})

# Train WGAN with WandB logging
train_wgan()
