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
wandb.init(project="gan-vs-wgan", name="gan_experiment", config={
    "epochs": 10000,
    "batch_size": 32,
    "latent_dim": 100,
    "save_interval": 1000  # Save checkpoint every 1000 iterations
})
config = wandb.config

# Load and preprocess data
(x_train, _), (_, _) = mnist.load_data()
x_train = (x_train - 127.5) / 127.5  # Normalize images
x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension
batch_size = config.batch_size

# Build GAN
def build_generator():
    model = Sequential([
        Dense(128, input_dim=config.latent_dim),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dense(784, activation='tanh'),
        Reshape((28, 28, 1))
    ])
    return model

def build_discriminator():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128),
        LeakyReLU(alpha=0.2),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_gan(generator, discriminator):
    model = Sequential([generator, discriminator])
    return model

generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Training function with checkpoints every 1000 iterations
def train_gan(epochs=10000, save_interval=1000):
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, 'gan_generator_weights_epoch_{epoch:04d}.weights.h5')
    
    # Check for the latest checkpoint
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.weights.h5')]
    checkpoint_files.sort()  # Sort by file name to get the latest one
    if checkpoint_files:
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
        print(f"Loading weights from checkpoint: {latest_checkpoint}")
        generator.load_weights(latest_checkpoint)
        # Extract epoch from filename
        initial_epoch = int(checkpoint_files[-1].split('_')[-1].split('.')[0])
    else:
        initial_epoch = 0

    # Main training loop
    for epoch in range(initial_epoch, epochs):
        # Train discriminator
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_images = x_train[idx]
        fake_images = generator.predict(np.random.normal(0, 1, (batch_size, config.latent_dim)))
        d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
        
        # Train generator
        g_loss = gan.train_on_batch(np.random.normal(0, 1, (batch_size, config.latent_dim)), np.ones((batch_size, 1)))
        print(f"Epoch {epoch}/{epochs} - D Loss Real: {d_loss_real} - D Loss Fake: {d_loss_fake} - G Loss: {g_loss}")
        
        # Log training losses to WandB
        wandb.log({"D Loss Real": d_loss_real, "D Loss Fake": d_loss_fake, "G Loss": g_loss})
        
        # Save model checkpoint every 1000 iterations
        if (epoch + 1) % save_interval == 0:
            generator.save_weights(checkpoint_path.format(epoch=epoch + 1))
            save_generated_images(epoch + 1, generator)

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
    img_path = f'gan_generated_image_epoch_{epoch}.png'
    plt.savefig(img_path)
    plt.close()
    
    # Log images to WandB
    wandb.log({"Generated Images": wandb.Image(img_path)})

# Train GAN with WandB logging and checkpointing
train_gan()
