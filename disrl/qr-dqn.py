import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gym
import wandb
from collections import deque
import random

class QRDQNModel(tf.keras.Model):
    def __init__(self, action_space, num_quantiles, num_hidden_units=64):
        super(QRDQNModel, self).__init__()
        self.num_actions = action_space.n
        self.num_quantiles = num_quantiles

        self.fc1 = layers.Dense(num_hidden_units, activation='relu')
        self.fc2 = layers.Dense(num_hidden_units, activation='relu')
        self.quantiles = layers.Dense(self.num_actions * num_quantiles)

        self.delta = 1.0  # Initialize delta (this can be a parameter to be tuned)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        quantiles = self.quantiles(x)
        quantiles = tf.reshape(quantiles, (-1, self.num_actions, self.num_quantiles))
        return quantiles

    def compute_action_distribution(self, state):
        quantiles = self.call(state)
        return tf.reduce_mean(quantiles, axis=-1)
    
def compute_target_distribution(next_states, rewards, dones, gamma=0.99):
    next_quantiles = target_model(next_states)
    
    rewards = tf.reshape(rewards, [-1, 1, 1])  # Shape: [batch_size, 1, 1]
    dones = tf.reshape(dones, [-1, 1, 1])      # Shape: [batch_size, 1, 1]

    target_quantiles = rewards + (1 - dones) * gamma * next_quantiles  # Shape: [batch_size, num_actions, num_quantiles]
    
    target_quantiles = tf.reduce_mean(target_quantiles, axis=1)  # Shape: [batch_size, num_quantiles]

    return target_quantiles

def quantile_huber_loss(y_true, y_pred, delta=1.0):
    batch_size = tf.shape(y_true)[0]
    num_quantiles = tf.shape(y_pred)[1]

    y_true = tf.reshape(y_true, [batch_size, num_quantiles])
    y_pred = tf.reshape(y_pred, [batch_size, num_quantiles])
    
    error = y_true - y_pred
    abs_error = tf.abs(error)
    huber_loss = tf.where(abs_error < delta, 0.5 * tf.square(abs_error), delta * (abs_error - 0.5 * delta))
    
    return tf.reduce_mean(tf.reduce_mean(huber_loss, axis=-1))

def train_step(batch):
    states, actions, rewards, next_states, dones = zip(*batch)
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        quantiles = model(states)
        
        action_quantiles = tf.reduce_sum(
            quantiles * tf.one_hot(actions, model.num_actions)[:, :, tf.newaxis],
            axis=1
        )
        
        target_quantiles = compute_target_distribution(next_states, rewards, dones)

        loss = quantile_huber_loss(target_quantiles, action_quantiles, delta=model.delta)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def select_action(state):
    quantiles = model.call(state)
    mean_quantiles = tf.reduce_mean(quantiles, axis=-1)
    action = tf.argmax(mean_quantiles, axis=-1).numpy().item()
    return int(action)

def evaluate_policy():
    total_rewards = []
    for _ in range(evaluation_episodes):
        state = env.reset()
        state = state[0] if isinstance(state, tuple) else state
        state = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        episode_reward = 0
        while True:
            action = select_action(state)
            step_result = env.step(action)
            
            if len(step_result) == 4:
                next_state, reward, done, _ = step_result
            elif len(step_result) == 5:
                next_state, reward, done, _, _ = step_result
            else:
                raise ValueError("Unexpected number of values returned by env.step()")

            next_state = tf.convert_to_tensor(next_state[None, :], dtype=tf.float32)
            episode_reward += reward
            state = next_state
            if done:
                break
        total_rewards.append(episode_reward)
    avg_reward = np.mean(total_rewards)
    return avg_reward

# Hyperparameters
num_quantiles = 200
num_hidden_units = 64
learning_rate = 1e-4
batch_size = 32
gamma = 0.99
epsilon = 0.1
episodes = 1000
buffer_size = 10000
target_update_freq = 1000
evaluation_freq = 100
evaluation_episodes = 10

# Setup Gym environment
env = gym.make('CartPole-v1')

# Initialize models
model = QRDQNModel(env.action_space, num_quantiles, num_hidden_units)
target_model = QRDQNModel(env.action_space, num_quantiles, num_hidden_units)
target_model.set_weights(model.get_weights())
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Replay buffer
buffer = deque(maxlen=buffer_size)

# WandB initialization
wandb.init(project="qr-dqn-cartpole")

# Main training loop
for episode in range(episodes):
    state = env.reset()
    state = state[0] if isinstance(state, tuple) else state
    state = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
    episode_reward = 0
    loss = None

    while True:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = select_action(state)

        action = int(action)

        step_result = env.step(action)
        
        if len(step_result) == 4:
            next_state, reward, done, _ = step_result
        elif len(step_result) == 5:
            next_state, reward, done, _, _ = step_result
        else:
            raise ValueError("Unexpected number of values returned by env.step()")

        next_state = tf.convert_to_tensor(next_state[None, :], dtype=tf.float32)

        buffer.append((state.numpy(), action, reward, next_state.numpy(), done))

        if len(buffer) >= batch_size:
            batch = random.sample(buffer, batch_size)
            loss = train_step(batch)
        
        state = next_state
        episode_reward += reward

        if done:
            wandb.log({"episode_reward": episode_reward, "loss": loss})
            break

    if (episode + 1) % evaluation_freq == 0:
        avg_reward = evaluate_policy()
        wandb.log({"evaluation_avg_reward": avg_reward})

    if (episode + 1) % target_update_freq == 0:
        target_model.set_weights(model.get_weights())

    print(f"Episode {episode + 1}: Reward: {episode_reward}, Loss: {loss}")

env.close()
