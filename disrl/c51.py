import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gym
import wandb
from collections import deque
import random

class C51Model(tf.keras.Model):
    def __init__(self, action_space, num_atoms, Vmin, Vmax, num_hidden_units=64):
        super(C51Model, self).__init__()
        self.num_actions = action_space.n
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.delta_z = (Vmax - Vmin) / (num_atoms - 1)
        self.z = tf.constant(np.linspace(Vmin, Vmax, num_atoms), dtype=tf.float32)

        self.fc1 = layers.Dense(num_hidden_units, activation='relu')
        self.fc2 = layers.Dense(num_hidden_units, activation='relu')
        self.logits = layers.Dense(self.num_actions * num_atoms)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        logits = self.logits(x)
        logits = tf.reshape(logits, (-1, self.num_actions, self.num_atoms))
        return logits

    def compute_action_distribution(self, state):
        logits = self.call(state)
        return tf.nn.softmax(logits, axis=-1)

# Hyperparameters
num_atoms = 51
Vmin = -10
Vmax = 10
num_hidden_units = 64
learning_rate = 1e-4
batch_size = 32
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 100000  # Decay steps for epsilon
episodes = 1000
buffer_size = 10000
target_update_freq = 1000
evaluation_freq = 100
evaluation_episodes = 10

# Setup Gym environment
env = gym.make('CartPole-v1')

# Initialize models
model = C51Model(env.action_space, num_atoms, Vmin, Vmax, num_hidden_units)
target_model = C51Model(env.action_space, num_atoms, Vmin, Vmax, num_hidden_units)  # Target network
target_model.set_weights(model.get_weights())  # Initialize target network with same weights
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Replay buffer
buffer = deque(maxlen=buffer_size)

# WandB initialization
wandb.init(project="disrl")

def compute_target_distribution(next_states, rewards, dones):
    z = tf.constant(np.linspace(Vmin, Vmax, num_atoms), dtype=tf.float32)
    next_logits = target_model.compute_action_distribution(next_states)  # Use target model
    next_dist = tf.reduce_sum(next_logits * tf.expand_dims(z, axis=0), axis=-1)  # Shape: [batch_size, num_atoms]
    
    rewards = tf.expand_dims(rewards, axis=-1)  # Shape: [batch_size, 1]
    dones = tf.expand_dims(dones, axis=-1)      # Shape: [batch_size, 1]
    
    target = rewards + (1 - dones) * gamma * next_dist  # Shape: [batch_size, num_atoms]
    
    # Make sure target_distribution is correctly shaped
    target_distribution = tf.one_hot(tf.argmax(target, axis=-1), num_atoms)  # Shape: [batch_size, num_atoms]
    
    return target_distribution


def train_step(batch):
    states, actions, rewards, next_states, dones = zip(*batch)
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        logits = model(states)  # Shape: [batch_size, num_actions, num_atoms]
        dist = tf.nn.softmax(logits, axis=-1)  # Shape: [batch_size, num_actions, num_atoms]

        action_dist = tf.reduce_sum(dist * tf.one_hot(actions, env.action_space.n)[:, :, tf.newaxis], axis=1)  # Shape: [batch_size, num_atoms]
        target_dist = compute_target_distribution(next_states, rewards, dones)  # Shape: [batch_size, num_atoms]

        # Ensure that the target_dist and action_dist have the same shape
        action_dist = tf.ensure_shape(action_dist, (batch_size, num_atoms))
        target_dist = tf.ensure_shape(target_dist, (batch_size, num_atoms))

        # Compute loss
        loss = tf.reduce_mean(tf.keras.losses.KLD(target_dist, action_dist))
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def select_action(state, epsilon, step):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        logits = model.compute_action_distribution(state)
        z = np.linspace(Vmin, Vmax, num_atoms)
        action_probs = tf.reduce_sum(tf.nn.softmax(logits, axis=-1) * z, axis=-1)
        action = tf.argmax(action_probs, axis=-1).numpy()
        return int(action)  # Ensure action is an integer

def evaluate_policy():
    total_rewards = []
    for _ in range(evaluation_episodes):
        state = env.reset()
        state = state[0] if isinstance(state, tuple) else state  # Ensure state is extracted correctly
        state = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        episode_reward = 0
        done = False
        while not done:
            action = select_action(state, epsilon_end, 0)  # No exploration during evaluation
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
        total_rewards.append(episode_reward)
    avg_reward = np.mean(total_rewards)
    return avg_reward

# Main training loop
total_steps = 0
for episode in range(episodes):
    state = env.reset()
    state = state[0] if isinstance(state, tuple) else state  # Ensure state is extracted correctly
    state = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
    episode_reward = 0
    loss = None  # Initialize loss to avoid NameError

    done = False
    while not done:
        epsilon = max(epsilon_end, epsilon_start - total_steps / epsilon_decay)
        action = select_action(state, epsilon, total_steps)

        action = int(action)  # Ensure action is an integer

        # Step through the environment
        step_result = env.step(action)

        if len(step_result) == 4:
            next_state, reward, done, _ = step_result
        elif len(step_result) == 5:
            next_state, reward, done, _, _ = step_result
        else:
            raise ValueError("Unexpected number of values returned by env.step()")

        next_state = tf.convert_to_tensor(next_state[None, :], dtype=tf.float32)

        # Store experience in replay buffer
        buffer.append((state.numpy(), action, reward, next_state.numpy(), done))

        if len(buffer) >= batch_size:
            batch = random.sample(buffer, batch_size)
            loss = train_step(batch)  # Define loss here
        
        state = next_state
        episode_reward += reward
        total_steps += 1

        if done:
            wandb.log({"episode_reward": episode_reward, "loss": loss})
            break

    # Periodic evaluation
    if (episode + 1) % evaluation_freq == 0:
        avg_reward = evaluate_policy()
        wandb.log({"evaluation_avg_reward": avg_reward})

    # Periodic target network update
    if (episode + 1) % target_update_freq == 0:
        target_model.set_weights(model.get_weights())

    print(f"Episode {episode + 1}: Reward: {episode_reward}, Loss: {loss}")

env.close()