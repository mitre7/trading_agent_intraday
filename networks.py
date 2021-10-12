import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense


class CriticNetwork(keras.Model):

    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256, name='critic', checkpoint_dir='sac_dir/'):
        super(CriticNetwork, self).__init__()

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_sac')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)
        q = self.q(action_value)

        return q


class ValueNetwork(keras.Model):

    def __init__(self, fc1_dims=256, fc2_dims=256, name='value', checkpoint_dir='sac_dir/'):
        super(ValueNetwork, self).__init__()

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_sac')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.v = Dense(1, activation=None)

    def call(self, state):
        state_value = self.fc1(state)
        state_value = self.fc2(state_value)
        v = self.v(state_value)

        return v


class ActorNetwork(keras.Model):

    def __init__(self, max_action, n_actions, fc1_dims=512, fc2_dims=512, name='actor', checkpoint_dir='sac_dir/'):
        super(ActorNetwork, self).__init__()

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.max_action = max_action
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_sac')
        self.noise = 1e-6

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation=None)
        self.sigma = Dense(self.n_actions, activation=None)

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = tf.clip_by_value(sigma, self.noise, 1)

        return mu, sigma

    def sample_normal(self, state, parametrize=False):
        mu, sigma = self.call(state)

        probabilities = tfp.distributions.Normal(mu, tf.exp(sigma))

        actions = probabilities.sample()

        # if not parametrize:
        #     actions = probabilities.sample()
        # else:
        #     actions = tf.stop_gradient(probabilities.sample())

        squashed_actions = tf.math.tanh(actions)

        log_probs = probabilities.log_prob(actions) - tf.math.log(1 - tf.math.pow(squashed_actions, 2) + self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return squashed_actions * self.max_action, log_probs
