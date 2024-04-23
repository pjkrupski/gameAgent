import os
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Reinforce(tf.keras.Model):
    def __init__(self, state_size, state_size2, num_actions):
        """
        The Reinforce class that inherits from tf.keras.Model
        The forward pass calculates the policy for the agent given a batch of states.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this,
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(Reinforce, self).__init__()
        self.num_actions = num_actions

        # TODO: Define network parameters and optimizer

        self.optimizer = tf.optimizers.Adam(learning_rate=0.1)     #.00005

        #self.D1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides = (1,1), activation = 'relu', input_shape=[state_size,state_size2,8])

        self.D2 = tf.keras.layers.Dense(num_actions, kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03))

    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        for each state in the episode
        """
        
        logits = tf.keras.layers.Flatten()(states)

        logits = self.D2(logits)
        probs = tf.nn.softmax(logits)

        return probs

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Make sure to understand the handout clearly when implementing this.

        :param states: A batch of states of shape [episode_length, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a Tensorflow scalar
        """
        # TODO: implement this
        # Hint: Use gather_nd to get the probability of each action that was actually taken in the episode.


        probs = self.call(np.array(states))
        actions = tf.expand_dims(actions,1)

        logged = tf.math.log(tf.gather_nd(probs, actions, 1))

        return -tf.reduce_sum(logged * discounted_rewards)