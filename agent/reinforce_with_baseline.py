import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class ReinforceWithBaseline(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The ReinforceWithBaseline class that inherits from tf.keras.Model.

        The forward pass calculates the policy for the agent given a batch of states. During training,
        ReinforceWithBaseLine estimates the value of each state to be used as a baseline to compare the policy's
        performance with.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(ReinforceWithBaseline, self).__init__()
        self.num_actions = num_actions

        # TODO: Define actor network parameters, critic network parameters, and optimizer
        self.optimizer = tf.optimizers.Adam(0.0002)
        self.D1 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides = (1,1), activation = 'relu', input_shape=[state_size,state_size,8])
        
        self.D2 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides = (1,1), activation = 'relu')
        
        self.D3 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides = (1,1), activation = 'relu')
        
        self.D4 = tf.keras.layers.Dense(num_actions, kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03))
        
        self.D5 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides = (1,1), activation = 'relu', input_shape=[state_size,state_size,8])
                
        self.D6 = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))
        
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
        # TODO: implement this!
        
        logits = self.D1(states)
        #logits = tf.nn.relu(logits)
        logits = self.D2(logits)
        #logits = tf.nn.relu(logits)
        logits = self.D3(logits)
        logits = tf.keras.layers.Flatten()(logits)
        logits = self.D4(logits)
        probs = tf.nn.softmax(logits)
        
        return probs

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An [episode_length, state_size] dimensioned array representing the history of states
        of an episode.
        :return: A [episode_length] matrix representing the value of each state.
        """
        # TODO: implement this :D
        
        logits = self.D5(states)
        logits = tf.keras.layers.Flatten()(logits)
        logits = self.D6(logits)
        
        return logits

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Refer to the lecture slides referenced in the handout to see how this is done.

        Remember that the loss is similar to the loss as in part 1, with a few specific changes.

        1) In your actor loss, instead of element-wise multiplying with discounted_rewards, you want to element-wise multiply with your advantage. 
        See handout/slides for definition of advantage.
        
        2) In your actor loss, you must use tf.stop_gradient on the advantage to stop the loss calculated on the actor network 
        from propagating back to the critic network.
        
        3) See handout/slides for how to calculate the loss for your critic network.

        :param states: A batch of states of shape (episode_length, state_size)
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a TensorFlow scalar
        """
        # TODO: implement this :)
        # Hint: use tf.gather_nd (https://www.tensorflow.org/api_docs/python/tf/gather_nd) to get the probabilities of the actions taken by the model
        value = self.value_function(np.array(states))
        value = tf.squeeze(value)
        adv = discounted_rewards - value 
        
        closs = tf.reduce_sum(adv * adv)
        
        probs = self.call(np.array(states))
        actions = tf.expand_dims(actions,1)
        
        logged = tf.math.log(tf.gather_nd(probs, actions, 1))
        
        return -tf.reduce_sum(logged * adv) + closs
