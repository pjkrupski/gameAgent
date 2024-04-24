from reinforce import Reinforce, ValueNN
from tttproblem import TTTProblem
from adversarialsearch import minimax
import tensorflow as tf
import numpy as np



ACTIONS = 9

model2 = ValueNN(3, 3, ACTIONS)

class StudentBot:
    

    def __init__(self):

        self.model = Reinforce(3, 3, ACTIONS)
        self.rewards = []
        self.states = []
        self.actions = []

        self.graph = []

        self.games = 0
        self.wins = 0

    def discount(self, rewards, discount_factor=.95):
        """
        Takes in a list of rewards for each timestep in an episode, and
        returns a list of the discounted rewards for each timestep, which
        are calculated by summing the rewards for each future timestep, discounted
        by how far in the future it is.
        For example, in the simple case where the episode rewards are [1, 3, 5]
        and discount_factor = .99 we would calculate:
        dr_1 = 1 + 0.99 * 3 + 0.99^2 * 5 = 8.8705
        dr_2 = 3 + 0.99 * 5 = 7.95
        dr_3 = 5
        and thus return [8.8705, 7.95 , 5].
        Refer to the slides for more details about how/why this is done.

        :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
        :param discount_factor: Gamma discounting factor to use, defaults to .99
        :returns: discounted_rewards: list containing the discounted rewards for each timestep in the original rewards list
        """

        if len(rewards) == 1:
            return rewards

        indices = np.arange(len(rewards))
        total = rewards[0]

        total = total + np.sum(rewards[1:] * np.power(discount_factor, indices[1:]))

        return np.concatenate((np.array([total]), self.discount(rewards[1:], discount_factor)))

    def decide(self, asp: TTTProblem):
        """
        Input: asp, a GoTProblem
        Output: A direction in {'U','D','L','R'}
        To get started, you can get the current
        state by calling asp.get_start_state()
        """

        cstate = asp.get_start_state()

        cells = 8 #Why does cells start at 8? 

        board = cstate.board

        state = np.zeros((len(board),len(board[0]), cells))
       

        for row in range(len(board)):
            for col in range(len(board[row])):
                if board[row][col] == 'X':
                    state[row][col][0] = 1     
                elif board[row][col] == 'O':
                    state[row][col][1] = 1

        pred = self.model(tf.expand_dims(state, 0))[0]
        safe = asp.get_available_actions(cstate)

        probs = []
        choices = []
        idx = []

        for i in range(3):
            for j in range(3):
                choices.append((i,j))

                if (i,j) in safe:
                    idx.append(i*3+j)
                    probs.append(pred[i*3+j])

        probs = np.array(probs)

        if probs.sum() == 0:
            probs += 0.1

        probs /= probs.sum()

        output = np.random.choice(idx, 1, p=probs)[0]
    
        self.rewards.append(0)
        self.actions.append(output)
        self.states.append(state)

        return choices[output]

    def cleanup(self, win):

        self.games += 1

        if win:
          self.rewards[-1] = 1
          self.wins += 1
        #  print("win")
        #else:
        #  print("lose")

        if len(self.rewards) > 0:

            with tf.GradientTape() as tape:
                discounted_rewards = self.discount(self.rewards)
                loss2 = self.model.loss(self.states, self.actions, discounted_rewards)

            gradients = tape.gradient(loss2, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        else:
            print("NO REWARDS!")

        if self.games%100 == 0:
            print("Wins last 100: " + str(self.wins))
            print("Games: " + str(self.games))

            self.graph.append(self.wins)
            self.wins = 0

        self.rewards = []
        self.states = []
        self.actions = []

        pass


class RandomBot:

    def __init__(self):
      pass

    def decide(self, asp: TTTProblem):
        """
        Input: asp, a GoTProblem
        Output: A direction in {'U','D','L','R'}
        To get started, you can get the current
        state by calling asp.get_start_state()
        """

        choices = []

        for i in range(3):
           for j in range(3):
              choices.append((i,j))

        idx = []

        safe = asp.get_available_actions(asp.get_start_state())

        for i in range(3):
            for j in range(3):
                choices.append((i,j))

                if (i,j) in safe:
                    idx.append(i*3+j)
                    
        #min_max_output = minimax(asp)
        #print(type(min_max_output), min_max_output, " is minmax return \n\n\n", flush=True)

        output = np.random.choice(idx)
        #print(type(choices[output]), choices[output], "is choices ", flush=True)
        return choices[output]

    def cleanup(self):

        pass


class StudentBot2:
    

    def __init__(self):


        self.rewards = []
        self.states = []
        self.actions = []

        self.graph = []

        self.games = 0
        self.wins = 0

    def discount(self, rewards, discount_factor=.95):
        """
        Takes in a list of rewards for each timestep in an episode, and
        returns a list of the discounted rewards for each timestep, which
        are calculated by summing the rewards for each future timestep, discounted
        by how far in the future it is.
        For example, in the simple case where the episode rewards are [1, 3, 5]
        and discount_factor = .99 we would calculate:
        dr_1 = 1 + 0.99 * 3 + 0.99^2 * 5 = 8.8705
        dr_2 = 3 + 0.99 * 5 = 7.95
        dr_3 = 5
        and thus return [8.8705, 7.95 , 5].
        Refer to the slides for more details about how/why this is done.

        :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
        :param discount_factor: Gamma discounting factor to use, defaults to .99
        :returns: discounted_rewards: list containing the discounted rewards for each timestep in the original rewards list
        """

        if len(rewards) == 1:
            return rewards

        indices = np.arange(len(rewards))
        total = rewards[0]

        total = total + np.sum(rewards[1:] * np.power(discount_factor, indices[1:]))

        return np.concatenate((np.array([total]), self.discount(rewards[1:], discount_factor)))

    def decide(self, asp: TTTProblem):
        """
        Input: asp, a GoTProblem
        Output: A direction in {'U','D','L','R'}
        To get started, you can get the current
        state by calling asp.get_start_state()
        """

        cstate = asp.get_start_state()

        cells = 8 #Why does cells start at 8? 

        board = cstate.board

        state = np.zeros((len(board),len(board[0]), cells))
       

        for row in range(len(board)):
            for col in range(len(board[row])):
                if board[row][col] == 'X':
                    state[row][col][0] = 1     
                elif board[row][col] == 'O':
                    state[row][col][1] = 1

        pred = model2(tf.expand_dims(state, 0))[0]
        safe = asp.get_available_actions(cstate)

        probs = []
        choices = []
        idx = []

        for i in range(3):
            for j in range(3):
                choices.append((i,j))

                if (i,j) in safe:
                    idx.append(i*3+j)
                    probs.append(pred[i*3+j])

        probs = np.array(probs)

        if probs.sum() == 0:
            probs += 0.1

        probs /= probs.sum()

        output = np.random.choice(idx, 1, p=probs)[0]
    
        self.rewards.append(0)
        self.actions.append(output)
        self.states.append(state)

        return choices[output]

    def cleanup(self, win):

        self.games += 1

        if win:
          self.rewards[-1] = 1
          self.wins += 1
        #  print("win")
        #else:
        #  print("lose")

        if len(self.rewards) > 0:

            with tf.GradientTape() as tape:
                discounted_rewards = self.discount(self.rewards)
                loss2 = model2.loss(self.states, self.actions, discounted_rewards)

            gradients = tape.gradient(loss2, model2.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model2.trainable_variables))
        else:
            print("NO REWARDS!")

        if self.games%100 == 0:
            print("Wins last 100: " + str(self.wins))
            print("Games: " + str(self.games))

            self.graph.append(self.wins)
            self.wins = 0

        self.rewards = []
        self.states = []
        self.actions = []

        pass


class MinmaxBot:

    def __init__(self):
      pass

    def decide(self, asp: TTTProblem):
        
        output = minimax(asp)

        return output

    def cleanup(self):

        pass



