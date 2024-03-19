#!/usr/bin/env python
import gym
import numpy as np
import random
import tabular_sarsa as Tabular_SARSA
import matplotlib.pyplot as plt


class SARSA(Tabular_SARSA.Tabular_SARSA):
    def __init__(self):
        super(SARSA, self).__init__()

    def learn_policy(
        self, env, gamma, learning_rate, epsilon, lambda_value, num_episodes
    ):
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        rewards_each_learning_episode = []
        total_steps = 0
        for i in range(num_episodes):
            state = env.reset()
            action = self.LearningPolicy(state)
            episodic_reward = 0
            steps = 0
            while True:
                next_state, reward, done, info = env.step(action)  # take a random
                "*** Fill in the rest of the algorithm!! ***"
                next_action = self.LearningPolicy(next_state)
                if done:
                    error = reward - self.qtable[state][action]
                else:
                    error = (
                        reward
                        + (self.gamma * self.qtable[next_state][next_action])
                        - self.qtable[state][action]
                    )
                self.qtable[state][action] += self.alpha * error
                steps += 1
                state = next_state
                action = next_action
                episodic_reward += reward
                if done:
                    # print "episode :"+ str(i)+ " : "+ str(steps)
                    # total_steps +=steps
                    break
            rewards_each_learning_episode.append(episodic_reward)
        # avg_steps = total_steps/num_episodes
        # print "Average Steps:" +str(avg_steps)
        np.save("qvalues_taxi_sarsa", self.qtable)
        np.save("policy_taxi_sarsa", self.policy)

        return self.policy, self.qtable, rewards_each_learning_episode

    def LearningPolicy(self, state):
        return Tabular_SARSA.Tabular_SARSA.learningPolicy(self, state)


if __name__ == "__main__":

    rewardslist = []
    for x in range(10000):
        rewardslist.append(0)

    for i in range(10):
        env = gym.make("Taxi-v2")
        env.reset()
        sarsaLearner = SARSA()
        policySarsa, QValues, episodeRewards = sarsaLearner.learn_policy(
            env, 0.95, 0.2, 0.1, 0.1, 10000
        )

        for each in range(10000):
            rewardslist[each] += episodeRewards[each]

    for each in range(10000):
        rewardslist[each] = rewardslist[each] / 10

    plt.plot(rewardslist)
    plt.ylabel("rewards per episode")
    plt.ion()
    plt.savefig("rewards_plot_taxi_sarsa.png")
