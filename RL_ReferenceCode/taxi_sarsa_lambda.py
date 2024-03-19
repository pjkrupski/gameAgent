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
        self.lambda_value = lambda_value
        total_steps = 0
        rewards_each_learning_episode = []
        for i in range(num_episodes):
            state = env.reset()
            action = self.LearningPolicy(state)
            steps = 0
            episodic_reward = 0
            self.etable = np.zeros((self.num_states, self.num_actions))
            while True:
                next_state, reward, done, info = env.step(action)
                "*** Fill in the rest of the algorithm!! ***"
                next_action = self.LearningPolicy(next_state)
                if done:
                    error = reward - self.qtable[state][action]
                else:
                    error = (
                        reward
                        + self.gamma * self.qtable[next_state][next_action]
                        - self.qtable[state][action]
                    )
                # for X in range(self.num_states):
                #     for Y in range(self.num_actions):
                #         if(X == state and Y == action):
                #             self.etable[X][Y] = 1
                #         else:
                #             self.etable[X][Y] = self.gamma*self.lambda_value*self.etable[X][Y]

                self.etable = self.etable * self.lambda_value * self.gamma

                steps += 1
                self.etable[state][action] = 1

                self.qtable += self.alpha * error * self.etable

                state = next_state
                action = next_action

                episodic_reward += reward
                if done:
                    # total_steps+=steps
                    # print "episode :"+ str(i)+ " : "+ str(steps)
                    break
            rewards_each_learning_episode.append(episodic_reward)
        # avg_steps = total_steps/num_episodes
        # print "Average steps:" +str(avg_steps)
        np.save("qvalues_taxi_sarsa_lambda", self.qtable)
        np.save("policy_taxi_sarsa_lambda", self.policy)

        return self.policy, self.qtable, rewards_each_learning_episode

    def LearningPolicy(self, state):
        return Tabular_SARSA.Tabular_SARSA.learningPolicy(self, state)


if __name__ == "__main__":

    rewardslist = []
    for x in range(15000):
        rewardslist.append(0)

    for i in range(10):
        env = gym.make("Taxi-v2")
        env.reset()
        sarsaLearner = SARSA()
        policySarsa, QValues, episodeRewards = sarsaLearner.learn_policy(
            env, 0.95, 0.2, 0.1, 0.8, 15000
        )

        for each in range(15000):
            rewardslist[each] += episodeRewards[each]

    for each in range(15000):
        rewardslist[each] = rewardslist[each] / 10

    plt.plot(rewardslist)
    plt.ylabel("rewards per episode")
    plt.ion()
    plt.savefig("rewards_plot_lambda.png")

    # state=env.reset()
    # env.render()

    # while True:
    #     next_state, reward, done, info = env.step(sarsaLearner.policy[state,0])
    #     env.render()
    #     print reward
    #     state=next_state
    #     if done:
    #         break
