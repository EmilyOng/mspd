import random
import numpy as np
from mspd import solve


class QEnvironment:
    def __init__(self, N, objectiveN, inputDf):
        self.N = N
        self.objectiveN = objectiveN
        self.inputDf = inputDf

        self.n_states = N
        self.n_actions = N

        # Include 0 as the initial starting point.
        self.selected_vertices = [0]
        # All vertices except the root (0) is selectable initially.
        self.selectable_vertices = np.full(self.N, True)
        self.selectable_vertices[0] = False
        self.previous_reward = 0


    def get_state(self):
        return tuple(sorted(self.selected_vertices))


    def get_reward(self):
        wl, skew = solve(self.N, self.selected_vertices[1:], self.inputDf)
        objective = wl + skew
        new_reward = (1 / objective) - self.previous_reward
        self.previous_reward = new_reward
        return new_reward


    def get_actions(self):
        return np.where(self.selectable_vertices)[0]


    def reset(self):
        self.selected_vertices = [0]
        self.previous_reward = 0
        self.selectable_vertices = np.full(self.N, True)
        self.selectable_vertices[0] = False

        return self.get_state()


    def step(self, vertex):
        self.selected_vertices.append(vertex)
        self.selectable_vertices[vertex] = False

        new_state = self.get_state()
        reward = self.get_reward()
        done = len(self.selected_vertices) - 1 == self.objectiveN

        return new_state, reward, done



class QLearningAgent:
    def __init__(self, environment):
        self.environment = environment

        # Exploitation vs exploration tradeoff when selecting actions during training.
        self.exploration_prob = 1
        self.exploration_decay = 0.99
        self.min_exploration_prob = 0.1
        # Settings
        self.learning_rate = 0.3
        self.discount_factor = 0.5
        self.visit_threshold = 5
        self.optimistic_estimate = float("inf")
        self.num_episodes = 100

        # Q(s, a) is the expected total discounted reward if the agent takes action
        # a in state s and acts optimally after. Initially 0.
        self.Q = {}
        # A table of frequencies for state-action pairs, initialy 0
        self.N_sa = {}


    def get_Q(self, state, action):
        key = tuple((state, action))
        if key not in self.Q:
            self.Q[key] = 0

        return self.Q[key]


    def get_N(self, state, action):
        key = tuple((state, action))
        if key not in self.N_sa:
            self.N_sa[key] = 0

        return self.N_sa[key]


    def f(self, utility, num_visits): # Exploration function
        if num_visits < self.visit_threshold:
            return self.optimistic_estimate
        else:
            return utility


    def update_q_table(self, state, action, new_state, reward_signal):
        # Reference: AIMA 4.0, Page 854 (Q-Learning Agent)
        if state is not None:
            key = tuple((state, action))
            if key in self.N_sa:
                self.N_sa[key] += 1
            else:
                self.N_sa[key] = 1

            # Get actions in the new state
            actions = self.environment.get_actions()
            best_action = max(actions, key=lambda action: self.get_Q(new_state, action))
            # Estimate the new Q-value
            self.Q[key] = \
                self.get_Q(state, action) + self.learning_rate * self.get_N(state, action) *\
                (reward_signal + self.discount_factor * self.get_Q(new_state, best_action) - self.get_Q(state, action))


    def choose_action(self):
        if random.random() <= self.exploration_prob:
            # Exploration: Choose a random action
            return np.random.choice(self.environment.get_actions())
        else:
            # Exploitation: Choose an action with the highest Q-value for the
            # current state
            state = self.environment.get_state()
            actions = self.environment.get_actions()
            best_action, best_score = None, float("-inf")
            for action in actions:
                Q_value = self.get_Q(state, action)
                if Q_value >= best_score:
                    best_action = action
                    best_score = Q_value
            return best_action


    def train(self):
        rewards = []
        for episode in range(self.num_episodes):
            # Reset the environment
            state = self.environment.reset()
            episode_reward = 0

            while True:
                # Perform epsilon-greedy policy
                action = self.choose_action()

                new_state, reward, done = self.environment.step(action)
                self.update_q_table(state, action, new_state, reward)
                episode_reward += reward

                if done:
                    break

                state = new_state

            self.exploration_prob = max(
                self.min_exploration_prob,
                self.exploration_prob * self.exploration_decay
            )

            print(f"Episode {episode + 1}: {episode_reward}", self.environment.selected_vertices)
            rewards.append(episode_reward)
        return rewards


    def select_best_vertices(self):
        import matplotlib.pyplot as plt

        rewards = self.train()
        plt.plot(range(1, self.num_episodes + 1), rewards)
        plt.show()

        self.environment.reset()

        while True:
            actions = self.environment.get_actions()
            state = self.environment.get_state()
            best_action, best_score = None, float("-inf")
            for action in actions:
                Q_value = self.get_Q(state, action)
                if Q_value >= best_score:
                    best_action = action
                    best_score = Q_value
            new_state, _, done = self.environment.step(best_action)
            if done:
                break

        selected_vertices = self.environment.selected_vertices[1:]
        self.environment.reset()
        return selected_vertices



import pandas as pd


inputDf = pd.read_csv("testcases/input_stt_45.csv.gz", compression="gzip")


q_env = QEnvironment(45, 2, inputDf[inputDf["netIdx"] == 299])
q_agent = QLearningAgent(q_env)
print(q_agent.select_best_vertices())
