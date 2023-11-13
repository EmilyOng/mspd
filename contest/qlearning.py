import random
from mspd import solve


class QEnvironment:
    ACTION_REPLACE_VERTEX = 'action_replace_vertex'
    ACTION_MODIFY_ALPHA = 'action_modify_alpha'

    def __init__(self, N, objectiveN, inputDf):
        self.N = N
        self.objectiveN = objectiveN
        self.inputDf = inputDf
        self.initial_objective = 2

        # Complete state formulation consisting of the selected vertices
        # and alpha value for MSPD.
        self.selected_vertices = self.randomize_start_state()
        self.alpha = 0.1

        self.rewards_store = {}

        # All vertices except the root (0) is selectable initially.
        self.selectable_vertices = [True for i in range(self.N)]
        self.selectable_vertices[0] = False
        self.previous_objective = 2 # Normalized wire length and skew (1 + 1 = 2)


    def randomize_start_state(self):
        return random.sample(range(1, self.N), self.objectiveN)


    def get_state(self):
        return tuple(sorted(self.selected_vertices)), self.alpha


    def get_reward(self):
        current_state = self.get_state()
        wl, skew = 0, 0
        if current_state in self.rewards_store:
            wl, skew = self.rewards_store[current_state]
        else:
            wl, skew = solve(self.N, self.alpha, self.selected_vertices, self.inputDf)
            self.rewards_store[current_state] = (wl, skew)

        # The smaller the better
        objective = wl + skew
        # Rewards improvement in objective scores
        reward = objective if self.previous_objective is None\
            else self.previous_objective - objective
        self.previous_objective = objective
        return reward, -objective


    def get_actions(self):
        # Actions include perturbing the current complete state formulation by
        # replacing a source vertex with another vertex, or changing the
        # alpha values.

        actions = []
        for vertex in range(self.N):
            if self.selectable_vertices[vertex]:
                for replaceable_index in range(self.objectiveN):
                    # Replacing a source vertex
                    actions.append((QEnvironment.ACTION_REPLACE_VERTEX, (replaceable_index, vertex)))

        alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for alpha_value in alpha_values:
            if alpha_value == self.alpha:
                continue
            # Modifying alpha values
            actions.append((QEnvironment.ACTION_MODIFY_ALPHA, alpha_value))
        return actions


    def reset(self):
        self.selected_vertices = self.randomize_start_state()
        self.alpha = 0.1

        self.selectable_vertices = [True for i in range(self.N)]
        self.selectable_vertices[0] = False
        self.previous_objective = self.initial_objective

        return self.get_state()


    def step(self, action):
        action_type, info = action
        if action_type == QEnvironment.ACTION_REPLACE_VERTEX:
            # Replacing a source vertex
            replaceable_index, vertex = info

            original_vertex = self.selected_vertices[replaceable_index]
            self.selected_vertices[replaceable_index] = vertex

            self.selectable_vertices[vertex] = False
            self.selectable_vertices[original_vertex] = True
        elif action_type == QEnvironment.ACTION_MODIFY_ALPHA:
            self.alpha = info

        new_state = self.get_state()
        reward, _ = self.get_reward()

        return new_state, reward



class QLearningAgent:
    def __init__(self, environment):
        self.environment = environment

        # Exploitation vs exploration tradeoff when selecting actions during training.
        self.exploration_prob = 1
        self.exploration_decay = 0.95
        self.min_exploration_prob = 0.1
        # Settings
        self.learning_rate = 0.8
        self.learning_rate_decay = 0.95
        self.min_learning_rate = 0.02
        self.discount_factor = 0.9
        self.visit_threshold = 1
        self.optimistic_estimate = float("inf")

        self.num_episodes = 300
        self.num_iterations_per_episode = 50

        # Q(s, a) is the expected total discounted reward if the agent takes action
        # a in state s and acts optimally after. Initially 0.
        self.Q = {}
        # A table of frequencies for state-action pairs, initialy 0
        self.N_sa = {}


    def get_Q(self, state, action):
        if (state, action) not in self.Q:
            self.Q[(state, action)] = 0

        return self.Q[(state, action)]


    def get_N(self, state, action):
        if (state, action) not in self.N_sa:
            self.N_sa[(state, action)] = 0

        return self.N_sa[(state, action)]


    def f(self, utility, num_visits): # Exploration function
        if num_visits < self.visit_threshold:
            return self.optimistic_estimate
        else:
            return utility


    def update_q_table(self, state, action, new_state, reward_signal):
        # Reference: AIMA 4.0, Page 854 (Q-Learning Agent)
        if state is not None:
            key = (state, action)
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


    def choose_action(self, greedy=False):
        if not greedy and random.random() <= self.exploration_prob:
            # Exploration: Choose a random action
            return random.choice(self.environment.get_actions())
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

            for _ in range(self.num_iterations_per_episode):
                # Perform epsilon-greedy policy
                action = self.choose_action()

                new_state, reward = self.environment.step(action)
                self.update_q_table(state, action, new_state, reward)
                episode_reward += reward

                state = new_state

            self.exploration_prob = max(
                self.min_exploration_prob,
                self.exploration_prob * self.exploration_decay
            )
            self.learning_rate = max(
                self.min_learning_rate,
                self.learning_rate * self.learning_rate_decay
            )

            print(f"Episode {episode + 1}: {episode_reward}", self.environment.selected_vertices)
            rewards.append(episode_reward)
        return rewards


    def select_best_vertices(self):
        import matplotlib.pyplot as plt

        rewards = self.train()
        plt.plot(range(1, self.num_episodes + 1), rewards)
        plt.show()

        best_state, best_score = None, float("-inf")
        # Try different randomly generated initial state
        max_rounds = self.environment.N * self.environment.objectiveN
        max_iters_per_round = 100
        for _ in range(max_rounds):
            # Start at a randomly generated initial state
            state = self.environment.reset()

            for _ in range(max_iters_per_round):
                # Move greedily to a local maximum
                action = self.choose_action(greedy=True)

                new_state, reward = self.environment.step(action)
                if reward >= 0: # Represents an improvement from the current state
                    state = new_state
                else:
                    break

            _, objective = self.environment.get_reward()
            if objective >= best_score:
                best_state, best_score = state, objective

        return best_state, best_score


import pandas as pd


inputDf = pd.read_csv("testcases/input_stt_45.csv.gz", compression="gzip")
inputDf = inputDf[inputDf["netIdx"] == 299]

q_env = QEnvironment(45, 2, inputDf)
q_agent = QLearningAgent(q_env)
print(q_agent.select_best_vertices())
