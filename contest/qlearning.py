import random


class BitTwiddling:
    @staticmethod
    def bitscan_lsb(bits):
        # Referenced from: https://stackoverflow.com/questions/5520655/return-index-of-least-significant-bit-in-python
        # Index of the least significant bit
        return (bits & -bits).bit_length() - 1

    @staticmethod
    def pop_lsb(bits):
        # Removes the least significant bit from the given binary
        return bits & ~(bits & -bits)

    @staticmethod
    def population_count(bits):
        # https://www.chessprogramming.org/Population_Count
        count = 0
        while bits:
            count += 1
            bits &= (bits - 1)
        return count

    @staticmethod
    def get_on_bits(bits):
        on_bits = []
        while bits:
            on_bits.append(BitTwiddling.bitscan_lsb(bits))
            bits = BitTwiddling.pop_lsb(bits)
        return on_bits


class QLearningAgent:
    # Settings
    learning_rate = 0.5
    discount_factor = 0.5
    visit_threshold = 5
    optimistic_estimate = float("inf")
    num_episodes = 100


    def __init__(self, N, objectiveN, inputDf):
        self.N = N
        self.objectiveN = objectiveN
        self.inputDf = inputDf

        self.bitmask = (1 << N) - 1

        # Exploitation vs exploration tradeoff when selecting actions during training.
        self.exploration_prob = 1.0
        self.exploration_decay = 0.9
        self.min_exploration_prob = 0.1

        # Persistent
        self.Q = {} # A table of action values indexed by state and action, initially 0
        self.N_sa = {} # A table of frequencies for state-action pairs, initialy 0


    # Each state is represented as a N-length bit vector with at most
    # objectiveN bits turned on. An action = k involves turning on the
    # k-th bit.
    def get_neighbours(self, state):
        num_vertices = BitTwiddling.population_count(state)
        if num_vertices == self.objectiveN:
            return []

        possible_vertices = self.bitmask & (~state)
        neighbours = []
        while possible_vertices:
            index = BitTwiddling.bitscan_lsb(possible_vertices)
            possible_vertices = BitTwiddling.pop_lsb(possible_vertices)

            next_state = state | (1 << index) # Inefficient
            neighbours.append((next_state, index))

        return neighbours


    def get_Q(self, state, action):
        if (state, action) in self.Q:
            return self.Q[(state, action)]

        # Q-table values are initialized to 0
        self.Q[(state, action)] = 0
        return 0


    def get_N(self, state, action):
        if (state, action) in self.N_sa:
            return self.N_sa[(state, action)]

        self.N_sa[(state, action)] = 0
        return 0


    def f(self, utility, num_visits): # Exploration function
        if num_visits < QLearningAgent.visit_threshold:
            return QLearningAgent.optimistic_estimate
        else:
            return utility


    def update_q_table(self, prev_state, prev_action, current_state, reward_signal):
        # Next actions from the current state
        neighbours = self.get_neighbours(current_state)

        # Reference: AIMA 4.0, Page 854 (Q-Learning Agent)
        if prev_state != 0:
            if (prev_state, prev_action) in self.N_sa:
                self.N_sa[(prev_state, prev_action)] += 1
            else:
                self.N_sa[(prev_state, prev_action)] = 1

            max_Q = 0 if len(neighbours) == 0 else\
                max(neighbours, key=lambda neighbour: self.get_Q(current_state, neighbour[1]))

            # Estimate the new Q-value
            self.Q[(prev_state, prev_action)] = \
                self.get_Q(prev_state, prev_action) +\
                QLearningAgent.learning_rate * (self.N_sa[(prev_state, prev_action)]) *\
                (reward_signal + QLearningAgent.discount_factor * max_Q - self.get_Q(prev_state, prev_action))

        is_terminal = len(neighbours) == 0
        return is_terminal


    def choose_action(self, state):
        neighbours = self.get_neighbours(state)
        if random.random() <= self.exploration_prob:
            # Exploration: Choose a random action
            return random.choice(neighbours)[1]
        else:
            # Exploitation: Choose an action with the highest Q-value for
            # the current state
            best_action, best_score = None, -1
            for _, action in neighbours:
                score = self.f(self.get_Q(state, action), self.get_N(state, action))
                if score >= best_score:
                    best_action, best_score = action, score
            return best_action


    def perform_action(self, current_state, action):
        return current_state | (1 << action) # Inefficent


    def get_reward(self, state):
        wl, skew = solve(self.N, BitTwiddling.get_on_bits(state), self.inputDf)
        objective = wl + skew
        return -objective


    def train(self):
        for episode in range(QLearningAgent.num_episodes):
            # Reset the environment
            state = 0
            total_reward = 0

            while True:
                action = self.choose_action(state)
                new_state = self.perform_action(state, action)
                reward = self.get_reward(new_state)

                is_terminal = self.update_q_table(state, action, new_state, reward)
                total_reward += reward
                if is_terminal:
                    break

                state = new_state

            self.exploration_prob = max(
                self.min_exploration_prob,
                self.exploration_prob * self.exploration_decay
            )

    def select_best_vertices(self):
        self.train()

        state = 0
        vertices = []

        while len(vertices) < self.objectiveN:
            neighbours = self.get_neighbours(state)
            best_action, best_Q_value = None, -1
            for _, action in neighbours:
                Q_value = self.get_Q(state, action)
                if Q_value >= best_Q_value:
                    best_action, best_Q_value = action, Q_value
            vertices.append(best_action)
            state = self.perform_action(state, best_action)

        return vertices



# import pandas as pd
#
# inputDf = pd.read_csv("testcases/input_stt_45.csv.gz", compression="gzip")
#
#
# q_agent = QLearningAgent(45, 2, inputDf[inputDf["netIdx"] == 299])
# print(q_agent.select_best_vertices())
