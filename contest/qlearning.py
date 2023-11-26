import random
import math
import time

from tqdm import tqdm
from mspd import solve


random.seed(24219284)

# Skips the timer
TIMER_OVERRIDE = False


class QEnvironment:
    ACTION_ADD_VERTEX = 'action_add_vertex'
    ACTION_MODIFY_ALPHA = 'action_modify_alpha'

    def __init__(self, N, objectiveN, inputDf):
        self.N = N
        self.objectiveN = objectiveN
        self.inputDf = inputDf
        self.alpha_values = alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Partial state formulation consisting of the selected vertices
        # and alpha value for MSPD.
        self.selected_vertices = []
        self.alpha = 0.5 # random.choice(self.alpha_values)

        self.objective_store = {}

        self.previous_objective = None
        # All vertices except the root (0) is selectable initially.
        self.selectable_vertices = [True for i in range(self.N)]
        self.selectable_vertices[0] = False


    def get_state(self):
        # Note that the number of selected vertices is at most 3 by the
        # constraints of the problem
        return tuple(sorted(self.selected_vertices)), self.alpha


    def compute_objective_score(self, alpha, selected_vertices):
        wl, skew = 0, 0
        state = tuple(sorted(selected_vertices)), alpha
        if state in self.objective_store:
            wl, skew = self.objective_store[state]
        else:
            wl, skew = solve(self.N, alpha, selected_vertices, self.inputDf)
            self.objective_store[state] = wl, skew

        return wl, skew


    def get_reward(self):
        current_state = self.get_state()
        wl, skew = self.compute_objective_score(self.alpha, self.selected_vertices)

        # The smaller the (wl + skew) value the better
        objective = wl + skew
        reward = objective if self.previous_objective is None\
            else self.previous_objective - objective
        self.previous_objective = objective

        return reward, wl + skew


    def get_actions(self):
        # Actions include adding an unselected vertex to the source set, or
        # changing the alpha values.

        actions = []
        for vertex in range(self.N):
            if self.selectable_vertices[vertex]:
                # Add an unselected vertex
                actions.append((QEnvironment.ACTION_ADD_VERTEX, vertex))

        # for alpha_value in self.alpha_values:
        #     if alpha_value == self.alpha:
        #         continue
        #     # Modify alpha values
        #     actions.append((QEnvironment.ACTION_MODIFY_ALPHA, alpha_value))
        return actions


    def reset(self):
        self.selected_vertices = []
        self.alpha = 0.5 # random.choice(self.alpha_values)

        self.selectable_vertices = [True for i in range(self.N)]
        self.selectable_vertices[0] = False
        self.previous_objective = None

        return self.get_state()


    def step(self, action):
        action_type, info = action
        if action_type == QEnvironment.ACTION_ADD_VERTEX:
            # Add an unselected vertex
            self.selected_vertices.append(info)
            self.selectable_vertices[info] = False
        elif action_type == QEnvironment.ACTION_MODIFY_ALPHA:
            self.alpha = info

        new_state = self.get_state()
        reward, _ = self.get_reward()

        done = len(self.selected_vertices) == self.objectiveN

        return new_state, reward, done



class QLearningAgent:
    def __init__(self, environment):
        self.environment = environment

        # Exploitation vs exploration tradeoff when selecting actions during training.
        self.exploration_prob = 1
        self.exploration_decay = 0.95
        self.min_exploration_prob = 0.0005
        # Settings
        self.learning_rate = 0.7
        self.learning_rate_decay = 0.95
        self.min_learning_rate = 0.05
        self.discount_factor = 0.15
        self.visit_threshold = 2
        self.optimistic_estimate = float("inf")

        self.num_episodes = 400
        self.num_local_search_iterations = 100

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


    def choose_action(self, eps_greedy=True):
        if eps_greedy and random.random() <= self.exploration_prob:
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


    def local_search(
        self,
        initial_state,
        initial_time=None,
        max_elapsed=float("inf")
    ):
        state = initial_state
        best_state, best_objective = None, float("inf")

        selectable_vertices = [True for i in range(self.environment.N)]
        # Root vertex is unselectable
        selectable_vertices[0] = False
        # Selected vertices are unselectable
        for vertex in state[0]:
            selectable_vertices[vertex] = False

        # Simulated annealing
        max_temperature = 10
        temperature_decay = 0.95

        temperature_schedule = lambda time: max_temperature * (alpha ** time)

        for i in range(self.num_local_search_iterations):
            selected_vertices, alpha = state
            # Pick a random action
            replace_index = random.choice(range(len(selected_vertices)))
            vertex = random.choice([i for i in range(self.environment.N) if selectable_vertices[i]])

            current_temperature = temperature_schedule(i)

            # Compute the scores of the current and next state
            curr_wl, curr_skew = self.environment.compute_objective_score(alpha, selected_vertices)

            new_selected_vertices = [x for x in selected_vertices]
            new_selected_vertices[replace_index] = vertex
            next_wl, next_skew = self.environment.compute_objective_score(alpha, new_selected_vertices)

            improvement = (curr_wl + curr_skew) - (next_wl + next_skew)
            if improvement > 0 or random.random() >= math.exp(improvement / current_temperature):
                state = (new_selected_vertices, alpha)
                selectable_vertices[vertex] = False
                selectable_vertices[selected_vertices[replace_index]] = True

            wl, skew = self.environment.compute_objective_score(state[1], state[0])
            curr_objective = wl + skew
            if best_state is None or curr_objective < best_objective:
                best_state, best_objective = state, curr_objective

            # Estimated time check
            if not TIMER_OVERRIDE and initial_time is not None and time.time() - initial_time >= max_elapsed:
                break

        return best_state, best_objective


    def train(
        self,
        reward_shaping=True,
        eps_greedy=True,
        initial_time=None,
        max_elapsed=float("inf")
    ):
        rewards = []
        for episode in range(self.num_episodes):
            # Reset the environment
            state = self.environment.reset()
            episode_reward = 0
            done = False

            while not done:
                # Perform epsilon-greedy policy
                action = self.choose_action(eps_greedy=eps_greedy)

                new_state, reward, done = self.environment.step(action)

                reward_signal = reward

                if reward_shaping:
                    # Perform local search (the higher the reward, the better).
                    # Take the inverse because the local search procedure returns
                    # the objective score which we want to minimize.
                    reward_shaping_curr = 0 if len(state[0]) == 0 else\
                        (1 / self.local_search(state)[1]) *\
                        (len(state[0]) / self.environment.objectiveN)
                    reward_shaping_new = 0 if len(new_state[0]) == 0 else\
                        (1 / self.local_search(new_state)[1]) *\
                        (len(new_state[0]) / self.environment.objectiveN)
                    # Reward shaping
                    reward_signal += self.discount_factor * reward_shaping_new - reward_shaping_curr

                self.update_q_table(state, action, new_state, reward_signal)
                episode_reward += reward_signal

                state = new_state

            self.exploration_prob = max(
                self.min_exploration_prob,
                self.exploration_prob * self.exploration_decay
            )
            self.learning_rate = max(
                self.min_learning_rate,
                self.learning_rate * self.learning_rate_decay
            )

            # print(f"Episode {episode + 1}: {episode_reward}", self.environment.get_state())
            rewards.append(episode_reward)

            # Estimated time check
            if not TIMER_OVERRIDE and initial_time is not None and time.time() - initial_time >= max_elapsed:
                break

        return rewards


    def select_best_vertices(
        self,
        # Whether to include reward shaping during training
        reward_shaping=True,
        # Whether to include epsilon greedy during training
        eps_greedy=True,
        # Whether to refine solution from RL with local search
        refine_soln=True
    ):

        # The time check is given to be lax since it is not refined.

        curr_time = time.time()
        rewards = self.train(
            reward_shaping=reward_shaping,
            eps_greedy=eps_greedy,
            initial_time=curr_time,
            max_elapsed=6.0 if refine_soln else 9.8
        )

        # import matplotlib.pyplot as plt
        # plt.scatter(range(1, self.num_episodes + 1), rewards)
        # plt.show()

        self.environment.reset()
        done = False
        while not done:
            action = self.choose_action(eps_greedy=False)
            _, _, done = self.environment.step(action)

        initial_state, alpha = self.environment.get_state()

        if refine_soln:
            curr_time=time.time()
            state, score = self.local_search((
                initial_state,
                alpha
            ), initial_time=curr_time, max_elapsed=3.8)
            return tuple(state[0])
        else:
            return initial_state



# Test program
# import pandas as pd
#
# inputDf = pd.read_csv("testcases/input_stt_45.csv.gz", compression="gzip")
# inputDf = inputDf[inputDf["netIdx"] == 299]
#
# q_env = QEnvironment(45, 2, inputDf)
# q_agent = QLearningAgent(q_env)
# vertices = q_agent.select_best_vertices(reward_shaping=True, eps_greedy=True, refine_soln=True)
# print(vertices)
