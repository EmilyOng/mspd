import random
import time
import math

from mspd import solve


class LocalSearchAgent:
    ACTION_ADD_VERTEX = 'action_add_vertex'
    ACTION_REPLACE_VERTEX = 'action_replace_vertex'
    ACTION_REMOVE_VERTEX = 'action_remove_vertex'


    def __init__(self, N, objectiveN, inputDf):
        self.N = N
        self.objectiveN = objectiveN
        self.inputDf = inputDf

        self.objective_store = {}


    def compute_objective_score(self, state):
        selected_vertices, _ = state
        alpha = 0.5

        wl, skew = 0, 0
        state = tuple(sorted(selected_vertices)), alpha
        if state in self.objective_store:
            wl, skew = self.objective_store[state]
        else:
            wl, skew = solve(self.N, alpha, selected_vertices, self.inputDf)
            self.objective_store[state] = wl, skew

        objective = wl + skew if self.objectiveN == 1\
            else 3 * wl + skew if self.objectiveN == 2\
            else wl + 3 * skew
        return wl, skew, objective


    def get_actions(self, state):
        selected_vertices, selectable_vertices = state

        actions = []
        for vertex in range(self.N):
            if selectable_vertices[vertex]:
                # Add an unselected vertex
                if len(selected_vertices) < 3:
                    actions.append((LocalSearchAgent.ACTION_ADD_VERTEX, vertex))

                if len(selected_vertices) > 0:
                    # Replacing a vertex
                    for i in range(len(selected_vertices)):
                        actions.append((LocalSearchAgent.ACTION_REPLACE_VERTEX, (i, vertex)))

        if len(selected_vertices) > 1:
            for i in range(len(selected_vertices)):
                # Removing a vertex
                actions.append((LocalSearchAgent.ACTION_REMOVE_VERTEX, i))

        return actions


    def perform_action(self, state, action):
        selected_vertices, selectable_vertices = state

        action_type, info = action
        if action_type == LocalSearchAgent.ACTION_ADD_VERTEX:
            # Add an unselected vertex
            selected_vertices.append(info)
            selectable_vertices[info] = False
        elif action_type == LocalSearchAgent.ACTION_REPLACE_VERTEX:
            # Replace with an unselected vertex
            index_to_replace, new_vertex = info
            selectable_vertices[selected_vertices[index_to_replace]] = True
            selectable_vertices[new_vertex] = False
            selected_vertices[index_to_replace] = new_vertex
        elif action_type == LocalSearchAgent.ACTION_REMOVE_VERTEX:
            # Remove a vertex
            selectable_vertices[selected_vertices[info]] = True
            selected_vertices.pop(info)

        return selected_vertices, selectable_vertices


    def search(self, initial_time, max_elapsed):
        # State representation
        selected_vertices = []
        selectable_vertices = [True for i in range(self.N)]
        # Source vertex is unselectable
        selectable_vertices[0] = False

        state = (selected_vertices, selectable_vertices)

        best_state = state
        _, _, best_objective = self.compute_objective_score(state)

        # Simulated annealing
        max_temperature = 10
        temperature_decay = 0.95

        temperature_schedule = lambda time: max_temperature * (temperature_decay ** time) + 0.000001
        curr_time = 0

        while True:
            # Pick a random action
            available_actions = self.get_actions(state)
            action = random.choice(available_actions)

            next_state = self.perform_action(state, action)

            current_temperature = temperature_schedule(curr_time)

            # Compute the scores of the current and next state
            curr_wl, curr_skew, curr_objective = self.compute_objective_score(state)

            new_selected_vertices = next_state[0]
            next_wl, next_skew, next_objective = self.compute_objective_score(next_state)

            improvement = curr_objective - next_objective
            if improvement > 0 or random.random() >= math.exp(improvement / current_temperature):
                state = next_state

            wl, skew, curr_objective = self.compute_objective_score(state)
            if best_state is None or curr_objective < best_objective:
                best_state, best_objective = state, curr_objective

            curr_time += 1

            # Estimated time check
            if time.time() - initial_time >= max_elapsed:
                break

        return best_state, best_objective


    def select_best_vertices(self):
        best_state, best_objective = self.search(initial_time=time.time(), max_elapsed=9.8)
        return best_state[0]


# Test program
# import pandas as pd
#
# inputDf = pd.read_csv("testcases/input_stt_45.csv.gz", compression="gzip")
# inputDf = inputDf[inputDf["netIdx"] == 299]
#
# ls_agent = LocalSearchAgent(45, 1, inputDf)
# vertices = ls_agent.select_best_vertices()
# print(vertices)
