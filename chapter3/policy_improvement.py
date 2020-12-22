import numpy as np


class PolicyImprovement:

    def __init__(self):
        pass

    @staticmethod
    def policy_improvement(V, P, gamma = 1.0):

        num_states = len(P)
        num_actions = len(P[0]) # assuming all states have the same actions

        Q = np.zeros((num_states, num_actions), dtype=np.float64)

        for s in range(num_states):
            for a in range(num_actions):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s, a] += prob * (reward + (gamma * V[next_state] if not done else 0))

        best_pi = np.argmax(Q, axis=1)

        best_pi_as_function = lambda s: {s:a for s, a in enumerate(best_pi)}[s]

        return best_pi_as_function
