import numpy as np


class Policy_Evaluation:

    def __init__(self):
        pass

    def policy_evaluation(self, pi, P, gamma=1.0, theta=1e-10):

        num_states = len(P)

        prev_V = np.zeros(num_states)

        while True:

            V = np.zeros(num_states)

            for s in range(num_states):

                for prob, next_state, reward, done in P[s][pi(s)]:

                    V[s] += prob * (reward + (gamma * prev_V[next_state] if not done else 0))

            if np.max(np.abs(V - prev_V)) < theta:
                print('Converged')
                break

            prev_V = V.copy()

        return V