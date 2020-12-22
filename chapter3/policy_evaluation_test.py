import gym, gym_walk
from policy_evaluation import PolicyEvaluation
from policy_improvement import PolicyImprovement
from policy_utils import PolicyUtils


env = gym.make('SlipperyWalkFive-v0')
P = env.env.P

LEFT, RIGHT = range(2)
action_symbols=('<', '>')
n_cols=7

pi = lambda s: {
    0: LEFT, 1: LEFT, 2: LEFT, 3: LEFT, 4: LEFT, 5: LEFT, 6: LEFT
}[s]

print('Original Policy: ')
PolicyUtils.print_policy(pi, P, action_symbols=action_symbols, n_cols=n_cols)

V = PolicyEvaluation.policy_evaluation(pi, P)

print('Values of States for Original Policy: ')
print(V)

improved_pi = PolicyImprovement.policy_improvement(V, P)

print("Improved Policy: ")
PolicyUtils.print_policy(improved_pi, P, action_symbols=action_symbols, n_cols=n_cols)