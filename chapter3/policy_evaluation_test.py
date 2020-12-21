import gym, gym_walk
from policy_evaluation import Policy_Evaluation

policy_evaluation = Policy_Evaluation()

env = gym.make('SlipperyWalkFive-v0')
P = env.env.P

LEFT, RIGHT = range(2)
pi = lambda s: {
    0:LEFT, 1:LEFT, 2:LEFT, 3:LEFT, 4:LEFT, 5:LEFT, 6:LEFT
}[s]

V = policy_evaluation.policy_evaluation(pi, P)

print(V)