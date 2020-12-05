import gym, gym_walk

P = gym.make('BanditWalk-v0').env.P
print(P)

P = gym.make('BanditSlipperyWalk-v0').env.P
print(P)

P = gym.make('FrozenLake-v0').env.P
print(P)
