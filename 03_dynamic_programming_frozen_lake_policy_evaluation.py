import gymnasium as gym
import numpy as np
import pprint

map = '4x4'
# map = '8x8'

# SLIPPERY = False    # deterministic
SLIPPERY = True    # stochastic

env = gym.make('FrozenLake-v1', desc=None, map_name=map, is_slippery=SLIPPERY)

GAMMA = 1.0 # discounting rate
THETA = 1e-5
num_states = env.observation_space.n
num_actions = env.action_space.n
transitions = env.unwrapped.P
print('num_states:', num_states)
print('num_actions:', num_actions)
pprint.pprint(transitions) # dynamics of the evironment


V = np.zeros(num_states)
policy = np.ones([num_states, num_actions]) * 0.25


while True:
    delta = 0
    for s in range(num_states):
        old_value = V[s]
        new_value = 0
        for a, prob_action in enumerate(policy[s]):
            for prob, s_, reward, _ in transitions[s][a]:
                new_value += prob_action * prob * (reward + GAMMA * V[s_])
        V[s] = new_value
        delta = max(delta, np.abs(old_value - V[s]))        
        
    if delta < THETA:
        break

print('Optimal Value = \n', V.reshape(4, 4) if map == '4x4' else V.reshape(8, 8))
