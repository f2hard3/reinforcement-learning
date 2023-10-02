import gymnasium as gym
import numpy as np 

map = '4x4'
# map = '8x8'

SLIPPERY = False    # deterministic
# SLIPPERY = True    # stochastic

env = gym.make('FrozenLake-v1', desc=None, map_name=map, is_slippery=SLIPPERY)

GAMMA = 1.0
THETA = 1e-7

num_states = env.observation_space.n
num_actions = env.action_space.n
transitions = env.unwrapped.P

V = np.zeros(num_states)
pi = np.ones([num_states, num_actions]) * 0.25

policy_stable = False
while not policy_stable:
    while True:
        delta = 0
        for s in range(num_states):
            old_value = V[s]
            new_value = 0
            for a, prob_a in enumerate(pi[s]):
                for prob, s_, r, _ in transitions[s][a]:
                    new_value += prob_a * prob * (r + GAMMA * V[s_])
            V[s] = new_value
            delta = max(delta, np.abs(old_value - V[s]))
        if delta < THETA:
            break
    
    old_pi = pi
    for s in range(num_states):
        new_action_values = np.zeros(num_actions)
        for a in range(num_actions):
            for prob, s_, r, _ in transitions[s][a]:
                new_action_values[a] += prob * (r + GAMMA * V[s_])
        new_action = np.argmax(new_action_values)
        pi[s] = np.eye(num_actions)[new_action]
    
    if np.array_equal(old_pi, pi):
        policy_stable = True

print('Optimal Value Function = \n', V.reshape(4, 4) if map == '4x4' 
        else V.reshape(8, 8))
print('Optimal Policy = \n', pi)
print('Optimal Action = \n', np.argmax(pi, axis=1).reshape(4, 4) if map == '4x4'
        else np.argmax(pi, axis=1).reshape(8, 8))