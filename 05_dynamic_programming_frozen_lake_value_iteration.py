import gymnasium as gym
import numpy as np

map = '4x4'
# map = '8x8'

SLIPPERY = False    # deterministic
# SLIPPERY = True   # stochastic

env = gym.make('FrozenLake-v1', desc=None, map_name=map, is_slippery=SLIPPERY)

GAMMA = 0.95
THETA = 1e-5

num_states = env.observation_space.n
num_actions = env.action_space.n
transitions = env.unwrapped.P

V = np.zeros(num_states)

while True:
    delta = 0
    for s in range(num_states):
        old_value = V[s]
        new_action_values = np.zeros(num_actions)
        for a in range(num_actions):
            for prob, s_, r, _ in transitions[s][a]:
                new_action_values[a] += prob * (r + GAMMA * V[s_])
        v_max = max(new_action_values)
        delta = max(delta, np.abs(v_max - old_value))
        V[s] = v_max 
    if delta < THETA:
        break

pi = np.zeros((num_states, num_actions))

for s in range(num_states):
    action_values = np.zeros(num_actions)
    for a in range(num_actions):
        for prob, s_, r, _ in transitions[s][a]:
            action_values[a] += prob * (r + GAMMA * V[s_])
        new_action = np.argmax(action_values)
        pi[s] = np.eye(num_actions)[new_action]

print('Optimal Value Function = \n', V.reshape(4, 4) if map == '4x4' 
        else V.reshape(8, 8))
print('Optimal Policy = \n', pi)
print('Optimal Action = \n', np.argmax(pi, axis=1).reshape(4, 4) if map == '4x4'
        else np.argmax(pi, axis=1).reshape(8, 8))