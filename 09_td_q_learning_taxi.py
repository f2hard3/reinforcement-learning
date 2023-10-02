# Q-Learning (off-policy TD control) for estimating pi=pi*
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

"""
6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger
    
state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
          5 * 5 * 5 * 4 = 500

Rewards:
    per-step : -1,
    delivering the passenger : +20,
    executing "pickup" and "drop-off" actions illegally : -10
    
blue: passenger
magenta: destination
yellow: empty taxi
green: full taxi
"""

env = gym.make('Taxi-v3')
n_states = env.observation_space.n
n_actions = env.action_space.n

GAMMA = 0.99
ALPHA = 0.9

epsilon = 0.7
epsilon_final = 0.1
epsilon_decay = 0.9999

Q = defaultdict(lambda: np.zeros(n_actions))

n_episodes = 1000

scores = []
steps = []
greedy = []

for episode in range(n_episodes):
    if episode > n_episodes * 0.995:
        env = gym.make('Taxi-v3', render_mode='human')

    s, _ = env.reset()
    step = 0
    score = 0
    
    while True:
        step += 1
        if np.random.rand() < epsilon:
            a = env.action_space.sample()
        else:
            a = np.argmax(Q[s])

        if epsilon > epsilon_final:
            epsilon *= epsilon_decay

        s_, r, terminated, truncated, _ = env.step(a)
        score += r

        Q[s][a] = Q[s][a] + ALPHA * (r + GAMMA * np.max(Q[s_]) - Q[s][a])

        if terminated or truncated:
            break
        
        s = s_

    steps.append(step)
    scores.append(score)
    greedy.append(epsilon)

    if episode % 100 == 0:
        print(f'average score and step of last 100 episodes: score = {np.mean(scores[-100:])} step = {np.mean(steps[-100:])}')

plt.bar(np.arange(len(steps)), steps)
plt.title(f'Steps of Taxi-v3, GAMMA: {GAMMA}, ALPHA: {ALPHA}')
plt.xlabel('episode')
plt.ylabel('steps per episode')
plt.show()

plt.bar(np.arange(len(scores)), scores)
plt.title(f'Scores of Taxi-v3, GAMMA: {GAMMA}, ALPHA: {ALPHA}')
plt.xlabel('episode')
plt.ylabel('score per episode')
plt.show()

plt.bar(np.arange(len(greedy)), greedy)
plt.title(f'Decay history of Taxi-v3, epsilon: {epsilon}, decay: {epsilon_decay}')
plt.xlabel('episode')
plt.ylabel('epsilon per episode')
plt.show()