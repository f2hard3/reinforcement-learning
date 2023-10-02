import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

GAMMA = 0.99
ALPHA = 0.9

epsilon = 0.3
n_episodes = 10_000

is_slippery = False

env = gym.make('FrozenLake-v1', is_slippery=is_slippery, render_mode=None)

num_actions = env.action_space.n

win_pct = []
scores = []

Q = defaultdict(lambda: np.zeros(num_actions))

for episode in range(n_episodes):
    if episode > n_episodes * 0.999:
        env = gym.make('FrozenLake-v1', is_slippery=is_slippery, render_mode='human')
    
    s, _ = env.reset()
    score = 0
    while True:
        if np.random.rand() < epsilon:
            a = env.action_space.sample()
        else:
            a = np.argmax(Q[s])
        
        s_, r, terminated, truncated, _ = env.step(a)
        score += r

        # update Q(S,A): Q(S,A) <- Q(S,A) + alpha[R + gamma*max_aQ(S',a) - Q(S, A)]
        Q[s][a] = Q[s][a] + ALPHA * (r + GAMMA * np.max(Q[s_]) - Q[s][a])

        if terminated or truncated:
            break

        s = s_
    
    scores.append(score)

    if episode % 1000 and episode > 0.8 * n_episodes:
        average = np.mean(scores[-10:])
        win_pct.append(average)

print('Stochastic' if is_slippery else 'Deterministic')
print(f'GAMMA={GAMMA}, epsilon={epsilon}, ALPHA={ALPHA}')

plt.plot(win_pct)
plt.xlabel('episode')
plt.ylabel('success ratio')
plt.title(f'average success ration of last 10 games\n - {"Stochastic Env" if is_slippery else "Deterministic Env"}')
plt.show()

WIDTH = 4
HEIGHT = 4
GOAL = (3, 3)
actions = ['L', 'D', 'R', 'U']

optimal_policy = []
for i in range(HEIGHT):
    optimal_policy.append([])
    for j in range(WIDTH):
        optimal_action = Q[i * WIDTH + j].argmax()
        if (i, j) == GOAL:
            optimal_policy[i].append('G')
        else:
            optimal_policy[i].append(actions[optimal_action])

for row in optimal_policy:
    print(row)