import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

win = 0
lose = 0
draw = 0
GAMMA = 1
e = 0.2
num_episodes = 100_000

env = gym.make('Blackjack-v1', sab=True, render_mode=None)
num_actions = env.action_space.n

# e-soft
pi = defaultdict(lambda: np.ones(num_actions, dtype=float) / num_actions)
# Q(s, a)
Q = defaultdict(lambda: np.zeros(num_actions))
# returns(s, a)
returns = defaultdict(list)

for n_episode in range(num_episodes):
    episode = []
    s, _ = env.reset()
    while True:
        P = pi[s]
        a = np.random.choice(np.arange(len(P)), p=P) # 0: stick, 1: hit
        s_, r, terminated, truncated, _ = env.step(a)
        episode.append((s, a, r))
        if terminated or truncated:
            if r == 1: 
                win += 1
            elif r == -1:
                lose += 1
            else:
                draw += 1
            break
        s = s_
    
    G = 0
    visited_state_action_pair = []
    for s, a, r in episode[::-1]:
        G = GAMMA * G + r
        if (s, a) not in visited_state_action_pair:
            returns[(s, a)].append(G)
            Q[s][a] = np.mean(returns[(s, a)])
            visited_state_action_pair.append((s, a))
        
        A_star = np.argmax(Q[s])
        for a in range(num_actions):
            if a == A_star:
                pi[s][a] = 1 - e + e / num_actions
            else:
                pi[s][a] = e / num_actions
    
    if n_episode % 5000 == 0:
        print(f'{n_episode} is completed.')

print(f'win ratio = {win / num_episodes * 100:.2f}%')
print(f'lose ratio = {lose / num_episodes * 100:.2f}%')
print(f'draw ratio = {draw / num_episodes * 100:.2f}%')

V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value

sample_state = (21, 3, True)
optimal_action = np.argmax(Q[sample_state])
print(f"state {sample_state}'s value = {V[sample_state]:.2f}", 'stick' if optimal_action == 0 else 'hit')


sample_state = (4, 1, False)
optimal_action = np.argmax(Q[sample_state])
print(f"state {sample_state}'s value = {V[sample_state]:.2f}", 'stick' if optimal_action == 0 else 'hit')


sample_state = (14, 8, True)
optimal_action = np.argmax(Q[sample_state])
print(f"state {sample_state}'s value = {V[sample_state]:.2f}", 'stick' if optimal_action == 0 else 'hit')

X, Y = np.meshgrid(
    np.arange(1, 11),
    np.arange(12, 22)
)
no_usable_ace = np.apply_along_axis(lambda idx: V[(idx[1], idx[0], False)], 2, np.dstack([X, Y]))
usable_ace = np.apply_along_axis(lambda idx: V[(idx[1], idx[0], True)], 2, np.dstack([X, Y]))

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 3), subplot_kw={'projection': '3d'})

ax0.plot_surface(X, Y, no_usable_ace, cmap=plt.cm.YlGnBu_r)
ax0.set_xlabel('Dealer open Cards')
ax0.set_ylabel('Player Cards')
ax0.set_zlabel('MC Estimated Value')
ax0.set_title('Useable Ace')

ax1.plot_surface(X, Y, usable_ace, cmap=plt.cm.YlGnBu_r)
ax1.set_xlabel('Dealer open Cards')
ax1.set_ylabel('Player Cards')
ax1.set_zlabel('MC Estimated Value')
ax1.set_title('Useable Ace')

plt.show()