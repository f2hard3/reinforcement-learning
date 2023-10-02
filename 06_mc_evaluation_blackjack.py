import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

stick_threshold = 17
win_cnt = 0
lose_cnt = 0
draw_cnt = 0
num_episodes = 100_000
GAMMA = 1 # no discounting

env = gym.make('Blackjack-v1', sab=True)

# state : (Player Cards, Dealer open Cards, Usable Ace) ex) (6, 1, False)
def pi(state):
    return 0 if state[0] >= stick_threshold else 1

V = defaultdict(float)
returns = defaultdict(list)

for i in range(num_episodes):
    episode = []
    s, _ = env.reset()
    while True:
        a = pi(s)
        s_, r, terminated, truncated, _ = env.step(a)
        episode.append((s, a, r))        
        if terminated or truncated:
            if r == 1:
                win_cnt += 1
            elif r == -1:
                lose_cnt += 1
            else:
                draw_cnt += 1
            break        
        s = s_
    G = 0
    visited_states = []
    for s, a, r in episode[::-1]:
        G = GAMMA * G + r
        if s not in visited_states:
            returns[s].append(G)
            V[s] = np.mean(returns[s])
            visited_states.append(s)
    if i % 5000 == 0:
        print(f'episode {i} complted...')

print(f'stick threshold = {stick_threshold}')
print(f'win ratio = {win_cnt / num_episodes * 100:.2f}%')
print(f'lose ratio = {lose_cnt / num_episodes * 100:.2f}%')
print(f'draw ratio = {draw_cnt / num_episodes * 100:.2f}%')

sample_state = (21, 3, True)
print(f"state {sample_state}'s value =  {V[sample_state]:.2f}")
print(f'while the player holding {sample_state[0]} and the dealer showing {sample_state[1]}')

sample_state = (14, 1, False)
print(f"state {sample_state}'s value =  {V[sample_state]:.2f}")
print(f'while the player holding {sample_state[0]} and the dealer showing {sample_state[1]}')

X, Y = np.meshgrid(
    np.arange(12, 22),
    np.arange(1, 11)
)
no_usable_ace = np.apply_along_axis(lambda idx: V[(idx[0], idx[1], False)], 2, np.dstack([X, Y]))
usable_ace = np.apply_along_axis(lambda idx: V[(idx[0], idx[1], True)], 2, np.dstack([X, Y]))

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4), subplot_kw={'projection': '3d'})

ax0.plot_surface(Y, X, no_usable_ace, cmap=plt.cm.YlGnBu_r)
ax0.set_xlabel('Dealer open Cards')
ax0.set_ylabel('Player Cards')
ax0.set_zlabel('MC Estimated Value')
ax0.set_title('No Usable Ace')

ax1.plot_surface(Y, X, usable_ace, cmap=plt.cm.YlGnBu_r)
ax1.set_xlabel('Dealer open Cards')
ax1.set_ylabel('Player Cards')
ax1.set_zlabel('MC Estimated Value')
ax1.set_title('Usable Ace')

plt.show()