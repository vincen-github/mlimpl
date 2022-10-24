# import gym
#
# env_name = "CartPole-v1"
#
# env = gym.make(env_name, render_mode="human")
#
# state = env.reset()
#
# for i in range(100):
#     env.render()
#
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)
#
#     if terminated or truncated:
#         break
#
# env.close()
import sys
from math import log

from numpy import sqrt, pi

a = lambda x: sqrt(2 * log(x))
b = lambda x: (log(log(x)) + log(4 * pi)) / (2 * sqrt(2 * log(x)))
c = lambda x: sqrt(2 * log(x) - log(log(x)) - log(4 * pi))
d = lambda x: 1 / sqrt(2 * log(x))

f = lambda x: (a(x) - b(x) - c(x)) / d(x)

x = [1e10, 1e30, 1e100, 1e200, 9e307]
for t in x:
    print(f(t))
    print(t < sys.maxsize)