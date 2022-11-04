import gym

env = gym.make('CliffWalking-v0', render_mode='rgb_array')
env.reset()
terminated, truncated = False, False
env.render()

temp = env.P

for i in range(100):

    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print("terminated = {}, truncated = {}".format(terminated, truncated))
    if terminated or truncated:
        break
    env.close()

