from collections import deque
from random import sample
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, terminated, truncated):
        self.buffer.append([state, action, reward, next_state, terminated, truncated])

    def sample_batch(self, batch_size):
        transitions = sample(self.buffer, batch_size)
        # state, action, reward, next_state, terminated, truncated
        return transitions

    def size(self):
        return len(self.buffer)