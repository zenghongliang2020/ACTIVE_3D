import numpy as np

class Replaybuffer:
    def __init__(self, state_dim, num_sw, action_dim, max_size, batch_size):
        self.mem_size = max_size
        self.mem_cnt = 0
        self.batch_size = batch_size

        self.state_memory = np.zeros((self.mem_size, num_sw, state_dim))
        self.mask_memory = np.zeros((self.mem_size, 9))
        self.action_memory = np.zeros((self.mem_size, action_dim))
        self.reward_memory = np.zeros((self.mem_size, ))
        self.next_state_memory = np.zeros((self.mem_size, num_sw, state_dim))
        self.next_mask_memory = np.zeros((self.mem_size, 9))
        self.terminal_memory = np.zeros((self.mem_size, ), dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        mem_idx = self.mem_cnt % self.mem_size
        self.state_memory[mem_idx] = state['observation']
        self.mask_memory[mem_idx] = state['mask']
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.next_state_memory[mem_idx] = state_['observation']
        self.next_mask_memory[mem_idx] = state_['mask']
        self.terminal_memory[mem_idx] = done

        self.mem_cnt += 1

    def sample_buffer(self):
        mem_len = min(self.mem_size, self.mem_cnt)

        batch = np.random.choice(mem_len, self.batch_size, replace=False)

        observations = self.state_memory[batch]
        masks = self.mask_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        observations_ = self.next_state_memory[batch]
        masks_ = self.next_mask_memory[batch]
        terminals = self.terminal_memory[batch]

        states = {'observation': observations, 'mask': masks}
        states_ = {'observation': observations_, 'mask': masks_}
        return states, actions, rewards, states_, terminals

    def ready(self):
        return self.mem_cnt > self.batch_size