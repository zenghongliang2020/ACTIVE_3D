import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from RelayBuffer import Replaybuffer

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

class Dueling_net(nn.Module):
    def __init__(self, alpha, state_dim, action_coor_dim, action_angle_dim):
        super(Dueling_net, self).__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.V = nn.Linear(256, 1)
        self.A_1 = nn.Linear(256, action_coor_dim)
        self.A_2 = nn.Linear(256, action_angle_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state):
        obs = state['observation'].cuda()
        action_mask = state['mask'].cuda()
        x = T.relu(self.fc1(obs))

        V = self.V(x)
        A_1 = self.A_1(x)
        A_2 = self.A_2(x)

        return V, A_1 * action_mask, A_2

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class DuelingDQN:
    def __init__(self, alpha, state_dim, action_1_dim, action_2_dim, ckpt_dir, gamma=0.99, tau=0.005,
                 epsilon=1.0, eps_end=0.01, eps_dec=5e-4, max_size=1000000, batch_size=256):
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.checkpoint_dir = ckpt_dir
        self.q_eval = Dueling_net(alpha=alpha, state_dim=state_dim, action_1_dim=action_1_dim, action_2_dim=action_2_dim)
        self.q_target = Dueling_net(alpha=alpha, state_dim=state_dim, action_1_dim=action_1_dim, action_2_dim=action_2_dim)
        self.action_1_space = [i for i in range(action_1_dim)]
        self.action_2_space = [i for i in range(action_2_dim)]

        self.memory = Replaybuffer(state_dim=state_dim, action_1_dim=action_1_dim, action_2_dim=action_2_dim,
                                   max_size=max_size, batch_size=batch_size)

        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)


    def choose_action(self, observation, isTrain=True):
        state = T.tensor([observation], dtype=T.float).to(device)
        _, A_1, A_2 = self.q_eval.forward(state)
        action_1 = T.argmax(A_1).item()
        action_2 = T.argmax(A_2).item()

        if (np.random.random() < self.epsilon) and isTrain:
            action_1 = np.random.choice(self.action_1_space)
            action_2 = np.random.choice(self.action_2_space)

        return action_1, action_2


    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def learn(self):
        if not self.memory.ready():
            return

        states, actions, rewards, next_states, terminals = self.memory.sample_buffer()
        batch_idx = T.arange(self.batch_size, dtype=T.long).to(device)
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        actions_tensor = T.tensor(actions, dtype=T.long).to(device)
        rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
        next_states_tensor = T.tensor(next_states, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)

        with T.no_grad():
            V_, A_1_, A_2_ = self.q_target.forward(next_states_tensor)
            q_ = V_ + A_1_ + A_2_ - T.mean(A_1_, dim=-1, keepdim=True) - T.mean(A_2_, dim=-1, keepdim=True)
            q_[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * T.max(q_, dim=-1)[0]
            V, A_1, A_2 = self.q_eval.forward(states_tensor)
            q = (V + A_1 - T.mean(A_1, dim=-1, keepdim=True) - T.mean(A_2, dim=-1, keepdim=True))[batch_idx, actions_tensor]

            loss = F.mse_loss(q, target.detach())
            self.q_eval.optimizer.zero_grad()
            loss.backward()
            self.q_eval.optimizer.step()

            self.update_network_parameters()
            self.decrement_epsilon()

    def save_models(self, episode):
        self.q_eval.save_chekpoint(self.checkpoint_dir + 'Q_eval/DuelingDQN_q_eval_{}.pth'.format(episode))
        print('Saving Q_eval network successfully!')

        self.q_target.save_chekpoint(self.checkpoint_dir + 'Q_target/DuelingDQN_q_target_{}.pth'.format(episode))
        print('Saving Q_target network successfully!')

    def load_models(self, episode):
        self.q_eval.load_checkpoint(self.checkpoint_dir + 'Q_eval/DuelingDQN_q_eval_{}.pth'.format(episode))
        self.q_target.load_checkpoint(self.checkpoint_dir + 'Q_target/DuelingDQN_q_target_{}.pth'.format(episode))




































