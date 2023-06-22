import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model.ReplayBuffer import Replaybuffer
from model.transformer_block import TransformerEncoder

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

class Dueling_net(nn.Module):
    def __init__(self, alpha, state_dim, num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                 dropout, num_sw, action_coor_dim, action_angle_dim):
        super(Dueling_net, self).__init__()
        self.dense1 = nn.Linear(state_dim, num_hiddens)
        self.transform_layer = TransformerEncoder(key_size=num_hiddens, query_size=num_hiddens,
                                                  value_size=num_hiddens, num_hiddens=num_hiddens,
                                                  norm_shape=norm_shape, ffn_num_input=ffn_num_input,
                                                  ffn_num_hiddens=ffn_num_hiddens, num_heads=num_heads,
                                                  num_layers=num_layers, dropout=dropout)
        self.dense2 = nn.Linear(num_sw * num_hiddens, num_hiddens)
        self.dense3 = nn.Linear(num_hiddens, 256)
        self.V = nn.Linear(256, 1)
        self.A_coor = nn.Linear(256, action_coor_dim)
        self.A_angle = nn.Linear(256, action_angle_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state, action_mask):
        x = T.relu(self.dense1(state))
        x_trans = self.transform_layer(x)
        x = T.flatten(x_trans, start_dim=1)
        x = T.relu(self.dense2(x))
        x = T.relu(self.dense3(x))

        V = self.V(x)
        A_coor = T.softmax(self.A_coor(x), dim=-1) * action_mask
        A_angle = T.softmax(self.A_angle(x), dim=-1)

        return V, A_coor, A_angle

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class DuelingDQN:
    def __init__(self, alpha, state_dim, num_hiddens, num_sw, trans_norm_shape, ffn_num_input,
                 ffn_num_hiddens, trans_num_heads, trans_num_layers, trans_dropout,
                 action_coor_dim, action_angle_dim, ckpt_dir, gamma=0.99, tau=0.005,
                 epsilon=1.0, eps_end=0.01, eps_dec=5e-4, max_size=10000, batch_size=256):
        self.gamma = gamma
        self.tau = tau
        self.num_sw = num_sw
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.checkpoint_dir = ckpt_dir
        self.q_eval = Dueling_net(alpha, state_dim, num_hiddens, trans_norm_shape,
                                  ffn_num_input, ffn_num_hiddens, trans_num_heads, trans_num_layers,
                                  trans_dropout, num_sw, action_coor_dim, action_angle_dim)
        self.q_target = Dueling_net(alpha, state_dim, num_hiddens, trans_norm_shape,
                                    ffn_num_input, ffn_num_hiddens, trans_num_heads, trans_num_layers,
                                    trans_dropout, num_sw, action_coor_dim, action_angle_dim)

        self.action_angle_space = [i for i in range(action_angle_dim)]

        self.memory = Replaybuffer(state_dim, self.num_sw, 2, max_size, batch_size)

        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)


    def choose_action(self, state, isTrain=True):
        obs = T.tensor([state['observation']], dtype=T.float32).to(device)
        action_mask = T.tensor([state['mask']], dtype=T.float32).to(device)
        _, A_coor, A_angle = self.q_eval.forward(obs, action_mask)
        action_coor = T.argmax(A_coor).item()
        action_angle = T.argmax(A_angle).item()

        if (np.random.random() < self.epsilon) and isTrain:
            action_coor_mask = state['mask']
            action_coor_space = []
            for i in range(len(action_coor_mask)):
                if action_coor_mask[i] > 0:
                    action_coor_space.append(i)
            action_coor = np.random.choice(action_coor_space)
            action_angle = np.random.choice(self.action_angle_space)

        return action_coor, action_angle


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
        states_tensor = T.tensor(states['observation'], dtype=T.float).to(device)
        mask_tensor = T.tensor(states['mask'], dtype=T.float).to(device)
        actions_tensor = T.tensor(actions, dtype=T.long).to(device)
        rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
        next_states_tensor = T.tensor(next_states['observation'], dtype=T.float).to(device)
        next_mask_tensor = T.tensor(next_states['mask'], dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)

        with T.no_grad():
            V_, A_coor_, A_angle_ = self.q_target.forward(next_states_tensor, next_mask_tensor)
            q_ = self.Q_cal(V_, A_coor_, A_angle_)
            q_[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * T.max(q_, dim=-1)[0]
        V, A_coor, A_angle = self.q_eval.forward(states_tensor, mask_tensor)
        q = self.Q_cal(V, A_coor, A_angle)
        actions_idx = T.empty(actions_tensor.shape[0])
        for i in range(actions_tensor.shape[0]):
            actions_idx[i] = actions_tensor[i][1] * 9 + actions_tensor[i][0]
        actions_idx_tensor = T.tensor(actions_idx, dtype=T.long).to(device)
        q = q[batch_idx, actions_idx_tensor]
        loss = F.mse_loss(q, target.detach())
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        self.update_network_parameters()
        self.decrement_epsilon()

    def Q_cal(self, V, A_coor, A_angle):
        A_coor_m = T.unsqueeze(A_coor - T.mean(A_coor, dim=-1, keepdim=True), dim=1)
        A_angle_m = T.unsqueeze(A_angle - T.mean(A_angle, dim=-1, keepdim=True), dim=1)
        A_ = A_coor_m * A_angle_m.permute(0, 2, 1)
        A_ = T.flatten(A_, start_dim=1)
        return V + A_

    def save_models(self, episode):
        self.q_eval.save_checkpoint(self.checkpoint_dir + 'Q_eval/DuelingDQN_q_eval_{}.pth'.format(episode))
        print('Saving Q_eval network successfully!')

        self.q_target.save_checkpoint(self.checkpoint_dir + 'Q_target/DuelingDQN_q_target_{}.pth'.format(episode))
        print('Saving Q_target network successfully!')

    def load_models(self, episode):
        self.q_eval.load_checkpoint(self.checkpoint_dir + 'Q_eval/DuelingDQN_q_eval_{}.pth'.format(episode))
        self.q_target.load_checkpoint(self.checkpoint_dir + 'Q_target/DuelingDQN_q_target_{}.pth'.format(episode))




































