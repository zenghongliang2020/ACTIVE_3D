import gym
import numpy as np
from model.DuelingDQN import DuelingDQN
from Utils.arguments import get_args
from Utils.utils import create_directory
from ENV.Active_env.Active_3d import Active_3d


args = get_args()
env = gym.make('Active_3d-v0')
agent = DuelingDQN(alpha=0.0003, state_dim=256 * 259, num_hiddens=1024,
                   num_sw=5, trans_norm_shape=[5, 1024], ffn_num_input=1024,
                   ffn_num_hiddens=2048, trans_num_heads=4, trans_num_layers=2,
                   trans_dropout=0.1, action_coor_dim=9, action_angle_dim=8, ckpt_dir=args.ckpt_dir)

def main():
    create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
    total_rewards, avg_rewards, eps_history = [], [], []

    for episode in range(args.max_episodes):
        total_reward = 0
        done = False
        state = env.reset()
        while not done:
            action_coor, action_angle = agent.choose_action(state, isTrain=True)
            action = [action_coor, action_angle]
            state_, reward, done, info = env.step(action)
            agent.remember(state, action, reward, state_, done)
            agent.learn()
            total_reward += reward
            state = state_
        total_rewards.append(total_reward)
        avg_reward = np.mean(total_reward)
        avg_rewards.append(avg_reward)
        eps_history.append(agent.epsilon)
        print('EP:{} reward:{} avg_reward:{} epsilon:{}'.format(episode+1, total_reward, avg_reward, agent.epsilon))

        if (episode + 1) % 5 == 0:
            agent.save_models(episode + 1)

    episodes = [i for i in range(args.max_episodes)]


if __name__ == '__main__':
    main()


