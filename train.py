import gym
import numpy as np
from model.DuelingDQN import DuelingDQN
from Utils.arguments import get_args
from Utils.utils import create_directory

env = gym.make('Active_3d-v0')
agent = DuelingDQN()
args = get_args()


def main():
    create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
    total_rewards, avg_rewards, eps_history = [], [], []

    for episode in range(args.max_episodes):
        total_reward = 0
        done = False
        observation = env.resrt()
        while not done:
            action_1, action_2 = agent.choose_action(observation, isTrain=True)
            observation_, reward, done, info = env.step(action_1, action_2)
            agent.remember(observation, action_1, action_2, reward, observation_, done)
            agent.learn()
            total_reward += reward
            observation = observation_

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_reward[-100:])
        avg_rewards.append(avg_reward)
        eps_history.append(agent.epsilon)
        print('EP:{} reward:{} avg_reward:{} epsilon:{}'.format(episode+1, total_reward, avg_reward, agent.epsilon))

        if (episode + 1) % 50 == 0:
            agent.save_models(episode + 1)

    episodes = [i for i in range(args.max_episodes)]


if __name__ == '__main__':
    main()


