import gym
import os
import random
import torch

class Active_3d(gym.Env):
    def __init__(self):
        self.data_root_path = ''
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(shape=(5,256,259), low=-100, high=100)
        self.train_Home_list = []
        self.angle_list = []
        self.done = 0
        self.step_num = 0

    def reset(self):
        Home = random.choice(self.train_Home_list)
        self.data_path = self.data_root_path + Home + '/'
        self.coors_list = os.listdir(self.data_path)
        self.coor_now = random.choice(self.coors_list)
        self.angle_now = random.choice(self.angle_list)



    def step(self, action):
        coor_int = [int(i) for i in self.coor_now.split(' ')]

        coor_space = self.coor_space()
        action_space = torch.FloatTensor(coor_space)

        reward = self.reward_cal()

        self.step_num += 1

        if self.step_num == 100:
            self.done = True

        return {'mask': action_space}, reward, self.done, {}

    def coor_space(self):
        coor_int = [int(i) for i in self.coor_now.split(' ')]
        coor_f = ' '.join([str(i) for i in [coor_int[0], coor_int[1] + 1]])
        coor_b = ' '.join([str(i) for i in [coor_int[0], coor_int[1] - 1]])
        coor_l = ' '.join([str(i) for i in [coor_int[0] - 1, coor_int[1]]])
        coor_r = ' '.join([str(i) for i in [coor_int[0] + 1, coor_int[1]]])
        coor_fl = ' '.join([str(i) for i in [coor_int[0] - 1, coor_int[1] + 1]])
        coor_fr = ' '.join([str(i) for i in [coor_int[0] + 1, coor_int[1] + 1]])
        coor_bl = ' '.join([str(i) for i in [coor_int[0] - 1, coor_int[1] - 1]])
        coor_br = ' '.join([str(i) for i in [coor_int[0] + 1, coor_int[1] - 1]])

        coor_space = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        if coor_f in self.coors_list:
            coor_space[1] = 1
        if coor_b in self.coors_list:
            coor_space[2] = 1
        if coor_l in self.coors_list:
            coor_space[3] = 1
        if coor_r in self.coors_list:
            coor_space[4] = 1
        if coor_fl in self.coors_list:
            coor_space[5] = 1
        if coor_fr in self.coors_list:
            coor_space[6] = 1
        if coor_bl in self.coors_list:
            coor_space[7] = 1
        if coor_br in self.coors_list:
            coor_space[8] = 1

        return coor_space


    def reward_cal(self):
        reward = 0
        return reward