import gym
import os
import random

import numpy as np
import torch
import open3d as o3d


class Active_3d(gym.Env):
    def __init__(self):
        self.data_root_path = ''
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(shape=(5, 256, 259), low=-100, high=100)
        self.train_Home_list = ['Home_3', 'Home_4', 'Home_5', 'Home_6', 'Home_7']
        self.angle_list = ['0', '45', '90', '135', '180', '225', '270', '315']

        self.votenet_pre = torch.load('')
        self.votenet_train = torch.load('')
        self.meta = []

        self.done = 0
        self.step_num = 0

    def reset(self):
        Home = random.choice(self.train_Home_list)
        self.data_path = self.data_root_path + Home + '/'
        self.coors_list = os.listdir(self.data_path)
        self.coor_now = random.choice(self.coors_list)
        self.angle_now = random.choice(self.angle_list)
        self.obs_buffer = []
        obs_empty = np.zeros((1, 256, 259))
        for i in range(4):
            self.obs_buffer.append(obs_empty)

    def step(self, action):
        coor_int = [int(i) for i in self.coor_now.split(' ')]
        action_coor = action['coor']
        action_angle = action['angle']

        self.coor_now = self.coor_now_cal(action_coor, coor_int)
        self.angle_now = self.angle_list[action_angle]

        cloud_path, _, _, _ = self.path()
        assert os.path.exists(cloud_path)
        reward, obs = self.reward_cal(cloud_path)

        self.obs_buffer.append(obs.detach().cpu().squeeze())

        coor_space = self.coor_space()
        action_space = torch.FloatTensor(coor_space)


        self.step_num += 1

        if self.step_num == 100:
            self.done = True

        return {'observation': self.obs_buffer[self.step_num - 1: self.step_num + 4],
                'mask': action_space}, reward, self.done, {}

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

    def coor_now_cal(self, action_coor, coor_int):
        coor_f = ' '.join([str(i) for i in [coor_int[0], coor_int[1] + 1]])
        coor_b = ' '.join([str(i) for i in [coor_int[0], coor_int[1] - 1]])
        coor_l = ' '.join([str(i) for i in [coor_int[0] - 1, coor_int[1]]])
        coor_r = ' '.join([str(i) for i in [coor_int[0] + 1, coor_int[1]]])
        coor_fl = ' '.join([str(i) for i in [coor_int[0] - 1, coor_int[1] + 1]])
        coor_fr = ' '.join([str(i) for i in [coor_int[0] + 1, coor_int[1] + 1]])
        coor_bl = ' '.join([str(i) for i in [coor_int[0] - 1, coor_int[1] - 1]])
        coor_br = ' '.join([str(i) for i in [coor_int[0] + 1, coor_int[1] - 1]])
        while True:
            if action_coor == 0:
                return coor_int
            elif action_coor == 1:
                return coor_f
            elif action_coor == 2:
                return coor_b
            elif action_coor == 3:
                return coor_l
            elif action_coor == 4:
                return coor_r
            elif action_coor == 5:
                return coor_fl
            elif action_coor == 6:
                return coor_fr
            elif action_coor == 7:
                return coor_bl
            elif action_coor == 8:
                return coor_br
            else:
                print('action wrong!!!!')

    def path(self):
        cloud_path = ''.join([self.data_path, self.coor_now, '/', 'cloud ', self.angle_now, '.pcd'])
        color_path = ''.join([self.data_path, self.coor_now, '/', 'color ', self.angle_now, '.png'])
        depth_path = ''.join([self.data_path, self.coor_now, '/', 'depth ', self.angle_now, '.png'])
        anno_path = ''.join([self.data_path, self.coor_now, '/', 'anno ', self.angle_now, '.txt'])
        return cloud_path, color_path, depth_path, anno_path

    def read_pc(self, path):
        pcd = o3d.io.read_point_cloud(path)
        xyz = pcd.points
        # o3d format(yz-x) to depth format
        xyz = np.asarray(xyz)[:, [0, 2, 1]]
        xyz[:, 1] = -xyz[:, 1]
        xyz[:, 2] = xyz[:, 2] + 1.1

        RGB = pcd.colors
        RGB = np.asarray(RGB)

        pc = np.concatenate((xyz, RGB), axis=1)

        if pc.shape[0]<20000:
            Replace = True
        else:
            Replace=False
        choices = np.random.choice(pc.shape[0], 20000, replace=Replace)
        pc = pc[choices]

        floor_height = np.percentile(pc[:, 2], 0.99)
        height = pc[:, 2] - floor_height
        pc = np.concatenate([pc, np.expand_dims(height, 1)], 1)

        return torch.from_numpy(pc).float().cuda()

    def reward_cal(self, path):
        reward = 0
        points = []
        points.append(self.read_pc(path))
        self.bbox_results = self.votenet_train.simple_test(points, self.meta, None, True)
        obsret = self.votenet_train.backbone.fp_ret
        obs = torch.cat((obsret['fp_xyz'][0], obsret['fp_features'][0].permute(0, 2, 1)), axis=2)
        return reward, obs

