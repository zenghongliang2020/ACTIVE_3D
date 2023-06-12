import gym


class Active_3d(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Discrete(3)
        self.train_home_list = []

    def reset(self):
        pass

    def step(self, action):
        pass
