import gym


class Active_3d(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(shape=(5,256,259), low=-100, high=100)
        self.train_home_list = []

    def reset(self):
        pass

    def step(self, action):
        pass
