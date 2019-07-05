from flappyBird_env import FlappyBirdEnv

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class FlappyBirdCnnEnv(FlappyBirdEnv):
    
    def __init__(self):
        FlappyBirdEnv.__init__(self)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.height, self.width, 3),
            dtype=np.uint8
        )
        
    @property
    def state(self):
        state = self.renderGame()
        bw_img = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        return (bw_img[:,:,0]/256)*2 - 1.