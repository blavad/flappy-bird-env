from flappyBird_env import FlappyBirdEnv

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import cv2

class FlappyBirdCnnEnv(FlappyBirdEnv):
    
    def __init__(self, width=84, height=84, n_frames=1):
        FlappyBirdEnv.__init__(self)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(height, width, n_frames),
            dtype=np.uint8
        )
        self.w = width
        self.h = height
        self.n_frames = n_frames
                
    @property
    def state(self):
        state = self.renderGame()
        bw_img = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        w,h = bw_img.shape
        bw_img_norm = cv2.resize(bw_img, (self.w,self.h))
        bw_img_norm = np.reshape(bw_img_norm, (self.w,self.h,1))
            
        return (bw_img_norm[:,:,:]/256)*2 - 1.