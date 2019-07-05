"""
Classic flappy bird game.
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import os
import cv2

from bird import Bird
from plateforme import Plateforme
from flappyBird_env import FlappyBirdEnv
import flappyBird_env
import tools


class FlappyBirdDuoEnv(gym.Env):
    """
    Description:
        A bird is flying 
    Observation: 
        Type: Box(4)
        Num	Observation                                 Min         Max
        0	Bird X Distance to next platform            -Inf        Inf
        1	Bird Y Distance to next platform            -Inf        Inf
        2	Bird X Velocity                             -Inf        Inf
        3	Bird Y Velocity                             -Inf        Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Do nothing
        1	Fly

    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    actions = [
        'none',
        'fly'
    ]

    high_score = 0

    def __init__(self):

        self.flappy_env = FlappyBirdEnv()

        self.birds = {
            "human": None,
            'ai': self.flappy_env.bird
        }
        self.steps_beyond_done = None

    @property
    def state(self):
        return self.flappy_env.state

    @property
    def score(self):
        return self.flappy_env.score
    
    @property
    def score_max(self):
        return self.flappy_env.score_max
    
    def is_dead(self, player):
        return not self.birds[player].alive

    def reset(self):
        self.flappy_env.reset()
        self.birds['ai'] = self.flappy_env.bird = Bird(
            self.flappy_env.width//4, self.flappy_env.height//2, img_name='flappy', ghost_rate=0.)
        self.birds['human'] = Bird(
            self.flappy_env.width//4 - 1.5*self.birds['ai'].rayon, self.flappy_env.height//2, 'flappy2')

        self.bulle = tools.loadImage("bulle2", self.birds['ai'].rayon+10, self.birds['ai'].rayon + 10)
        return np.array(self.state)

    def step(self, action):
        state, rew, ai_done, _ = self.flappy_env.step(action['ai'], duo=True)
        if self.birds['human'].alive:
            self.birds['human'].y -= self.flappy_env.massbird if self.birds['human'].y > 0 else 0
            self.birds['human'].y += self.flappy_env.powerbird if FlappyBirdEnv.actions[action['human']
                                                                                        ] == "fly" and self.birds['human'].y+self.birds['human'].rayon < self.flappy_env.height else 0
            if self.flappy_env.checkCollision(self.birds['human']):
                self.birds['human'].kill(self.flappy_env.score)
        else:
            self.birds['human'].update_death_state()
            if self.birds['human'].end_death_time() and not self.flappy_env.checkCollision(self.birds['human']):
                self.birds['human'].backToLife()

        if not self.birds['ai'].alive:
            self.birds['ai'].update_death_state()
            if self.birds['ai'].end_death_time() and not self.flappy_env.checkCollision(self.birds['ai']):
                self.birds['ai'].backToLife()

        done = (not self.birds['human'].alive and not self.birds['ai'].alive) or self.birds['human'].num_life <= 0 or self.birds['ai'].num_life <= 0 or self.score > self.score_max

        # Calcul du score
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
            if self.flappy_env.score > FlappyBirdDuoEnv.high_score:
                FlappyBirdDuoEnv.high_score = self.score
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.flappy_env.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward + rew, done, {}

    def render(self, mode='human'):
        if mode == 'human':
            return self.render_human(mode)
        return self.render_array()

    def render_human(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer.imshow(self.render(mode='rbg_array'))

    def render_array(self):
        color = np.array([250, 240, 170])
        img = self.flappy_env.renderGame(color_background=color)
        img = self.birds['human'].draw(img)
        if not self.birds['human'].alive:
            img = self.drawBulle(img, self.birds['human'])
        if not self.birds['ai'].alive:
            img = self.drawBulle(img, self.birds['ai'])
        img = self.displayLives(img)
        return np.concatenate((img, self.flappy_env.renderInfos(high_score=FlappyBirdDuoEnv.high_score, color_background=color-50)), axis=0)

    def drawBulle(self, img, bird):
        max_rate = 0.8*(1-bird.ghost_rate)
        rate = max_rate - max_rate*(bird.step_death / bird.num_step_death)
        rate = max(rate, 0)
        for idy, y in enumerate(range(img.shape[0]-bird.y - bird.rayon-5, img.shape[0]-bird.y+5)):
            for idx, x in enumerate(range(bird.x-5, bird.x+bird.rayon+5)):
                if self.bulle[idy, idx, 3] > 250 and tools.are_valide_coord(img, x, y):
                    img[y, x, :] = rate*self.bulle[idy,
                                                   idx, :] + (1-rate)*img[y, x, :]
        return img
    
    def displayLives(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0,0,0)
        tools.putImg(img, self.birds['ai'].img, 5, self.flappy_env.height-20, 13,12)
        tools.putImg(img, self.birds['human'].img, 5, self.flappy_env.height-40, 13,12)
        
        cv2.putText(img, " {}".format(self.birds["ai"].num_life), (20,self.flappy_env.height-10), font, 0.3, color, 1, cv2.LINE_AA)
        cv2.putText(img, " {}".format(self.birds["human"].num_life), (20,self.flappy_env.height-30), font, 0.3, color, 1, cv2.LINE_AA)
        return img
