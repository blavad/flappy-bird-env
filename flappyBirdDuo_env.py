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
        self.resurrect_rate = 4

    @property
    def state(self):
        return self.getState(self.birds['ai'])

    @property
    def score(self):
        return self.flappy_env.score

    def reset(self):
        self.flappy_env.reset()
        self.birds['ai'] = self.flappy_env.bird = Bird(
            self.flappy_env.width//4, self.flappy_env.height//2, img_name='flappy2',ghost_rate=0.5)
        self.birds['human'] = Bird(
            self.flappy_env.width//4, self.flappy_env.height//2, 'flappy')

        self.bulle = tools.loadImage(
            "bulle2", self.birds['ai'].rayon+10, self.birds['ai'].rayon + 10)
        return np.array(self.state)

    def step(self, action):
        state, rew, ai_done, _ = self.flappy_env.step(action['ai'], duo=True)
        self.birds['human'].y -= self.flappy_env.massbird if self.birds['human'].y > 0 else 0
        self.birds['human'].y += self.flappy_env.powerbird if FlappyBirdEnv.actions[action['human']] == "fly" and self.birds['human'].y+self.birds['human'].rayon < self.flappy_env.height else 0
        if self.birds['human'].alive:
            if self.flappy_env.checkCollision(self.birds['human']):
                self.birds['human'].kill(self.flappy_env.score)

        self.birds['human'].update_death_state()
        self.birds['ai'].update_death_state()

        done = not self.birds['human'].alive and not self.birds['ai'].alive
        done = bool(done)

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
        return np.concatenate((img, self.flappy_env.renderInfos(high_score=FlappyBirdDuoEnv.high_score, color_background=color-50)), axis=0)

    def drawBulle(self, img, bird):
        max_rate = 0.8
        rate = max_rate - max_rate*(bird.step_death / bird.num_step_death)
        for idy, y in enumerate(range(img.shape[0]-bird.y - bird.rayon-5, img.shape[0]-bird.y+5)):
            for idx, x in enumerate(range(bird.x-5, bird.x+bird.rayon+5)):
                if self.bulle[idy, idx, 3] > 250 and tools.are_valide_coord(img, x, y):
                    img[y, x, :] = rate*self.bulle[idy,idx, :] + (1-rate)*img[y, x, :]
        return img

    def getState(self, bird):
        c_p = self.flappy_env._get_current_plateform()
        dx = c_p.x - (bird.x + bird.rayon)
        dy = bird.y - c_p.get_pos_ouv()
        sp_x = self.flappy_env.speedbird
        sp_y = self.flappy_env.massbird
        return (dx, dy, sp_x, sp_y)
