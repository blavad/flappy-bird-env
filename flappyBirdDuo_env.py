"""
Classic flappy bird game.
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

from bird import Bird
from plateforme import Plateforme
from flappyBird_env import FlappyBirdEnv


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

    def __init__(self):

        self.flappy_env = FlappyBirdEnv()

        self.birds = {
            "human": None,
            'ai': self.flappy_env.bird
        }

        self.resurrect_rate = 4

    @property
    def state(self):
        return self.getState(self.birds['ai'])

    def reset(self):
        self.flappy_env.reset()
        self.birds['ai'] = self.flappy_env.bird
        self.birds['human'] = Bird(
            self.flappy_env.width//4, self.flappy_env.height//2, 'flappy2')
        return np.array(self.state)

    def step(self, action):

        state, rew, ai_done, _ = self.flappy_env.step(action['ai'])
        if self.birds['human'].alive:
            self.birds['human'].y += self.flappy_env.powerbird if FlappyBirdEnv.actions[action['human']] == "fly" else 0
            self.birds['human'].y -= self.flappy_env.massbird
            if self.flappy_env.checkCollision(self.birds['human']):
                self.birds['human'].kill(self.flappy_env.score)
        else:
            if self.birds['human'].dead_plat + self.resurrect_rate <= self.flappy_env.score :
                self.birds['human'].backToLife()

        if self.birds['ai'].dead_plat+self.resurrect_rate <= self.flappy_env.score:
            self.birds['ai'].backToLife()

        done = not self.birds['human'].alive and not self.birds['ai'].alive
        done = bool(done)

        # Calcul du score
        if not done:
            reward = 1.0
        elif self.flappy_env.steps_beyond_done is None:
            self.flappy_env.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.flappy_env.steps_beyond_done == 0:
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
        color = np.array([200,200,200])
        img = self.flappy_env.renderGame(color_background=color)
        img = self.birds['human'].draw(img)
        return np.concatenate((img, self.flappy_env.renderInfos(color-50)), axis=0)

    def getState(self, bird):
        c_p = self.flappy_env._get_current_plateform()
        dx = c_p.x - (bird.x + bird.rayon)
        dy = bird.y - c_p.get_pos_ouv()
        sp_x = self.flappy_env.speedbird
        sp_y = self.flappy_env.massbird
        return (dx, dy, sp_x, sp_y)
