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


class FlappyBirdEnv(gym.Env):
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
        'video.frames_per_second' : 50
    }

    actions = [
        'none',
        'fly'
    ]

    def __init__(self):
        
        # Dimensions de l ecran d affichage
        self.width = 200
        self.height = 200
        
        self.score = 0
        self.score_max = 100
        
        # Parametres lies aux plateformes
        self.plateformes = None
        self.nb_plateform = 2
        self.dist_plat = self.width/self.nb_plateform
        self.size_ouv = 60
        
        # Parametres lies à l oiseau
        self.bird = None
        self.massbird = 5
        self.speedbird = 4.0
        self.powerbird = 10
        
        # Autres parametres d etats
        self.new_plat = False
        self.current_plat = None
        self.steps_beyond_done = None

        high = np.array([
            np.finfo(np.float32).max, 
            np.finfo(np.float32).max, 
            np.finfo(np.float32).max, 
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype = np.float32)

        self.seed()
        self.viewer = None

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    @property
    def state(self):
        # Mets a jour l etat
        c_p = self._get_current_plateform()
        dx = c_p.x - (self.bird.x + self.bird.rayon)
        dy = self.bird.y - c_p.get_pos_ouv()
        sp_x = self.speedbird
        sp_y = self.massbird
        return (dx, dy, sp_x, sp_y)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        action = FlappyBirdEnv.actions[action]
        # Applique l action sur flappy bird
        self.bird.y += self.powerbird if action == 'fly' else 0
        self.bird.y -= self.massbird
        
        # Si plateforme hors plateau, on la supprime et on en ajoute une autre a la suite
        if (self.plateformes[0].x + self.plateformes[0].epaisseur < 0):
            self.new_plat = True
            new_p = Plateforme(self.plateformes[-1].x + self.dist_plat, self.height, self.size_ouv)
            self.plateformes = np.delete(self.plateformes,0)
            self.plateformes = np.append(self.plateformes, new_p) 
            
        # Decale les plateformes a la vitesse "speedbird"
        for p in self.plateformes :
            p.updatePos(self.speedbird)
        
        # Verifie que flappy bird n est pas mort
        c_p = self._get_current_plateform()
        done =  self.bird.y <= 0\
                or self.bird.y+self.bird.rayon > self.height \
                or ((self.bird.x+self.bird.rayon > c_p.x) and ((self.bird.y < c_p.get_pos_ouv()) or (self.bird.y+self.bird.rayon > c_p.get_pos_ouv() + c_p.get_size_ouv())))\
                or (self.score > self.score_max)
                
        done = bool(done)
        
        # Calcul du score 
        if not done:
            if (self.current_plat != c_p):
                self.score += 1
                self.current_plat = c_p
            reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.score = 0
        self.bird = Bird(self.width//3, self.height//2)
        self.plateformes = [Plateforme(self.width + delta*self.dist_plat, self.height, self.size_ouv) for delta in range(self.nb_plateform)]
        self.current_plat = self._get_current_plateform()
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        if mode == 'human':
            return self.render_human(mode)

        return self.render_array()

    def render_human(self,mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer.imshow(self.render(mode='rbg_array'))

    def render_array(self):
        img = np.full(
            ( self.height,self.width, 3),
            255,
            dtype=np.uint8,
        )
        img = self.bird.draw(img)
        for p in self.plateformes :
            p.draw(img)
        return img


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        
        
    def _get_current_plateform(self):
        for p in self.plateformes :
            if (p.x+p.epaisseur > self.bird.x ):
                return p