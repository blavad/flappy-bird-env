import os
import math
import numpy as np
import cv2

import tools
import flappyBird_env

class Bird:

    def __init__(self, px, py, img_name = "flappy"):
        self.__x = int(px)
        self.__y = int(py)
        self.__y_init = int(py)
            
        self.__rayon = 15
        
        self.__alive = True
        self.__step_death = None
        self.num_step_death = 100

        self.__img = tools.loadImage(img_name, self.__rayon, self.__rayon)

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, y):
        self.__y = int(y)

    @property
    def rayon(self):
        return self.__rayon
    
    @property
    def alive(self):
        return self.__alive
    
    @property
    def step_death(self):
        return self.__step_death
    
    @step_death.setter
    def step_death(self, step_death):
        self.__step_death = step_death
    
    def kill(self, plat_num):
        self.__alive = False
        if (self.__step_death is None):
            self.__step_death = 1
        
    def backToLife(self):
        self.__alive = True
        self.__step_death = None

    def draw(self, img):
        for idy, y in enumerate(range(img.shape[0]-self.y - self.rayon,img.shape[0]-self.y)):
            for idx, x in enumerate(range(self.x, self.x+self.rayon)):
                if self.__img[idy,idx,3]>250 and tools.are_valide_coord(img, x,y):
                    img[y,x,:] = self.__img[idy,idx,:]   
        return img