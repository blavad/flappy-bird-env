import os
import math
import numpy as np
import cv2

import flappyBird_env

class Bird:

    def __init__(self, px, py, img_name = "flappy"):
        self.__x = int(px)
        self.__y = int(py)
        self.__y_init = int(py)
            
        self.__rayon = 15
        
        self.__alive = True
        self.__dead_plat = 100

        self.loadImages(img_name)
        
    def loadImages(self, img_name):
        url_flappy = os.path.join(os.path.dirname(flappyBird_env.__file__), 'assets/{}.png'.format(img_name))
        img_flappy = cv2.imread(url_flappy, cv2.IMREAD_UNCHANGED)
        self.__img = cv2.resize(img_flappy, (self.__rayon, self.__rayon))

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
    def dead_plat(self):
        return self.__dead_plat
    
    def kill(self, plat_num):
        self.__alive = False
        self.__dead_plat = plat_num
        self.y = self.__y_init
        
    def backToLife(self):
        self.__alive = True

    def draw(self, img):
        if (self.y > 0 and self.y + self.rayon < img.shape[0]):
            for idy, y in enumerate(range(img.shape[0]-self.y - self.rayon,img.shape[0]-self.y)):
                for idx, x in enumerate(range(self.x, self.x+self.rayon)):
                    if self.__img[idy,idx,3]>250:
                         img[y,x,:] = self.__img[idy,idx,:] 
        return img