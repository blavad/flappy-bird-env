import os
import math
import numpy as np
import cv2

import flappyBird_env

class Bird:

    def __init__(self, px, py):
        self.__x = int(px)
        self.__y = int(py)
        self.__rayon = 15
        url_flappy = os.path.join(os.path.dirname(flappyBird_env.__file__), 'assets/flappy_gris.png')
        img_flappy = cv2.imread(url_flappy,1)
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

    def draw(self, img):
        if (self.y > 0 and self.y + self.rayon < img.shape[0]):
            img[img.shape[0]-self.y - self.rayon : img.shape[0]-self.y , self.x:self.x + self.rayon, :] = self.__img
        return img
