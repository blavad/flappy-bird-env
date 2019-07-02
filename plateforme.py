import math
import numpy as np
import cv2
import os

import flappyBird_env

class Plateforme:

    def __init__(self, x, y_max, taille_o):
        self.__x = int(x)
        self.__epaisseur = 20
        self.y_ouv = np.random.randint(10, high=y_max-taille_o-10)
        self.size_ouv = taille_o
        url_ph = os.path.join(os.path.dirname(flappyBird_env.__file__), 'assets/plateform_haut.png')
        url_pb = os.path.join(os.path.dirname(flappyBird_env.__file__), 'assets/plateform_bas.png') 
        
        plat_haut = cv2.imread(url_ph, cv2.IMREAD_UNCHANGED)
        plat_bas = cv2.imread(url_pb, cv2.IMREAD_UNCHANGED)
        self.plat_haut = cv2.resize(plat_haut, (self.__epaisseur,y_max - self.y_ouv-self.size_ouv))
        self.plat_bas = cv2.resize(plat_bas, (self.__epaisseur, self.y_ouv))
        
    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, x):
        self.__x = int(x)

    @property
    def epaisseur(self):
        return self.__epaisseur

    def get_size_ouv(self):
        return self.size_ouv

    def get_pos_ouv(self):
        return self.y_ouv

    def updatePos(self, sp):
        self.x = self.x - sp

    def draw(self, img):
        if (self.x+self.epaisseur<img.shape[1] and self.x > 0):
            img[0:img.shape[0]-self.get_pos_ouv()-self.get_size_ouv(),self.x:self.x + self.__epaisseur,:] = self.plat_haut
            img[img.shape[0]-self.get_pos_ouv():, self.x:self.x+self.__epaisseur, :] = self.plat_bas
        return img
