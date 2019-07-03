import flappyBird_env

import os
import cv2


def loadImage(img_name, w, h):
    url = os.path.join(os.path.dirname(flappyBird_env.__file__),
                       'assets/{}.png'.format(img_name))
    img_cv2 = cv2.imread(url, cv2.IMREAD_UNCHANGED)
    return cv2.resize(img_cv2, (w, h))


def are_valide_coord(arr, x, y):
    return (x > 0 and x < arr.shape[1] and y > 0 and y < arr.shape[0])
