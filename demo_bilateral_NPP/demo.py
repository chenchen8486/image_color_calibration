import os

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    image = cv2.imread(os.path.join(BASE_DIR, 'input', 'test.bmp'))
    result = cv2.bilateralFilter(image, 9, 25, 25)
    cv2.imwrite(os.path.join(BASE_DIR, 'output', 'resulg.bmp'), result)
