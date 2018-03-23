# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 16:02:11 2018

@author: sakurai
"""

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('model.pkl', 'rb') as f:
        model = pickle.loads(f.read())

    cap = cv2.VideoCapture(0)
    while cv2.waitKey(1) != ord('q'):  # Press 'q' key to stop capture.
        ret, image = cap.read()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        x_eval = hsv.reshape(-1, 3)
        y_eval = model.predict(x_eval)
        mask_eval = y_eval.reshape(hsv.shape[:2])
        cv2.imshow('pred', mask_eval.astype('f'))
        cv2.imshow('camera', image)

    cap.release()
    cv2.destroyAllWindows()
