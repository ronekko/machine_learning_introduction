# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 13:57:41 2018

@author: sakurai
"""

from pathlib import Path
import pickle

import cv2
import numpy as np
import sklearn.ensemble

if __name__ == '__main__':
    # load data
    positives = []
    for filepath in Path('positive').glob('*'):
        bgr = cv2.imread(str(filepath))
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        positives.append(hsv.reshape(-1, 3))
    positives = np.concatenate(positives, 0)

    negatives = []
    for filepath in Path('negative').glob('*'):
        bgr = cv2.imread(str(filepath))
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        negatives.append(hsv.reshape(-1, 3))
    negatives = np.concatenate(negatives, 0)

    n_positives = len(positives)
    n_negatives = len(negatives)
    n_examples = n_positives + n_negatives
    target = np.zeros(n_examples, int)
    target[:n_positives] = 1
    x = np.concatenate((positives, negatives))

    # train
    model = sklearn.ensemble.GradientBoostingClassifier()
    model.fit(x, target)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # evaluate
    for filepath in Path('evaluate').glob('*'):
        bgr = cv2.imread(str(filepath))
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        x_eval = hsv.reshape(-1, 3)
        y_eval = model.predict(x_eval)
        mask_eval = y_eval.reshape(hsv.shape[:2])
        cv2.imshow('gbr', bgr)
        cv2.imshow('pred', mask_eval.astype('f'))
        cv2.waitKey()
    cv2.destroyAllWindows()
