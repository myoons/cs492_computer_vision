import os
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier



if __name__ == '__main__':
    classes = os.listdir(os.path.abspath('Caltech_101'))
    dataset = dict.fromkeys(classes, {'train_img': [],
                                      'train_desc': [],
                                      'test_img': [],
                                      'test_desc': [],
                                      })

    random_forest_classifier = RandomForestClassifier(
                                              n_estimators=10,
                                              criterion='gini', 
                                              max_depth=None,
                                              min_samples_split=2,
                                              min_samples_leaf=1, 
                                              min_weight_fraction_leaf=0.0, 
                                              max_features='auto', 
                                              max_leaf_nodes=None, 
                                              bootstrap=True, 
                                              oob_score=False,
                                              n_jobs=1, 
                                              random_state=None,
                                              verbose=0, 
                                              warm_start=False, 
                                              class_weight=None)
      