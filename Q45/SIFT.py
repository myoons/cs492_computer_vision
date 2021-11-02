import os
import cv2
import numpy as np
from glob import glob


if __name__ == '__main__':
    classes = os.listdir(os.path.abspath('Caltech_101'))
    dataset = dict.fromkeys(classes, {'train_img': [],
                                      'train_desc': [],
                                      'test_img': [],
                                      'test_desc': [],
                                      })
    sift = cv2.SIFT.create()
    a = 0
    for c in classes:
        images = glob(f'Caltech_101/{c}/**.jpg')

        # randomly select 15 images each class without replacement. (For both training & testing)
        target_images = np.random.choice(images, 30, replace=False)
        dataset[c]['train_img'] = [cv2.imread(img) for img in target_images[:15]]
        dataset[c]['test_img'] = [cv2.imread(img) for img in target_images[15:]]

        dataset[c]['train_desc'] = np.concatenate([sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)[1] for img in dataset[c]['train_img']], axis=0)
        dataset[c]['test_desc'] = np.concatenate([sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)[1] for img in dataset[c]['test_img']], axis=0)

        print(np.concatenate([sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)[1] for img in dataset[c]['train_img']], axis=0).shape)