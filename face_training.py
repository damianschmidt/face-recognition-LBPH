import cv2
import numpy as np
import pickle
from os import path, walk, makedirs


def labels_for_training_data():
    current_id = 0
    label_ids = dict()
    faces, faces_ids = list(), list()

    for root, dirs, files in walk('data/'):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = path.join(root, file)
                label = path.basename(root).replace(' ', '-').lower()
                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]

                test_img = cv2.imread(img_path)
                test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
                if test_img is None:
                    print('Image not loaded properly')
                    continue

                faces.append(test_img)
                faces_ids.append(id_)

    if not path.exists('labels/face-labels.pickle'):
        makedirs('labels/')
    with open('labels/face-labels.pickle', 'wb') as file:
        pickle.dump(label_ids, file)

    return faces, faces_ids


def train_classifier(faces, faces_ids):
    recognizer_lbph = cv2.face.LBPHFaceRecognizer_create()
    recognizer_lbph.train(faces, np.array(faces_ids))
    recognizer_lbph.save('trainner.yml')
    print('Model training complete!')


faces, faces_ids = labels_for_training_data()
train_classifier(faces, faces_ids)
