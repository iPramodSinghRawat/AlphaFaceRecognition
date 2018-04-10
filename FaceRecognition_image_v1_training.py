'''
Face_Recognition_image_v1_training.py
By: Pramod Singh Rawat
'''

import cv2
import numpy as np

from AlphaFaceRecognitionFunc import *
from FaceRecDataSet import *

face_rec_training_data_file = 'face_recognizer_training_data/face_rec_training_data1.yml'

print ("Select From Below To train Your System:")

i=1
while i < len(subjects):
    print("Enter "+str(i)+" to train for: "+subjects[i])
    i += 1

try:
    #train_for_label = int(input('input label(int) for training and Press Enter : '))
    train_for_label = int(input(' Provide input label(int) for training and Press Enter : '))
    print("Entered: "+str(train_for_label))
except ValueError:
    print("This is not a whole number.")
    exit()

if train_for_label >= len(subjects) or train_for_label < 1:
    print("Invalid Label Entered")
else:
    print ("Training for : "+subjects[train_for_label])
    print ("Training Data from Directory  : "+subjects_dir[train_for_label])

    faces_main = [] #list to hold all subject faces
    labels_main = [] #list to hold labels for all subjects(Labels)

    print("Preparing data...")

    #Training Method 2
    faces, labels = prepare_training_data_type2(subjects_dir[train_for_label],train_for_label)
    faces_main = faces_main + faces
    labels_main =  labels_main + labels

    print("Data prepared")
    #Total Faces & Labels
    print("Total faces: ", len(faces_main))
    print("Total labels: ", len(labels_main))

    #LBPH FaceRecognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    if os.path.isfile(face_rec_training_data_file):
        print('file exist')
        face_recognizer.read(face_rec_training_data_file)
        face_recognizer.update(faces_main, np.array(labels_main)) #updates face_recognizer with new data
        print('Updated Face Recognizer Data')
    else:
        face_recognizer.train(faces_main, np.array(labels_main))

    #face_recognizer.train(faces_main, np.array(labels_main))
    face_recognizer.write(face_rec_training_data_file)
    print("Training Done ...")
