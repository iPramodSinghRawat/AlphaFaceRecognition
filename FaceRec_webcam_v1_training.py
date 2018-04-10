'''
FaceRec_webcam_v1_training.py
By: Pramod Singh Rawat
'''

import cv2
import sys

import numpy as np

from AlphaFaceRecognitionFunc import *
from FaceRecDataSet import *

'''
Notes: Need Proper lighting on face for better detects
'''

face_rec_training_data_file = 'face_recognizer_training_data/face_rec_training_data_vid1.yml'

faces_main = [] #list to hold all subject faces
labels_main = [] #list to hold labels for all subjects(Labels)

face_recognizer = cv2.face.LBPHFaceRecognizer_create() #create our LBPH face recognizer

if os.path.isfile(face_rec_training_data_file):
    face_recognizer.read(face_rec_training_data_file)#Loading face recognizer with Old Data for Label 1

def start_training(for_label):

    video_capture = cv2.VideoCapture(0)

    while True:

        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            pass

        # Capture frame-by-frame
        ret, frame = video_capture.read()
        #Detecting Face From Frame
        face, rect = detect_face(frame)

        if face is not None:
            faces_main.append(face)
            labels_main.append(for_label)

            draw_rectangle(frame, rect)
            draw_text(frame, "Training for: "+subjects[for_label], rect[0], rect[1]-5)

        if (len(labels_main)>0):
            #face_recognizer.train(faces_main, np.array(labels_main))
            if os.path.isfile(face_rec_training_data_file):
                face_recognizer.update(faces_main, np.array(labels_main)) #updates face_recognizer with new data
            else:
                face_recognizer.train(faces_main, np.array(labels_main))

        face_recognizer.write(face_rec_training_data_file)

        #Display the result frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #Release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    print("Training End")

#start_training(1)

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
    start_training(train_for_label)
