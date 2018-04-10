'''
FaceRec_webcam_v1_recog.py
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
#face_rec_training_data_file = 'face_recognizer_training_data/face_rec_training_data1.yml'

face_recognizer = cv2.face.LBPHFaceRecognizer_create()#LBPH face recognizer
face_recognizer.read(face_rec_training_data_file)
#face_recognizer.read('face_recognizer_training_data/face_rec_training_data1.yml')

def start_recogniz():

    video_capture = cv2.VideoCapture(0)

    while True:

        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            pass

        # Capture frame-by-frame
        ret, frame = video_capture.read()

        #gray, faces = detect_face(frame)

        faces = detect_multiple_o_face(frame)

        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.COLOR_BGR2HSV
        )
        '''

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            rect = (x, y, w, h)

            detc_face = frame[y:y+h, x:x+w]

            label, confidence = face_recognizer.predict(cv2.cvtColor(detc_face, cv2.COLOR_BGR2GRAY))
            print(label)
            match_percentage=round(100-float(confidence),2)

            try:
                if match_percentage < 0:
                    label_text = "Detect"
                else:
                    label_text = subjects[label]+", Match: "+str(match_percentage)+" %"

            except IndexError:
                label_text = 'UnKnown'

            print(label_text)
            #draw a rectangle around face detected
            draw_rectangle(frame, rect)
            #draw name of predicted person
            draw_text(frame, label_text, rect[0], rect[1]-5)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #Release the Camera
    video_capture.release()
    cv2.destroyAllWindows()

    print("Recog End")

start_recogniz()
