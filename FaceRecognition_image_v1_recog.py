'''
Face_Recognition_image_v1_recog.py
By: Pramod Singh Rawat
'''

import cv2
from AlphaFaceRecognitionFunc import *
from FaceRecDataSet import *

#create our LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_recognizer_training_data/face_rec_training_data1.yml')

print("Predicting Images ...")

#loading images to Test
test_img1 = cv2.imread("testing_data/known_image.jpg")
if not test_img1 is None:
    predicted_img1 = predict(face_recognizer, subjects, test_img1)
    if predicted_img1 is None:
        print("No face To Predict in 1st Image")
    else:
        cv2.imshow("Testing 1st", cv2.resize(predicted_img1, (500, 600))) #subjects[1]

test_img2 = cv2.imread("testing_data/a_kumar.jpg")
if not test_img2 is None:
    predicted_img2 = predict(face_recognizer, subjects, test_img2)

    if predicted_img2 is None:
        print("No face To Predict In 2nd Image")
    else:
        cv2.imshow("Testing 2nd", cv2.resize(predicted_img2, (500, 600))) #subjects[2]

test_img3 = cv2.imread("testing_data/test_kat1.jpg")
if not test_img3 is None:
    predicted_img3 = predict(face_recognizer, subjects, test_img3)
    if predicted_img3 is None:
        print("No face To Predict In 3rd Image")
    else:
        cv2.imshow("Testing 3rd", cv2.resize(predicted_img3, (500, 600))) #subjects[3]

#predict_from_multiple #todo: to be debug
test_img_multi_face = cv2.imread("testing_data/test_multiple3.jpg")
if not test_img_multi_face is None:
    predicted_img4 = predict_from_multiple(face_recognizer, subjects, test_img_multi_face)
    #cv2.imshow("From_MultiDetect", cv2.resize(predicted_img3, (500, 600)))
    if predicted_img4 is None:
        print("No face To Predict In This Image")
    else:
        cv2.imshow("Testing Image with Multiple face", cv2.resize(predicted_img4, (500, 600))) #subjects[3]

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Prediction Complete")
