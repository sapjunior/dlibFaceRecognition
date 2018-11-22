embeddingFeaturesFile = 'feat.npy'
minMatchDistance = 0.5
camNo = 2

import time

stLoad = time.time()
import dlib
import numpy as np
import cv2

faceDetector = dlib.get_frontal_face_detector()
facePoseDetector = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
faceEncoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

storeEmbedingFeature = np.load(embeddingFeaturesFile)

endLoad = time.time()
print('\tLoad Library and Model in', endLoad-stLoad, 's')

stCam = time.time()
inputStream = cv2.VideoCapture(camNo)
endCam = time.time()
print('\tOpen Camera in', endCam-stCam, 's')
foundMatch = False
stFaceRecog = time.time()
while True:
    _, currentFrame = inputStream.read()

    faceBBoxes = faceDetector(currentFrame)
    if len(currentFrame.shape) == 2:
        currentFrame = cv2.cvtColor(currentFrame, cv2.COLOR_GRAY2RGB)

    for faceIdx, faceBBox in enumerate(faceBBoxes):
        x1, y1, x2, y2 = faceBBox.left(), faceBBox.top(), faceBBox.right(), faceBBox.bottom()
        cv2.rectangle(currentFrame, (x1, y1), (x2, y2), (0,255,0), 3)

        faceLandmarks = facePoseDetector(currentFrame,faceBBox)
        embedingFeature = np.array(faceEncoder.compute_face_descriptor(currentFrame, faceLandmarks))[:,np.newaxis]
        distance = np.linalg.norm(embedingFeature-storeEmbedingFeature, axis=0)
        minIdx = np.argmin(distance)
        minDistance = distance[minIdx]

        if minDistance < minMatchDistance:
            print('Winning Model',minIdx,'Distance', minDistance)
            foundMatch = True
            break
    if foundMatch:
        break
            
    #cv2.imshow('Output', currentFrame)
    #cv2.waitKey(1)

endFaceRecog = time.time()
print('\tFace Recognition in', endFaceRecog-stFaceRecog, 's')
print('Total Time', endFaceRecog - stLoad,'s')
