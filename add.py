embeddingFeaturesFile = 'feat.npy'
camNo = 2

import time
start = time.time()

stLoad = time.time()
import os
import dlib
import numpy as np
import cv2

faceDetector = dlib.get_frontal_face_detector()
facePoseDetector = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
faceEncoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
if os.path.exists(embeddingFeaturesFile):
    storeEmbedding = np.load(embeddingFeaturesFile)
else:
    storeEmbedding = np.zeros((128,0))
    
endLoad = time.time()
print('Load Library and Model in', endLoad-stLoad, 's')

stCam = time.time()
inputStream = cv2.VideoCapture(camNo)
endCam = time.time()
print('Open Camera in', endCam-stCam, 's')

frameIdx = 0
getEmbeddingFeature = False

while frameIdx < 60:
    _, currentFrame = inputStream.read()
    

    faceBBoxes = faceDetector(currentFrame)
    if len(currentFrame.shape) == 2:
        currentFrame = cv2.cvtColor(currentFrame, cv2.COLOR_GRAY2RGB)

    for faceIdx, faceBBox in enumerate(faceBBoxes):
        faceLandmarks = facePoseDetector(currentFrame,faceBBox)
        embeddingFeature = faceEncoder.compute_face_descriptor(currentFrame, faceLandmarks)
        getEmbeddingFeature = True
        
    if getEmbeddingFeature:
        embeddingFeature = np.array(embeddingFeature)[:,np.newaxis]
        storeEmbedding = np.concatenate((embeddingFeature,storeEmbedding),axis=1)
        print('Total Face Models', storeEmbedding.shape[1])
        np.save(embeddingFeaturesFile, storeEmbedding)
        break
    frameIdx += 1

if getEmbeddingFeature == False:
    print('Timeout Face not Found')
end = time.time()
print('In', end-start)