from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import time
import random




# https://blog.naver.com/jadeinher20s/221869607726
# https://blog.naver.com/jadeinher20s/221879555059



# 모델 로드
def loadModel():
    # faceNet : 얼굴을 찾는 모델
    faceNet = cv2.dnn.readNet('face_detector/deploy.prototxt', 'face_detector/res10_300x300_ssd_iter_140000.caffemodel')
    # maskNet : 마스크 검출 모델
    maskNet = load_model('models/mask/mask_detector.model')
    return faceNet, maskNet


# 웹캠 작동
def cam():
    cap = cv2.VideoCapture(0)
    #time.sleep(2.0)
    play(vs)
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

# 서비스 시작
def play(vs):
    faceNet, maskNet = loadModel()
    
    while True:
        frame = vs.read()
        frame = cv2.flip(frame, flipCode=1)
        frame = imutils.resize(frame, width=400)

        # 얼굴 좌표값 + 마스크착용여부 예측값 추출
        locs, preds = detectFacePredictMask(frame, faceNet, maskNet)
        ret = checkMask(frame, locs, preds) # 마스크 안썼다면 True 리턴
        if ret: # 마스크 안쓴 상태라면
            pass









        cv2.imshow('check_attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break





# 얼굴부분 좌표값 + 마스크착용여부 예측값 추출
def detectFacePredictMask(frame, faceNet, maskNet):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0)) # 이미지 전처리
    faceNet.setInput(blob) # faceNet의 input으로 blob을 설정
    detections = faceNet.forward() # faceNet 결과 저장

    faces = []
    locs = []
    preds = []
    
    # 마스크 착용여부 확인
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2] # confidence : 신뢰도
        if confidence < 0.5: # 신뢰도가 0.5보다 작으면 다시 확인
            continue
        # 바운딩 박스를 구함
        startX = int(detections[0, 0, i, 3] * w)
        startY = int(detections[0, 0, i, 4] * h)
        endX   = int(detections[0, 0, i, 5] * w)
        endY   = int(detections[0, 0, i, 6] * h)

        # 바운딩박스가 프레임 크기 내에 있는지 확인
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY)     = (min(w - 1, endX), min(h - 1, endY))

        # 원본 이미지에서 얼굴영역 추출
        face = frame[startY:endY, startX:endX]

        # 추출한 얼굴영역을 전처리
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)

        faces.append(face)
        locs.append((startX, startY, endX, endY))

    if len(faces) > 0: # 감지된 얼굴이 1개라도 있으면 실행
        # 더 빠른 추론을 위해 위의 'for'루프에서 일대일 예측이 아닌 
        # * 모든 * 얼굴에 대한 일괄 예측을 동시에 수행
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32) # 대략 이런느낌 [[0.5380048  0.46199515]]
    return locs, preds

# 마스크 착용여부 감지 -> 표현
def checkMask(frame, locs, preds):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        # 마스크 착용여부 확인
        label = ''
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255) , 2)
        if mask > withoutMask: # 마스크 썼다면
            label = "Take off your Mask"
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (startX, startY - 50), font, 1.0, (0, 0, 255) , 1)
        else : # 마스크 안썼다면
            label = "No Mask"
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (startX, startY - 50), font, 1.0, (0, 255, 0) , 1)
            return True

def getMission():
    mission = random.randint(1,2)
    if mission == 1:
        pass
    else: 
        pass
    pass





# 테스트
if __name__ == '__main__':
    #cam()

    import random

    ran_num = random.randint(1,2)
    print(ran_num)
    print(random.randint(1, 6))
    print(random.randint(1, 6))
