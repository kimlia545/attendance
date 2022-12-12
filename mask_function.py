# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array
# import cv2, dlib
# import numpy as np
# import random
# from imutils import face_utils
# import pickle
# from file_manager import load_save_filelist
# from face_recog import get_frame
# import face_recognition             # 얼굴인식(트래킹 포함)
# from datetime import datetime
# import time
# from time import strftime # 화면 현재 시간
# import os

# print('loading faceNet.....') # faceNet : 얼굴을 찾는 모델
# faceNet = cv2.dnn.readNet('face_detector/deploy.prototxt', 'face_detector/res10_300x300_ssd_iter_140000.caffemodel')
# print('loading maskNet.....') # maskNet : 마스크 검출 모델
# maskNet = load_model('models/mask/mask_detector.model')
# print('loading detector.....') # detector : 얼굴 검출기
# detector = dlib.get_frontal_face_detector()
# print('loading predictor.....') # predictor : 68개 랜드마크 검출기
# predictor = dlib.shape_predictor('dlib-models/shape_predictor_68_face_landmarks.dat')
# print('loading blinkModel.....') # blinkModel : 눈깜박임여부 학습된 모델
# blinkModel = load_model('models/blink/2020_10_22_14_47_26.h5') 
# load_save_filelist() # 얼굴인식을 위해 폴더로부터 인코딩리스트, 네임리스트 로드하여 pickle로 저장
# face_classifier = cv2.CascadeClassifier('models/face/haarcascade_frontalface_default.xml') # 사진 촬영 얼굴 인식

# IMG_SIZE = (34, 26) # IMG_SIZE 설정
# font = cv2.FONT_HERSHEY_SIMPLEX # 폰트 설정

# ######################################## 함수 ########################################

# # 얼굴부분 좌표값 + 마스크착용여부 예측값 추출
# def detectFacePredictMask(frame):
#     h, w = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0)) # 이미지 전처리
#     faceNet.setInput(blob) # faceNet의 input으로 blob을 설정
#     detections = faceNet.forward() # faceNet 결과 저장

#     faces = []
#     locs = []
#     preds = []
    
#     # 마스크 착용여부 확인
#     for i in range(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2] # confidence : 신뢰도
#         if confidence < 0.5: # 신뢰도가 0.5보다 작으면 다시 확인
#             continue
#         # 바운딩 박스를 구함
#         startX = int(detections[0, 0, i, 3] * w)
#         startY = int(detections[0, 0, i, 4] * h)
#         endX   = int(detections[0, 0, i, 5] * w)
#         endY   = int(detections[0, 0, i, 6] * h)

#         # 바운딩박스가 프레임 크기 내에 있는지 확인
#         (startX, startY) = (max(0, startX), max(0, startY))
#         (endX, endY)     = (min(w - 1, endX), min(h - 1, endY))

#         face = frame[startY:endY, startX:endX] # 원본 이미지에서 얼굴영역 추출
        
#         # 추출한 얼굴영역을 전처리
#         face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#         face = cv2.resize(face, (224, 224))
#         face = img_to_array(face)
#         face = preprocess_input(face)

#         faces.append(face)
#         locs.append((startX, startY, endX, endY)) # 얼굴 사각 박스 좌표

#     if len(faces) > 0: # 감지된 얼굴이 1개라도 있으면 실행
#         # 더 빠른 추론을 위해 위의 'for'루프에서 일대일 예측이 아닌 
#         # * 모든 * 얼굴에 대한 일괄 예측을 동시에 수행
#         faces_arr = np.array(faces, dtype="float32")
#         #print('face는',face.shape)
#         preds = maskNet.predict(faces_arr, batch_size=32) # 대략 이런느낌 [[0.5380048  0.46199515]] 
#     return faces, locs, preds

# # 마스크 착용여부 감지 -> 표현
# def checkMask(frame, locs, preds):
#     # print('checkMask')
#     for (box, pred) in zip(locs, preds):
#         (startX, startY, endX, endY) = box
#         (mask, withoutMask) = pred # 마스크 착용 예측값
#         maskState = None
#         # 마스크 착용여부 확인
#         label = ''
        
#         if mask > withoutMask: # 마스크 썼다면
#             label = "Take off your Mask"
#             label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
#             cv2.putText(frame, label, (startX, startY - 50), font, 1.0, (0, 0, 255) , 2)
#             cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255) , 2)
#         else : # 마스크 안썼다면
#             label = "No Mask"
#             label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
#             cv2.putText(frame, label, (startX, startY - 50), font, 1.0, (0, 255, 0) , 2)
#             cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0) , 2)
#             maskState = True # 마스크 안썻다
#             return maskState

# # 미션1 (눈 깜박임)
# def blink_detecting(frame):
#     print('blink_detecting')
#     blink_state = None
#     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

#     faces = detector(gray) # Detect faces in the image

#     # mission_con = 'mission -> blink your eyes'
#     # cv2.putText(frame, mission_con, (50, 100 - 50), font, 1.0, (0, 0, 255) , 1)
 
#     for face in faces:
#         shapes = predictor(gray, face) # gray(캡처이미지), face(예측한 얼굴이미지리스트 중 1개의 얼굴)
#         shapes = face_utils.shape_to_np(shapes)

#         eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
#         eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

#         eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
#         eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
   
#         #eye_img_r = cv2.flip(eye_img_r, flipCode=1)
#         #########################################################
#         # mission_res = eye_is_outside(eye_img_l)
#         # print(mission_res)
#         #########################################################
#         #print(eye_is_outside(eye_img_r))

#         # cv2.imshow('l', eye_img_l)
#         # cv2.imshow('r', eye_img_r)
            
#         eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
#         eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

#         pred_l = blinkModel.predict(eye_input_l)
#         pred_r = blinkModel.predict(eye_input_r)

#         # visualize
#         # pred_l ===> 0.1 미만 : 눈감은것(-) / 0.1 초과 : 눈뜬것(O)
#         state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
#         state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

#         state_l = state_l % pred_l
#         state_r = state_r % pred_r

#         cv2.rectangle(frame, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
#         cv2.rectangle(frame, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)

#         cv2.putText(frame, state_l, tuple(eye_rect_l[0:2]), font, 0.7, (255,255,255), 2)
#         cv2.putText(frame, state_r, tuple(eye_rect_r[0:2]), font, 0.7, (255,255,255), 2)

#         if pred_l < 0.1: # 눈 감았다면
#             return True
#         else:
#             return False

# # 눈 부분만 자르는 함수
# def crop_eye(gray, eye_points): 
# 	x1, y1 = np.amin(eye_points, axis=0)
# 	x2, y2 = np.amax(eye_points, axis=0)
# 	cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    
# 	w = (x2 - x1) * 1.2
# 	h = w * IMG_SIZE[1] / IMG_SIZE[0]

# 	margin_x, margin_y = w / 2, h / 2

# 	min_x, min_y = int(cx - margin_x), int(cy - margin_y)
# 	max_x, max_y = int(cx + margin_x), int(cy + margin_y)

# 	eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

# 	eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

# 	return eye_img, eye_rect

# # 미션2 (동공 움직임)
# def eye_detecting(frame):
#     mission2_res = None
#     print('미션2 동공움직임')
#     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     faces = detector(gray) # Detect faces in the image

#     for face in faces:
#         shapes = predictor(gray, face) # gray(캡처이미지), face(예측한 얼굴이미지리스트 중 1개의 얼굴)
#         shapes = face_utils.shape_to_np(shapes)

#         eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
#         eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

#         eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
#         eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)

#         mission2_res = eye_is_outside(eye_img_l) # 왼쪽보는 미션 성공시 True리턴 / 실패시 False리턴

#         cv2.rectangle(frame, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
#         cv2.rectangle(frame, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)

#         cv2.putText(frame, 'text', tuple(eye_rect_l[0:2]), font, 0.7, (255,255,255), 2)
#         cv2.putText(frame, 'text', tuple(eye_rect_r[0:2]), font, 0.7, (255,255,255), 2)

#         if mission2_res: # 왼쪽보는 미션 성공시
#             return mission2_res

# # 미션 (동공)
# def eye_is_outside(eye_img):
# 	ret = False
# 	_, tmp = cv2.threshold(eye_img, 38, 255, cv2.THRESH_BINARY_INV) #38이상 0, 38이하는 최대값
# 	cv2.imshow('r', tmp)
# 	tmp = cv2.erode(tmp, None, iterations=2) #1
# 	tmp = cv2.dilate(tmp, None, iterations=4) #2
# 	tmp = cv2.medianBlur(tmp, 3) #3
# 	non_zeros = cv2.findNonZero(tmp) # 0이 아닌 좌표값들(=눈동자에 해당하는 255값만 non_zeros에 저장)
# 	if non_zeros is not None:
# 		avg_index = int(sum(non_zeros)[0][0] / len(non_zeros))
# 		ret = avg_index < 10 # 왼쪽으로 봤을 때 보통 10이하
# 	return ret

# # 얼굴인식을 위한 인코딩리스트, 네임리스트 로드
# def load_list_encoding_name():
#     print('load_list_encoding_name')
#     # 인코딩 리스트 로드하기
#     with open('known_face_encodings_p', 'rb') as fp:
#         known_face_encodings = pickle.load(fp)
#     # 이름 리스트 로드하기
#     with open('known_face_names_p', 'rb') as fp:
#         known_face_names = pickle.load(fp)

#     return known_face_encodings, known_face_names


# def check_Attendance(name): # 이름, 출석시간 파일 저장
#     print('check_Attendance')
#     now = datetime.now()  # 현재 시간 출력
#     nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
#     list = []
#     attend = name +' '+ nowDatetime
#     list.append(attend) # 출석 기록
#     with open(f'{name}.txt', 'a') as f:
#         f.write(str(list))   


# def face_extractor(img): # 얼굴 자르기
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray,1.3,5)
#     if faces is():
#         return None
#     for(x,y,w,h) in faces:
#         cropped_face = img[y:y+h+100, x:x+w+100] 
#     return cropped_face

# def pic(frame, count): # 사진 촬영 파일 저장
#     if face_extractor(frame) is not None:
#         face = cv2.resize(face_extractor(frame),(300,300))
#         file_name_path = 'knowns/'+str(new_name)+str(count)+'.jpg'        
#         cv2.imwrite(file_name_path,face)