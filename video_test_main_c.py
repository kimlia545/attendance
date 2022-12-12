# ---------------------------------------------- import ----------------------------------------------
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import cv2, dlib
import numpy as np
import random
from imutils import face_utils
import pickle
from datetime import datetime
import time
from time import strftime           # 화면 현재 시간
import os

from file_manager import load_save_filelist
from face_recog import get_frame
import face_recognition             # 얼굴인식(트래킹 포함)
from database import list_dao

# ---------------------------------------------- load ----------------------------------------------
print('loading faceNet.....') # faceNet : 얼굴을 찾는 모델
faceNet = cv2.dnn.readNet('face_detector/deploy.prototxt', 'face_detector/res10_300x300_ssd_iter_140000.caffemodel')
print('loading maskNet.....') # maskNet : 마스크 검출 모델
maskNet = load_model('models/mask/mask_detector.model')
print('loading detector.....') # detector : 얼굴 검출기
detector = dlib.get_frontal_face_detector()
print('loading predictor.....') # predictor : 68개 랜드마크 검출기
predictor = dlib.shape_predictor('dlib-models/shape_predictor_68_face_landmarks.dat')
print('loading blinkModel.....') # blinkModel : 눈깜박임여부 학습된 모델
blinkModel = load_model('models/blink/2020_10_22_14_47_26.h5') 
load_save_filelist() # 얼굴인식을 위해 폴더로부터 인코딩리스트, 네임리스트 로드하여 pickle로 저장
#face_classifier = cv2.CascadeClassifier('models/face/haarcascade_frontalface_default.xml') # 사진 촬영 얼굴 인식

# ---------------------------------------------- static var ----------------------------------------------
IMG_SIZE = (34, 26) # IMG_SIZE 설정
font = cv2.FONT_HERSHEY_SIMPLEX # 폰트 설정

# ---------------------------------------------- function ----------------------------------------------
# 얼굴부분 좌표값 + 마스크착용여부 예측값 추출
def detectFacePredictMask(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0)) # 이미지 전처리
    faceNet.setInput(blob) # faceNet의 input으로 blob을 설정
    detections = faceNet.forward() # faceNet 결과 저장

    faces = []
    locs = []
    preds = []
    startX, startY = (0,0)

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

        face = frame[startY:endY, startX:endX] # 원본 이미지에서 얼굴영역 추출
        
        # 추출한 얼굴영역을 전처리
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)

        faces.append(face)
        locs.append((startX, startY, endX, endY)) # 얼굴 사각 박스 좌표

    if len(faces) > 0: # 감지된 얼굴이 1개라도 있으면 실행
        # 더 빠른 추론을 위해 위의 'for'루프에서 일대일 예측이 아닌 
        # * 모든 * 얼굴에 대한 일괄 예측을 동시에 수행
        faces_arr = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces_arr, batch_size=32) # 대략 이런느낌 [[0.5380048  0.46199515]] 
    return faces, locs, preds

# 마스크 착용여부 감지 -> 표현
def checkMask(frame, locs, preds):
    global frame_count
    # print('checkMask')
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box        
        (mask, withoutMask) = pred # 마스크 착용 예측값        
        maskState = None
        # 마스크 착용여부 확인
        label = ''
        
        if mask > withoutMask: # 마스크 썼다면
            label = "Take off your Mask"
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (startX, startY - 50), font, 1.0, (0, 0, 255) , 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255) , 2)
        else : # 마스크 안썼다면
            print('test2')
            label = "No Mask"
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (startX, startY - 50), font, 1.0, (0, 255, 0) , 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0) , 2)
            frame_count += 1
            if frame_count >25:
                maskState = True # 마스크 안썻다
                return maskState

# 미션1 (눈 깜박임)
def blink_detecting(frame):
    print('blink_detecting')
    blink_state = None
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = detector(gray) # Detect faces in the image

    # mission_con = 'mission -> blink your eyes'
    # cv2.putText(frame, mission_con, (50, 100 - 50), font, 1.0, (0, 0, 255) , 1)
 
    for face in faces:
        shapes = predictor(gray, face) # gray(캡처이미지), face(예측한 얼굴이미지리스트 중 1개의 얼굴)
        shapes = face_utils.shape_to_np(shapes)

        eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
        eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

        eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
   
        #eye_img_r = cv2.flip(eye_img_r, flipCode=1)
        #########################################################
        # mission_res = eye_is_outside(eye_img_l)
        # print(mission_res)
        #########################################################
        #print(eye_is_outside(eye_img_r))

        # cv2.imshow('l', eye_img_l)
        # cv2.imshow('r', eye_img_r)
            
        eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
        eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

        pred_l = blinkModel.predict(eye_input_l)
        pred_r = blinkModel.predict(eye_input_r)

        # visualize
        # pred_l ===> 0.1 미만 : 눈감은것(-) / 0.1 초과 : 눈뜬것(O)
        state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
        state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

        state_l = state_l % pred_l
        state_r = state_r % pred_r

        cv2.rectangle(frame, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
        cv2.rectangle(frame, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)

        cv2.putText(frame, state_l, tuple(eye_rect_l[0:2]), font, 0.7, (255,255,255), 2)
        cv2.putText(frame, state_r, tuple(eye_rect_r[0:2]), font, 0.7, (255,255,255), 2)

        info = 'Please blink your eyes'
        cv2.putText(frame, info, (20, 110), font, 1.0, (0, 0, 255), 2) 

        if pred_l < 0.1: # 눈 감았다면
            return True
        else:
            return False

# 눈 부분만 자르는 함수
def crop_eye(gray, eye_points): 
	x1, y1 = np.amin(eye_points, axis=0)
	x2, y2 = np.amax(eye_points, axis=0)
	cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    
	w = (x2 - x1) * 1.2
	h = w * IMG_SIZE[1] / IMG_SIZE[0]

	margin_x, margin_y = w / 2, h / 2

	min_x, min_y = int(cx - margin_x), int(cy - margin_y)
	max_x, max_y = int(cx + margin_x), int(cy + margin_y)

	eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

	eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

	return eye_img, eye_rect

# 미션2 (동공 움직임)
def eye_detecting(frame):
    mission2_res = None
    print('미션2 동공움직임')
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector(gray) # Detect faces in the image

    for face in faces:
        shapes = predictor(gray, face) # gray(캡처이미지), face(예측한 얼굴이미지리스트 중 1개의 얼굴)
        shapes = face_utils.shape_to_np(shapes)

        eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
        eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

        eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)

        mission2_res = eye_is_outside(eye_img_l) # 왼쪽보는 미션 성공시 True리턴 / 실패시 False리턴

        cv2.rectangle(frame, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
        cv2.rectangle(frame, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)

        #cv2.putText(frame, 'text', tuple(eye_rect_l[0:2]), font, 0.7, (255,255,255), 2)
        #cv2.putText(frame, 'text', tuple(eye_rect_r[0:2]), font, 0.7, (255,255,255), 2)

        if mission2_res: # 왼쪽보는 미션 성공시
            return mission2_res

# 미션 (동공)
def eye_is_outside(eye_img):
	ret = False
	_, tmp = cv2.threshold(eye_img, 38, 255, cv2.THRESH_BINARY_INV) #38이상 0, 38이하는 최대값
	cv2.imshow('r', tmp)
	tmp = cv2.erode(tmp, None, iterations=2) #1
	tmp = cv2.dilate(tmp, None, iterations=4) #2
	tmp = cv2.medianBlur(tmp, 3) #3
	non_zeros = cv2.findNonZero(tmp) # 0이 아닌 좌표값들(=눈동자에 해당하는 255값만 non_zeros에 저장)
	if non_zeros is not None:
		avg_index = int(sum(non_zeros)[0][0] / len(non_zeros))
		ret = avg_index < 10 # 왼쪽으로 봤을 때 보통 10이하
	return ret

# 얼굴인식을 위한 인코딩리스트, 네임리스트 로드
def load_list_encoding_name():
    print('load_list_encoding_name')
    # 인코딩 리스트 로드하기
    with open('known_face_encodings_p', 'rb') as fp:
        known_face_encodings = pickle.load(fp)
    # 이름 리스트 로드하기
    with open('known_face_names_p', 'rb') as fp:
        known_face_names = pickle.load(fp)

    return known_face_encodings, known_face_names

#  # 이름, 출석시간 파일 저장
# def check_Attendance(name):
#     print('check_Attendance')
#     now = datetime.now()  # 현재 시간 출력
#     nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
#     list = []
#     attend = name +' '+ nowDatetime
#     list.append(attend) # 출석 기록
#     with open(f'{name}.txt', 'a') as f:
#         f.write(str(list))   

 # 얼굴 자르기
def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return None
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h+100, x:x+w+100] 
    return cropped_face

 # 사진 촬영 파일 저장
def pic(frame, count):
    if face_extractor(frame) is not None:
        face = cv2.resize(face_extractor(frame),(300,300))
        file_name_path = 'knowns/'+str(new_name)+str(count)+'.jpg'        
        cv2.imwrite(file_name_path,face)

# 초기화 함수
def init_env():
    global isgoing # 동작 

    global step  # 초기 마스크 인식 단계
    global mission_success_cnt # 미션 성공횟수
    global cnt_blink  # 첫 학생의 눈깜박임 횟수를 위한 초기화
    global mission_frame_count # 미션 실패시 프레임 계속 머물러서 미션 수행 시도 돕게 하는 것
    global notice
    global frame_count  # 프레임 대기시간 주기 위한 변수
    global recog_count # 언논일때 얼굴인식 기회를 여러번 주기 위함
    global new_name # 사진 촬영 후 등록할 새 이름
    global Attendance
    global known_face_encodings
    global known_face_names
    global name

    isgoing = True
    step = 0 # 초기 마스크 인식 단계
    mission_success_cnt = 0 # 미션 성공횟수
    cnt_blink = 0 # 첫 학생의 눈깜박임 횟수를 위한 초기화
    mission_frame_count = 0
    notice = None
    frame_count = 0 # 프레임 대기시간 주기 위한 변수
    recog_count = 0 # 언논일때 얼굴인식 기회를 여러번 주기 위함
    new_name = None # 사진 촬영 후 등록할 새 이름
    Attendance = False
    name = None

# ---------------------------------------------- init global var ----------------------------------------------
global isgoing # 동작 

global step  # 초기 마스크 인식 단계
global mission_success_cnt # 미션 성공횟수
global cnt_blink  # 첫 학생의 눈깜박임 횟수를 위한 초기화
global mission_frame_count # 미션 실패시 프레임 계속 머물러서 미션 수행 시도 돕게 하는 것
global notice
global frame_count  # 프레임 대기시간 주기 위한 변수
global recog_count # 언논일때 얼굴인식 기회를 여러번 주기 위함
global new_name # 사진 촬영 후 등록할 새 이름
global Attendance
global known_face_encodings
global known_face_names
global name

isgoing = True
step = 0 # 초기 마스크 인식 단계
mission_success_cnt = 0 # 미션 성공횟수
cnt_blink = 0 # 첫 학생의 눈깜박임 횟수를 위한 초기화
mission_frame_count = 0
notice = None
frame_count = 0 # 프레임 대기시간 주기 위한 변수
recog_count = 0 # 언논일때 얼굴인식 기회를 여러번 주기 위함
new_name = None # 사진 촬영 후 등록할 새 이름
Attendance = False
known_face_encodings, known_face_names = load_list_encoding_name() # 얼굴 인식 인코딩
name = None

# ---------------------------------------------- main function ----------------------------------------------
def attend_check( frame ):
    global isgoing # 동작 

    global step  # 초기 마스크 인식 단계
    global mission_success_cnt # 미션 성공횟수
    global cnt_blink  # 첫 학생의 눈깜박임 횟수를 위한 초기화
    global mission_frame_count # 미션 실패시 프레임 계속 머물러서 미션 수행 시도 돕게 하는 것
    global notice
    global frame_count  # 프레임 대기시간 주기 위한 변수
    global recog_count # 언논일때 얼굴인식 기회를 여러번 주기 위함
    global new_name # 사진 촬영 후 등록할 새 이름
    global Attendance
    global known_face_encodings
    global known_face_names
    global name

    # init_env()

    _, locs, preds = detectFacePredictMask(frame) # 얼굴 좌표값 + 마스크착용여부 예측값 추출

    #print('계속 호출된다', step)
    # frame = cv2.flip(frame, flipCode=1)

    # 0. 마스크 착용여부 ================================================================================================
    if step == 0 :
        maskState = checkMask(frame, locs, preds) # 마스크 안썼다면 True 리턴

        if maskState:
            step = 1 # 미션 수행하러 가기

    # 1. 눈 동작 미션 수행 ==============================================================================================
    elif step == 1:
        if mission_success_cnt < 3: # 미션 3번미만으로 달성했다면 아래 코드 실행(총 3번성공해야함)
            if random.randint(1,2) == 1: # 랜덤 미션 번호 (1,2 중에 선택)
                step = 6
                ######################################
                # print('미션1. 눈을 깜박여주세요')
                # if blink_detecting(frame): # 눈감았다면 True 리턴
                #     cnt_blink += 1 
                #     print('깜박임 확인성공 cnt_blink', cnt_blink)
                # else:
                #     cnt_blink = 0 # 연속적인 눈깜박임을 확인하기 위해서! -> 만약 눈 감다가 뜬 시간이 2frame보다 적다면 다시 0으로 초기화
                # if cnt_blink == 2: # 연속적(2frame)인 눈 깜박임이 있다면
                #     print('눈깜박임 미션 성공')
                #     mission_success_cnt += 1
                #     cnt_blink = 0
                #     print('깜박임 미션성공 mission_success_cnt', mission_success_cnt)
                ######################################
            elif random.randint(1,2) == 2: # 랜덤 미션 번호 (1,2 중에 선택)
                step = 7
                ######################################
                # print('미션2. 눈동자를 왼쪽으로 향하게 봐주세요')
                # if eye_detecting(frame): # 왼쪽방향 봐서 True 리턴되었다면
                #     print('동공방향 미션 성공')
                #     mission_success_cnt += 1
                #     print('동공방향 미션성공 mission_success_cnt', mission_success_cnt)
                ######################################
        else: # 미션 최종 성공했다면
            print('미션최종 성공')
            time.sleep(2.0)
            step = 2 # 얼굴 인식 단계

    # 2. 얼굴 인식 ================================================================================================
    elif step == 2:
        #locs에서 face값 추출
        for box in locs:
            (startX, startY, endX, endY) = box
        
        frame_copy = frame.copy() 
        face_frame = frame_copy[480-endY:480-startY, startX:endX, ::]
        cropped_face = frame_copy[startY:startY+endY, startX:startX+endX] # 사진 촬영 얼굴 자르기
        new_face = cv2.resize(cropped_face,(300,300)) # 촬영 얼굴 사이즈 조정

        # name = 'Unknown'
        name = get_frame( frame, known_face_encodings, known_face_names ) # 얼굴인식 루틴 호출
        
        if name == 'Unknown' : # 인식 되지 않은 얼굴 Unknown
            if recog_count < 3: # 3번 기회동안 언논이면 다시 얼굴인식 시도
                recog_count += 1
            else:
                step = 3 # 사진 촬영 스텝

        else: # 등록된 사람일 경우 출석기록
            if not Attendance: # 아직 출석체크 안했다면

                name_list = list_dao.get_name( name ) # DB에 로그 기록
                Attendance = True
                print('네임리스트는', name_list)
            # check_Attendance(name) # 이름, 시간 파일 저장
            step = 5 # 출석 기록 저장, 화면에 인식된 이름 길게 보여주는 스텝
                
    # 3. unknown 경고창 ================================================================================================
    elif step == 3:
        notice = ('Unknwon! Access is not possible.')
        #cv2.putText(frame, name, (startX + 6, startY - 6), font, 1.0, (0, 0, 255), 2) 
        cv2.putText(frame, notice, (20, 300), font, 1.0, (0, 0, 255), 2) 

        #time.sleep(1.0)   
        #new_name = input('Write your name : ') # 이름 쓰기
        #step = 4

    # 4. 사진 저장 여부 동의 ================================================================================================
    elif step == 4 :
        choice = input('do you want to save your photo? yes(1) or no(0)  answer:  ')
        if choice == '1':
            file_name_path = 'knowns/'+str(new_name)+'.jpg'        
            cv2.imwrite(file_name_path,new_face) # 사진 저장
            load_save_filelist()  # 얼굴인식을 위해 폴더로부터 인코딩리스트, 네임리스트 로드하여 pickle로 저장 
            known_face_encodings, known_face_names = load_list_encoding_name() # 얼굴인식을 위한 인코딩리스트, 네임리스트 로드
            init_env()

    # 5. 출석 체크후 완료 문구 표시 =============================================================================================
    elif step == 5:
        faces, _, _ = detectFacePredictMask(frame) # 얼굴 좌표값 + 마스크착용여부 예측값 추출

        if len(faces) > 0: # 감지된 얼굴이 1개라도 있으면 실행
            for box in locs: #locs에서 face값 추출
                (startX, startY, endX, endY) = box

            cv2.putText(frame, name, (startX, startY - 50), font, 1.0, (0, 255, 0) , 2)
            info = 'attendance completed' # 출석 완료
            cv2.putText(frame, info, (startX, startY - 20), font, 1.0, (0, 255, 0) , 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0) , 2)
        else: # 사람이 보이지 않는다면
            init_env()

    # 6. 미션1. 눈깜박임
    elif step == 6: # 눈깜박임 미션 수행
        notice = '미션1. 눈을 2초간 감아주세요'
        print( notice )
        if blink_detecting(frame): # 성공시
            cnt_blink += 1 
            print('깜박임 확인성공 cnt_blink', cnt_blink)
        else:
             cnt_blink = 0 # 연속적인 눈깜박임을 확인하기 위해서! -> 만약 눈 감다가 뜬 시간이 2frame보다 적다면 다시 0으로 초기화
        if cnt_blink == 2: # 연속적(2frame)인 눈 깜박임이 있다면
            print('눈깜박임 미션 성공')
            mission_success_cnt += 1
            cnt_blink = 0
            notice = '눈깜박임 미션 성공!!!'
            print( notice )
            print('깜박임 미션성공 mission_success_cnt', mission_success_cnt)
            step = 1
        else: # 실패시
            mission_frame_count += 1
            if mission_frame_count > 20 : # 프레임 20번도는데도 실패라면
                step = 1 # 스텝1로 돌아가 다시 랜덤미션 받기
                mission_frame_count = 0 # 다시 0으로 초기화

    # 7. 미션2. 동공 움직임
    elif step == 7: # 동공 미션 수행
        notice = '미션2. 눈동자를 왼쪽으로 향하게 봐주세요'
        print( notice )
        if eye_detecting(frame): # 성공시
            print('동공방향 미션 성공')
            mission_success_cnt += 1
            notice = '동공방향 미션 성공!!!'
            print( notice )
            print('동공방향 미션성공 mission_success_cnt', mission_success_cnt)
            step = 1
        else: # 실패시
            mission_frame_count += 1
            if mission_frame_count > 20 : # 프레임 20번도는데도 실패라면
                step = 1 # 스텝1로 돌아가 다시 랜덤미션 받기
                mission_frame_count = 0 # 다시 0으로 초기화
 