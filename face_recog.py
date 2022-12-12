import face_recognition             # 얼굴인식(트래킹 포함)
import cv2                          # OPEN_CV 로딩
import os                           # 파일 입출력을 위한 로딩
import numpy as np                  # 넘파이 로딩
from datetime import datetime
import pickle
import time
import re

font = cv2.FONT_HERSHEY_DUPLEX

def get_frame( face_frame, known_face_encodings, known_face_names ):

    face_locations = []
    face_encodings = []
    face_names = []

    small_frame = cv2.resize(face_frame, (0, 0), fx=0.25, fy=0.25) # 1/4 사이즈로 줄여서 small_frame에 저장(데이타 처리 속도 향상)
    rgb_small_frame = small_frame[:, :, ::-1] # BGR -> RGB로 변경

    # 입력 영상 두번중에 한번꼴로 known_face_encodings과 비교해서 거리가 0.5이하이면 face_names LIST에 이름 추가 시킴

    # 현재 비디오 프레임에서 모든 얼굴위치 및 얼굴 인코딩 찾기
    face_locations = face_recognition.face_locations(rgb_small_frame)  # 위치
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations) # 인코딩 데이타
    
    name = "Unknown"

    for face_encoding in face_encodings:       # 현재 입력된 영상의 엔코딩 데이타를 순서데로 불러옴.
        # 현재 캡쳐된 엔코딩 데이타(face_encoding)와 알고있는 엔코딩 데이타(self.known_face_encodings)와 비교하여 거리(차)를 구함
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        min_value = min(distances)  # 거리 데이타중 최소값만 취함.
        
        # 일치하는것으로 간주할 거리값 비교, tolerance(공차)는 적을 수록 일치함,실험적으로 0.6이 최적
        # name에 Unknown으로 설정한뒤 일치하는 사람이 나오면 해당 이름으로 변경
        # name = "Unknown"
        if min_value < 0.4:                     # 최소값이 0.4(실험값중에 이상적인값에 해당)보다 적으면
            index = np.argmin(distances)        # 최소값 조건에 해당하는 색인(index) 값을 찾기
            name = known_face_names[index]     # 색인에 해당하는 이름 데이타 불러 오기

        face_names.append(name)                # face_names에 확인된 이름 face_names LIST에 추가

    # 정규화
    update_name = re.findall('[A-Za-z]',name) # 인식된 이름에서 숫자를 제외한 문자열만 찾기
    update_name = ''.join(update_name) # 한글자씩 떨어져 있는 것을 join하기
    print('숫자 떼고 문자열만 있는 이름', update_name)

    return update_name

if __name__ == '__main__':
    pass
    # 인코딩 리스트 로드하기
    with open('known_face_encodings_p', 'rb') as fp:
        known_face_encodings = pickle.load(fp)
    print('인코딩리스트는', known_face_encodings)

    # 이름 리스트 로드하기
    with open('known_face_names_p', 'rb') as fp:
        known_face_names = pickle.load(fp)
    print('이름리스트는', known_face_names)

    cap = cv2.VideoCapture(0)

    # locs =  [(263, 210, 438, 443)] # 임시로 넣은 실험값
    while 1:
        ret, frame = cap.read()
        frame = cv2.flip(frame, flipCode=1)

        name = get_frame( frame, known_face_encodings, known_face_names ) # 얼굴인식 루틴 호출

        cv2.imshow('result', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()                     # 영상표시 윈도우를 닫음
    print('프로그램 종료!')




