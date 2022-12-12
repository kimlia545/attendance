# ---------------------------------------------- import ----------------------------------------------
import cv2
import numpy as np
import time

face_classifier = cv2.CascadeClassifier('models/face/haarcascade_frontalface_default.xml') # 사진 촬영 얼굴 인식

# ---------------------------------------------- static var ----------------------------------------------
IMG_SIZE = (34, 26) # IMG_SIZE 설정
font = cv2.FONT_HERSHEY_SIMPLEX # 폰트 설정

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
def pic(frame):
    if face_extractor(frame) is not None:
        face = cv2.resize(face_extractor(frame),(400,400))
    return face    


def register(frame):    
    #     info = 'Taking pictures'
    #     cv2.putText(frame, info, (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) , 2)    
    #new_face = pic(frame)
    file_name_path = 'knowns/'+'user'+'.jpg'        
    cv2.imwrite(file_name_path,frame) # 사진 저장    

'''
def register(frame):
    new_name = input('Write your name : ') # 이름 쓰기
    new_face = pic(frame)
    choice = input('do you want to save your photo? yes(1) or no(0)  answer:  ')
    if choice == '1':
        file_name_path = 'knowns/'+str(new_name)+'.jpg'        
        cv2.imwrite(file_name_path,new_face) # 사진 저장    
'''
# main
if __name__ == "__main__":
    print('main')
    cap = cv2.VideoCapture(0)

    while 1:
        ret, frame = cap.read()
        print("Frame Reading")
        frame = cv2.flip(frame, flipCode=1)

        #register( frame )

        # 실시간 시간 화면 출력
        cv2.imshow('result', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()    # 영상표시 윈도우를 닫음
    print('프로그램 종료!')
