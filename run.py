from flask import Flask, render_template, Response, request, session as sess, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import cv2, dlib
import numpy as np
import random
from imutils import face_utils
import pickle
import face_recognition             # 얼굴인식(트래킹 포함)
from datetime import datetime
import time
from time import strftime           # 화면 현재 시간
import os

from file_manager import load_save_filelist
from face_recog import get_frame
from video_test_main_c import attend_check, init_env
from database import list_dao
from database.list_dao import register_student
import os
from face_register import register

from database import selectLogin
from database.list_dao import call_list

from filename import name_change

app = Flask(__name__)
#app.secret_key = 'lkawfndlnmcdewlocnmewqocnmewlck'
#app.secret_key = os.urandom(24)
msg = {
    "logout":'로그 아웃'
}


app = Flask(__name__)
camera = cv2.VideoCapture(1) # use 0 for web camera

global name
name = None

# 출석체크 프레임
def gen_frames():  # generate frame by frame from camera
    global name

    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if (not success):
            break
        else:
            #print('영상이 갱신된다')
            # ret, frame = cap.read()            
            frame = cv2.flip(frame, flipCode=1)
            # 실시간 시간 화면 출력            
            cv2.putText(frame, strftime("%H:%M:%S"), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)                      
            attend_check( frame )            
            # cv2.imshow('result', frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   # concat frame one by one and show result

# 출석체크 비디오
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    print(' video_feed() ')
    init_env()
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# 사진촬영 비디오
@app.route('/camera_feed')
def camera_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    print(' camera_feed() ')
    #init_env()
    return Response(gen_camera(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 사진촬영 프레임
def gen_camera():  # generate frame by frame from camera
    no_picture = True
    frame_count = 0
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if (not success):
            break
        else:    
            frame = cv2.flip(frame, flipCode=1)
            
            if frame_count < 100: # 사진 촬영 전 문구 화면에 띄우기
                cv2.putText(frame, 'Taking pictures', (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) , 2)                
            else:
                if no_picture: # 사진 촬영 1회
                    register ( frame ) 
                    
                    frame_count = 2000
                    no_picture = False                      
                else:
                    if frame_count <=100:
                        frame_count = 0

            if frame_count > 2000:     # 사진 촬영 완료 문구                  
                info2 = 'Completed. Write your info'
                cv2.putText(frame, info2, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) , 2)

            frame_count += 1
            # print( frame_count  )
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   # concat frame one by one and show result                    



# 첫화면
@app.route('/')
def home():
    """home page."""
    try:
        return render_template('home.html')
    except:
        return render_template('error.html')
    
# 출석 페이지
@app.route('/attendance', methods=['GET','POST'])
def attendance():
    if request.method == 'POST':
        return jsonify( list_dao.call_name() )
    else:
        """attendance page."""
        # name_list = list_dao.get_name( 'KimLyeeun' ) # DB 로그 기록
        rows = list_dao.call_name() # DB 로그 기록 가져오기 -> 웹

        return render_template('attendance.html', abs=rows)
        # return render_template('attendance.html', abs=rows, list=name_list)

# 404 에러 페이지
@app.route('/error')
def error():
    """error page."""
    return render_template('error.html')

# 제작자 페이지
@app.route('/member')
def member():
    """member page."""
    return render_template('member.html')

# 관리자 출결 확인
@app.route('/admin')
def admin():
    rows = call_list()
    """admin page."""
    return render_template('admin.html', list=rows)

# 신규 등록 사진 촬영
@app.route('/register', methods=['POST','GET'])
def new_register():
    if request.method == 'POST':
        """register page."""

        name = request.form.get('name')
        my_class = request.form.get('my_class')
        birdate = request.form.get('birdate')
        gender = request.form.get('gender')
        abs_rate = request.form.get('abs_rate')

        print( name )
        print( my_class )
        print( birdate )
        print( gender )
        print( abs_rate )

        name_change(name) #사진파일 변경

        # 디비입력
        if register_student(name, my_class, birdate, gender, abs_rate):
            # 성공
            return jsonify({ "code":1, "msg":'등록성공'})
        else:
            # 실패
            return jsonify({ "code":-1, "msg":'등록실패'})
    else:
        # 등록화면 랜더링
        return render_template('register.html')

# 로그인
@app.route('/login', methods=['GET', 'POST']) # guest 1
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:# POST
        uid = request.form.get('uid')
        upw = request.form.get('upw')
        print( uid, upw )
        user = selectLogin(uid, upw)
        print( user )
        if user:
            sess['uid'] = uid
            return redirect( url_for('admin') )       
        else:
            return render_template('alert.html')
        #return '로그인 처리 페이지 %s %s' % (uid, upw)  

# 로그아웃
@app.route('/logout')
def logout():
    if 'uid' in sess:
        sess.pop('uid', None)
    return redirect( url_for('home') )

if (__name__ == '__main__'):
    app.secret_key = 'lkawfndlnmcdewlocnmewqocnmewlck'
    app.config['SESSION_TYPE'] = 'filesystem'


    app.run(host='127.0.0.1', port=8000, debug=True) # True
    