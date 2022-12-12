import cv2
import os
import face_recognition
import pickle



def load_save_filelist():
    # Using OpenCV to capture from device 0. If you have trouble capturing
    # from a webcam, comment the line below out and use a video file
    # instead.
    #self.camera = camera.VideoCamera()              # device 0에 해당되는 카메라 초기화
    known_face_encodings = []                  # known_face_encodings LIST 초기화
    known_face_names = []                      # known_face_names LIST 초기화
    

    # Load sample pictures and learn how to recognize it.
    dirname = 'knowns'                              # 디렉토리명 = knowns
    files = os.listdir(dirname)                     # files LIST에 디렉토리내 파일명 저장
    print(files)    # files의 list 데이타 출력 => ['HA.jpg', 'KIM.jpg', 'PARK.jpg']

    # knowns 디렉토리에 있는 jpg파일 로딩후 파일명(이름)은 known_face_names LIST에 추가하고 엔코딩값은 known_face_encodings LIST에 추가
    for filename in files:                          # 디렉토리내 파일명 하나씩 불러옴
        name, ext = os.path.splitext(filename)      # 파일이름과 확장자 분리
        if ext == '.jpg':                           # JPG 확장자로 되어 있는 파일 일 경우
            known_face_names.append(name)          # known_face_names에 파일명 known_face_names LIST에 추가
            pathname = os.path.join(dirname, filename)  # 해당파일 디렉토리명 저장

            # 사진에서 얼굴 영역을 알아내고, face landmarks라 불리는 68개 얼굴 특징의 위치를 분석한 데이터를
            # known_face_encodings에 저장합니다.
            img = face_recognition.load_image_file(pathname)    # 해당파일 로딩후 저장
            face_encoding = face_recognition.face_encodings(img)[0]     # 저장된 해당파일 엔코딩 처리
            known_face_encodings.append(face_encoding)             # 해당파일 엔코딩값 known_face_encodings LIST에 추가

    # 인코딩 파일 저장
    with open('known_face_encodings_p', 'wb') as fp:
        pickle.dump(known_face_encodings, fp)

    # 이름 파일 저장
    with open('known_face_names_p', 'wb') as fp:
        pickle.dump(known_face_names, fp)

    # # 로드하기
    # with open('known_folder_list', 'rb') as fp:
    #     ttt = pickle.load(fp)
    # print('ttt는', ttt)


if __name__ == '__main__':

    load_save_filelist()
    
    print('finish')
