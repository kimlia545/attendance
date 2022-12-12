from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import numpy as np
import pickle

# 기존 이미지목록 pickle로 로드
def load_imglist_pic():
    with open('image_list_p', 'rb') as fp:
        existing_image_list = pickle.load(fp)
    return existing_image_list

# 증식완료된 이미지목록 pickle로 저장 [KCY.jpg, JYT.jpg.....]
def save_imglist(image_list):
    with open('image_list_p', 'wb') as fp:
        pickle.dump(image_list, fp)

# 전체 이미지리스트 목록 구성하기
def get_all_imglist():
    path = 'knowns/' #원본 이미지 위치
    image_list = os.listdir(path) # ['ASH.jpg', 'JYT.jpg', 'KCY.jpg', 'KRU.jpg', 'ODS.jpg', 'OUK.jpg', 'PJM.jpg', 'PSE.jpg']
    return image_list # 'knowns/' 폴더 내 전체 파일목록 들고오기

# 이미지 증식이 진행되지 않은 이미지리스트 구성하기
def pick_gen_imglist():
    need_list = [] # 비교후, 증식이 필요한 이미지 담을 리스트
    image_list = get_all_imglist()
    print("known폴더 내 전체 이미지 목록",image_list )

    if os.path.isfile('image_list_p'):
        print('이전에 저장된 파일이 존재합니다')
        # 비교 시작
        existing_image_list = load_imglist_pic()
        for i in image_list: # 새로 추가된 이미지가 있는 리스트로부터 한개씩 뽑은 i
            if i not in existing_image_list: # 그 i가 기존 이미지리스트에 없다면
                need_list.append(i)
        # 없는 애들만 증식하고 저장하기
        print('증식이 필요한 아이들', need_list)
        img_gen(need_list)
        save_imglist(image_list) # pickle로 파일 저장
    else: # 이전에 저장된 pickle파일이 없다면(최초 실행이라면)
        img_gen(image_list)
        save_imglist(image_list) # pickle로 파일 저장

# 이미지 증식
def img_gen(image_list): # 여기엔 실행 당시 전체 이미지목록이 들어와야 함
    datagen = ImageDataGenerator(
            rotation_range = 20, # 이미지 회전
            width_shift_range = 0.2, #이미지 좌우로 움직이기
            height_shift_range = 0.2, #이미지 위아래로 움직이기
            shear_range = 0.5, # 0.5라디안내외로 시계반대방향으로 변형
            zoom_range = 0.3, # 1을 기준으로 0.7배~1.3배로 크기 변화
            horizontal_flip = True, # 수평방향으로 뒤집기
            #vertical_flip = True, # 수직방향으로 뒤집기
            fill_mode = 'nearest') #빈값 채우기

    # image_list = os.listdir(path) 
    path = 'knowns/' #원본 이미지 위치
    save_dir = 'images/' #늘릴 이미지가 저장될 위치

    for f in image_list:
        image_filepath = path+f # 'knowns/ASH.jpg
        img = load_img(image_filepath)  # PIL 이미지
        x = img_to_array(img)  # (3, 150, 150) 크기의 NumPy 배열
        x = x.reshape((1,) + x.shape)  # (1, 3, 150, 150) 크기의 NumPy 배열

        # 아래 .flow() 함수는 임의 변환된 이미지를 배치 단위로 생성해서
        # 지정된 `preview/` 폴더에 저장합니다.
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                save_to_dir=save_dir, save_prefix=f[:3], save_format='jpg'):
            i += 1
            if i > 5:
                break  # 이미지 20장을 생성하고 마칩니다


if __name__ == '__main__':
    pick_gen_imglist()