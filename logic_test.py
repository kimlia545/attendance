
############################### 미션 1개만 적용 ###############################
# While 1:
    # 상태변수 = 0 (초기화, 선언)

    # 프레임 1장 찍어 가져오기
    # 마스크 착용여부 판단(안썼다면 True 리턴)

    # while 1:  
        # if 마스크를 썼다면(리턴값이 None라면):
            # print(마스크 벗으세요!)
            # -> 다시 마스크 착용여부 확인하러 돌아가기
        # else 마스크 안썼다면(리턴값이 True라면):
            # -> 눈깜박임 미션 주기
                # if 눈을 깜박이지 않았다면: (리턴값이 0이라면)
                    # continue   => while문으로 다시 올라감
                # if 눈을 깜박였다면:(리턴값이 1이라면)    ==> 블링크함수 내에 젤 위에 상태변수 0초기화 후 성공시 1, 실패는 그대로 0 주고 상태변수 리턴
                    # if 상태변수라면:        ==> 미션 성공해서 리턴값이 1이라면
                        # 얼굴인식 시도
                        # 마스크 안썼을 때 사진가져와서 시도(위의 프레임)
                        # 얼굴찾았으면 화면에 표시
                        # 이름, 출석일시 저장
                        # break  ==> 가장 가까운 반복문 빠져나감


############################### WHILE문 연습 ###############################
# i = 0      
# while i < 3:
#     print('첫번째 와일문 안 i' , i)
#     while True:
#         print('두번째 와일문 안 i ', i)
#         i += 1
#         break




############################### 미션 3가지 적용 ###############################
# While 1:
    # 상태변수 = 0 (초기화, 선언)

    # 프레임 1장 찍어 가져오기
    # 마스크 착용여부 판단(안썼다면 True 리턴)

    # while 1:  
        # if 마스크를 썼다면(리턴값이 None라면):
            # print(마스크 벗으세요!)
            # -> 다시 마스크 착용여부 확인하러 돌아가기
        # elif 마스크 안썼다면(리턴값이 True라면):
            # -> 눈깜박임 미션 주기
                # if 눈을 깜박이지 않았다면: (리턴값이 0이라면)
                    # continue   => while문으로 다시 올라감
                # elif 눈을 깜박였다면:(리턴값이 1이라면)    ==> 블링크함수 내에 젤 위에 상태변수 0초기화 후 성공시 1, 실패는 그대로 0 주고 상태변수 리턴
                    # if 상태변수라면:        ==> 미션 성공해서 리턴값이 1이라면
                        # 얼굴인식 시도
                        # 마스크 안썼을 때 사진가져와서 시도(위의 프레임)
                        # 얼굴찾았으면 화면에 표시
                        # 이름, 출석일시 저장
                        # break  ==> 가장 가까운 반복문 빠져나감
############################### 연습 ###############################
# blink_list = []
# cnt = 0
# i = 0
# pred_l = 0


# while 1:
#     blink_list.append(pred_l)
#     pred_l += 1

#     if len(blink_list) >= 5: # 5개 이상 쌓였다면 실행
#         for pred in range(i, i+5): # 최초 0 ~ 5 ( 0,1,2,3,4 )
#             if blink_list[i] < 0.1: # 눈 감았다면
#                 cnt += 1
#         if cnt == 5: # 연속으로 눈감음이 유지된 상태

#             cnt = 0 # 초기화
#             i = 0 # 초기화
#             del blink_list[:] # 초기화

#             print(blink_list)
#             print('끝')
#             blink_state = 1
#             return blink_state # 감으면 1 안감으면 0
#         else:
#             i += 1 # 다음번째부터 연속 5번째 체크하기 위해 + 1

            


############################### 연습 ###############################

# import random
# case = random.randint(1,2)
# print(case)


        
# locs =  [(263, 210, 438, 443)]
# print(locs)
# # box = ()
# for box in locs:
#     (startX, startY, endX, endY) = box
#     locs = [(startX, endY, endX, startY)]

# print( startX )


############################### 자르기 ###############################

import cv2

# src = cv2.imread("test.jpg", cv2.IMREAD_COLOR) # (480, 640, 3)
# print(src.shape) # (605, 805, 3)

# dst = src.copy() 
# dst = src[183:250, 407:421]

# cv2.imshow("src", src)
# cv2.imshow("dst", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##############################################################3
# main
# if __name__ == "__main__":
#     cap = cv2.VideoCapture(0)
#     while 1:

#         ret, frame = cap.read()
#         frame = cv2.flip(frame, flipCode=1)
#         print('frame은', frame.shape) #(480, 640, 3)

#         dst = frame.copy() 
#         dst = dst[480-endY:480-startY, startX:endX]


 

#         cv2.imshow('result', frame)
#         cv2.imshow('result1', dst)
#         if cv2.waitKey(1) == ord('q'):
#             break
#     cv2.destroyAllWindows()    # 영상표시 윈도우를 닫음
#     print('프로그램 종료!')


##############################################################



# for box in locs:
#     (startX, startY, endX, endY) = box

# cv2.putText(frame, label, (startX, startY - 50), font, 1.0, (0, 0, 255) , 2)
# cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255) , 2)


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


##############################################################


# if step == 0 :
#     pass

# elif step == 1: # 눈동작 미션 수행
#     if mission_success_cnt < 3:
#         if 랜덤으로부터 1나와서 눈깜박임 미션을 받았다면:
#             step = 6
#         elif 랜덤으로부터 2나와서 동공 미션을 받았다면:
#             step = 7
#     else: # 미션 최종 성공했다면
#         print('미션최종 성공')
#         missionState = True 
#         time.sleep(2.0)
#         step = 2 # 얼굴 인식 단계        


# ###########################################3

# elif step == 6: # 눈깜박임 미션 수행
#     if blink_detecting(frame): # 성공시
#         mission_success_cnt += 1
#         step = 1
#     else: # 실패시
#         mission_frame_count += 1
#         if mission_frame_count > 20 : # 프레임 20번도는데도 실패라면
#             step = 1 # 스텝1로 돌아가 다시 랜덤미션 받기
#             mission_frame_count = 0 # 다시 0으로 초기화

# elif step == 7: # 동공 미션 수행
#     if eye_detecting(frame): # 성공시
#         mission_success_cnt += 1
#     else: # 실패시
#         mission_frame_count += 1
#         if mission_frame_count > 20 : # 프레임 20번도는데도 실패라면
#             step = 1 # 스텝1로 돌아가 다시 랜덤미션 받기
#             mission_frame_count = 0 # 다시 0으로 초기화

# ###########################################

		

#     # 1. 눈 동작 미션 수행 ==============================================================================================
#     elif step == 1:
#         if mission_success_cnt < 3: # 미션 3번미만으로 달성했다면 아래 코드 실행(총 3번성공해야함)
#             if random.randint(1,2) == 1: # 랜덤 미션 번호 (1,2 중에 선택)
#                 step = 6
#                 ######################################
#                 print('미션1. 눈을 깜박여주세요')
#                 if blink_detecting(frame): # 눈감았다면 True 리턴
#                     cnt_blink += 1 
#                     print('깜박임 확인성공 cnt_blink', cnt_blink)
#                 else:
#                     cnt_blink = 0 # 연속적인 눈깜박임을 확인하기 위해서! -> 만약 눈 감다가 뜬 시간이 2frame보다 적다면 다시 0으로 초기화
#                 if cnt_blink == 2: # 연속적(2frame)인 눈 깜박임이 있다면
#                     print('눈깜박임 미션 성공')
#                     mission_success_cnt += 1
#                     cnt_blink = 0
#                     print('깜박임 미션성공 mission_success_cnt', mission_success_cnt)
#                 ######################################
#             elif random.randint(1,2) == 2: # 랜덤 미션 번호 (1,2 중에 선택)
#                 step = 7
#                 ######################################
#                 print('미션2. 눈동자를 왼쪽으로 향하게 봐주세요')
#                 if eye_detecting(frame): # 왼쪽방향 봐서 True 리턴되었다면
#                     print('동공방향 미션 성공')
#                     mission_success_cnt += 1
#                     print('동공방향 미션성공 mission_success_cnt', mission_success_cnt)
#                 ######################################
#         else: # 미션 최종 성공했다면
#             print('미션최종 성공')
#             time.sleep(2.0)
#             step = 2 # 얼굴 인식 단계



step = 0
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if step == 0: # 마스크 착용여부 확인
        step = 1 # 성공시
    elif step == 1: # 눈 동작 미션 수행
        step = 2 # 성공시
    elif step == 2: # 얼굴 인식
        step = 0 #  # 성공시 이름,시간 저장후 초기화



