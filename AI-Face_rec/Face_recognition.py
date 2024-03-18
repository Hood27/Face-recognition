import cv2
import face_recognition
import os
from datetime import datetime
import numpy as np

#step 1: Load ảnh từ kho ảnh nhận dạng
path="Images"
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(len(images))
print(classNames)

#step 2: encoding (mã hóa)
def Mahoa(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode =  face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnow = Mahoa(images)
print("Ma hoa thanh cong")
print(len(encodeListKnow))

def quanly(name):
    with open('quanly.csv', 'r+') as f:
        myDataList = f.readlines()
        print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',') # tách theo dấu ,
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now() # trả về 2021-12-18 16:43:30.709791
            dtString = now.strftime('%H:%M:%S') # biểu thị string giờ phút giây
            f.writelines(f'\n{name},{dtString}')

#step 3: Khoi dong webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    framS = cv2.resize(frame, (0,0), None, fx=0.5, fy=0.5)
    framS = cv2.cvtColor(framS, cv2.COLOR_BGR2RGB)

    #Xác định vị trí khuôn mặt trên cam và encode hình ảnh trên cam
    facecurFrame = face_recognition.face_locations(framS)
    encodecurFrame = face_recognition.face_encodings(framS)

    for encodeFace, faceLoc in zip(encodecurFrame, facecurFrame):
        matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis) #đẩy về index của giá trị nhỏ nhất


        if faceDis[matchIndex] <0.65 :
            name = classNames[matchIndex].upper()
        else:
            name = "Unknown"

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*2, x2*2, y2*2, x1*2
        cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, name, (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)



    cv2.imshow('Test', frame)
    if cv2.waitKey(1) == ord("q"): #độ trễ 1/1000s, bấm q để thoát
        break

cap.release() #giải phóng camera
cv2.destroyAllWindows() #thoát tất cả cửa sổ
