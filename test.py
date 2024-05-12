import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import threading
from playsound import playsound

model = YOLO('best.pt')

def audio():
    tap=playsound('alarm.mp3')
    tap.release()
    cv2.destroyAllWindows()

def RGB(event, x, y,flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point=[x,y]
        print(point)

def video():
    cv2.namedWindow('RGB')
    cv2.setMouseCallback('RGB',RGB)

    nap=cv2.VideoCapture('new1.avi')
    return nap
    
cap=video()

tap=threading.Thread(target=video)
tap.start()
tap.join()


my_file = open("coco1.txt","r")
data=my_file.read()	
class_list=data.split("\n")


count = 0
area1=[(69,172),(464,95),(574,157),(62,319)] #(new1.avi file)
#area1=[(669,39),(563,17),(548,59),(666,82)] #(new2.avi file)
#area1=[(245,119),(316,91),(474,131),(411,189)] #(new3.avi file)
while True:
    ret,frame=cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    count+=1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
   

#    print(px)
    list1=[]

    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        cx=int(x1+x2)//2
        cy=int(y1+y2)//2
        w,h=x2-x1,y2-y1
        result=cv2.pointPolygonTest(np.array(area1,np.int32),((cx,cy)),False)
        if result >= 0:
#        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),-1)
            cvzone.cornerRect(frame,(x1,y1,w,h),3,2)
            cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
            cvzone.putTextRect(frame,f'person',(x1,y1),1,1)
            list1.append(cx)

            
        
    ctr=len(list1)

    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,255,0),2)
    #cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,0,255),2)
    cvzone.putTextRect(frame,f'Counter:- {ctr}',(50,60),1,1)

    if ctr > 8:
        print("Queue Full!")
        cvzone.putTextRect(frame,f'Queue Full! Raise Alarm!',(350,40),1,1)
        aud=threading.Thread(target=audio)
        aud.start()
        aud.join()
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
    
    
cap.release()
cv2.destroyAllWindows()

