from ultralytics import YOLO
import cv2
model = YOLO("yolov8n.pt")  # Load pretrained YOLOv8
from scipy.integrate import quad

       


##########################
import requests
import math
import time

#hardcoding the ip adress, because the connection is gonna be on the local network
ip_addr = '192.168.4.1'
#consts for implementing PI control
I =0
_time=0
#PI control
def PI(K, Ki, x_desired, x_actual):
    global I, _time
    now=time.monotonic()
    if (_time ==0):
        dt = 0.05
    else:
        dt = max(1e-3,now-_time)
    _time = now
    e = (x_desired - x_actual)

    if(e>255):
        e=255
    if(e<-255):
        e=-255
    if(e<3 and e>-3):
        e=0
    
    #when increasing the tork does not cause a trouble, only then we change the tork
    u_ = Ki*I + K*e
    #checking if the theoretical PI sygnal is bigger that possible signal if so variable would_sat = True
    would_sat   = abs(u_) > 255
    #checking if the sing of the error and theoretical PI sygnal are the same. It is bad for us in case of big tork , wich is checked in would_sat
    push_worse = (u_*e)>0
    if not(would_sat and push_worse):
        I+=e*dt

    u = Ki*I + K*e
    u = max(-255, min(255, u))
    return u



def reduce_error_x(x_coord,x_center_coord):
    #negative value of an error means that we should be turning left
    pwm = int(round(PI(1,1,x_center_coord , x_coord)))

    if(pwm<0):
                command = f'{{"T":11,"L":0,"R":{(abs(pwm))}}}' 
    elif(pwm >0):
                command = f'{{"T":11,"L":{(abs(pwm))},"R":0}}' 
    else:
                command = f'{{"T":11,"L":0,"R":0}}' 
    url = "http://" + ip_addr + "/js?json=" + command
    response = requests.get(url)
    content = response.text
    print(content)
##########################
def stop_robot():
        command = f'{{"T":11,"L":0,"R":0}}' 
        url = "http://" + ip_addr + "/js?json=" + command
        response = requests.get(url)
        content = response.text

#iterating over model.predict objects- each for the frame 
for results in model.predict(source = 0,stream=True,show=False,classes = [47],save =False,device=0, max_det=1, vid_stride=5 ):
    #creating naked image(just taking image from camera)
    frame = results.orig_img.copy()
    #checking whether or not there are some bb, if not, just show regular image
    if results.boxes is None or len(results.boxes)==0:
        cv2.imshow("YOLO livestream", frame)
        stop_robot()

    else:
    #if some bb was detecter using formulas we are calculating the center of the bb
        
        original_cords = results.boxes.data
        #calculating the center of the bounding box. 
        x_center = (original_cords[0][0]+(original_cords[0][2] -original_cords[0][0])/2)
        y_center = (original_cords[0][1]+(original_cords[0][3]- original_cords[0][1])/2)
        center_of_bb = [int(x_center.item()),int(y_center.item())]
        print(center_of_bb)
        reduce_error_x(320,center_of_bb[0])
        #drawing dot over the "frame" image
        cv2.circle(frame, center=(center_of_bb[0], center_of_bb[1]), radius=1, color=(0, 255, 0), thickness=5)
    #showing what
    cv2.imshow("YOLO livestream", frame)
    #handaling exceptions
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
