from ultralytics import YOLO
import cv2
model = YOLO("yolov8n.pt")  # Load pretrained YOLOv8
#itetating over model.predict objects- each for the frame 
for results in model.predict(source = 0,stream=True,show=False,classes = [39],save =False,device=0, max_det=1, vid_stride=2 ):
    #creating naked image(just taking image from camera)
    frame = results.orig_img.copy()
    #checking whether or not there are some bb, if not, just show regular image
    if results.boxes is None or len(results.boxes)==0:
        cv2.imshow("YOLO livestream", frame)
    else:
    #if some bb was detecter using formulas we are calculating the center of the bb
        
        original_cords = results.boxes.data
        #calculating the center of the bounding box. 
        x_center = (original_cords[0][0]+(original_cords[0][2] -original_cords[0][0])/2)
        y_center = (original_cords[0][1]+(original_cords[0][3]- original_cords[0][1])/2)
        center_of_bb = [int(x_center.item()),int(y_center.item())]
        #drawing dot over the "frame" image
        cv2.circle(frame, center=(center_of_bb[0], center_of_bb[1]), radius=1, color=(0, 255, 0), thickness=5)
    #showing what
    cv2.imshow("YOLO livestream", frame)
    #handaling exceptions
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
