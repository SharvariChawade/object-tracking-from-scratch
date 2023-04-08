from ultralytics import YOLO
import cv2
import numpy as np
import math

import torch
print(torch.backends.mps.is_available())

cap = cv2.VideoCapture("Test Video.mp4")

model = YOLO("yolov8m.pt")
count = 0
center_points_prev_frame = []

tracking_object = {}
track_id = 0

while True:

    ret,frame = cap.read()
    count += 1
    if not ret:
        break

    center_points_curr_frame = []
    results = model(frame,device="mps")
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(),dtype="int")
    classes = np.array(result.boxes.cls.cpu(),dtype="int")
    
    for cls,bbox in zip(classes,bboxes):
        if result.names[cls] == 'car':
            (x,y,x2,y2) = bbox
            cx = int((x+x2)/2)
            cy = int((y+y2)/2)
            center_points_curr_frame.append((cx,cy))
            print("frame no",count,"coords",x,y,x2,y2)
            
            cv2.rectangle(frame,(x,y),(x2,y2),(0,0,255),2)
            # cv2.putText(frame,str(cls),(x,y-5),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
            # cv2.putText(frame,result.names[cls],(x,y-5),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
    if count <= 2:
        for pt in center_points_curr_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0]-pt[0],pt2[1]-pt[1])
                if(distance<30):
                    tracking_object[track_id] = pt
                    track_id += 1
    else:

        tracking_objects_copy = tracking_object.copy()
        center_points_cur_frame_copy = center_points_curr_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Update IDs position
                if distance < 30:
                    tracking_object[object_id] = pt
                    object_exists = True
                    if pt in center_points_curr_frame:
                        center_points_curr_frame.remove(pt)
                    continue

            # Remove IDs lost
            if not object_exists:
                tracking_object.pop(object_id)

        # Add new IDs found
        for pt in center_points_curr_frame:
            tracking_object[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_object.items():
        cv2.circle(frame,pt,5,(0,0,255),-1)
        cv2.putText(frame,str(object_id),(pt[0],pt[1]-7),0,1,(0,0,255),2)

        print("tracking obj",tracking_object)

                

    print("Current frame:", center_points_curr_frame)
    print("previous frame: ", center_points_prev_frame)
    cv2.imshow("Img",frame)
    center_points_prev_frame=center_points_curr_frame.copy()
    key = cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()