import numpy as np
import cv2

cap = cv2.VideoCapture(1)

image_count=0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #define region of interest
    roi=frame[100:350, 100:350]
    cv2.rectangle(frame,(100,100),(350,350),(0,255,0),2) 

    image_name="dataset\\medium\\"+ str(image_count) +".jpg"

    cv2.imwrite(image_name,roi)
    image_count+=1



    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()