import numpy as np
import cv2

#Capture video
cap = cv2.VideoCapture(0)

while(True):
    #Capture frame-by-frame
    ret, frame = cap.read()
    ret, frame2 = cap.read()

    #Varible to smoother the shape detection
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    cnn = cv2.Canny(gray, 10, 150)
    cnn = cv2.dilate(cnn, None, iterations=1)
    cnn = cv2.erode(cnn, None, iterations=1)
    cnts,_ = cv2.findContours(cnn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame2, cnts, -1, (0,255,0), 2)

    #Shape detection
    for c in cnts:
      area = cv2.contourArea(c)
      perimeter = cv2.arcLength(c,True)
      epsilon = 0.01*cv2.arcLength(c,True)
      approx = cv2.approxPolyDP(c,epsilon,True)
      x,y,w,h = cv2.boundingRect(approx)
      #Triangulo & perimetro
      if len(approx)==3:
        cv2.putText(frame2,'Triangulo: ' + str(int(perimeter)), (x,y-5),1,1.5,(0,255,0),2)
      #Cuadrado & perimetro
      if len(approx)==4:
        aspect_ratio = float(w)/h
        print('aspect_ratio= ', aspect_ratio)
        if aspect_ratio == 1:
          cv2.putText(frame2,'Cuadrado: ' + str(int(perimeter)), (x,y-5),1,1.5,(0,255,0),2)
        #Rectangulo & perimetro  
        else:
          cv2.putText(frame2,'Rectangulo: ' + str(int(perimeter)), (x,y-5),1,1.5,(0,255,0),2)
      #Pentagono & perimetro    
      if len(approx)==5:
        cv2.putText(frame2,'Pentagono: ' + str(int(perimeter)), (x,y-5),1,1.5,(0,255,0),2)
      #Circulo & perimetro  
      if len(approx)>10:
        cv2.putText(frame2,'Circulo: ' + str(int(perimeter)), (x,y-5),1,1.5,(0,255,0),2)

    #Color segmentation
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # Red Color
    lower_blue = np.array([161,155,84])
    upper_blue = np.array([179,255,255])
    red_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    red = cv2.bitwise_and(frame, frame, mask = red_mask)

    #Green color
    lower_green = np.array([25,52,72])
    upper_green = np.array([102,255,255])
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    green = cv2.bitwise_and(frame, frame, mask = green_mask)

    #Print results
    win = cv2.hconcat([red, green]) 
    cv2.imshow('Color Segmentation', win)
    cv2.imshow('Original', frame)
    cv2.imshow('Contour & Area', frame2)
    cv2.drawContours(frame2, [approx], 0, (0,255,0),2)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()