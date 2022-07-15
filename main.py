import cv2 
from deepface import DeepFace
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0) # might also be 1

# check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret,frame = cap.read()# read one image from a video

    result = DeepFace.analyze(frame,enforce_detection=False, actions=['emotion','age'])
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    # draw a rectangle around the faces
    for(x,y,w,h)in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # use putText()method for inserting text on video
    cv2.putText(frame,
                "Emotion: "+result['dominant_emotion'],
                (50,50),
                font,2,
                (0,0,255),
                2)
    cv2.putText(frame,
                "Age: "+str(result['age']),
                (50,450),
                font,2,
                (255,0,0),
                2)

    # might change the window size
    imS = cv2.resize(frame, (1350, 630))
    cv2.imshow('Original video',imS)

    # press q to quit
    if cv2.waitKey(2)&0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllwindows()
