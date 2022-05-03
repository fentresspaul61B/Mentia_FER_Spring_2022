print("Running FER")
import cv2
from time import sleep
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import pickle
from PIL import ImageFont, ImageDraw, Image
font_path = "kredit back.ttf"
font = ImageFont.truetype(font_path, 80)



# model_path = "CNN_model.pkl"
#
# model = pickle.load( open( model_path, "rb") )
model_path = "CNN_FER.h5"

classifier = load_model(model_path)
# emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
emotion_labels = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# https://www.youtube.com/watch?v=Bb4Wvl57LIk

# Use on video:
video_path = "Branda_schmitz_64.mp4"
# video_capture = cv2.VideoCapture(video_path)

# Use on webcam
# video_capture = cv2.VideoCapture(0)



face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# face_classifier = cv2.CascadeClassifier(r'C:\Users\Admin\Desktop\PythonProject\EmotionDetectionCNN\haarcascade_frontalface_default.xml')

def video_test(video_path: str):
    """
    Arguments:
        None.
    Returns:
        Nothing.
    Purpose:
        Run the FER on a test video.
    Algorithm:
        1. Open video
        2. Start forever loop.
            3. Detect faces using pretrained CV2 face_classifier
            4. Draw box around face object.
            5. Collect data inside face box.
            6. Make prediction inside face box.
            7. Write preidction to screen.
    """
    video_capture = cv2.VideoCapture(video_path)

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        labels = []
        cv2.imshow("Window", frame)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # parameters: scaleFactor=1.3, minNeighbors=8
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=8)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                print(label + " with: " + str(max(prediction)) + " confidence.")
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Faces',(30,80),1,cv2.FONT_HERSHEY_SIMPLEX,(0,255,0),2)


        cv2.imshow('Emotion Detector',frame)

        #This breaks on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def web_cam():
    """
    Arguments:
        None.
    Returns:
        Nothing.
    Purpose:
        Run the FER on webcam.
    Algorithm:
        1. Open Webcam
        2. Start forever loop.
            3. Detect faces using pretrained CV2 face_classifier
            4. Draw box around face object.
            5. Collect data inside face box.
            6. Make prediction inside face box.
            7. Write preidction to screen.
    """
    video_capture = cv2.VideoCapture(0)
    cv2.namedWindow("Window")

    while True:
        ret, frame = video_capture.read()
        labels = []
        cv2.imshow("Window", frame)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                print(prediction)
                label = emotion_labels[prediction.argmax()]
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Faces',(30,80),1,cv2.FONT_HERSHEY_SIMPLEX,(0,255,0),2)


        cv2.imshow('Emotion Detector',frame)

        #This breaks on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


# video_test(video_path)

web_cam()
