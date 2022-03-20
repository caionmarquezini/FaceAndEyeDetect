import cv2
import sys

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        eyes = eyeCascade.detectMultiScale(
            roi,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x2, y2, w2, h2) in eyes:
            if eyes.size >= 2:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                for eye in eyes:
                    if eye[2] * eye[3] < 9000:
                        cv2.rectangle(frame, (x2 + x, y2 + y), (x + x2 + w2, y + y2 + h2), (0, 0, 255), 1)


    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
