import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2 as cv2
import cvlib as cv
import numpy as np

model = load_model('my_model.h5')


classes = ["unmasked", "masked"]


capture = cv2.VideoCapture(0)

while capture.isOpened():

    status, frame = capture.read()

    face, confidence = cv.detect_face(frame)

    for idx, f in enumerate(face):

        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        rectangle = cv2.rectangle(
            frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        face_crop = np.copy(frame[startY:endY, startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        face_crop = cv2.resize(face_crop, (300, 300))
        face_crop = face_crop.astype(float)/255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        conf = model.predict(face_crop)[0]

        if conf > 0.6:
            idx = 0
        else:
            idx = 1
        label = classes[idx]

        cv2.putText(frame, label, (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('uwu', frame)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
model
capture.release()
cv2.destroyAllWindows()
