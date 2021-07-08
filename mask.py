import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2 as cv2
import cvlib as cv
import numpy as np

model = load_model('model_V3.h5')


classes = ["Masked", "Incorrectly masked", "Unmasked"]


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

        face_crop = cv2.resize(face_crop, (128, 128))
        face_crop = face_crop.astype(float)/255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        conf = model.predict(face_crop)[0]
        (mask, incorrect, unmask) = conf
        print((mask, incorrect, unmask))

        if mask == max(conf):
            idx = 0
        elif incorrect == max(conf):
            idx = 1
        else:
            idx = 2
        label = classes[idx]

        cv2.putText(frame, label, (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
model
capture.release()
cv2.destroyAllWindows()
