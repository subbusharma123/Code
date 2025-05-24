import cv2
import numpy as np
import time
from collections import defaultdict, Counter

# Model paths - update to your actual file locations
FACE_PROTO = r"C:\Users\subra\Documents\GitHub\m\Camera\file\haarcascade_frontalface_default.xml"
AGE_PROTO = r"C:\Users\subra\Documents\GitHub\m\Camera\file\age_deploy.prototxt"
AGE_MODEL = r"C:\Users\subra\Documents\GitHub\m\Camera\file\age_net.caffemodel\age_net.caffemodel"
GENDER_PROTO = r"C:\Users\subra\Documents\GitHub\m\Camera\file\gender_deploy.prototxt"
GENDER_MODEL = r"C:\Users\subra\Documents\GitHub\m\Camera\file\gender_net.caffemodel\gender_net.caffemodel"
EMOTION_MODEL = r"C:\Users\subra\Documents\GitHub\m\Camera\file\emotion-ferplus-8.onnx"

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
EMOTION_LIST = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']

# Load models
face_cascade = cv2.CascadeClassifier(FACE_PROTO)
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
emotion_net = cv2.dnn.readNetFromONNX(EMOTION_MODEL)

def ensure_color(img):
    # Convert grayscale to BGR if needed
    if img.ndim == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Convert BGRA to BGR if needed
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def preprocess_age_gender(face_img):
    face_img = ensure_color(face_img)
    blob = cv2.dnn.blobFromImage(face_img, scalefactor=1.0, size=(227, 227),
                                 mean=(78.426, 87.769, 114.896), swapRB=False)
    # print("Age/Gender blob shape:", blob.shape)  # (1,3,227,227)
    return blob

def preprocess_emotion(face_img):
    face_img = ensure_color(face_img)
    face_resized = cv2.resize(face_img, (64, 64))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    blob = cv2.dnn.blobFromImage(face_rgb, scalefactor=1.0/255, size=(64, 64),
                                 mean=(0, 0, 0), swapRB=False, crop=False)
    # print("Emotion blob shape:", blob.shape)  # (1,3,64,64)
    return blob

def predict_age_gender(face_img):
    blob = preprocess_age_gender(face_img)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_LIST[age_preds[0].argmax()]
    return gender, age

def predict_emotion(face_img):
    blob = preprocess_emotion(face_img)
    emotion_net.setInput(blob)
    preds = emotion_net.forward()
    emotion = EMOTION_LIST[preds[0].argmax()]
    return emotion


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    start_time = time.time()
    duration = 15  # seconds to lock predictions

    predictions = defaultdict(lambda: {'gender': [], 'age': [], 'emotion': []})

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        elapsed = time.time() - start_time

        for (x, y, w, h) in faces:
            # Clamp bounding box inside frame
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            face_img = frame[y1:y2, x1:x2].copy()

            key = (x // 10, y // 10, w // 10, h // 10)

            if elapsed < duration:
                gender, age = predict_age_gender(face_img)
                emotion = predict_emotion(face_img)
                predictions[key]['gender'].append(gender)
                predictions[key]['age'].append(age)
                predictions[key]['emotion'].append(emotion)

                display_gender = gender
                display_age = age
                display_emotion = emotion
            else:
                display_gender = Counter(predictions[key]['gender']).most_common(1)[0][0] if predictions[key]['gender'] else "N/A"
                display_age = Counter(predictions[key]['age']).most_common(1)[0][0] if predictions[key]['age'] else "N/A"
                display_emotion = Counter(predictions[key]['emotion']).most_common(1)[0][0] if predictions[key]['emotion'] else "N/A"

            label = f"{display_gender}, {display_age}, {display_emotion}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            text_pos = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20)
            cv2.putText(frame, label, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Age, Gender and Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
