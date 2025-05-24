import cv2
emotion_net = cv2.dnn.readNetFromONNX(r"C:\Users\subra\Documents\GitHub\m\Camera\file\emotion-ferplus-8.onnx")

img = cv2.imread(r"C:\Users\subra\Documents\GitHub\m\Camera\file\dummy.jpg")  # Make sure this is color image

def preprocess_emotion(face_img):
    if face_img.ndim == 2 or face_img.shape[2] == 1:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
    face_resized = cv2.resize(face_img, (64, 64))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    blob = cv2.dnn.blobFromImage(face_rgb, scalefactor=1.0/255, size=(64, 64),
                                 mean=(0, 0, 0), swapRB=False, crop=False)
    return blob

blob = preprocess_emotion(img)
print(blob.shape)  # Should be (1, 3, 64, 64)
emotion_net.setInput(blob)
preds = emotion_net.forward()
print(preds)
