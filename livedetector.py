import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# 4 MUDRAS
model = tf.keras.models.load_model('mudra_model.keras')

class_names = ['Alapadmam', 'Chandrakala', 'Kapitham', 'Shikaram']

CONF_THRESHOLD = 60.0
PREDICTION_WINDOW = 7
recent_preds = deque(maxlen=PREDICTION_WINDOW)

def predict_mudra_live(frame):
    h, w = frame.shape[:2]
    crop = frame[h//4:3*h//4, w//4:3*w//4]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (224, 224))
    crop = crop.astype(np.float32)  
    crop = np.expand_dims(crop, 0)
    pred = model.predict(crop, verbose=0)
    probs = pred[0]
    top_idx = int(np.argmax(probs))
    return class_names[top_idx], 100 * float(probs[top_idx]), probs

cap = cv2.VideoCapture(0)
cap.set(3, 640); cap.set(4, 480)

print(" LIVE Detection: Alapadmam | Chandrakala | Kapitham | Shikaram")
print("'q' = quit, 's' = screenshot")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    mudra, conf, probs = predict_mudra_live(frame)

    recent_preds.append(np.argmax(probs))
    stable_idx = int(np.bincount(np.array(recent_preds), minlength=len(class_names)).argmax())
    stable_mudra = class_names[stable_idx]

    if conf < CONF_THRESHOLD:
        label = f'Not sure ({conf:.0f}%)'
        color = (0, 165, 255)
    else:
        label = f'{stable_mudra}: {conf:.0f}%'
        color = (0, 255, 0)
    
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = w//4, h//4, 3*w//4, 3*h//4

    cv2.putText(frame, label,
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    cv2.imshow('LIVE Mudra Detector', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('s'):
        cv2.imwrite(f'screenshot_{stable_mudra}.jpg', frame)
        print(f"📸 Saved: screenshot_{stable_mudra}.jpg")

cap.release()
cv2.destroyAllWindows()
