import cv2
import numpy as np
import tensorflow as tf
import time
import pygame 

# --- إعدادات ---
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = 'haarcascade_eye.xml'
MODEL_PATH = 'my_best_eye_model.h5'
SOUND_FILE_PATH = 'WAKY WAKY.wav' 


pygame.mixer.init()
try:
    alarm_sound = pygame.mixer.Sound(SOUND_FILE_PATH)
    print("Sound file loaded successfully!")
except:
    print(f"Error: Could not load sound file '{SOUND_FILE_PATH}'. Make sure it's in the folder.")
    exit()


print("Loading detection cascades...")
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

if face_cascade.empty() or eye_cascade.empty():
    print("Error: Could not load Haar Cascades.")
    exit()

print("Loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("All loaded! Starting camera...")

def prepare_eye_image(eye_img_gray):
    resized = cv2.resize(eye_img_gray, (64, 64))
    img_array = np.expand_dims(resized, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array / 255.0


drowsy_start_time = None
ALARM_TRIGGER_TIME = 2.0 

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    status = "No Face"
    color = (200, 200, 200)
    is_curr_drowsy = False 

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
        
        eyes_detected = False
        prediction_score = 0

        for (ex, ey, ew, eh) in eyes:
            eyes_detected = True
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            eye_crop = roi_gray[ey:ey+eh, ex:ex+ew]
            
            processed_eye = prepare_eye_image(eye_crop)
            prediction = model.predict(processed_eye, verbose=0)
            prediction_score = prediction[0][0]
            break 

        if eyes_detected:
            
            if prediction_score > 0.5:
                status = "Awake :)"
                color = (0, 255, 0)
                is_curr_drowsy = False
            else:
                status = "DROWSY!"
                color = (0, 0, 255)
                is_curr_drowsy = True
        else:
            status = "No Eyes"
            color = (0, 255, 255)
            is_curr_drowsy = False

   
    if is_curr_drowsy:
        if drowsy_start_time is None:
            drowsy_start_time = time.time()
        
        elapsed_time = time.time() - drowsy_start_time
        
        cv2.putText(frame, f"Time: {elapsed_time:.1f}s", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if elapsed_time >= ALARM_TRIGGER_TIME:
            status = "WAKY WAKY!!!"
       
            if not pygame.mixer.get_busy():
                alarm_sound.play()
            
    else:
        drowsy_start_time = None
      
        alarm_sound.stop() 

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0,0,0), -1)
    cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    cv2.imshow('Pro Drowsiness Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()