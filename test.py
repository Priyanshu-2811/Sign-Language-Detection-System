import cv2
import numpy as np
import math
import time
import tensorflow as tf
import mediapipe as mp

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

def get_hand_bbox(landmarks, w, h):
    x_coords = [lm.x * w for lm in landmarks.landmark]
    y_coords = [lm.y * h for lm in landmarks.landmark]
    return int(min(x_coords)), int(min(y_coords)), int(max(x_coords) - min(x_coords)), int(max(y_coords) - min(y_coords))

model = tf.keras.models.load_model("Model/keras_model.h5")

offset = 20  # Offset for cropping
imgSize = 224  # Size to resize cropped image


# Load class labels
with open("Model/labels.txt", "r") as f:
    labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    
    # Process with MediaPipe instead of cvzone
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get hand bounding box
            imgH, imgW, _ = img.shape
            x, y, w, h = get_hand_bbox(hand_landmarks, imgW, imgH)
            
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            
            # Calculate crop boundaries with bounds checking
            y1 = max(0, y - offset)
            y2 = min(imgH, y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(imgW, x + w + offset)
            
            # Only crop and display if we have valid dimensions
            if y2 > y1 and x2 > x1:
                imgCrop = img[y1:y2, x1:x2]
                
                aspectRatio= h/w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    if wCal > imgSize:  # Add bounds check
                        wCal = imgSize
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal)/2)
                    imgWhite[:, wGap:wGap+wCal] = imgResize  # Fixed indexing
                
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    if hCal > imgSize:  # Add bounds check
                        hCal = imgSize
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal)/2)
                    imgWhite[hGap:hGap+hCal, :] = imgResize  # Fixed indexing
                    
                # Prepare the image for prediction
                img_array = np.expand_dims(imgWhite, axis=0)
                img_array = img_array.astype(np.float32) / 255.0
                prediction = model.predict(img_array, verbose=0)
                index = np.argmax(prediction)
                confidence = np.max(prediction)

                # Display prediction with confidence
                if confidence > 0.7:  # Only show confident predictions
                    if labels[index] == "YES":
                        box_color = (0, 255, 0)  # Green for YES
                    elif labels[index] == "NO":
                        box_color = (0, 0, 255)  # Red for NO
                    else:
                        box_color = (128, 128, 128)  # Gray for unknown (shouldn't happen)
                    
                    # Display prediction with appropriate color
                    cv2.rectangle(imgOutput, (x1, y1-50), (x1+120, y1), box_color, -1) 
                    cv2.putText(imgOutput, labels[index], (x1+5, y1-30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  
                    cv2.putText(imgOutput, f'{confidence:.2f}', (x1+5, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1) 
                    
                    print(f"Prediction: {labels[index]} ({confidence:.2f})")

                cv2.imshow("Cropped Image", imgCrop)
                cv2.imshow("White Image", imgWhite)

    cv2.imshow("Image", imgOutput)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break

#Close the camera and windows
cap.release()
cv2.destroyAllWindows()