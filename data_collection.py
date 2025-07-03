import cv2
import numpy as np
import math
import time
import mediapipe as mp

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
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

offset = 20  # Offset for cropping
imgSize = 300  # Size to resize cropped image

folderPath = "Data/NO"  # Folder to collect data for training
counter = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Add this line to draw hand skeleton
            mp.solutions.drawing_utils.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
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
                
                aspectRatio = h/w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    if wCal > imgSize:  # Add bounds check
                        wCal = imgSize
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal)/2)
                    imgWhite[:, wGap:wGap+wCal] = imgResize
                
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    if hCal > imgSize:  # Add bounds check
                        hCal = imgSize
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal)/2)
                    imgWhite[hGap:hGap+hCal, :] = imgResize

                cv2.imshow("Cropped Image", imgCrop)
                cv2.imshow("White Image", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    
    # Save the image when 's' is pressed
    if key == ord('s'):
        if 'imgWhite' in locals():  # Check if image exists
            counter += 1
            cv2.imwrite(f'{folderPath}/Image_{time.time()}.jpg', imgWhite)
            print(f"Saved image {counter}")
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()