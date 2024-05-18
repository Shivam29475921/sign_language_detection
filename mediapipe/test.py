from HandDetector import HandDetector
from FrameCounter import FrameCounter
from keras.models import load_model
import cv2
import numpy as np

# custom model for prediction
customModel = load_model('model.keras')

# objects
feed = cv2.VideoCapture(0)  # video feed
detector = HandDetector()  # custom mediapipe model
fps_count = FrameCounter()  # fps tracker

# constants
IMG_SIZE = 175
classes = [chr(i) for i in range(65, 91)]

# variables
PREDICTION = '-'

while True:
    success, frame = feed.read()
    # copy of the actual image but without hand points for clearer output
    imgOutput = frame.copy()
    
    is_hand_visible, frame = detector.find_hands(frame, draw=True)  # returns the tracking points of hands
    height, width, channel = frame.shape
    fps_count.calc_fps()  # calculates the fps
    
    temp_list = detector.find_cords(frame, emphasize_points=False)  # returns the co-ordinates of hands
    
    if is_hand_visible:
        
        # cropped image dimensions
        bottom_lim = int(np.clip(max([x[2] for x in temp_list]) + 30, 0, height))
        top_lim = int(np.clip(min([x[2] for x in temp_list]) - 30, 0, height))
        left_lim = int(np.clip(min(x[1] for x in temp_list) - 30, 0, width))
        right_lim = int(np.clip(max(x[1] for x in temp_list) + 30, 0, width))
        
        # crop the image
        cropped_img = frame[top_lim:bottom_lim, left_lim:right_lim]
        
        # placing the cropped image on a fixed white bg for same sized dataset
        imgWhite = np.ones((IMG_SIZE, IMG_SIZE), np.uint8) * 255
        
        # if the cropped image is smaller than the fixed dataset size
        if cropped_img.shape[0] <= IMG_SIZE and cropped_img.shape[1] <= IMG_SIZE:
            # converting the image to gray to reduce computation
            imgWhite[0:cropped_img.shape[0], 0:cropped_img.shape[1]] = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            # preparing the image for prediction
            scaled_imgWhite = (imgWhite / 255).reshape((-1, IMG_SIZE, IMG_SIZE))
            # the index of max value of our predict output is the required index of our classes list
            PREDICTION = classes[np.argmax(customModel.predict(scaled_imgWhite))]
            # display formatting
            cv2.rectangle(imgOutput, (left_lim + 10, top_lim - 40), (left_lim + 60, top_lim), (255, 100, 255),
                          cv2.FILLED)
            cv2.putText(imgOutput, PREDICTION, ((left_lim + 20), (top_lim - 10)), cv2.FONT_HERSHEY_TRIPLEX, 1,
                        (255, 255, 255), 1)
            cv2.rectangle(imgOutput, (left_lim, top_lim), (right_lim, bottom_lim), (255, 100, 255), 2)
            cv2.imshow('scaled', imgWhite)
    
    cv2.putText(imgOutput, f"FPS: {str(int(fps_count.fps))}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 255), 2)
    
    cv2.imshow("Feed", imgOutput)
    key = cv2.waitKey(1)
    # press q to exit
    if key == ord('q'):
        break
