from HandDetector import HandDetector
from FrameCounter import FrameCounter
import cv2
import numpy as np
import time

# objects
feed = cv2.VideoCapture(0)  # video feed
detector = HandDetector()  # custom mediapipe model
fps_count = FrameCounter()  # fps tracker

# variables
counter = 0
# constants
IMG_SIZE = 175

while True:
    success, frame = feed.read()
    is_hand_visible, frame = detector.find_hands(frame, draw=True)  # returns the tracking points of hands
    height, width, level = frame.shape
    fps_count.calc_fps()  # calculates the fps

    temp_list = detector.find_cords(frame, emphasize_points=False)  # returns the co-ordinates of hands

    if is_hand_visible:
        # cropped image dimensions
        bottom_lim = np.clip(max([x[2] for x in temp_list]) + 30, 0, height)
        top_lim = np.clip(min([x[2] for x in temp_list]) - 30, 0, height)
        left_lim = np.clip(min(x[1] for x in temp_list) - 30, 0, width)
        right_lim = np.clip(max(x[1] for x in temp_list) + 30, 0, width)

        # crop the image
        cropped_img = frame[top_lim:bottom_lim, left_lim:right_lim]
        imgWhite = np.ones((IMG_SIZE, IMG_SIZE), np.uint8) * 255

        if cropped_img.shape[0] <= IMG_SIZE and cropped_img.shape[1] <= IMG_SIZE:
            imgWhite[0:cropped_img.shape[0], 0:cropped_img.shape[1]] = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            cv2.imshow("cropped", imgWhite)

    cv2.putText(frame, f"FPS: {str(int(fps_count.fps))}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 255), 2)
    cv2.imshow("Feed", frame)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f"../Data/C/Image_{time.time()}.jpg", imgWhite)
        print(counter)
