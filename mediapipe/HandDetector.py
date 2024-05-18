import cv2
import mediapipe as mp


# custom mediapipe hand detector class
class HandDetector:
    def __init__(self, mode=False, max_hands=1, model_comp=1, detection_confidence=0.5,
                 tracking_confidence=0.5):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(mode, max_hands, model_comp, detection_confidence, tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.result = None
        self.hand_cords = []

    # returns whether the hand is visible and also the drawn points
    def find_hands(self, img, draw=True):
        is_hand_visible = False
        # converting the BGR image to RGB as the hand tracking module operates on RGB images
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(img_rgb)
        if self.result.multi_hand_landmarks:
            is_hand_visible = True
            # in case of two-handed landmarks
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return is_hand_visible, img

    # returns the co-ordinates of each of the 20 hand landmarks
    def find_cords(self, img, emphasize_points=True):
        temp_list = []
        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                for l_id, landmark in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    # landmarks are in the range [0,1] so we upscale it using the dimensions of the image
                    px, py = int(landmark.x * w), int(landmark.y * h)
                    if emphasize_points:  # drawing circles to emphasize each point
                        cv2.circle(img, (px, py), 15, (255, 0, 255), 5, cv2.FILLED)
                    self.hand_cords.append([l_id, px, py])
                    temp_list.append([l_id, px, py])

        return temp_list
