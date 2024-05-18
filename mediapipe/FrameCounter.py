import time


class FrameCounter:

    def __init__(self):
        self.p_time = 0.0
        self.c_time = 0.0
        self.fps = 0.0

    def calc_fps(self):
        self.c_time = time.time()
        self.fps = 1 / (self.c_time - self.p_time)  # calculating fps
        self.p_time = self.c_time
