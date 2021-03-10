import cv2
from threading import Thread
from queue import Queue


class CameraStream:

    def __init__(self):
        self.is_on = False
        self.cap = None
        self.queue = Queue(maxsize=1)

    def get_images(self):
        while self.is_on:
            yield self.queue.get()

    def capture_image(self):
        self.cap = cv2.VideoCapture(0)
        while self.cap.isOpened() and self.is_on:
            success, image = self.cap.read()
            if not success:
                raise Exception("Bad reading of input camera")
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            if not self.queue.full():
                self.queue.put(image)

    def open(self):
        self.is_on = True
        thrd = Thread(target=self.capture_image, args=())
        thrd.daemon = True
        thrd.start()

    def close(self):
        self.is_on = False
        self.cap.release()
