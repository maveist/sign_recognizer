import cv2


class CameraStream:

    def __init__(self):
        self.is_on = False
        self.cap = None

    def get_images(self):
        while self.cap.isOpened() and self.is_on:
            success, image = self.cap.read()
            if not success:
                raise Exception("Bad reading of input camera")
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            yield image

    def open(self):
        self.is_on = True
        self.cap = cv2.VideoCapture(0)

    def close(self):
        self.is_on = False
        self.cap.release()
