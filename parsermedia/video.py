import cv2


class VideoStream:

    def __init__(self, video_path):
        # TODO Ensure the file given in agument is a video file (check extension at least)
        # If not a video file, raise an exception to alert user about the issue
        self.video_path = video_path
        self.is_on = False
        self.cap = None

    def get_images(self):
        while (self.cap.isOpened()) and self.is_on:
            ret, frame = self.cap.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                yield image
            else:
                self.close()
                break

    def open(self):
        self.is_on = True
        self.cap = cv2.VideoCapture(self.video_path)

    def close(self):
        self.is_on = False
        self.cap.release()
