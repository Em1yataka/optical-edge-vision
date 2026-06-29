import cv2


class Camera:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

    def read(self):
        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def release(self):
        self.cap.release()