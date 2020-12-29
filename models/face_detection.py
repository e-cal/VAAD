import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN


class FaceDetector(object):
    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def overlay(self, frame, boxes, probs):
        print(f"boxes: {boxes}\tprobs: {probs}")
        for box, prob in zip(boxes, probs):
            print("here")
            cv2.rectangle(frame,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          (0, 0, 255),
                          thickness=2)

            cv2.putText(frame,
                        str(prob),
                        (box[2], box[3]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2,
                        cv2.LINE_AA)

    def run(self):
        video = cv2.VideoCapture(0)

        while True:
            ret, frame = video.read()
            try:
                boxes, probs, ld = self.mtcnn.detect(frame, landmarks=True)
                self.overlay(frame, boxes, probs)
            except:
                pass

            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()


mtcnn = MTCNN()
detector = FaceDetector(mtcnn)
detector.run()
