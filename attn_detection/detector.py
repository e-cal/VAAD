import cv2
from facenet_pytorch import MTCNN


class FaceDetector():
    def __init__(self, cam=None):
        self.mtcnn = MTCNN()
        # self.app = False
        # if cam:
        #     self.cam = cam
        # else:
        #     self.cam = cv2.VideoCapture(2)

    def release(self):
        self.cam.release()

    def detect(self, frame):
        box, prob = self.mtcnn.detect(frame)
        if not box is None:
            return zip(box, prob)
        return None

    def overlay(self, frame):
        # _, frame = self.cam.read()
        if frame is None:
            return
        det_data = self.detect(frame)
        if det_data is None:
            return frame
        for box, prob in det_data:
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
        return frame

    def run(self):
        cam = cv2.VideoCapture(0)
        while True:
            ret, frame = cam.read()
            box, prob, ld = self.mtcnn.detect(frame, landmarks=True)
            if not type(box) == type(None):
                print(f"facebox: {box}\nprob: {prob[0]}\n")
                for box, prob in zip(box, prob):
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
            else:
                print("No face detected")
            cv2.imshow('Face Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = FaceDetector()
    detector.run()
    prob = detector.detect()
    if prob == None:
        print("No face detected")
    else:
        print(f"face detected with {(prob * 100):.2f}% certainty")
