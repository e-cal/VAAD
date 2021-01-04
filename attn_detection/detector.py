import cv2
from facenet_pytorch import MTCNN


class FaceDetector(object):
    def __init__(self):
        self.mtcnn = MTCNN()

    def detect(self):
        video = cv2.VideoCapture(0)
        ret, frame = video.read()
        box, prob, ld = self.mtcnn.detect(frame, landmarks=True)
        if not type(box) == type(None):
            video.release()
            # return box, prob[0], ld
            return prob[0]
        video.release()

    def overlay(self, frame, box, prob):
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

    def run(self):
        video = cv2.VideoCapture(0)
        while True:
            ret, frame = video.read()
            box, prob, ld = self.mtcnn.detect(frame, landmarks=True)
            if not type(box) == type(None):
                print(f"facebox: {box}\nprob: {prob[0]}\n")
                self.overlay(frame, box, prob)
            else:
                print("No face detected")
            cv2.imshow('Face Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = FaceDetector()
    detector.run()
    prob = detector.detect()
    if prob == None:
        print("No face detected")
    else:
        print(f"face detected with {(prob * 100):.2f}% certainty")
