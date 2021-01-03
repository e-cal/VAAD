from detector import *


def main():
    detector = FaceDetector()
    box, prob, ld = detector.detect()
    print(f"face detected with {(prob[0] * 100):.2f}% certainty")
    print("No face detected")


main()
